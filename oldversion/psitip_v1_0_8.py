# Python Symbolic Information Theoretic Inequality Prover
# Copyright (C) 2020  Cheuk Ting Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Python Symbolic Information Theoretic Inequality Prover
Version 1.0.8
Copyright (C) 2020  Cheuk Ting Li

Based on the general method of using linear programming for proving information 
theoretic inequalities described in the following work:

R. W. Yeung, "A new outlook on Shannon's information measures,"
 IEEE Trans. Inform. Theory, vol. 37, pp. 466-474, May 1991.

R. W. Yeung, "A framework for linear information inequalities,"
 IEEE Trans. Inform. Theory, vol. 43, pp. 1924-1934, Nov 1997.

Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
 IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.

This linear programming approach was used in the ITIP software developed by 
Raymond W. Yeung and Ying-On Yan ( http://user-www.ie.cuhk.edu.hk/~ITIP/ ).


Usage:

This library requires Python 3. Python 2 is not supported.
This library requires either PuLP, Pyomo or scipy for sparse linear programming.
Open an interactive Python console in the directory containing psitip.py.

from psitip import *
X, Y, Z = rv("X", "Y", "Z")

# Check for always true statements
# Use H(X + Y | Z + W) for the entropy H(X,Y|Z,W)
# Use I(X & Y + Z | W) for the mutual information I(X;Y,Z|W)
# As in the case in ITIP, a False return value does not mean the inequality is
# false. It only means that it cannot be deduced by Shannon-type inequalities.

bool(I(X & Y) - H(X + Z) <= 0)  # return True
bool(I(X & Y) == H(X) - H(X | Y))  # return True

# Each constraint (e.g. I(X & Y) - H(X + Z) <= 0 above) generates a Region object
# that can be intersected with each other (using the "&" operator). Casting a
# Region to bool returns whether the constraints in the Region are always satisfied.

# Use A >> B or A <= B to check if the conditions in A implies those in B
# The "<=" operator checks whether the region on the left is a subset of the 
# region on the right.
# Note that "<=" does NOT denote implication (which is the opposite direction).

bool(((I(X & Y) == 0) & (I(X+Y & Z) == 0)) >> (I(X & Z) == 0))  # return True


See test.py for more usage examples.


WARNING: Nested implication may produce incorrect results for some cases.
Use at your own risk. It is advisable to check whether the output auxiliary 
random variables are indeed valid.

"""

import itertools
import collections
import array
import fractions
import warnings
import types
import functools
import math
import time


try:
    import numpy
    import numpy.linalg
except ImportError:
    numpy = None

try:
    import scipy
    import scipy.sparse
    import scipy.optimize
    import scipy.spatial
    import scipy.special
except ImportError:
    scipy = None

try:
    import pulp
except ImportError:
    pulp = None

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
except ImportError:
    pyo = None


try:
    import cdd
except ImportError:
    cdd = None


try:
    import graphviz
except ImportError:
    graphviz = None
    
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches
    import matplotlib.lines
    import matplotlib.patheffects
    from matplotlib.collections import PatchCollection
except ImportError:
    matplotlib = None
    plt = None

try:
    import torch
    import torch.optim
except ImportError:
    torch = None

try:
    import torch.linalg
except ImportError:
    pass


try:
    import IPython
    import IPython.display
except ImportError:
    IPython = None


class LinearProgType:
    NIL = 0
    H = 1    # H(X_C)
    HC1BN = 2  # H(X | Y_C)
    
class PsiOpts:
    """ Options
    Attributes:
        solver      : The linear programming solver used
                      "scipy" : scipy.optimize.linprog
                      "pulp.cbc"  : PuLP with CBC
                      "pulp.glpk"  : PuLP with GLPK
                      "pyomo.glpk"  : Pyomo with GLPK
        
        str_style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
                      STR_STYLE_LATEX : I(X,Y;Z|W)
                      
        lptype      : Linear programming mode
                      LinearProgType.H : Classical
                      LinearProgType.HC1BN : Bayesian Network optimization
                      
        verbose_auxsearch : Print auxiliary RV search progress
        
        verbose_lp        : Print linear programming parameters
        
        rename_char : Char appended when renaming auxiliary RV
        
        eps         : Epsilon for float comparison
    """
    
    STR_STYLE_STANDARD = 1
    STR_STYLE_PSITIP = 2
    STR_STYLE_LATEX = 4
    STR_STYLE_LATEX_ARRAY = 1 << 5
    STR_STYLE_LATEX_FRAC = 1 << 6
    STR_STYLE_LATEX_QUANTAFTER = 1 << 7
    STR_STYLE_MARKOV = 1 << 8
    
    SFRL_LEVEL_SINGLE = 1
    SFRL_LEVEL_MULTIPLE = 2
    
    settings = {
        "ent_coeff": 1.0 / math.log(2.0),
        
        "eps": 1e-10,
        "eps_lp": 1e-5,
        "eps_check": 1e-6,
        "max_denom": 1000000,
        "max_denom_mul": 10000,
        
        "str_style": 2 + (1 << 8),
        "str_style_std": 1 + (1 << 8),
        "str_style_latex": 4 + (1 << 8),
        "str_style_repr": 2 + (1 << 8),
        "str_tosort": False,
        "str_lhsreal": True,
        "str_eqn_prefer_ge": False,
        "rename_char": "_",
        "fcn_suffix": "@@32768@@#fcn",
        
        "truth": None,
        "proof": None,
        
        "timer_start": None,
        "timer_end": None,
        
        "solver": "scipy",
        "lptype": LinearProgType.HC1BN,
        "lp_bounded": False,
        "lp_ubound": 1e4,
        "lp_eps": 1e-3,
        "lp_eps_obj": 1e-4,
        "lp_zero_cutoff": -1e-5,
        "fcn_mode": 1,
        "solver_scipy_maxsize": -1,
        "pulp_options": None,
        "pyomo_options": {},
        
        "simplify_enabled": True,
        "simplify_quick": False,
        "simplify_reduce_coeff": True,
        "simplify_remove_missing_aux": True,
        "simplify_aux": True,
        "simplify_aux_combine": True,
        "simplify_aux_commonpart": True,
        "simplify_aux_xor_len": 1,
        "simplify_aux_empty": False,
        "simplify_aux_recombine": False,
        "simplify_pair": True,
        "simplify_redundant": True,
        "simplify_redundant_full": True,
        "simplify_bayesnet": True,
        "simplify_expr_exhaust": True,
        "simplify_redundant_op": True,
        "simplify_union": False,
        "simplify_aux_relax": True,
        "simplify_sort": True,
        "simplify_aux_hull": False,
        "simplify_num_iter": 1,
        
        "istorch": False,
        "opt_optimizer": "SLSQP", #"sgd",
        "opt_eps_denom": 1e-8,
        "opt_eps_tol": 1e-7,
        "opt_learnrate": 0.04,
        "opt_learnrate2": 0.02,
        "opt_momentum": 0.0,
        "opt_num_iter": 800,
        "opt_num_iter2": 0,
        "opt_num_points": 1,
        "opt_eps_converge": 1e-9,
        "opt_num_hop": 1,
        "opt_hop_temp": 0.05,
        "opt_hop_prob": 0.2,
        "opt_alm_rho": 1.0,
        "opt_alm_rho_pow": 1.0,
        "opt_alm_step": 5,
        "opt_alm_penalty": 20.0,
        "opt_aux_card": 5,
        
        "imp_noncircular": True,
        "imp_noncircular_allaux": False,
        "imp_simplify": False,
        "prefer_expand": True,
        "tensorize_simplify": False,
        "eliminate_rays": False,
        "ignore_must": False,
        "forall_multiuse": True,
        "forall_multiuse_numsave": 128,
        "auxsearch_local": True,
        "auxsearch_leaveone": False,
        "auxsearch_leaveone_add_ineq": True,
        "auxsearch_aux_strengthen": True,
        "init_leaveone": True,
        "auxsearch_max_iter": 0,
        "auxsearch_op_casesteplimit": 16,
        "auxsearch_op_caselimit": 512,
        "auxsearch_sandwich": True,
        "auxsearch_sandwich_inc": False,
        "flatten_minmax_elim": False,
        "flatten_distribute": True,
        "flatten_distribute_multi": False,

        "bayesnet_semigraphoid_iter": 1000,
        
        "proof_enabled": False,
        "proof_nowrite": False,
        "proof_step_dualsum": True,
        "proof_step_simplify": False,
        
        "repr_simplify": True,
        "repr_check": False,
        "repr_latex": False,
        
        "venn_latex": False,
        
        "discover_hull_frac_enabled": True,
        "discover_hull_frac_denom": 1000,
        
        "latex_H": "H",
        "latex_I": "I",
        "latex_rv_delim": ",",
        "latex_cond": "|",
        "latex_sup": "\\sup",
        "latex_inf": "\\inf",
        "latex_max": "\\max",
        "latex_min": "\\min",
        "latex_exists": "\\exists",
        "latex_forall": "\\forall",
        "latex_quantifier_sep": ":\\,",
        "latex_indep": "{\\perp\\!\\!\\perp}",
        "latex_markov": "\\leftrightarrow",
        "latex_mi_delim": ";",
        "latex_list_bracket_l": "[",
        "latex_list_bracket_r": "]",
        "latex_matimplies": "\\Rightarrow",
        "latex_equiv": "\\Leftrightarrow",
        "latex_implies": "\\Rightarrow",
        "latex_times": "\\cdot",
        "latex_prob": "\\mathbf{P}",
        "latex_rv_empty": "\\emptyset",
        "latex_region_universe": "\\top",
        "latex_region_empty": "\\emptyset",
        "latex_contradiction": "\\bot",
        "latex_or": "\\vee",
        "latex_and": "\\wedge",
        "latex_infty": "\\infty",
        
        "latex_color": None,
        
        "verbose_lp": False,
        "verbose_lp_cons": False,
        "verbose_auxsearch": False,
        "verbose_auxsearch_step": False,
        "verbose_auxsearch_result": False,
        "verbose_auxsearch_cache": False,
        "verbose_auxsearch_step_cached": False,
        "verbose_auxsearch_op": False,
        "verbose_auxsearch_op_step": False,
        "verbose_auxsearch_op_detail": False,
        "verbose_auxsearch_op_detail2": False,
        "verbose_subset": False,
        "verbose_sfrl": False,
        "verbose_flatten": False,
        "verbose_eliminate": False,
        "verbose_eliminate_toreal": False,
        "verbose_semigraphoid": False,
        "verbose_proof": False,
        "verbose_discover": False,
        "verbose_discover_detail": False,
        "verbose_discover_outer": False,
        "verbose_discover_terms": False,
        "verbose_discover_terms_inner": False,
        "verbose_discover_terms_outer": False,
        "verbose_opt": False,
        "verbose_opt_step": False,
        "verbose_opt_step_var": False,
        "verbose_float_dp": 8,
        "verbose_commmodel": False,
        "verbose_codingmodel": False,
        
        "sfrl_level": 0,
        "sfrl_maxsize": 1,
        "sfrl_gap": ""
    }
    
    @staticmethod
    def set_setting_dict(d, key, value):
        if key == "sfrl":
            if value == "no":
                d["sfrl_level"] = 0
            else:
                d["sfrl_level"] = max(d["sfrl_level"], PsiOpts.SFRL_LEVEL_SINGLE)
                if value == "frl":
                    d["sfrl_gap"] = ""
                elif value.startswith("sfrl_gap."):
                    d["sfrl_gap"] = value[value.index(".") + 1 :]
                elif value == "sfrl_nogap":
                    d["sfrl_gap"] = "zero"
                    
        elif key == "str_style":
            
            d["str_style"] = iutil.convert_str_style(value)
                    
        elif key == "timer":
            if value is None:
                d["timer_start"] = None
                d["timer_end"] = None
            else:
                curtime = time.time() * 1000
                d["timer_start"] = curtime
                d["timer_end"] = curtime + float(value)
            
            
        elif key == "simplify_level":
            d["simplify_enabled"] = value >= 1
            d["simplify_quick"] = value <= 2
            # d["simplify_remove_missing_aux"] = True
            d["simplify_aux"] = value >= 4
            d["simplify_aux_combine"] = value >= 4
            d["simplify_aux_commonpart"] = value >= 5
            
            if value >= 9:
                d["simplify_aux_xor_len"] = 4
            elif value >= 7:
                d["simplify_aux_xor_len"] = 3
            else:
                d["simplify_aux_xor_len"] = 1
                
            d["simplify_aux_empty"] = value >= 6
            d["simplify_aux_recombine"] = value >= 10
            d["simplify_pair"] = value >= 2
            d["simplify_redundant"] = value >= 3
            d["simplify_redundant_full"] = value >= 4
            d["simplify_bayesnet"] = value >= 4
            d["simplify_redundant_op"] = value >= 4
            d["simplify_union"] = value >= 8
            d["simplify_num_iter"] = 2 if value >= 9 else 1

        elif key == "ent_base":
            d["ent_coeff"] = 1.0 / math.log(value)
                    
        elif key == "verbose_auxsearch_all":
            d["verbose_auxsearch"] = value
            d["verbose_auxsearch_step"] = value
            d["verbose_auxsearch_result"] = value
                    
        elif key == "verbose_proof":
            d["verbose_proof"] = value
            if value:
                d["proof_enabled"] = value
                if d["proof"] is None:
                    d["proof"] = ProofObj.empty()
                    
        elif key == "proof_enabled":
            d["proof_enabled"] = value
            if value:
                if d["proof"] is None:
                    d["proof"] = ProofObj.empty()
                    
        elif key == "pulp_solver":
            d["solver"] = "pulp.other"
            iutil.pulp_solver = value
                    
        elif key == "lptype":
            if value == "H":
                d[key] = LinearProgType.H
            elif isinstance(value, str):
                d[key] = LinearProgType.HC1BN
            else:
                d[key] = value
                    
        elif key == "truth":
            if value is None:
                d["truth"] = None
            else:
                d["truth"] = value.copy()
                    
        elif key == "truth_add":
            if value is not None:
                if d["truth"] is None:
                    d["truth"] = value.copy()
                else:
                    d["truth"] = d["truth"] & value
                    
        elif key == "proof_add":
            if d["proof"] is None:
                d["proof"] = ProofObj.empty()
            
            if PsiOpts.settings.get("verbose_proof", False):
                print(value.tostring(prev = d["proof"]))
                print("")
                
            d["proof"] += value
                    
        elif key == "proof_clear":
            if value:
                if d["proof"] is not None:
                    d["proof"].clear()
            
                    
        elif key == "proof_new":
            if value:
                d["proof_enabled"] = value
                d["proof"] = ProofObj.empty()
                    
        elif key == "proof_branch":
            if value:
                d["proof_enabled"] = value
                if d["proof"] is None:
                    d["proof"] = ProofObj.empty()
                else:
                    d["proof"] = d["proof"].copy()
                    
        elif key == "proof_step_in":
            if d["proof"] is None:
                d["proof"] = ProofObj.empty()
                
            d["proof"] = d["proof"].step_in(value)
                    
        elif key == "proof_step_out":
            if d["proof"] is None:
                d["proof"] = ProofObj.empty()
                
            d["proof"] = d["proof"].step_out()
                    
        elif key == "opt_singlepass":
            d["opt_learnrate"] = 0.04
            d["opt_num_iter"] = 800
            d["opt_num_iter2"] = 0
            d["opt_num_points"] = 1
            d["opt_num_hop"] = 1
            
        elif key == "opt_basinhopping":
            if value:
                d["opt_learnrate"] = 0.12
                d["opt_learnrate2"] = 0.02 #0.001
                d["opt_num_points"] = 5
                d["opt_num_iter"] = 15
                d["opt_num_iter2"] = 500
                d["opt_num_hop"] = 20
            else:
                PsiOpts.set_setting_dict(d, "opt_singlepass", True)
            
            
        elif key == "opt_learnrate_mul":
            d["opt_learnrate"] *= value
            d["opt_learnrate2"] *= value
            
        elif key == "opt_num_iter_mul":
            d["opt_num_iter"] = int(d["opt_num_iter"] * value)
            d["opt_num_iter2"] = int(d["opt_num_iter2"] * value)
            
        elif key == "opt_num_points_mul":
            d["opt_num_points"] = int(d["opt_num_points"] * value)
            
        else:
            if key not in d:
                raise KeyError("Option '" + str(key) + "' not found.")
            d[key] = value
    
    
    @staticmethod
    def apply_dict(d):
        IBaseObj.set_repr_latex(d["repr_latex"])
        
    @staticmethod
    def set_setting(**kwargs):
        for key, value in kwargs.items():
            PsiOpts.set_setting_dict(PsiOpts.settings, key, value)
        PsiOpts.apply_dict(PsiOpts.settings)
    
    @staticmethod
    def setting(**kwargs):
        PsiOpts.set_setting(**kwargs)
    
    @staticmethod
    def get_setting(key, defaultval = None):
        if key in PsiOpts.settings:
            return PsiOpts.settings[key]
        return defaultval
    
    @staticmethod
    def get_proof():
        return PsiOpts.settings["proof"]
    
    @staticmethod
    def get_truth():
        return PsiOpts.settings["truth"]
    
    @staticmethod
    def timer_left():
        if PsiOpts.settings["timer_end"] is None:
            return None
        curtime = time.time() * 1000
        return PsiOpts.settings["timer_end"] - curtime
    
    @staticmethod
    def timer_left_sec():
        r = PsiOpts.timer_left()
        if r is None:
            return None
        return int(round(r / 1000.0))
    
    @staticmethod
    def is_timer_ended():
        if PsiOpts.settings["timer_end"] is None:
            return False
        curtime = time.time() * 1000
        return curtime > PsiOpts.settings["timer_end"]
    
    @staticmethod
    def get_pyomo_options():
        r = dict(PsiOpts.settings["pyomo_options"])
        
        timelimit = PsiOpts.timer_left_sec()
        if timelimit is not None:
            csolver = iutil.get_solver()
            if csolver == "pyomo.glpk":
                r["tmlim"] = timelimit
            elif csolver == "pyomo.cplex":
                r["timelimit"] = timelimit
            elif csolver == "pyomo.gurobi":
                r["TimeLimit"] = timelimit
            elif csolver == "pyomo.cbc":
                r["seconds"] = timelimit
                
        return r
        
    def __init__(self, **kwargs):
        """
        Options.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.cur_settings = PsiOpts.settings.copy()
        for key, value in kwargs.items():
            PsiOpts.set_setting_dict(self.cur_settings, key, value)
    
    def __enter__(self):
        PsiOpts.settings, self.cur_settings = self.cur_settings, PsiOpts.settings
        PsiOpts.apply_dict(PsiOpts.settings)
        return PsiOpts.settings
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        PsiOpts.settings, self.cur_settings = self.cur_settings, PsiOpts.settings
        PsiOpts.apply_dict(PsiOpts.settings)
    
    
class iutil:
    """Common utilities
    """
    
    
    solver_list = ["pulp.glpk", "pyomo.glpk", "pulp.cbc", "scipy"]
    pulp_solver = None
    pulp_solvers = {}
    
    cur_count = 0
    cur_count_name = {}
    
    @staticmethod
    def display_latex(s, ismath = True, metadata = None):
        color = PsiOpts.settings["latex_color"]
        if color is not None:
            s = "\\color{" + color + "}{" + s + "}"
        
        if ismath:
            r = IPython.display.Math(s, metadata = metadata)
        else:
            r = IPython.display.Latex(s, metadata = metadata)
        IPython.display.display(r)
        # return r
    
    
    @staticmethod
    def float_tostr(x, style = 0, bracket = True):
        if abs(x) <= 1e-10:
            return "0"
        elif abs(x - round(x)) <= 1e-10:
            return str(int(round(x)))
        else:
            frac = fractions.Fraction(abs(x)).limit_denominator(
                        PsiOpts.settings["max_denom"])
            if x > 0:
                if style & PsiOpts.STR_STYLE_LATEX_FRAC:
                    return "\\frac{" + str(frac.numerator) + "}{" + str(frac.denominator) + "}"
                else:
                    if bracket:
                        return "(" + str(frac) + ")"
                    else:
                        return str(frac)
            else:
                if style & PsiOpts.STR_STYLE_LATEX_FRAC:
                    return "-\\frac{" + str(frac.numerator) + "}{" + str(frac.denominator) + "}"
                else:
                    if bracket:
                        return "-(" + str(frac) + ")"
                    else:
                        return "-" + str(frac)

    @staticmethod
    def float_snap(x):
        t = float(fractions.Fraction(x).limit_denominator(PsiOpts.settings["max_denom"]))
        if abs(x - t) <= PsiOpts.settings["eps"]:
            return t
        return x
    
    @staticmethod
    def tostr_verbose(x):
        if isinstance(x, float):
            dp = PsiOpts.settings["verbose_float_dp"]
            if dp is None:
                return str(x)
            return ("{:." + str(dp) + "f}").format(x)
        elif isinstance(x, list):
            return "[" + ", ".join(iutil.tostr_verbose(a) for a in x) + "]"
        elif isinstance(x, tuple):
            return "(" + ", ".join(iutil.tostr_verbose(a) for a in x) + ")"
        else:
            return str(x)

    @staticmethod
    def num_open_brackets(s):
        return s.count("(") + s.count("[") + s.count("{") - (
            s.count("}") + s.count("]") + s.count(")"))
        
    @staticmethod
    def split_comma(s, delim = ", "):
        t = s.split(delim)
        c = ""
        r = []
        for a in t:
            if c != "":
                c += delim
            c += a
            if iutil.num_open_brackets(c) == 0:
                r.append(c)
                c = ""
        if c != "":
            r.append(c)
            
        return r

    @staticmethod
    def get_count(counter_name = None):
        if counter_name is None:
            iutil.cur_count += 1
            return iutil.cur_count
        
        if counter_name not in iutil.cur_count_name:
            iutil.cur_count_name[counter_name] = 0
        iutil.cur_count_name[counter_name] += 1
        return iutil.cur_count_name[counter_name]
        
    @staticmethod
    def gcd(a, b):
        while b > 0:
            a, b = b, a % b
        return a

    @staticmethod
    def lcm(a, b):
        return (a // iutil.gcd(a, b)) * b
    
    @staticmethod
    def hasinstance(a, t):
        if isinstance(a, t):
            return True
        if isinstance(a, (tuple, list)):
            return any(iutil.hasinstance(x, t) for x in a)
    
    @staticmethod
    def convert_str_style(style):
        if style == "standard" or style == "std":
            return PsiOpts.settings["str_style_std"]
        elif style == "psitip" or style == "code":
            return PsiOpts.settings["str_style_repr"]
        elif style == "latex":
            return PsiOpts.settings["str_style_latex"] | PsiOpts.STR_STYLE_LATEX_ARRAY | PsiOpts.STR_STYLE_LATEX_FRAC
        elif style == "latex_noarray":
            return PsiOpts.settings["str_style_latex"] | PsiOpts.STR_STYLE_LATEX_FRAC
        else:
            return style
        
    @staticmethod
    def reverse_eqnstr(eqnstr):
        if eqnstr == "<=":
            return ">="
        elif eqnstr == ">=":
            return "<="
        elif eqnstr == "<":
            return ">"
        elif eqnstr == ">":
            return "<"
        else:
            return eqnstr
    
    @staticmethod
    def eqnstr_style(eqnstr, style):
        if style & PsiOpts.STR_STYLE_LATEX:
            if eqnstr == "<=":
                return "\\le"
            elif eqnstr == ">=":
                return "\\ge"
            elif eqnstr == "==":
                return "="
            elif eqnstr == "!=":
                return "\\neq"
            else:
                return eqnstr
        return eqnstr
    
    @staticmethod
    def str_python_multiline(s):
        s = str(s)
        return "(\"" + "\\n\"\n\"".join(s.split("\n")) + "\")"
    
    @staticmethod
    def istensor(a):
        return hasattr(a, "shape")
        # return isinstance(a, (numpy.array, IBaseArray)) or (torch is not None and isinstance(a, torch.Tensor))
    
    @staticmethod
    def hash_short(s):
        s = str(s)
        return hash(s) % 99991
        
    @staticmethod
    def get_solver(psolver = None):
        csolver_list = [PsiOpts.settings["solver"]] + iutil.solver_list
        if psolver is not None:
            csolver_list = [psolver] + csolver_list
        for s in csolver_list:
            if s == "scipy" and (scipy is not None):
                return s
            if s.startswith("pulp.") and (pulp is not None):
                return s
            if s.startswith("pyomo.") and (pyo is not None):
                return s
        return ""
    
    @staticmethod
    def pulp_get_solver(solver):
        coptions = PsiOpts.settings["pulp_options"]
        copt = solver[solver.index(".") + 1 :].upper()
        if copt == "OTHER":
            return iutil.pulp_solver
        
        if copt in iutil.pulp_solvers:
            return iutil.pulp_solvers[copt]
        
        r = None
        if copt == "GLPK":
            #r = pulp.solvers.GLPK(msg = 0, options = coptions)
            r = pulp.GLPK(msg = False, timeLimit = PsiOpts.timer_left_sec(), options = coptions)
        elif copt == "CBC" or copt == "PULP_CBC_CMD":
            #r = pulp.solvers.PULP_CBC_CMD(options = coptions)
            r = pulp.PULP_CBC_CMD(msg = False, timeLimit = PsiOpts.timer_left_sec(), options = coptions)
        elif copt == "GUROBI":
            r = pulp.GUROBI(msg = False, timeLimit = PsiOpts.timer_left_sec(), options = coptions)
        elif copt == "CPLEX":
            r = pulp.CPLEX(msg = False, timeLimit = PsiOpts.timer_left_sec(), options = coptions)
        elif copt == "MOSEK":
            r = pulp.MOSEK(msg = False, timeLimit = PsiOpts.timer_left_sec(), options = coptions)
        elif copt == "CHOCO_CMD":
            r = pulp.CHOCO_CMD(msg = False, timeLimit = PsiOpts.timer_left_sec(), options = coptions)
        
        iutil.pulp_solvers[copt] = r
        return r
    
    @staticmethod
    def istorch(x):
        return torch is not None and isinstance(x, torch.Tensor)
    
    @staticmethod
    def ensure_torch(x):
        if hasattr(x, "get_x"):
            x = x.get_x()
        if iutil.istorch(x):
            return x
        return torch.tensor(x, dtype=torch.float64)
        
    
    @staticmethod
    def log(x):
        if iutil.istorch(x):
            return torch.log(x)
        else:
            return numpy.log(x)
    
    @staticmethod
    def xlogxoy(x, y):
        if iutil.istorch(x) or iutil.istorch(y):
            ceps_d = PsiOpts.settings["opt_eps_denom"]
            return x * iutil.log((x + ceps_d) / (y + ceps_d))
        else:
            ceps = PsiOpts.settings["eps"]
            if x <= ceps:
                return 0.0
            else:
                return x * numpy.log(x / y)
    
    @staticmethod
    def xlogxoy2(x, y):
        if iutil.istorch(x) or iutil.istorch(y):
            ceps_d = PsiOpts.settings["opt_eps_denom"]
            return x * (iutil.log((x + ceps_d) / (y + ceps_d)) ** 2)
        else:
            ceps = PsiOpts.settings["eps"]
            if x <= ceps:
                return 0.0
            else:
                return x * (numpy.log(x / y) ** 2)
            
    
    @staticmethod
    def sqrt(x):
        if iutil.istorch(x):
            return torch.sqrt(x)
        else:
            return numpy.sqrt(x)
    
    @staticmethod
    def product(x):
        r = 1
        for a in x:
            r *= a
        return r
    
    @staticmethod
    def bitcount(x):
        r = 0
        while x != 0:
            x &= x - 1
            r += 1
        return r
    
    @staticmethod
    def strpad(*args):
        r = ""
        tgtlen = 0
        for i in range(0, len(args), 2):
            r += str(args[i])
            if i + 1 < len(args):
                tgtlen += int(args[i + 1])
                while len(r) < tgtlen:
                    r += " "
        return r
    
    @staticmethod
    def list_tostr(x, tuple_delim = ", ", list_delim = ", ", inden = 0):
        r = " " * inden
        if isinstance(x, list):
            if len([a for a in x if isinstance(a, list) or isinstance(a, tuple)]) > 0:
                r += "["
                for i in range(len(x)):
                    if i == 0:
                        r += iutil.list_tostr(x[i], tuple_delim, list_delim, inden + 2)[inden + 1:]
                    else:
                        r += list_delim + "\n" + iutil.list_tostr(x[i], tuple_delim, list_delim, inden + 2)
                r += " ]"
                return r
            else:
                r += "[" + list_delim.join([iutil.list_tostr(a, tuple_delim, list_delim, 0) for a in x]) + "]"
                return r
        elif isinstance(x, tuple):
            r += "(" + tuple_delim.join([iutil.list_tostr(a, tuple_delim, list_delim, 0) for a in x]) + ")"
            return r
        
        r += str(x)
        #r += x.tostring()
        return r
    
    @staticmethod
    def list_tostr_std(x):
        return iutil.list_tostr(x, tuple_delim = ": ", list_delim = "; ")

    @staticmethod
    def list_iscomplex(x):
        if not isinstance(x, list):
            return True
        for a in x:
            if not isinstance(a, tuple):
                return True
            if len(a) != 2:
                return True
            if isinstance(a[1], list):
                return True
        return False
        
    @staticmethod
    def str_inden(s, ninden, spacestr = " "):
        return " " * ninden + s.replace("\n", "\n" + spacestr * ninden)
    
    @staticmethod
    def enum_partition(n):
        def enum_partition_recur(mask):
            if mask == 0:
                return [[]]
            r = []
            for i in range(n):
                if mask & (1 << i) != 0:
                    mask2 = mask - (1 << i)
                    while True:
                        mask3 = (1 << i) | mask2
                        r += [[mask3] + a
                              for a in enum_partition_recur(mask - mask3)]
                        if mask2 == 0:
                            break
                        mask2 = (mask2 - 1) & (mask - (1 << i))
                    break
            return r
        return enum_partition_recur((1 << n) - 1)
    
    @staticmethod
    def tsort(x):
        """Topological sort."""
        n = len(x)
        ninc = [0] * n
        for i in range(n):
            for j in range(n):
                if x[i][j]:
                    ninc[j] += 1
        cstack = [i for i in range(n) if ninc[i] == 0]
        r = []
        while len(cstack) > 0:
            i = cstack.pop()
            r.append(i)
            for j in range(n):
                if x[i][j]:
                    ninc[j] -= 1
                    if ninc[j] == 0:
                        cstack.append(j)
        return r
            
    @staticmethod
    def iscyclic(x):
        return len(iutil.tsort(x)) < len(x)
        
    @staticmethod
    def signal_type(x):
        if isinstance(x, tuple) and len(x) > 0 and isinstance(x[0], str):
            return x[0]
        return ""
    
    @staticmethod
    def mhash(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return hash(tuple(iutil.mhash(y) for y in x))
        return hash(x)
    
    @staticmethod
    def list_unique(x):
        r = []
        s = set()
        for a in x:
            h = iutil.mhash(a)
            if h not in s:
                s.add(h)
                r.append(a)
        return r
    
    @staticmethod
    def list_sorted_unique(x):
        x = sorted(x, key = lambda a: iutil.mhash(a))
        return [x[i] for i in range(len(x)) if i == 0 or not x[i] == x[i - 1]]
    
    @staticmethod
    def sumlist(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return sum(iutil.sumlist(a) for a in x)
        return x
    
    @staticmethod
    def find_similarity(s, x):
        t = s.split("@@")
        x2 = x.split("@@")
        r = 0
        for i in range(0, len(t), 2):
            for j in range(0, len(x2), 2):
                if x2[j] != "" and x2[j] in t[i]:
                    r = max(r, 10000 + len(x2[j]) - len(t[i]))
        return r
    
    @staticmethod
    def set_suffix_num(s, k, schar, replace_mode = "set", style = None):
        t = s.split("@@")
        if len(t) >= 2:
            for i in range(0, len(t), 2):
                t[i] = iutil.set_suffix_num(t[i], k, schar, replace_mode, style = None if i == 0 else int(t[i - 1]))
            return "@@".join(t)
            
        if replace_mode != "suffix":

            s, v0 = iutil.break_subscript_latex(s)
            if replace_mode == "append":
                v0 += str(k)
            elif replace_mode == "set":
                v0 = str(k)
            elif replace_mode == "add":
                if v0.isdigit():
                    v0 = str(int(v0) + k)
                else:
                    v0 += str(k)

            if len(v0) > 1 and style == PsiOpts.STR_STYLE_LATEX:
                return s + "_{" + v0 + "}"
            else:
                return s + "_" + v0
            
            # i = s.rfind(schar)
            # if i >= 0:
            #     if replace_mode == "append":
            #         return s + str(k)
            #     elif replace_mode == "set":
            #         return s[:i] + schar + str(k)
            #     else:
            #         if s[i + 1 :].isdigit():
            #             if replace_mode == "add":
            #                 return s[:i] + schar + str(int(s[i + 1 :]) + k)
            #             else:
            #                 return s[:i] + schar + str(k)

        return s + schar + str(k)
    
    @staticmethod
    def get_name_fromcat(s, style):
        t = s.split("@@")
        for i in range(1, len(t) - 1, 2):
            if int(t[i]) & style:
                return t[i + 1]
        return t[0]
    
    @staticmethod
    def break_subscript_latex(s):
        t = s.split("_")
        if len(t) == 0:
            return "", ""
        if len(t) == 1:
            return t[0], ""
        v = "_".join(t[1:])
        if v.startswith("{") and v.endswith("}"):
            return t[0], v[1:-1]
        return t[0], v
    
    @staticmethod
    def is_single_term(x):
        if isinstance(x, (int, float, tuple, list, IVar)):
            return True
        if isinstance(x, (Comp, Expr)):
            return len(x) == 1
        return False
    
    @staticmethod
    def fcn_name_maker(name, v, pname = None, lname = None, cropi = False, infix = False, latex_group = False, fcn_suffix = True):
        if not isinstance(v, list) and not isinstance(v, tuple):
            v = [v]
        r = ""
        for style in [PsiOpts.STR_STYLE_STANDARD, PsiOpts.STR_STYLE_PSITIP, PsiOpts.STR_STYLE_LATEX]:
            if style != PsiOpts.STR_STYLE_STANDARD:
                r += "@@" + str(style) + "@@"
                
                
            for i, a in enumerate(v):
                a_single_term = iutil.is_single_term(a)
                
                if (i == 0) ^ infix:
                    if style & PsiOpts.STR_STYLE_STANDARD:
                        r += name
                    elif style & PsiOpts.STR_STYLE_PSITIP:
                        r += pname or name
                    elif style & PsiOpts.STR_STYLE_LATEX:
                        r += lname or name
                if not infix:
                    if i == 0:
                        r += "("
                    if i:
                        r += ","
                    
                else:
                    if latex_group and style & PsiOpts.STR_STYLE_LATEX:
                        r += "{"
                    if not a_single_term:
                        r += "("
                t = ""
                if isinstance(a, (IVar, Comp, Term, Expr, Region)):
                    if (style & PsiOpts.STR_STYLE_STANDARD or style & PsiOpts.STR_STYLE_LATEX) and isinstance(a, Comp):
                        t = a.tostring(style = style, add_bracket = True)
                    else:
                        t = a.tostring(style = style)
                elif isinstance(a, str):
                    if style & PsiOpts.STR_STYLE_PSITIP:
                        t = repr(a)
                    else:
                        t = str(a)
                else:
                    t = str(a)
                    
                if cropi and isinstance(a, Term) and len(t) >= 3 and (t.startswith("I(") or t.startswith("H(")):
                    r += t[2:-1]
                else:
                    r += t
                    
                if not infix:
                    if i == len(v) - 1:
                        r += ")"
                else:
                    if not a_single_term:
                        r += ")"
                    if latex_group and style & PsiOpts.STR_STYLE_LATEX:
                        r += "}"
        
        if fcn_suffix:
            r += PsiOpts.settings["fcn_suffix"]
        return r
    
    @staticmethod
    def add_subscript_latex(s, v):
        s, v0 = iutil.break_subscript_latex(s)
        if len(v0):
            v0 += "," + str(v)
        else:
            v0 = str(v)
        
        if len(v0) > 1:
            return s + "_{" + v0 + "}"
        else:
            return s + "_" + v0
    
    @staticmethod
    def copy(x):
        if (x is None or isinstance(x, int) or isinstance(x, float) 
            or isinstance(x, str)):
            return x
        if isinstance(x, list):
            return [iutil.copy(a) for a in x]
        if isinstance(x, tuple):
            return tuple(iutil.copy(a) for a in x)
        if isinstance(x, dict):
            return {a: iutil.copy(b) for a, b in x.items()}
        
        return x.copy()
    
    @staticmethod
    def str_join(x, delim = ""):
        if isinstance(x, list) or isinstance(x, tuple):
            return delim.join(iutil.str_join(a, delim) for a in x)
        return str(x)
    
    
    @staticmethod
    def tostring_join(x, style, delim = ""):
        if isinstance(x, list) or isinstance(x, tuple):
            return delim.join(iutil.tostring_join(a, style, delim) for a in x)
        if hasattr(x, "tostring"):
            return x.tostring(style = style)
        
        if style & PsiOpts.STR_STYLE_LATEX:
            return "\\text{" + str(x) + "}"
        
        return str(x)
    
    
    @staticmethod
    def bit_reverse(x, n):
        r = 0
        for i in range(n):
            if x & (1 << i):
                r += 1 << (n - 1 - i)
        return r
    
    @staticmethod
    def bin_to_gray(x):
        """Binary to Gray code. From en.wikipedia.org/wiki/Gray_code
        """
        return x ^ (x >> 1)
    
    @staticmethod
    def gray_to_bin(x):
        """Gray code to binary. From en.wikipedia.org/wiki/Gray_code
        """
        m = x
        while m:
            m >>= 1
            x ^= m
        return x
    
    @staticmethod
    def gbsearch(fcn, num_iter = 30):
        """Binary search.
        """
        lb = None
        ub = None
        for it in range(num_iter):
            m = 0.0
            if it == 0:
                m = 0.0
            elif ub is None:
                m = max(lb * 2, 1.0)
            elif lb is None:
                m = min(ub * 2, -1.0)
            else:
                m = (lb + ub) * 0.5
            
            if fcn(m):
                ub = m
            else:
                lb = m
        
        if ub is None:
            return lb
        if lb is None:
            return ub
        return (lb + ub) * 0.5
        
    
    @staticmethod
    def polygon_frompolar(poly, inf_value = 1e6):
        """
        Convert polygon from polar representation to positive and negative portions.

        Parameters
        ----------
        poly : list
            List of 3-tuples of vertices.
        inf_value : float, optional
            The value used as infinity. The default is 1e6.

        Returns
        -------
        r : list
            List of two polygons (positive and negative portions, both are lists of 2-tuples).

        """
        r = [[], []]
        x = [[0 if abs(a[0]) <= 1e-10 else 1 if a[0] > 0 else -1] + a[1:] for a in poly]
        x.append(x[0])
        
        for i in range(len(x) - 1):
            if x[i][0] > 0:
                r[0].append(x[i][1:])
            elif x[i][0] < 0:
                r[1].append([-y for y in x[i][1:]])
                
            ray = None
            if x[i][0] == 0:
                ray = x[i]
            elif x[i][0] * x[i + 1][0] < 0:
                ray = [y0 + y1 for y0, y1 in zip(x[i], x[i+1])]
            
            if ray is not None:
                ray = ray[1:]
                norm = max(abs(y) for y in ray)
                ray = [y / norm * inf_value for y in ray]
                r[0].append(ray)
                r[1].append([-y for y in ray])
            
        return r
    

class MHashSet:
    def __init__(self, x = None, s = None):
        if x is None:
            self.x = []
        else:
            self.x = x
        if s is None:
            self.s = set()
        else:
            self.s = s
        
    def add(self, y):
        h = iutil.mhash(y)
        
        if h in self.s:
            return False
        self.x.append(y)
        self.s.add(h)
        return True
    
    def clear(self):
        self.x[:] = []
        self.s.clear()
        
    
    def __iadd__(self, other):
        for y in other:
            self.add(y)
        return self
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, key):
        return self.x[key]
        #return self.x[len(self.x) - 1 - key]
    
    def copy(self):
        return MHashSet(self.x[:], self.s.copy())
    
    def __eq__(self, other):
        return self.s == other.s
    
    def __ne__(self, other):
        return self.s != other.s
    
    def __hash__(self):
        return hash(frozenset(self.s))
    
    
def fcn_substitute(fcn):
    
    @functools.wraps(fcn)
    def wrapper(cself, *args, **kwargs):
        i = 0
        while i < len(args):
            if isinstance(args[i], dict):
                for key, val in args[i].items():
                    fcn(cself, key, val)
                i += 1
            elif isinstance(args[i], CompArray):
                for key, val in args[i].to_dict().items():
                    fcn(cself, key, val)
                i += 1
            elif isinstance(args[i], list):
                for key, val in args[i]:
                    fcn(cself, key, val)
                i += 1
            elif i + 1 < len(args):
                fcn(cself, args[i], args[i + 1])
                i += 2
            else:
                i += 1
        
        for key, val in kwargs.items():
            found = cself.find_name(key)
            if isinstance(found, Comp) and not found.isempty():
                fcn(cself, found, val)
            elif isinstance(found, Expr) and not found.iszero():
                fcn(cself, found, val)
                
                
    return wrapper

    
def fcn_list_to_list(fcn):
    
    @functools.wraps(fcn)
    def wrapper(*args, **kwargs):
        islist = False
        maxlen = -1
        maxshape = tuple()
        for a in itertools.chain(args, kwargs.values()):
            if CompArray.isthis(a) or ExprArray.isthis(a):
                islist = True
                # maxlen = max(maxlen, len(a))
                if len(a) > maxlen:
                    maxlen = len(a)
                    if isinstance(a, list):
                        maxshape = (len(a),)
                    else:
                        maxshape = a.shape
        
        if not islist:
            return fcn(*args, **kwargs)
        
        r = []
        for i in range(maxlen):
            targs = []
            for a in args:
                if CompArray.isthis(a):
                    if i < len(a):
                        targs.append(a[i])
                    else:
                        targs.append(Comp.empty())
                elif ExprArray.isthis(a):
                    if i < len(a):
                        targs.append(a[i])
                    else:
                        targs.append(Expr.zero())
                else:
                    targs.append(a)
                    
            tkwargs = dict()
            for key, a in kwargs.items():
                if CompArray.isthis(a):
                    if i < len(a):
                        tkwargs[key] = a[i]
                    else:
                        tkwargs[key] = Comp.empty()
                elif ExprArray.isthis(a):
                    if i < len(a):
                        tkwargs[key] = a[i]
                    else:
                        tkwargs[key] = Expr.zero()
                else:
                    tkwargs[key] = a
                    
            r.append(fcn(*targs, **tkwargs))
                    
        if len(r) > 0:
            if isinstance(r[0], Region):
                return alland(r)
            elif isinstance(r[0], Comp):
                return CompArray(r, maxshape)
            elif isinstance(r[0], Expr):
                return ExprArray(r, maxshape)
            else:
                return r
        else:
            return None
    
    return wrapper
        

class PsiRec:
    num_lpprob = 0
    
class IVarType:
    NIL = 0
    RV = 1
    REAL = 2
    
    
class IBaseObj:
    """Base class of objects
    """
    def __init__(self):
        pass
        
    def _latex_(self):
        return ""
        
    def latex(self, skip_simplify = False):
        """LaTeX code
        """
        if skip_simplify:
            r = ""
            with PsiOpts(repr_simplify = False):
                r = self._latex_()
            return r
        
        return self._latex_()
    
    def display(self, skip_simplify = False):
        """IPython display
        """
        iutil.display_latex(self.latex(skip_simplify = skip_simplify))
    
    def display_bool(self, s = "{region} \\;\\mathrm{{is}}\\;\\mathrm{{{truth}}}", skip_simplify = True):
        """IPython display, show truth value
        """
        iutil.display_latex(s.format(region = self.latex(skip_simplify = skip_simplify), 
                                            truth = str(bool(self))))
    
    @staticmethod
    def set_repr_latex(enabled):
        hasa = hasattr(IBaseObj, "_repr_latex_")
        if enabled and not hasa:
            setattr(IBaseObj, "_repr_latex_", lambda s: "$" + s.latex() + "$")
        if not enabled and hasa:
            delattr(IBaseObj, "_repr_latex_")
    
    
class IVar(IBaseObj):
    """Random variable or real variable
    Do NOT use this class directly. Use Comp instead
    """
    
    def __init__(self, vartype, name, reg = None, reg_det = False, markers = None):
        self.vartype = vartype
        self.name = name
        self.reg = reg
        self.reg_det = reg_det
        self.markers = markers
        
    @staticmethod
    def rv(name):
        return IVar(IVarType.RV, name)
        
    @staticmethod
    def real(name):
        return IVar(IVarType.REAL, name)
        
    @staticmethod
    def eps():
        return IVar(IVarType.REAL, "EPS")
        
    @staticmethod
    def one():
        return IVar(IVarType.REAL, "ONE")
        
    @staticmethod
    def inf():
        return IVar(IVarType.REAL, "INF")
    
    def isrealvar(self):
        return self.vartype == IVarType.REAL and self.name != "ONE" and self.name != "EPS" and self.name != "INF"
    
    def tostring(self, style = 0):
        return iutil.get_name_fromcat(self.name, style)
    
    def __str__(self):
        
        return self.tostring(PsiOpts.settings["str_style"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.settings["str_style_repr"])
    
    def _latex_(self):
        return self.tostring(iutil.convert_str_style("latex"))
    
    def __hash__(self):
        return hash(self.name)
        
    def __eq__(self, other):
        return self.name == other.name
        
    def copy(self):
        return IVar(self.vartype, self.name, None if self.reg is None else self.reg.copy(), 
                    self.reg_det, None if self.markers is None else self.markers[:])
        
    def copy_noreg(self):
        return IVar(self.vartype, self.name, None, 
                    False, None if self.markers is None else self.markers[:])

    
class Comp(IBaseObj):
    """Compound random variable or real variable
    """
    
    def __init__(self, varlist):
        self.varlist = varlist
        
    @staticmethod
    def empty():
        """
        The empty random variable.

        Returns
        -------
        Comp

        """
        return Comp([])
        
    @staticmethod
    def rv(name):
        """
        Random variable.

        Parameters
        ----------
        name : str
            Name of the random variable.

        Returns
        -------
        Comp

        """
        return Comp([IVar(IVarType.RV, name)])
        
    @staticmethod
    def rv_reg(a, reg, reg_det = False):
        r = a.copy_noreg()
        for i in range(len(r.varlist)):
            r.varlist[i].reg = reg.copy()
            r.varlist[i].reg_det = reg_det
        return r
        #return Comp([IVar(IVarType.RV, str(a), reg.copy(), reg_det)])
        
    @staticmethod
    def real(name):
        return Comp([IVar(IVarType.REAL, name)])
    
    @staticmethod
    def array(name, st, en = None):
        if en is None:
            en = st
            st = 0
            
        t = []
            
        for i in range(st, en):
            istr = str(i)
            s = name + istr
            s += "@@" + str(PsiOpts.STR_STYLE_LATEX)
            s += "@@" + iutil.add_subscript_latex(name, istr)
            t.append(IVar(IVarType.RV, s))
        return Comp(t)
    
    def get_name(self):
        if len(self.varlist) == 0:
            return ""
        return self.varlist[0].name
    
    def find_name(self, *args):
        r = Comp.empty()
        for carg in args:
            cmax = 0
            cmaxa = Comp.empty()
            for a in self.varlist:
                t = iutil.find_similarity(a.name, carg)
                if t > cmax:
                    cmax = t
                    cmaxa = Comp([a.copy()])
            r += cmaxa
        if len(r) > 0 and r.get_type() == IVarType.REAL:
            r = ExprArray([Expr.fromcomp(t) for t in r])
            if len(r) == 1:
                r = r[0]
        return r
    
    def get_type(self):
        if len(self.varlist) == 0:
            return IVarType.NIL
        return self.varlist[0].vartype
        
    def allcomp(self):
        return self.copy()
    
    def swapped_id(self, i, j):
        if i >= len(self.varlist) or j >= len(self.varlist):
            return self.copy()
        r = self.copy()
        r.varlist[i], r.varlist[j] = r.varlist[j], r.varlist[i]
        return r
    
    def set_markers(self, markers):
        for a in self.varlist:
            if markers is None:
                a.markers = None
            else:
                a.markers = markers[:]
        return self
    
    def add_markers(self, markers):
        for a in self.varlist:
            if a.markers is None:
                a.markers = []
            a.markers += markers
        return self
    
    def add_marker(self, key, value = 1):
        self.add_markers([(key, value)])
        return self
    
    def add_marker_id(self, key):
        return self.add_marker(key, iutil.get_count())
    
    def mark(self, *args):
        for a in args:
            if a == "symm" or a == "disjoint" or a == "nonsubset":
                self.add_marker_id(a)
            else:
                self.add_marker(a)
        return self
    
    def get_marker_key(self, key):
        for a in self.varlist:
            if a.markers is not None:
                for v, w in reversed(a.markers):
                    if v == key:
                        return w
        return None
        
    
    def get_markers(self):
        r = []
        for a in self.varlist:
            if a.markers is not None:
                for v, w in a.markers:
                    if (v, w) not in r:
                        r.append((v, w))
        return r
        
    
    def set_card(self, m):
        self.add_marker("card", m)
        return self
    
    def get_card(self):
        r = 1
        for a in self:
            t = a.get_marker_key("card")
            if t is None:
                return None
            r *= t
        return r
    
    def get_shape(self):
        r = []
        for a in self:
            t = a.get_card()
            if t is None:
                raise ValueError("Cardinality of " + str(a) + " not set. Use " + str(a) + ".set_card(m) to set cardinality.")
                return
            r.append(t)
        return tuple(r)
        
    def add_suffix(self, csuffix):
        for a in self.varlist:
            a.name += csuffix
            
    def added_suffix(self, csuffix):
        r = self.copy()
        r.add_suffix(csuffix)
        return r
    
    def __getitem__(self, key):
        r = self.varlist[key]
        if isinstance(r, list):
            return Comp(r)
        return Comp([r])
        
    def __setitem__(self, key, value):
        if value.isempty():
            del self.varlist[key]
        self.varlist[key] = value.varlist[0]
        
    def __delitem__(self, key):
        del self.varlist[key]
        
    def __iter__(self):
        for a in self.varlist:
            yield Comp([a])
            
    def copy(self):
        return Comp([a.copy() for a in self.varlist])
            
    def copy_noreg(self):
        return Comp([a.copy_noreg() for a in self.varlist])
        
    def addvar(self, x):
        if x in self.varlist:
            return
        self.varlist.append(x.copy())
        
    def removevar(self, x):
        self.varlist = [a for a in self.varlist if a != x]
        
    def reg_excluded(self):
        r = Comp.empty()
        for a in self.varlist:
            if a.reg is None:
                r.varlist.append(a)
        return r
        
    def index_of(self, x):
        for i, a in enumerate(self.varlist):
            if x.ispresent(a):
                return i
        return -1
    
    def ispresent_shallow(self, x):
        
        if isinstance(x, str):
            if x == "real":
                return self.get_type() == IVarType.REAL
            if x == "realvar":
                for a in self.varlist:
                    if a.isrealvar():
                        return True
                return False
        
        if isinstance(x, Comp):
            for y in x.varlist:
                if y in self.varlist:
                    return True
            return False
        
        return x in self.varlist
    
    def ispresent(self, x):
        
        if isinstance(x, Expr) or isinstance(x, Region):
            x = x.allcomp()
        
        for a in self.varlist:
            if a.reg is not None and a.reg.ispresent(x):
                return True
                
        return self.ispresent_shallow(x)
        
    def __iadd__(self, other):
        for i in range(len(other.varlist)):
            self.addvar(other.varlist[i])
        return self
        
    def __add__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if isinstance(other, Expr):
            return other.__radd__(self)
        r = self.copy()
        if isinstance(other, Comp):
            r += other
        return r
        
    def __radd__(self, other):
        r = self.copy()
        if isinstance(other, Comp):
            r += other
        return r
        
    def __isub__(self, other):
        for i in range(len(other.varlist)):
            self.removevar(other.varlist[i])
        return self
        
    def __sub__(self, other):
        r = self.copy()
        r -= other
        return r
        
    def __imul__(self, other):
        ceps = PsiOpts.settings["eps"]
        if isinstance(other, bool) and not other:
            self.varlist = []
            return self
        if isinstance(other, int) and other == 0:
            self.varlist = []
            return self
        if isinstance(other, float) and abs(other) <= ceps:
            self.varlist = []
            return self
        if isinstance(other, Expr):
            other = other.get_const()
            if other is not None and abs(other) <= ceps:
                self.varlist = []
                return self
        return self
        
    def __mul__(self, other):
        r = self.copy()
        r *= other
        return r
        
    def __rmul__(self, other):
        r = self.copy()
        r *= other
        return r
    
    def inter(self, other):
        """Intersection."""
        return Comp([a for a in self.varlist if a in other.varlist])
    
    def interleaved(self, other):
        r = Comp([])
        for i in range(max(len(self.varlist), len(other.varlist))):
            if i < len(self.varlist):
                r.varlist.append(self.varlist[i].copy())
            if i < len(other.varlist):
                r.varlist.append(other.varlist[i].copy())
        return r
    
    def size(self):
        return len(self.varlist)
        
    def __len__(self):
        return len(self.varlist)
        
    def __bool__(self):
        return bool(self.varlist)
    
    def isempty(self):
        """Whether self is empty."""
        return (len(self.varlist) == 0)
    
    def from_mask(self, mask):
        """Return subset using bit mask."""
        r = []
        for i in range(len(self.varlist)):
            if mask & (1 << i) != 0:
                r.append(self.varlist[i])
        return Comp(r)
        
    # Get bit mask of Comp
    def get_mask(self, x):
        r = 0
        for i in range(len(self.varlist)):
            if self.varlist[i] in x:
                r |= (1 << i)
        return r
    
    def super_of(self, other):
        """Whether self is a superset of other."""
        for i in range(len(other.varlist)):
            if not (other.varlist[i] in self.varlist):
                return False
        return True
        
    def disjoint(self, other):
        """Whether self is disjoint from other."""
        for i in range(len(other.varlist)):
            if other.varlist[i] in self.varlist:
                return False
        return True
        
        
    def __contains__(self, other):
        if isinstance(other, IVar):
            return other in self.varlist
        return self.super_of(other)
        
    def __ge__(self, other):
        return self.super_of(other)
         
    def __le__(self, other):
        return other.super_of(self)
    
    def __eq__(self, other):
        return {a.name for a in self.varlist} == {a.name for a in other.varlist}
        #return self.super_of(other) and other.super_of(self)
    
    def __ne__(self, other):
        return {a.name for a in self.varlist} != {a.name for a in other.varlist}
    
    def __gt__(self, other):
        return self.super_of(other) and not other.super_of(self)
    
    def __lt__(self, other):
        return other.super_of(self) and not self.super_of(other)
    
    def tolist(self):
        return CompArray([a.copy() for a in self])
    
    
    
    def mean(self, f = None, name = None):
        """
        Returns the expectation of the function f.

        Parameters
        ----------
        f : function, numpy.array or torch.Tensor
            If f is a function, the number of arguments must match the number of
            random variables.
            If f is an array or tensor, shape must match the shape of the 
            joint distribution.
        Returns
        -------
        r : Expr
            The expectation as an expression.

        """
        
        if name is None:
            name = "mean_" + str(iutil.get_count("mean"))
            
        def fcncall(xdist):
            return xdist.mean(f)
    
        R = Expr.real(name)
        
        return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), Region.universe(), 0, fcncall, [self.copy()]))
    
    
    def prob(self, *args):
        """
        Returns the probability mass function at *args.

        Parameters
        ----------
        *args : int
            The indices to query. E.g. (X+Y).prob(2,3) is P(X=2, Y=3).
        Returns
        -------
        r : Expr
            The probability as an expression.

        """
        
        args = tuple(args)
        name = "P(" + ",".join((a.tostring(style = PsiOpts.STR_STYLE_STANDARD) + "=" + str(b)) for a, b in zip(self, args)) + ")"
        name += "@@" + str(PsiOpts.STR_STYLE_PSITIP) + "@@"
        if len(self) == 1:
            name += repr(self)
        else:
            name += "(" + repr(self) + ")"
        name += ".prob(" + ",".join(str(b) for b in args) + ")"
        name += "@@" + str(PsiOpts.STR_STYLE_LATEX) + "@@"
        name += PsiOpts.settings["latex_prob"] + "(" + ",".join((a.tostring(style = PsiOpts.STR_STYLE_LATEX) + "=" + str(b)) for a, b in zip(self, args)) + ")"
            
        def fcncall(xdist):
            return xdist[args]
    
        R = Expr.real(name)
        
        return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), Region.universe(), 0, fcncall, [self.copy()]))
    
    def pmf(self):
        """ Returns the probability mass function as ExprArray.
        """
        shape = self.get_shape()
        r = ExprArray.zeros(shape)
        for xs in itertools.product(*[range(x) for x in shape]):
            r[xs] = self.prob(*xs)
        return r
    
    def tostring(self, style = 0, tosort = False, add_bracket = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        style = iutil.convert_str_style(style)
        
        namelist = [a.tostring(style) for a in self.varlist]
        if len(namelist) == 0:
            if style & PsiOpts.STR_STYLE_PSITIP:
                return "rv_empty()"
            elif style & PsiOpts.STR_STYLE_LATEX:
                return PsiOpts.settings["latex_rv_empty"]
            return "!"
            
        if tosort:
            namelist.sort()
        r = ""
        if add_bracket and len(namelist) > 1:
            r += "("
            
        if style & PsiOpts.STR_STYLE_PSITIP:
            r += "+".join(namelist)
        elif style & PsiOpts.STR_STYLE_LATEX:
            r += (PsiOpts.settings["latex_rv_delim"] + " ").join(namelist)
        else:
            r += ",".join(namelist)
            
        if add_bracket and len(namelist) > 1:
            r += ")"
        
        return r
    
    
    def __str__(self):
        
        return self.tostring(PsiOpts.settings["str_style"],
                             tosort = PsiOpts.settings["str_tosort"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.settings["str_style_repr"])
    
    def _latex_(self):
        return self.tostring(iutil.convert_str_style("latex"))
        
        
    def __hash__(self):
        #return hash(self.tostring(tosort = True))
        return hash(frozenset(a.name for a in self.varlist))
    
    
        
    def isregtermpresent(self):
        for b in self.varlist:
            if b.reg is not None:
                return True
        return False
        
    
    
    def __and__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        return Term.H(self) & Term.H(other)
    
    def __or__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if isinstance(other, int):
            return Term.H(self)
        return Term.H(self) | Term.H(other)
        
    def rename_var(self, name0, name1):
        for a in self.varlist:
            if a.name == name0:
                a.name = name1
        for a in self.varlist:
            if a.reg is not None:
                a.reg.rename_var(name0, name1)
            
    def rename_map(self, namemap):
        """Rename according to name map
        """
        for a in self.varlist:
            a.name = namemap.get(a.name, a.name)
        for a in self.varlist:
            if a.reg is not None:
                a.reg.rename_map(namemap)
        return self
    
    @fcn_substitute
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound)"""
        
        if len(v0) > 1:
            for v0c in v0:
                self.substitute(v0c, v1)
            return
        
        v0s = v0.get_name()
        for i in range(len(self.varlist)):
            if self.varlist[i].name == v0s:
                nameset = set()
                for j in range(len(self.varlist)):
                    if j != i:
                        nameset.add(self.varlist[j].name)
                
                self.varlist = self.varlist[:i] + [t.copy() for t in v1.varlist if t.name not in nameset] + self.varlist[i+1:]
                break
        
        # r = Comp.empty()
        # for a in self.varlist:
        #     if v0.ispresent_shallow(a):
        #         r += v1
        #     else:
        #         r += Comp([a])
        # self.varlist = r.varlist
        
        for a in self.varlist:
            if a.reg is not None:
                a.reg.substitute(v0, v1)
    
    def substituted(self, *args, **kwargs):
        """Substitute variable v0 by v1 (v1 can be compound), return result"""
        r = self.copy()
        r.substitute(*args, **kwargs)
        return r
        
    @staticmethod
    def substitute_list(a, vlist, suffix = "", isaux = False):
        """Substitute variables in vlist into a"""
        if isinstance(vlist, list):
            for v in vlist:
                Comp.substitute_list(a, v, suffix, isaux)
        elif isinstance(vlist, tuple):
            if len(vlist) >= 2:
                w = vlist[1]
                if isinstance(w, list):
                    w = w[0] if len(w) > 0 else Comp.empty()
                if suffix != "":
                    if isaux:
                        a.substitute_aux(Comp.rv(vlist[0].get_name() + suffix), w)
                    else:
                        a.substitute(Comp.rv(vlist[0].get_name() + suffix), w)
                else:
                    if isaux:
                        a.substitute_aux(vlist[0], w)
                    else:
                        a.substitute(vlist[0], w)
        
    @staticmethod
    def substitute_list_to_dict(vlist):
        r = dict()
        if isinstance(vlist, list):
            for v in vlist:
                r.update(Comp.substitute_list_to_dict(v))
        elif isinstance(vlist, tuple):
            if len(vlist) >= 2:
                w = vlist[1]
                if isinstance(w, list):
                    w = w[0] if len(w) > 0 else Comp.empty()
                r[vlist[0]] = w
        return r
    

    def record_to(self, index):
        index.record(self)
        for a in self.varlist:
            if a.reg is not None:
                index.record(a.reg.allcomprv_noaux())
        
    def fcn_of(self, b):
        return Expr.Hc(self, b) == 0
    
        
    def table(self, *args, **kwargs):
        """Plot the information diagram as a Karnaugh map.
        """
        return universe().table(self, *args, **kwargs)
        
    def venn(self, *args, **kwargs):
        """Plot the information diagram as a Venn diagram.
        Can handle up to 5 random variables (uses Branko Grunbaum's Venn diagram for n=5).
        """
        return universe().venn(self, *args, **kwargs)
    
    
class IVarIndex:
    """Store index of variables
    Do NOT use this class directly
    """
    
    def __init__(self):
        self.dictrv = {}
        self.comprv = Comp.empty()
        self.dictreal = {}
        self.compreal = Comp.empty()
        self.prefavoid = "@@"
    
    def copy(self):
        r = IVarIndex()
        r.dictrv = self.dictrv.copy()
        r.comprv = self.comprv.copy()
        r.dictreal = self.dictreal.copy()
        r.compreal = self.compreal.copy()
        r.prefavoid = self.prefavoid
        return r
        
    def record(self, x):
        for i in range(len(x.varlist)):
            if not x.varlist[i].name.startswith(self.prefavoid):
                if x.varlist[i].vartype == IVarType.RV:
                    if not (x.varlist[i].name in self.dictrv):
                        self.dictrv[x.varlist[i].name] = self.comprv.size()
                        self.comprv.varlist.append(x.varlist[i])
                else:
                    if not (x.varlist[i].name in self.dictreal):
                        self.dictreal[x.varlist[i].name] = self.compreal.size()
                        self.compreal.varlist.append(x.varlist[i])
        
        
    def add_varindex(self, x):
        self.record(x.comprv)
        self.record(x.compreal)
        
    # Get index of IVar
    def get_index(self, x):
        if isinstance(x, Comp):
            x = x.varlist[0]
        if x.vartype == IVarType.RV:
            if x.name in self.dictrv:
                return self.dictrv[x.name]
            return -1
        else:
            if x.name in self.dictreal:
                return self.dictreal[x.name]
            return -1
        
    def __contains__(self, other):
        return self.get_index(other) >= 0
        
    # Get index of name
    def get_index_name(self, name):
        if name in self.dictrv:
            return self.dictrv[name]
        if name in self.dictreal:
            return self.dictreal[name]
        return -1
        
    # Get bit mask of Comp
    def get_mask(self, x):
        if x.get_type() != IVarType.RV:
            return 0
        r = 0
        for a in x.varlist:
            k = self.get_index(a)
            if k < 0:
                return -1
            r |= (1 << k)
        return r
        
    def from_mask(self, m):
        return self.comprv.from_mask(m)
        
    def num_rv(self):
        return self.comprv.size()
        
    def num_real(self):
        return self.compreal.size()
        
    def size(self):
        return self.comprv.size() + self.compreal.size()
        
        
    def name_avoid(self, name0):
        name1 = name0
        while self.get_index_name(name1) >= 0:
            name1 += PsiOpts.settings["rename_char"]
            
        return name1
    
    def calc_rename_map(self, a):
        m = dict()
        for x in a:
            xstr = ""
            if isinstance(x, Comp):
                xstr = x.get_name()
            else:
                xstr = str(x)
            tname = self.name_avoid(xstr)
            self.record(Comp.rv(tname))
            m[xstr] = tname
        return m
        

class TermType:
    NIL = 0
    IC = 1
    REAL = 2
    REGION = 3
    
class Term(IBaseObj):
    """A term in an expression
    Do NOT use this class directly. Use Expr instead
    """
    
    def __init__(self, x, z = None, reg = None, sn = 0, fcncall = None, fcnargs = None, reg_outer = None):
        self.x = x
        if z is None:
            self.z = Comp.empty()
        else:
            self.z = z
        self.reg = reg
        self.sn = sn
        self.fcncall = fcncall
        self.fcnargs = fcnargs
        self.reg_outer = reg_outer
        
    def copy(self):
        return Term([a.copy() for a in self.x], self.z.copy(), iutil.copy(self.reg), 
                    self.sn, self.fcncall, iutil.copy(self.fcnargs), iutil.copy(self.reg_outer))
        
    def copy_noreg(self):
        return Term([a.copy_noreg() for a in self.x], self.z.copy_noreg(), None, 0)
        
    def zero():
        return Term([], Comp.empty())
    
    def isempty(self):
        """Whether self is empty."""
        return len(self.x) == 0 or any(len(a) == 0 for a in self.x)
    
    def setzero(self):
        self.x = []
        self.z = Comp.empty()
        self.reg = None
        self.reg_outer = None
        self.sn = 0
        self.fcncall = None
        self.fcnargs = None
        
    def iszero(self):
        if self.get_type() == TermType.REGION:
            return False
        else:
            if len(self.x) == 0:
                return True
            for a in self.x:
                if a.isempty():
                    return True
        return False
        
    def fromcomp(x):
        return Term([x.copy()], Comp.empty())
        
    def H(x):
        return Term([x.copy()], Comp.empty())
        
    def I(x, y):
        return Term([x.copy(), y.copy()], Comp.empty())
        
    def Hc(x, z):
        return Term([x.copy()], z.copy())
        
    def Ic(x, y, z):
        return Term([x.copy(), y.copy()], z.copy())
        
    def fcn(fcnname, fcncall, fcnargs):
        cname = fcnname
        return Term(Comp.real(cname), Comp.empty(), reg = Region.universe(), 
                    sn = 0, fcncall = fcncall, fcnargs = fcnargs)
    
    def allcomp(self):
        r = Comp.empty()
        for a in self.x:
            r += a
        r += self.z
        return r
        
    def allcomprv_shallow(self):
        r = Comp.empty()
        for a in self.x:
            r += a
        r += self.z
        return Comp([a for a in r.varlist if a.vartype == IVarType.RV])
        
        
    def get_name(self):
        return self.allcomp().get_name()
    
    def size(self):
        r = self.z.size()
        for a in self.x:
            r += a.size()
        return r
        
    def complexity(self):
        # return (len(self.z) + sum(len(a) for a in self.x)) * 2 + len(self.x) + (not self.z.isempty())
        r = len(self.z) + sum(len(a) for a in self.x) * 2
        if len(self.x) == 1:
            r += 1
        elif len(self.x) >= 3:
            r += len(self.x) * 2
        return r
    
    def get_type(self):
        if self.reg is not None:
            return TermType.REGION
        if len(self.x) == 0:
            return TermType.NIL
        if self.x[0].get_type() == IVarType.REAL:
            return TermType.REAL
        return TermType.IC
        
    def iseps(self):
        if self.get_type() != TermType.REAL:
            return False
        return self.x[0].varlist[0].name == "EPS"
        
    def isone(self):
        if self.get_type() != TermType.REAL:
            return False
        return self.x[0].varlist[0].name == "ONE"
        
    def isinf(self):
        if self.get_type() != TermType.REAL:
            return False
        return self.x[0].varlist[0].name == "INF"
        
    def isrealvar(self):
        if self.get_type() != TermType.REAL:
            return False
        return self.x[0].varlist[0].name != "EPS" and self.x[0].varlist[0].name != "ONE" and self.x[0].varlist[0].name != "INF"
        
    def isnonneg(self):
        if self.get_type() == TermType.IC:
            return len(self.x) <= 2
        if self.isone() or self.iseps() or self.isinf():
            return True
        return False
        
    def isic2(self):
        if self.get_type() == TermType.IC:
            if len(self.x) != 2:
                return False
            return (self.x[0]-self.z).disjoint(self.x[1]-self.z)
            #return ((self.x[0]+self.x[1]+self.z).size() 
            #        == self.x[0].size()+self.x[1].size()+self.z.size())
        return False
        
    def ishc(self):
        if self.get_type() == TermType.IC:
            if len(self.x) != 1:
                return False
            return True
        return False
        
    def isihc2(self):
        if self.get_type() == TermType.IC:
            if len(self.x) != 2:
                return False
            return True
        return False
        
    def ish(self):
        if self.get_type() == TermType.IC:
            if len(self.x) != 1:
                return False
            if not self.z.isempty():
                return False
            return True
        return False
        
    def record_to(self, index):
        if self.get_type() == TermType.REGION:
            if self.reg is not None:
                index.record(self.reg.allcomprv_noaux())
            if self.reg_outer is not None:
                index.record(self.reg_outer.allcomprv_noaux())
            if self.fcnargs is not None:
                for a in self.fcnargs:
                    if isinstance(a, (IVar, Comp, Term, Expr, Region)):
                        a.record_to(index)
            
        for a in self.x:
            a.record_to(index)
        self.z.record_to(index)
        
    def definition(self):
        """Return the definition of this term.
        """
        if self.get_type() == TermType.REGION:
            cname = self.x[0].get_name()
            if cname.find(PsiOpts.settings["fcn_suffix"]) >= 0:
                return self.substituted(self.x[0], Comp.real(cname.replace(PsiOpts.settings["fcn_suffix"], "")))
        return self.copy()
        
    
    def get_shape(self):
        r = []
        for a in itertools.chain(self.z, *(self.x)):
            t = a.get_card()
            if t is None:
                raise ValueError("Cardinality of " + str(a) + " not set. Use " + str(a) + ".set_card(m) to set cardinality.")
                return
            r.append(t)
        return tuple(r)
    
    
    def prob(self, *args):
        """
        Returns the conditional probability mass function at *args.

        Parameters
        ----------
        *args : int
            The indices to query. E.g. (X+Y|Z).prob(2,3,4) is P(X=3, Y=4 | Z=2).
        Returns
        -------
        r : Expr
            The conditional probability as an expression.

        """
        cx = []
        for a in self.x:
            cx += list(a)
        args = tuple(args)
        name = "P(" + ",".join((a.tostring(style = PsiOpts.STR_STYLE_STANDARD) + "=" + str(b)) for a, b in zip(cx, args[len(self.z):])) + "|"
        name += ",".join((a.tostring(style = PsiOpts.STR_STYLE_STANDARD) + "=" + str(b)) for a, b in zip(self.z, args[:len(self.z)])) + ")"
        name += "@@" + str(PsiOpts.STR_STYLE_PSITIP) + "@@"
        name += "(" + "&".join(repr(a) for a in cx) + "|" + repr(self.z) + ")"
        name += ".prob(" + ",".join(str(b) for b in args) + ")"
        name += "@@" + str(PsiOpts.STR_STYLE_LATEX) + "@@"
        name += PsiOpts.settings["latex_prob"] + "(" + ",".join((a.tostring(style = PsiOpts.STR_STYLE_LATEX) + "=" + str(b)) for a, b in zip(cx, args[len(self.z):])) + "|"
        name += ",".join((a.tostring(style = PsiOpts.STR_STYLE_LATEX) + "=" + str(b)) for a, b in zip(self.z, args[:len(self.z)])) + ")"
            
        def fcncall(xdist):
            return xdist[args]
    
        R = Expr.real(name)
        
        return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), Region.universe(), 0, fcncall, [self.copy()]))
    
        # def fcncall(P):
        #     return P[self][args]
    
        # R = Expr.real(name)
        
        # return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), Region.universe(), 0, fcncall, ["model"]))
    
    def pmf(self):
        """ Returns the probability mass function as ExprArray.
        """
        shape = self.get_shape()
        r = ExprArray.zeros(shape)
        for xs in itertools.product(*[range(x) for x in shape]):
            r[xs] = self.prob(*xs)
        return r
        
    @staticmethod
    def fcneval(fcncall, fcnargs):
        if fcncall == "*":
            return fcnargs[0] * fcnargs[1]
        elif fcncall == "/":
            return fcnargs[0] / fcnargs[1]
        elif fcncall == "**":
            return fcnargs[0] ** fcnargs[1]
        else:
            return fcncall(*fcnargs)
    
    def get_fcneval(self, fcnargs):
        return Term.fcneval(self.fcncall, fcnargs)
    
    def value(self, method = "", num_iter = 30, prog = None):
        if self.isone():
            return 1.0
        if self.iseps():
            return 0.0
        if self.isinf():
            return float("inf")
        if self.reg is None:
            return None
        
        if self.sn == 0:
            return None
        
        ms = method.split(",")
        if isinstance(self.reg, RegionOp):
            ms.append("bsearch")
            
        if "bsearch" in ms:
            selfterm = Expr.real(str(self))
            return self.sn * iutil.gbsearch(
                lambda x: self.reg.implies(self.sn * selfterm <= x), 
                num_iter = num_iter)
        else:
            selfterm = Expr.real(str(self))
            cs = self.reg.consonly().imp_flipped()
            index = IVarIndex()
            cs.record_to(index)
            
            r = []
            dual_enabled = None
            val_enabled = None
            if "dual" in ms:
                dual_enabled = True
            if "val" in ms:
                val_enabled = True
            
            cprog = cs.init_prog(index, lp_bounded = False, dual_enabled = dual_enabled, val_enabled = val_enabled)
            cprog.checkexpr_ge0(-self.sn * selfterm, optval = r)
            
            if prog is not None:
                prog.append(cprog)
                
            if len(r) == 0:
                return None
            return r[0] * -self.sn
        
    def get_reg_sgn_bds(self):
        if self.get_type() != TermType.REGION:
            return None
        if self.sn == 0:
            return None
        
        reg = self.reg.copy()
        if isinstance(reg, RegionOp):
            reg = reg.tosimple()
        if reg is None:
            return None
        
        sn = self.sn
        
        tbds = reg.get_lb_ub_eq(self)
        reg.eliminate_term(self)
        reg.simplify_quick()
        
        if sn > 0:
            return (reg, sn, tbds[1])
        else:
            return (reg, sn, tbds[0])
            
    
    def tostring(self, style = 0, tosort = False, add_bracket = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        style = iutil.convert_str_style(style)
        termType = self.get_type()
        
        if termType == TermType.NIL:
            return "0"
            
        elif termType == TermType.REAL:
            return self.x[0].tostring(style = style, tosort = tosort)
            
        elif termType == TermType.IC:
            r = ""
            if len(self.x) == 1:
                if style & PsiOpts.STR_STYLE_LATEX:
                    r += PsiOpts.settings["latex_H"]
                else:
                    r += "H"
            else:
                if style & PsiOpts.STR_STYLE_LATEX:
                    r += PsiOpts.settings["latex_I"]
                else:
                    r += "I"
            r += "("
            
            namelist = [a.tostring(style = style, tosort = tosort) for a in self.x]
            if tosort:
                namelist.sort()
                
            if style & PsiOpts.STR_STYLE_PSITIP:
                r += "&".join(namelist)
            elif style & PsiOpts.STR_STYLE_LATEX:
                r += (PsiOpts.settings["latex_mi_delim"] + " ").join(namelist)
            else:
                r += ";".join(namelist)
                
            if self.z.size() > 0:
                if style & PsiOpts.STR_STYLE_LATEX:
                    r += PsiOpts.settings["latex_cond"]
                else:
                    r += "|" 
                r += self.z.tostring(style = style, tosort = tosort)
            r += ")"
            return r
        
        elif termType == TermType.REGION:
            if self.x[0].varlist[0].name.find(PsiOpts.settings["fcn_suffix"]) >= 0:
                return self.x[0].tostring(style = style, tosort = tosort)
            
            reg = self.reg
            sn = self.sn
            bds = [self.copy_noreg()]
            
            rsb = self.get_reg_sgn_bds()
            if rsb is not None and (not style & PsiOpts.STR_STYLE_PSITIP
                                    or len(rsb[2]) == 1 or rsb[0].isuniverse()):
                reg, sn, bds = rsb
            # elif rsb is not None and rsb[0].isuniverse():
            #     reg, sn, bds = rsb
            #     sn *= -1
            reg_universe = reg.isuniverse(canon = True)
            
            if len(bds) == 0:
                if style & PsiOpts.STR_STYLE_LATEX:
                    if sn > 0:
                        return PsiOpts.settings["latex_infty"]
                    else:
                        if add_bracket:
                            return "(-" + PsiOpts.settings["latex_infty"] + ")"
                        else:
                            return "-" + PsiOpts.settings["latex_infty"]
                else:
                    if sn > 0:
                        return "INF"
                    else:
                        if add_bracket:
                            return "(-" + "INF" + ")"
                        else:
                            return "-" + "INF"

            if style & PsiOpts.STR_STYLE_LATEX:
                r = ""
                if not reg_universe:
                    if sn > 0:
                        r += PsiOpts.settings["latex_sup"]
                    else:
                        r += PsiOpts.settings["latex_inf"]
                    r += "_{"
                    r += reg.tostring(style = style & ~PsiOpts.STR_STYLE_LATEX_ARRAY, 
                                      tosort = tosort, small = True, skip_outer_exists = True)
                    r += "}"
                    
                if len(bds) > 1:
                    if sn > 0:
                        r += PsiOpts.settings["latex_min"]
                    else:
                        r += PsiOpts.settings["latex_max"]
                    r += "\\left("
                r += ",\\, ".join(b.tostring(style = style, tosort = tosort, add_bracket = len(bds) == 1 and (add_bracket or not reg_universe)) for b in bds)
                
                if len(bds) > 1:
                    r += "\\right)"
                    
                return r
            else:
                r = ""
                if not reg_universe:
                    r += "("
                    r += reg.tostring(style = style, tosort = tosort, small = True)
                    r += ")"
                    if sn > 0:
                        r += ".maximum"
                    else:
                        r += ".minimum"
                    r += "("
                    
                if len(bds) > 1:
                    if style & PsiOpts.STR_STYLE_PSITIP:
                        if sn > 0:
                            r += "emin"
                        else:
                            r += "emax"
                    else:
                        if sn > 0:
                            r += "min"
                        else:
                            r += "max"
                            
                    r += "("
                    
                r += ", ".join(b.tostring(style = style, tosort = tosort, add_bracket = len(bds) == 1 and (add_bracket or not reg_universe)) for b in bds)
                
                if len(bds) > 1:
                    r += ")"
                    
                if not reg_universe:
                    r += ")"
                return r
            
        
        return ""
    
        
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"],
                             tosort = PsiOpts.settings["str_tosort"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.settings["str_style_repr"])
    
    def _latex_(self):
        return self.tostring(iutil.convert_str_style("latex"))
        
        
    def __hash__(self):
        #return hash(self.tostring(tosort = True))
        return hash((frozenset(hash(a) for a in self.x), hash(self.z)))
    
    def simplify(self, reg = None, bnet = None):
        if self.get_type() == TermType.IC:
            for i in range(len(self.x)):
                self.x[i] -= self.z
                
            for i in range(len(self.x)):
                if self.x[i].isempty():
                    self.x = []
                    self.z = Comp.empty()
                    return self
                    
            for i in range(len(self.x)):
                for j in range(len(self.x)):
                    if i != j and (self.x[j] is not None) and self.x[i].super_of(self.x[j]):
                        self.x[i] = None
                        break
            self.x = [a for a in self.x if a is not None]
            
            if bnet is not None:
                for i in range(len(self.x) + 1):
                    cc = None
                    if i == len(self.x):
                        cc = self.z
                    else:
                        if len(self.x) == 1:
                            continue
                        cc = self.x[i]
                    j = 0
                    while j < len(cc.varlist):
                        if bnet.check_ic(Expr.Ic(cc[j],
                            sum((self.x[i2] for i2 in range(len(self.x)) if i2 != i), Comp.empty()),
                            (Comp.empty() if i == len(self.x) else self.z) + 
                            sum((cc[j2] for j2 in range(len(cc.varlist)) if j2 != j), Comp.empty()))):
                            
                            cc.varlist.pop(j)
                        else:
                            j += 1
                        
                if len(self.x) == 3:
                    for i in range(len(self.x)):
                        if bnet.check_ic(Expr.Ic(self.x[(i + 1) % 3], 
                                                 self.x[(i + 2) % 3],
                                                 self.x[i] + self.z)):
                            self.x.pop(i)
                            break
                        
                if len(self.x) == 2:
                    for i in range(2):
                        j = 0
                        while j < len(self.x[i]):
                            if bnet.check_ic(Expr.Ic(self.x[i][j], self.x[1 - i], self.z)):
                                self.z.varlist.append(self.x[i].varlist[j])
                                self.x[i].varlist.pop(j)
                            else:
                                j += 1

                
                
        return self
    
    def simplified(self, reg = None, bnet = None):
        r = self.copy()
        r.simplify(reg, bnet)
        return r
    
    def match_x(self, other):
        viss = [-1] * len(self.x)
        viso = [-1] * len(other.x)
        for i in range(len(self.x)):
            for j in range(len(other.x)):
                if viso[j] < 0 and self.x[i] == other.x[j]:
                    viss[i] = j
                    viso[j] = i
                    break
        return (viss, viso)
    
    def __eq__(self, other):
        if self.z != other.z:
            return False
        if len(self.x) != len(other.x):
            return False
        
        viso = [-1] * len(other.x)
        for i in range(len(self.x)):
            found = False
            for j in range(len(other.x)):
                if viso[j] < 0 and self.x[i] == other.x[j]:
                    found = True
                    viso[j] = i
                    break
            if not found:
                return False
        
#        (viss, viso) = self.match_x(other)
#        if -1 in viso:
#            return False
        return True
        
        
    def try_iadd(self, other):
        if self.iseps() and other.iseps():
            return True
        
        # if self.isinf():
        #     return True
        
        if self.get_type() == TermType.IC and other.get_type() == TermType.IC:
            
            # H(X|Y) + I(X;Y) = H(X)
            if len(self.x) + 1 == len(other.x):
                (viss, viso) = self.match_x(other)
                if viso.count(-1) == 1:
                    j = viso.index(-1)
                    if other.x[j].disjoint(other.z) and self.z == other.x[j] + other.z:
                        self.z -= other.x[j]
                        return True
                        
            # H(X|Y) + H(Y) = H(X,Y)
            if len(self.x) == len(other.x):
                (viss, viso) = self.match_x(other)
                if viss.count(-1) == 1 and viso.count(-1) == 1:
                    i = viss.index(-1)
                    j = viso.index(-1)
                    if other.x[j].disjoint(other.z) and self.z == other.x[j] + other.z:
                        self.x[i] += other.x[j]
                        self.z -= other.x[j]
                        return True
                        
        return False
        
        
    def try_isub(self, other):
        
        if self.get_type() == TermType.IC and other.get_type() == TermType.IC:
            
            # H(X) - I(X;Y) = H(X|Y)
            if len(self.x) + 1 == len(other.x) and self.z == other.z:
                (viss, viso) = self.match_x(other)
                if viso.count(-1) == 1:
                    j = viso.index(-1)
                    self.z += other.x[j]
                    return True
            
            # H(X) - H(X|Y) = I(X;Y)
            if len(self.x) == len(other.x) and other.z.super_of(self.z):
                (viss, viso) = self.match_x(other)
                if viso.count(-1) == 0:
                    self.x.append(other.z - self.z)
                    return True
            
            # H(X,Y) - H(X) = H(Y|X)
            if len(self.x) == len(other.x) and self.z == other.z:
                (viss, viso) = self.match_x(other)
                if viss.count(-1) == 1 and viso.count(-1) == 1:
                    i = viss.index(-1)
                    j = viso.index(-1)
                    if self.x[i].super_of(other.x[j]):
                        self.x[i] -= other.x[j]
                        self.z += other.x[j]
                        return True
            
            # H(X,Y) - H(X|Y) = H(Y)
            if len(self.x) == len(other.x) and other.z.super_of(self.z):
                (viss, viso) = self.match_x(other)
                if viss.count(-1) == 1 and viso.count(-1) == 1:
                    i = viss.index(-1)
                    j = viso.index(-1)
                    if self.x[i].super_of(other.x[j]):
                        if self.x[i] - other.x[j] == other.z - self.z:
                            self.x[i] -= other.x[j]
                            return True
                    
        return False
        
        
    def __and__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if isinstance(other, Comp):
            other = Term.H(other)
        return Term([a.copy() for a in self.x] + [a.copy() for a in other.x], self.z + other.z)
        
    
    def __or__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if isinstance(other, Comp):
            other = Term.H(other)
        if isinstance(other, int):
            return self.copy()
        return Term([a.copy() for a in self.x], self.z + other.allcomp())
        
        
        
    def istight(self, canon = False):
        if self.get_type() == TermType.REGION:
            if self.reg_outer is None:
                return True
            if canon:
                return False
            else:
                return self.reg_outer.implies(self.reg)
        return True

    def tighten(self):
        if self.istight(canon = False):
            self.reg_outer = None
    
    def lu_bound(self, sn, name = None):
        if self.get_type() != TermType.REGION:
            return self.copy()
        
        r = self.copy()
        if name is None:
            name = self.x[0].get_name()
            if sn > 0:
                name += "_LB"
            else:
                name += "_UB"

        r.substitute(self.x[0], Comp.real(name))
        if sn * self.sn < 0:
            if r.reg_outer is not None:
                r.reg = r.reg_outer
        r.reg_outer = None
        
        return r
    
    def lower_bound(self, name = None):
        return self.lu_bound(1, name = name)

    def upper_bound(self, name = None):
        return self.lu_bound(-1, name = name)
        


    def ispresent(self, x):
        """Return whether any variable in x appears here"""
        
        if isinstance(x, IVar):
            x = Comp([x])
        if not isinstance(x, str) and not isinstance(x, Comp):
            x = x.allcomp()
        
        if self.get_type() == TermType.REGION:
            if self.reg is not None and self.reg.ispresent(x):
                return True
            if self.reg_outer is not None and self.reg_outer.ispresent(x):
                return True
            if self.fcnargs is not None:
                for a in self.fcnargs:
                    if isinstance(a, (IVar, Comp, Term, Expr, Region)):
                        if a.ispresent(x):
                            return True
        
        xlist = []
        if isinstance(x, str):
            xlist = [x]
        else:
            xlist = x.varlist
            
        for y in xlist:
            for a in self.x:
                if a.ispresent(y):
                    return True
            if self.z.ispresent(y):
                return True
        return False
        
    def rename_var(self, name0, name1):
        if self.get_type() == TermType.REGION:
            if self.reg is not None:
                self.reg.rename_var(name0, name1)
            if self.reg_outer is not None:
                self.reg_outer.rename_var(name0, name1)
            if self.fcnargs is not None:
                for a in self.fcnargs:
                    if isinstance(a, (IVar, Comp, Term, Expr, Region)):
                        a.rename_var(name0, name1)
        for a in self.x:
            a.rename_var(name0, name1)
        self.z.rename_var(name0, name1)
            
    def rename_map(self, namemap):
        """Rename according to name map
        """
        if self.get_type() == TermType.REGION:
            if self.reg is not None:
                self.reg.rename_map(namemap)
            if self.reg_outer is not None:
                self.reg_outer.rename_map(namemap)
            if self.fcnargs is not None:
                for a in self.fcnargs:
                    if isinstance(a, (IVar, Comp, Term, Expr, Region)):
                        a.rename_var(name0, name1)
        for a in self.x:
            a.rename_map(namemap)
        self.z.rename_map(namemap)
        return self
    
    @fcn_substitute
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound)"""
        if self.get_type() == TermType.REGION:
            if self.reg is not None:
                self.reg.substitute(v0, v1)
            if self.reg_outer is not None:
                self.reg_outer.substitute(v0, v1)
            if self.fcnargs is not None:
                for a in self.fcnargs:
                    if isinstance(a, (IVar, Comp, Term, Expr, Region)):
                        a.substitute(v0, v1)
        for a in self.x:
            a.substitute(v0, v1)
        self.z.substitute(v0, v1)
        
    def substituted(self, *args, **kwargs):
        """Substitute variable v0 by v1 (v1 can be compound), return result"""
        r = self.copy()
        r.substitute(*args, **kwargs)
        return r
    
    def get_var_avoid(self, v):
        r = None
        for a in self.x + [Comp.empty()]:
            b = a + self.z
            if b.ispresent(v):
                b = b - v
                if r is None:
                    r = b
                else:
                    r = r.inter(b)
                    
        return r
    
    
    
class Expr(IBaseObj):
    """An expression
    """
    
    def __init__(self, terms, mhash = None):
        self.terms = terms
        self.mhash = mhash
    
        
    def copy(self):
        return Expr([(a.copy(), c) for (a, c) in self.terms], self.mhash)
        
        
    def copy_(self, other):
        self.terms = [(a.copy(), c) for (a, c) in other.terms]
        self.mhash = other.mhash
        
    def copy_noreg(self):
        return Expr([(a.copy_noreg(), c) for (a, c) in self.terms], None)
    
    @staticmethod
    def fromcomp(x):
        return Expr([(Term.fromcomp(x), 1.0)])
    
    @staticmethod
    def fromterm(x):
        return Expr([(x.copy(), 1.0)])
    
    @staticmethod
    def fcn(fcncall, name = None):
        """Wrap any function mapping a ConcModel to a number as an Expr. 
        E.g. the Hamming distortion is given by 
        Expr.fcn(lambda P: P[X+Y].mean(lambda x, y: float(x != y))). 
        For optimization using PyTorch, the return value should be a scalar 
        torch.Tensor with gradient information.
        """
        if name is None:
            name = "fcn_" + str(iutil.get_count("fcn"))
        return Expr.fromterm(Term([Comp.real(name)], Comp.empty(), Region.universe(), 0, fcncall, ["model"]))
    
    def find_name(self, *args):
        return self.allcomp().find_name(*args)
    
    def get_const(self):
        r = 0.0
        for (a, c) in self.terms:
            if a.isone():
                r += c
            else:
                return None
        return r
        
    def get_name(self):
        return self.allcomp().get_name()
    
    def __len__(self):
        return len(self.terms)
    
    def __bool__(self):
        return bool(self.terms)
    
    def __getitem__(self, key):
        r = self.terms[key]
        if isinstance(r, list):
            return Expr(r)
        return Expr([r])
        
        
    def allcomprealvar(self):
        r = Comp.empty()
        for a, c in self.terms:
            if a.isrealvar() and a.fcncall is None:
                r += a.x[0]
        return r
    
    def __iadd__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if not isinstance(other, Expr):
            other = Expr.const(other)
        self.terms += [(a.copy(), c) for (a, c) in other.terms]
        self.mhash = None
        return self
        
    def __add__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Expr([(a.copy(), c) for (a, c) in self.terms]
                    + [(a.copy(), c) for (a, c) in other.terms])
        
    def __radd__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Expr([(a.copy(), c) for (a, c) in other.terms]
                    + [(a.copy(), c) for (a, c) in self.terms])
        
    def __neg__(self):
        return Expr([(a.copy(), -c) for (a, c) in self.terms])
        
    def __isub__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if not isinstance(other, Expr):
            other = Expr.const(other)
        self.terms += [(a.copy(), -c) for (a, c) in other.terms]
        self.mhash = None
        return self
        
    def __sub__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Expr([(a.copy(), c) for (a, c) in self.terms]
                    + [(a.copy(), -c) for (a, c) in other.terms])
        
    def __rsub__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Expr([(a.copy(), c) for (a, c) in other.terms]
                    + [(a.copy(), -c) for (a, c) in self.terms])
        
    def __imul__(self, other):
        if isinstance(other, Expr):
            tother = other.get_const()
            if tother is None:
                return self * other
                # raise ValueError("Multiplication with non-constant expression is not supported.")
            else:
                other = tother
        self.terms = [(a, c * other) for (a, c) in self.terms]
        self.mhash = None
        return self
        
    def __mul__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        
        if isinstance(other, int) or isinstance(other, float):
            if other == 0:
                return Expr.zero()
            
        if isinstance(other, Expr):
            tother = other.get_const()
            if tother is None:
                tself = self.get_const()
                if tself is None:
                    # raise ValueError("Multiplication with non-constant expression is not supported.")
                    return Expr.fromterm(Term(Comp.real(
                        iutil.fcn_name_maker("*", [self, other], lname = PsiOpts.settings["latex_times"] + " ", infix = True)
                        ), reg = Region.universe(), fcncall = "*", fcnargs = [self, other]))
                else:
                    return other * tself
            else:
                other = tother
        return Expr([(a.copy(), c * other) for (a, c) in self.terms])
        
    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            if other == 0:
                return Expr.zero()
            
        if isinstance(other, Expr):
            tother = other.get_const()
            if tother is None:
                # raise ValueError("Multiplication with non-constant expression is not supported.")
                return Expr.fromterm(Term(Comp.real(
                    iutil.fcn_name_maker("*", [other, self], lname = PsiOpts.settings["latex_times"] + " ", infix = True)
                    ), reg = Region.universe(), fcncall = "*", fcnargs = [other, self]))
            else:
                other = tother
        return Expr([(a.copy(), c * other) for (a, c) in self.terms])
        
    def __itruediv__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        
        if isinstance(other, Expr):
            tother = other.get_const()
            if tother is None:
                return self / other
                # raise ValueError("In-place division with non-constant expression is not supported.")
            else:
                other = tother
        self.terms = [(a, c / other) for (a, c) in self.terms]
        self.mhash = None
        return self
    
    def __pow__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        return Expr.fromterm(Term(Comp.real(
            iutil.fcn_name_maker("**", [self, other], lname = "^", infix = True, latex_group = True)
            ), reg = Region.universe(), fcncall = "**", fcnargs = [self, other]))
        
    def __ipow__(self, other):
        return self ** other
        
    def __rpow__(self, other):
        return Expr.const(other) ** self
        
    def record_to(self, index):
        for (a, c) in self.terms:
            a.record_to(index)
    
        
    def allcomp(self):
        r = Comp.empty()
        for (a, c) in self.terms:
            r += a.allcomp()
        return r
        
    def allcomprv_shallow(self):
        r = Comp.empty()
        for (a, c) in self.terms:
            r += a.allcomprv_shallow()
        return r
        
    def size(self):
        return len(self.terms)
        
    def iszero(self):
        """Whether the expression is zero"""
        return len(self.terms) == 0
        
    def setzero(self):
        """Set expression to zero"""
        self.terms = []
        self.mhash = None
        
    def isnonneg(self):
        """Whether the expression is always nonnegative"""
        for (a, c) in self.terms:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if c < 0:
                return False
            if not a.isnonneg():
                return False
        return True
        
    def isnonpos(self):
        """Whether the expression is always nonpositive"""
        for (a, c) in self.terms:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if c > 0:
                return False
            if not a.isnonneg():
                return False
        return True
        
    def isnonneg_ic2(self):
        for (a, c) in self.terms:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if c < 0:
                return False
            if not a.isic2():
                return False
        return True
        
    def isnonpos_ic2(self):
        for (a, c) in self.terms:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if c > 0:
                return False
            if not a.isic2():
                return False
        return True
        
    def isnonpos_hc(self):
        for (a, c) in self.terms:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if c > 0:
                return False
            if not a.ishc():
                return False
        return True
    
    def isrealvar(self):
        if len(self.terms) != 1:
            return False
        a, c = self.terms[0]
        if abs(c - 1.0) > PsiOpts.settings["eps"]:
            return False
        return a.isrealvar()

    @staticmethod
    def zero():
        """The constant zero expression."""
        return Expr([])
    
    @staticmethod
    def H(x):
        """Entropy."""
        return Expr([(Term.H(x), 1.0)])
        
    @staticmethod
    def I(x, y):
        """Mutual information."""
        return Expr([(Term.I(x, y), 1.0)])
        
    @staticmethod
    def Hc(x, z):
        """Conditional entropy."""
        return Expr([(Term.Hc(x, z), 1.0)])
        
    @staticmethod
    def Ic(x, y, z):
        """Conditional mutual information."""
        return Expr([(Term.Ic(x, y, z), 1.0)])
        
        
    @staticmethod
    def real(name):
        """Real variable."""
        if isinstance(name, IVar):
            return Expr([(Term([Comp([name])], Comp.empty()), 1.0)])
        if isinstance(name, Comp):
            return Expr([(Term([name], Comp.empty()), 1.0)])
        return Expr([(Term([Comp.real(name)], Comp.empty()), 1.0)])
    
        
    @staticmethod
    def eps():
        """Epsilon."""
        return Expr([(Term([Comp([IVar.eps()])], Comp.empty()), 1.0)])
        
    @staticmethod
    def one():
        """One."""
        return Expr([(Term([Comp([IVar.one()])], Comp.empty()), 1.0)])
        
    @staticmethod
    def inf():
        """Infinity."""
        return Expr([(Term([Comp([IVar.inf()])], Comp.empty()), 1.0)])
        
    @staticmethod
    def const(c):
        """Constant."""
        if abs(c) <= PsiOpts.settings["eps"]:
            return Expr.zero()
        return Expr([(Term([Comp([IVar.one()])], Comp.empty()), float(c))])
    
    
    def commonpart_coeff(self, v):
        r = 0.0
        for (a, c) in self.terms:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if a.get_type() == TermType.IC:
                if not isinstance(v, list) and not isinstance(v, tuple):
                    if not a.z.ispresent(v) and all(t.ispresent(v) for t in a.x):
                        r += c
                else:
                    if all(not t.ispresent(vt) for t in a.x for vt in v):
                        continue
                    
                    # if len(a.x) > 2:
                    #     return None
                    
                    v2 = [vt for vt in v if not a.z.ispresent(vt)]
                    if len(v2) == 0:
                        continue
                    
                    if any(all(not t.ispresent(vt) for vt in v2) for t in a.x):
                        continue
                    
                    if len(a.x) > 2:
                        tr = (Expr.fromterm(Term(a.x[1:], a.z)) 
                              - Expr.fromterm(Term(a.x[1:], a.z+a.x[0]))).commonpart_coeff(v)
                        if tr is None:
                            return None
                        r += c * tr
                        
                    else:
                        for vt in v2:
                            tpres = [t.ispresent(vt) for t in a.x]
                            if all(tpres):
                                r += c * numpy.inf
                            elif not any(tpres):
                                break
                        else:
                            r += c
        return r
        
    
    def ent_coeff(self, v):
        r = 0.0
        for (a, c) in self.terms:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if a.get_type() == TermType.IC:
                if all(t.ispresent(v) for t in a.x) and not a.z.ispresent(v):
                    r += c
        return r
    
    def var_mi_only(self, v):
        return abs(self.ent_coeff(v)) <= PsiOpts.settings["eps"]
        
    def istight(self, canon = False):
        return all(a.istight(canon) for a, c in self.terms)

    def tighten(self):
        for a, c in self.terms:
            a.tighten()
            
    def lu_bound(self, sn, name = None):
        return Expr([(a.lu_bound(1 if sn * c >= 0 else -1, name = name), c) for a, c in self.terms])
    
    def lower_bound(self, name = None):
        return self.lu_bound(1, name = name)

    def upper_bound(self, name = None):
        return self.lu_bound(-1, name = name)
        

    def get_reg_sgn_bds(self):
        reg = Region.universe()
        sn = 0
        bds = []
        rest = Expr.zero()
        for (a, c) in self.terms:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            t = a.get_reg_sgn_bds()
            if t is None:
                rest.terms.append((a, c))
            else:
                if sn != 0:
                    return None
                reg, sn, bds = t
                bds = [b * c for b in bds]
                if c < 0:
                    sn = -sn
        
        return (reg, sn, [b + rest for b in bds])
        
        
    def tostring(self, style = 0, tosort = False, add_bracket = False, tosort_pm = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        
        style = iutil.convert_str_style(style)
        termlist = self.terms
        if tosort:
            termlist = sorted(termlist, key=lambda a: (-round(a[1] * 1000.0), 
                                   a[0].tostring(style = style, tosort = tosort)))
        elif tosort_pm:
            termlist = sorted(termlist, key=lambda a: a[1] < 0)


        use_bracket = add_bracket and len(termlist) >= 2
        
        r = ""
        if use_bracket:
            if style & PsiOpts.STR_STYLE_LATEX:
                r += "\\left("
            else:
                r += "("
                
        first = True
        for (a, c) in termlist:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if c > 0.0 and not first:
                r += "+"
            if a.isone():
                r += iutil.float_tostr(c)
            else:
                need_bracket = False
                if abs(c - 1.0) < PsiOpts.settings["eps"]:
                    pass
                elif abs(c + 1.0) < PsiOpts.settings["eps"]:
                    r += "-"
                    need_bracket = True
                else:
                    r += iutil.float_tostr(c, style)
                    if style & PsiOpts.STR_STYLE_PSITIP:
                        r += "*"
                    need_bracket = True
                r += a.tostring(style = style, tosort = tosort, add_bracket = need_bracket)
            first = False
            
        if r == "":
            return "0"
        
        if use_bracket:
            if style & PsiOpts.STR_STYLE_LATEX:
                r += "\\right)"
            else:
                r += ")"
                
        return r
        
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"], 
                             tosort = PsiOpts.settings["str_tosort"])
    
    def __repr__(self):
        if PsiOpts.settings.get("repr_simplify", False):
            return self.simplified().tostring(PsiOpts.settings["str_style_repr"])
        return self.tostring(PsiOpts.settings["str_style_repr"])
        
    
    def _latex_(self):
        if PsiOpts.settings.get("repr_simplify", False):
            return self.simplified().tostring(iutil.convert_str_style("latex"))
        return self.tostring(iutil.convert_str_style("latex"))
        
    
    def __hash__(self):
        if self.mhash is None:
            #self.mhash = hash(self.tostring(tosort = True))
            self.mhash = hash(tuple(sorted((hash(a), c) for a, c in self.terms)))
            
        return self.mhash
        
    def table(self, *args, **kwargs):
        """Plot the information diagram as a Karnaugh map.
        """
        return universe().table(*args, self, **kwargs)
        
    def venn(self, *args, **kwargs):
        """Plot the information diagram as a Venn diagram.
        Can handle up to 5 random variables (uses Branko Grunbaum's Venn diagram for n=5).
        """
        return universe().venn(*args, self, **kwargs)
        
    def isregtermpresent(self):
        for (a, c) in self.terms:
            rvs = a.allcomprv_shallow()
            for b in rvs.varlist:
                if b.reg is not None:
                    return True
            if a.get_type() == TermType.REGION:
                return True
        return False
    
    def sortIc(self):
        def sortkey(a):
            x = a[0]
            if x.isic2():
                return x.size()
            else:
                return 100000
            
        self.terms.sort(key=sortkey)
        self.mhash = None
        
    def __le__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([other - self], [], Comp.empty(), Comp.empty(), Comp.empty())
        
    def __lt__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([other - self - Expr.eps()], [], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __ge__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([self - other], [], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __gt__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([self - other - Expr.eps()], [], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __eq__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([], [other - self], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __ne__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return NotImplemented
        #return RegionOp.union([self > other, self < other])
        return ~RegionOp.inter([self == other])
        
    def equiv(self, other):
        """Whether self is equal to other"""
        return (self <= other).check() and (other <= self).check()

    def real_present(self):
        for (a, c) in self.terms:
            if a.get_type() == TermType.REAL:
                return True
        return False

    def complexity(self):
        max_denom = PsiOpts.settings["max_denom"]
        r = 0
        for (a, c) in self.terms:
            frac = fractions.Fraction(c).limit_denominator(max_denom)
            r += min(abs(frac.numerator) + abs(frac.denominator), 8)
            r += a.complexity() * 4
        return r
        
    def sorting_priority(self):
        return int(not self.real_present()) * 100000 + self.complexity()
    
    def get_coeff(self, b):
        """Get coefficient of Term b"""
        r = 0.0
        for (a, c) in self.terms:
            if a == b:
                r += c
        return r
    
    def get_sign(self, b):
        """Return whether Expr is increasing or decreasing in random variable b"""
        r = 0
        for (a, c) in self.terms:
            sn = 0
            if abs(c) <= PsiOpts.settings["eps"] or not a.ispresent(b):
                continue
            sn = 1 if c > 0 else -1
            if a.get_type() == TermType.IC:
                if len(a.x) > 2 or len(a.x) == 0:
                    return 0
                nx = 0
                nz = 0
                if a.z.ispresent(b):
                    nz += 1
                for x in a.x:
                    if x.ispresent(b):
                        nx += 1
                
                if nx + nz != 1:
                    return 0
                
                if len(a.x) == 2 and nz == 1:
                    return 0
                
                if nz == 1:
                    sn = -sn
            else:
                return 0
            
            if r != 0 and r != sn:
                return 0
            r = sn
        return r
    
    def get_var_avoid(self, x):
        r = None
        for (a, c) in self.terms:
            t = a.get_var_avoid(x)
            if r is None:
                r = t
            elif t is not None:
                r = r.inter(t)
        return r
    
    def remove_term(self, b):
        """Remove Term b in place."""
        self.terms = [(a, c) for (a, c) in self.terms if a != b]
        self.mhash = None
        return self
    
    def removed_term(self, b):
        """Remove Term b, return Expr after removal."""
        return Expr([(a, c) for (a, c) in self.terms if a != b])
        
    def symm_sort(self, terms):
        """Sort the random variables in terms assuming symmetry among those terms."""
        self.mhash = None
        index = IVarIndex()
        terms.record_to(index)
        terms = index.comprv
        n = len(terms.varlist)
        v = [0] * n
        
        for (a, c) in self.terms:
            cint = int(round(c * 1000))
            for b, bc in [(t, 2) for t in a.x] + [(a.z, 1)]:
                mask = index.get_mask(b)
                count = iutil.bitcount(mask)
                for i in range(n):
                    if mask & (1 << i):
                        v[i] += cint * bc + count * 5 + 11
        
        vs = sorted(list(range(n)), key = lambda k: v[k], reverse = True)
        tmpvar = Comp.array("#TMPVAR", 0, n)
        for i in range(n):
            self.substitute(terms[i], tmpvar[i])
        for i in range(n):
            self.substitute(tmpvar[i], terms[vs[i]])
        
    
    def coeff_sum(self):
        """Sum of coefficients"""
        return sum([c for (a, c) in self.terms])
    
    def simplify_mul(self, mul_allowed = 0):
        self.mhash = None
        if mul_allowed > 0:
            max_denom = PsiOpts.settings["max_denom"]
            max_denom_mul = PsiOpts.settings["max_denom_mul"]
            denom = 1
            for (a, c) in self.terms:
                denom = iutil.lcm(fractions.Fraction(c).limit_denominator(
                    max_denom).denominator, denom)
                if denom > max_denom_mul:
                    break
                
            if denom > 0 and denom <= max_denom_mul:
                if mul_allowed >= 2:
                    if self.coeff_sum() < 0:
                        denom = -denom
                if all(abs(c * denom - round(c * denom)) <= PsiOpts.settings["eps"] for (a, c) in self.terms):
                    self.terms = [(a, iutil.float_snap(c * denom)) for (a, c) in self.terms]
            
            num = None
            for (a, c) in self.terms:
                tnum = abs(fractions.Fraction(c).limit_denominator(max_denom).numerator)
                if tnum == 0:
                    continue
                if num is None:
                    num = tnum
                else:
                    num = iutil.gcd(num, tnum)
            if num is not None and num > 1:
                self.terms = [(a, iutil.float_snap(c / num)) for (a, c) in self.terms]
                
    def mi_disjoint(self):
        i = 0
        while i < len(self.terms):
            a, c = self.terms[i]
            if a.get_type() == TermType.IC and len(a.x) >= 2:
                xt = a.x[0]
                for j in range(1, len(a.x)):
                    xt = xt.inter(a.x[j])
                if not xt.isempty():
                    self.terms.insert(i + 1, (Term.Hc(xt, a.z), c))
                    for j in range(len(a.x)):
                        a.x[j] = a.x[j] - xt
                    a.z = a.z + xt
                    i += 1
            i += 1

    def simplify(self, reg = None, bnet = None):
        """Simplify the expression in place"""
        ceps = PsiOpts.settings["eps"]
        reduce_coeff = PsiOpts.settings.get("simplify_reduce_coeff", False)

        self.mhash = None
        
        for (a, c) in self.terms:
            a.simplify(reg, bnet)
            
        self.mi_disjoint()

        did = True
        while did:
            did = False
            
            for i in range(len(self.terms)):
                for j in range(i):
                    if self.terms[i][0] == self.terms[j][0]:
                        self.terms[j] = (self.terms[j][0], self.terms[j][1] + self.terms[i][1])
                        self.terms[i] = (self.terms[i][0], 0.0)
                        did = True
                        break
            
            self.terms = [(a, c) for (a, c) in self.terms if abs(c) > ceps and not a.iszero()]
            
            for i in range(len(self.terms)):
                if abs(self.terms[i][1]) > ceps:
                    for j in range(len(self.terms)):
                        if i != j and abs(self.terms[j][1]) > ceps:
                            ci = self.terms[i][1]
                            cj = self.terms[j][1]
                            if abs(ci - cj) <= ceps:
                                if self.terms[i][0].try_iadd(self.terms[j][0]):
                                    self.terms[j] = (self.terms[j][0], 0.0)
                                    did = True
                            elif reduce_coeff and ci * cj > 0:
                                if abs(ci) > abs(cj):
                                    ti = self.terms[i][0].copy()
                                    if self.terms[i][0].try_iadd(self.terms[j][0]):
                                        self.terms[i] = (self.terms[i][0], cj)
                                        self.terms[j] = (ti, ci - cj)
                                        did = True
                                else:
                                    if self.terms[i][0].try_iadd(self.terms[j][0]):
                                        self.terms[j] = (self.terms[j][0], cj - ci)
                                        did = True

                            elif abs(ci + cj) <= ceps:
                                if self.terms[i][0].try_isub(self.terms[j][0]):
                                    self.terms[j] = (self.terms[j][0], 0.0)
                                    did = True
                            elif reduce_coeff and ci * cj < 0:
                                if abs(ci) > abs(cj):
                                    ti = self.terms[i][0].copy()
                                    if self.terms[i][0].try_isub(self.terms[j][0]):
                                        self.terms[i] = (self.terms[i][0], -cj)
                                        self.terms[j] = (ti, ci + cj)
                                        did = True
                                else:
                                    if self.terms[i][0].try_isub(self.terms[j][0]):
                                        self.terms[j] = (self.terms[j][0], cj + ci)
                                        did = True
            
            self.terms = [(a, c) for (a, c) in self.terms if abs(c) > ceps and not a.iszero()]
        
        #self.terms = [(a, iutil.float_snap(c)) for (a, c) in self.terms 
        #              if abs(c) > ceps and not a.iszero()]
        

        return self

    
    def simplified(self, reg = None, bnet = None):
        """Simplify the expression, return simplified expression"""
        r = self.copy()
        r.simplify(reg, bnet)
        return r
    
    def simplified_exhaust_inner(self):
        if len(self.terms) <= 2:
            return None

        scom = self.complexity()
        for i, (a, c) in enumerate(self.terms):
            if a.get_type() == TermType.IC:
                a2 = a.copy()
                for ix in range(len(a2.x)):
                    if len(a2.x[ix]) <= 1:
                        continue
                    for x in a2.x[ix]:
                        ks = [Term([p - x if ip == ix else p.copy() for ip, p in enumerate(a2.x)], a2.z.copy()),
                                Term([x.copy() if ip == ix else p.copy() for ip, p in enumerate(a2.x)], a2.z + (a2.x[ix] - x))]
                        # print(str(a2) + "  " + str(ks[0]) + "  " + str(ks[1]))
                        for it in range(2):
                            expr = Expr([(a4.copy(), c4) if i4 != i else (ks[it].copy(), c) for i4, (a4, c4) in enumerate(self.terms)])
                            # print("  " + str(expr))
                            expr.simplify()
                            # print("  " + str(expr))
                            expr += Expr([(ks[1 - it].copy(), c)])
                            # print("  " + str(expr))
                            expr.simplify()
                            # print("  " + str(expr))
                            # print()
                            if expr.complexity() < scom:
                                return expr

        return None

    
    def simplify_exhaust(self):
        self.terms.sort(key = lambda a: a[0].complexity())
        self.mhash = None
        while True:
            t = self.simplified_exhaust_inner()
            if t is None:
                break
            self.terms = t.terms
            self.mhash = None

    
    def simplified_exhaust(self):
        r = self.copy()
        r.simplify_exhaust()
        return r

    def get_ratio(self, other, skip_simplify = False):
        """Try dividing self by other, return None if self is not scalar multiple of other"""
        
        es = self
        eo = other
        
        if not skip_simplify:
            es = self.simplified()
            eo = other.simplified()
        
        if es.iszero():
            return 0.0
        
        if len(es.terms) != len(eo.terms):
            return None
        
        if eo.iszero():
            return None
            
        rmax = -1e12
        rmin = 1e12
        
        vis = [False] * len(eo.terms)
        
        for i in range(len(es.terms)):
            found = False
            for j in range(len(eo.terms)):
                if not vis[j] and abs(eo.terms[j][1]) > PsiOpts.settings["eps"] and es.terms[i][0] == eo.terms[j][0]:
                    cr = es.terms[i][1] / eo.terms[j][1]
                    rmax = max(rmax, cr)
                    rmin = min(rmin, cr)
                    if rmax > rmin + PsiOpts.settings["eps"]:
                        return None
                    vis[j] = True
                    found = True
                    break
            if not found:
                return None
                
        if rmax <= rmin + PsiOpts.settings["eps"]:
            return (rmax + rmin) * 0.5
        else:
            return None
        
    def __truediv__(self, other):
        if isinstance(other, Expr):
            t = other.get_const()
            if t is None:
                t = self.get_ratio(other)
                if t is not None:
                    return t
                else:
                    return Expr.fromterm(Term(Comp.real(
                        iutil.fcn_name_maker("/", [self, other], lname = "/", infix = True)
                        ), reg = Region.universe(), fcncall = "*", fcnargs = [self, other]))
            other = t
        return Expr([(a.copy(), c / other) for (a, c) in self.terms])
        

    def ispresent(self, x):
        """Return whether any variable in x appears here"""
        for (a, c) in self.terms:
            if a.ispresent(x):
                return True
        return False
        
    def affine_present(self):
        """Return whether this expression is affine."""
        return self.ispresent((Expr.one() + Expr.eps() + Expr.inf()).allcomp())

    def rename_var(self, name0, name1):
        for (a, c) in self.terms:
            a.rename_var(name0, name1)
        self.mhash = None
        
    def rename_map(self, namemap):
        """Rename according to name map
        """
        for (a, c) in self.terms:
            a.rename_map(namemap)
        self.mhash = None
        return self

        
    def definition(self):
        """Return the definition of this expression.
        """
        return Expr([(a.definition(), c) for a, c in self.terms])

    def substitute_rate(self, v0, v1):
        self.mhash = None
        for i, v0c in enumerate(v0):
            if i > 0:
                self.substitute(v0c, v0[0])
        c = self.commonpart_coeff(v0[0])
        self.substitute(v0[0], Comp.empty())
        if c != 0:
            self += v1 * c
        return self
        
    @fcn_substitute
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), in place"""
        self.mhash = None
        if isinstance(v0, Expr):
            if len(v0.terms) > 0:
                t = v0.terms[0][0]
                tmpterms = self.terms
                self.terms = []
                for (a, c) in tmpterms:
                    if a == t:
                        self += v1 * c
                    else:
                        self.terms.append((a, c))
        elif isinstance(v1, Expr):
            self.substitute_rate(v0, v1)
        else:
            for (a, c) in self.terms:
                a.substitute(v0, v1)
        return self

    def substituted(self, *args, **kwargs):
        """Substitute variable v0 by v1 (v1 can be compound), return result"""
        r = self.copy()
        r.substitute(*args, **kwargs)
        return r

    def condition(self, b):
        """Condition on random variable b, in place"""
        for (a, c) in self.terms:
            if a.get_type() == TermType.IC:
                a.z += b
        self.mhash = None
        return self

    def conditioned(self, b):
        """Condition on random variable b, return result"""
        r = self.copy()
        r.condition(b)
        return r

    
    def var_neighbors(self, v):
        r = v.copy()
        for (a, c) in self.terms:
            if a.get_type() == TermType.IC:
                t = sum(a.x, Comp.empty()) + a.z
                if t.ispresent(v):
                    r += t
            elif a.get_type() == TermType.REGION:
                r += a.reg.var_neighbors(v)
        return r
    
    def __abs__(self):
        return eabs(self)
        #return emax(self, -self)
    
    
    def value(self, method = "", num_iter = 30, prog = None):
        r = 0.0
        for (a, c) in self.terms:
            if a.isone():
                r += c
            else:
                t = a.value(method = method, num_iter = num_iter, prog = prog)
                if t is None:
                    return None
                r += c * t
        return r
    
    def solve_prog(self, method = "", num_iter = 30):
        prog = []
        optval = self.value(method = method, num_iter = num_iter, prog = prog)
        if len(prog) > 0:
            return (optval, prog[0])
        else:
            return (optval, None)
        
    
    def __call__(self, method = "", num_iter = 30, prog = None):
        return self.value(method = method, num_iter = num_iter, prog = prog)
    
    def __float__(self):
        return float(self.value())
    
    def __int__(self):
        return int(round(float(self.value())))
    
    def split_lhs(self, lhsvar):
        lhs = []
        rhs = []
        for (a, c) in self.terms:
            index = IVarIndex()
            a.record_to(index)
            t = index.size()
            index2 = IVarIndex()
            lhsvar.record_to(index)
            lhsvar.record_to(index2)
            if index.size() != t + index2.size():
                lhs.append((a, c))
            else:
                rhs.append((a, c))
        return (Expr(lhs), Expr(rhs))
        
    
    def split_posneg(self):
        lhs = []
        rhs = []
        for (a, c) in self.terms:
            if c > 0:
                rhs.append((a, c))
            else:
                lhs.append((a, c))
        return (Expr(lhs), Expr(rhs))
        
    
    def tostring_eqn(self, eqnstr, style = 0, tosort = False, lhsvar = None, prefer_ge = None):
        if prefer_ge is None:
            prefer_ge = PsiOpts.settings["str_eqn_prefer_ge"]
        
        style = iutil.convert_str_style(style)
        lhs = self
        rhs = Expr.zero()
        if lhsvar is not None:
            lhs, rhs = self.split_lhs(lhsvar)
            
            if lhs.iszero() or rhs.iszero():
                lhs, rhs = self.split_posneg()
                if prefer_ge:
                    lhs, rhs = rhs, lhs
                if lhs.iszero():
                    lhs, rhs = rhs, lhs
                elif lhs.get_const() is not None:
                    lhs, rhs = rhs, lhs
                
            rhs *= -1.0
                
            if lhs.coeff_sum() < 0 or (abs(lhs.coeff_sum()) <= PsiOpts.settings["eps"] and rhs.coeff_sum() < 0):
                lhs *= -1.0
                rhs *= -1.0
                eqnstr = iutil.reverse_eqnstr(eqnstr)
        
        return (lhs.tostring(style = style, tosort = tosort, tosort_pm = True) + " "
        + iutil.eqnstr_style(eqnstr, style) + " " + rhs.tostring(style = style, tosort = tosort, tosort_pm = True))
        
        
class BayesNet(IBaseObj):
    """Bayesian network"""
    
    def __init__(self, edges = None):
        self.index = IVarIndex()
        self.parent = []
        self.child = []
        self.fcn = []
        
        if edges is not None:
            self += edges
        
    def copy(self):
        r = BayesNet()
        r.index = self.index.copy()
        r.parent = [list(x) for x in self.parent]
        r.child = [list(x) for x in self.child]
        r.fcn = list(self.fcn)
        return r
    
    def allcomp(self):
        return self.index.comprv.copy()
        
    def get_parents(self, x):
        """
        Get the parents of node x.

        Parameters
        ----------
        x : Comp

        Returns
        -------
        Comp

        """
        i = self.index.get_index(x)
        if i < 0:
            return None
        return sum((self.index.comprv[x] for x in self.parent[i]), Comp.empty())
    
    def get_children(self, x):
        """
        Get the children of node x.

        Parameters
        ----------
        x : Comp

        Returns
        -------
        Comp

        """
        i = self.index.get_index(x)
        if i < 0:
            return None
        return sum((self.index.comprv[x] for x in self.child[i]), Comp.empty())
    
    def get_ancestors(self, x, descendant = False, include_self = True):
        """
        Get the ancestors of node x.

        Parameters
        ----------
        x : Comp

        Returns
        -------
        Comp

        """
        n = self.index.comprv.size()
        vis = [False] * n
        i = self.index.get_index(x)
        vis[i] = include_self
        cstack = [i]
        r = Comp.empty()
        while len(cstack):
            x = cstack.pop()
            if vis[x]:
                r += self.index.comprv[x]
            for y in (self.child[x] if descendant else self.parent[x]):
                if not vis[y]:
                    vis[y] = True
                    cstack.append(y)
        return r
        
    def get_descendants(self, x, **kwargs):
        """
        Get the descendants of node x.

        Parameters
        ----------
        x : Comp

        Returns
        -------
        Comp

        """
        return self.get_ancestors(x, descendant = True, **kwargs)
    
    def edges(self):
        """
        Generator over the edges of the network.

        Yields
        ------
        Pairs of Comp representing the edges.
        """
        n = self.index.comprv.size()
        for i in range(n):
            for j in self.child[i]:
                yield (self.index.comprv[i], self.index.comprv[j])
    
    def add_edge_id(self, i, j):
        if i < 0 or j < 0 or i == j:
            return
        if i not in self.parent[j]:
            self.parent[j].append(i)
        if j not in self.child[i]:
            self.child[i].append(j)
        
    def remove_edge_id(self, i, j):
        if i < 0 or j < 0 or i == j:
            return
        self.parent[j].remove(i)
        self.child[i].remove(j)
        
    def record(self, x):
        self.index.record(x)
        n = self.index.comprv.size()
        while len(self.parent) < n:
            self.parent.append([])
        while len(self.child) < n:
            self.child.append([])
        while len(self.fcn) < n:
            self.fcn.append(False)
        
    @fcn_list_to_list
    def set_fcn(self, x, v = True):
        """Mark variables in x to be functions of their parents."""
        self.record(x)
        for xa in x.varlist:
            i = self.index.get_index(xa)
            if i >= 0:
                self.fcn[i] = v
        
    def is_fcn(self, x):
        """Query whether x is a function of their parents."""
        for xa in x.varlist:
            i = self.index.get_index(xa)
            if i >= 0:
                if not self.fcn[i]:
                    return False
        return True
                
    @fcn_list_to_list
    def add_edge(self, x, y):
        """Add edges from every variable in x to every variable in y.
        Also add edges among variables in y.
        """
        self.record(x)
        self.record(y)
        
        for xa in x.varlist:
            for ya in y.varlist:
                self.add_edge_id(self.index.get_index(xa),
                                 self.index.get_index(ya))
        
        for yi in range(len(y.varlist)):
            for yj in range(yi + 1, len(y.varlist)):
                self.add_edge_id(self.index.get_index(y.varlist[yi]),
                                 self.index.get_index(y.varlist[yj]))
                
        return y
    
    def __iadd__(self, other):
        if isinstance(other, list):
            for x in other:
                self += x
            return self
        
        if isinstance(other, tuple):
            for i in range(len(other) - 1):
                self.add_edge(other[i], other[i + 1])
            return self
        elif isinstance(other, Term):
            self.add_edge(other.z, other.x[0])
            return self
        elif isinstance(other, Comp):
            self.add_edge(Comp.empty(), other)
            return self
        elif isinstance(other, BayesNet):
            for x, y in other.edges():
                self.add_edge(x, y)
            return self
                
        return self
    
    def communicate(self, a, b):
        return self.get_ancestors(a).ispresent(b) and self.get_ancestors(b).ispresent(a)
    
    def scc(self):
        n = self.index.comprv.size()
        r = BayesNet()
        vis = [False] * n
        for i in range(n):
            if vis[i]:
                continue
            cgroup = Comp.empty()
            cparent = Comp.empty()
            for j in range(i, n):
                if self.communicate(self.index.comprv[i], self.index.comprv[j]):
                    vis[j] = True
                    cgroup += self.index.comprv[j]
                    cparent += self.get_parents(self.index.comprv[j])
            
            cparent -= cgroup
            for k in range(len(cgroup)):
                tparent = cparent + cgroup[:k]
                r.add_edge(tparent, cgroup[k])
                if self.is_fcn(cgroup[k]) and tparent.super_of(self.get_parents(cgroup[k])):
                    r.set_fcn(cgroup[k])
                
                
        return r
        
    
    def tsorted(self):
        n = self.index.comprv.size()
        cstack = []
        cnparent = [0] * n
        nrec = 0
        
        r = BayesNet()
        
        for i in range(n - 1, -1, -1):
            cnparent[i] = len(self.parent[i])
            if cnparent[i] == 0:
                cstack.append(i)
                
        while len(cstack) > 0:
            i = cstack.pop()
            r.record(self.index.comprv[i])
            nrec += 1
            for j in reversed(self.child[i]):
                if cnparent[j] > 0:
                    cnparent[j] -= 1
                    if cnparent[j] == 0:
                        cstack.append(j)
                        
        if nrec < n:
            return None
        
        for i in range(n):
            for j in self.parent[i]:
                r.add_edge(self.index.comprv[j], self.index.comprv[i])
        
        for i in range(n):
            if self.fcn[i]:
                r.set_fcn(self.index.comprv[i])
                
        return r
    
    def iscyclic(self):
        return self.tsorted() is None

    def contracted_node(self, x):
        k = self.index.get_index(x)
        if k < 0:
            return self.copy()
        
        n = self.index.comprv.size()
        r = BayesNet()

        for i in range(n):
            if i == k:
                continue
            for j in self.parent[i]:
                if j == k:
                    for j2 in self.parent[k]:
                        if j2 == k:
                            continue
                        r.add_edge(self.index.comprv[j2], self.index.comprv[i])
                    continue
                r.add_edge(self.index.comprv[j], self.index.comprv[i])
        
        for i in range(n):
            if i == k:
                continue
            if self.fcn[i]:
                if self.fcn[k] or (k not in self.parent[i]):
                    r.set_fcn(self.index.comprv[i])
                
        return r
    
    def eliminated(self, x):
        r = self.copy()
        for a in x:
            r = r.contracted_node(a)
        return r

    def check_hc_mask(self, x, z):
        n = self.index.comprv.size()
        cstack = []
        vis = [False] * n
        
        x &= ~z
        
        for i in range(n):
            if x & (1 << i):
                cstack.append(i)
                vis[i] = True
            if z & (1 << i):
                vis[i] = True
        
        while cstack:
            i = cstack.pop()
            if not self.fcn[i]:
                return False
            for j in self.parent[i]:
                if not vis[j]:
                    cstack.append(j)
                    vis[j] = True
        
        return True
    
    def fcn_descendants_mask(self, x):
        n = self.index.comprv.size()
        did = True
        while did:
            did = False
            for i in range(n):
                if self.fcn[i] and not x & (1 << i):
                    if all(x & j for j in self.parent[i]):
                        x |= 1 << i
                        did = True
        return x
        
    def fcn_descendants(self, x):
        return self.index.from_mask(self.fcn_descendants_mask(self.index.get_mask(x)))
    
    def check_ic_mask(self, x, y, z):
        if x < 0 or y < 0 or z < 0:
            return False
        
        z = self.fcn_descendants_mask(z)
        
        x &= ~z
        y &= ~z
        
        if x & y != 0:
            # if not self.check_hc_mask(x & y, z):
            #     return False
            # z |= x & y
            # x &= ~z
            # y &= ~z
            
            return False
        
        if x == 0 or y == 0:
            return True
        
        n = self.index.comprv.size()
        desc = z
        
        cstack = []
        for i in range(n):
            if z & (1 << i) != 0:
                cstack.append(i)
                
        while len(cstack) > 0:
            i = cstack.pop()
            for j in self.parent[i]:
                if desc & (1 << j) == 0:
                    desc |= (1 << j)
                    cstack.append(j)
        
        vis = [0, x]
        cstack = []
        for i in range(n):
            if x & (1 << i) != 0:
                cstack.append((1, i))
        
        while len(cstack) > 0:
            (d, i) = cstack.pop()
            if y & (1 << i) != 0:
                return False
            if z & (1 << i) == 0:
                for j in self.child[i]:
                    if vis[0] & (1 << j) == 0:
                        vis[0] |= (1 << j)
                        cstack.append((0, j))
            if (d == 0 and desc & (1 << i) != 0) or (d == 1 and z & (1 << i) == 0):
                for j in self.parent[i]:
                    if vis[1] & (1 << j) == 0:
                        vis[1] |= (1 << j)
                        cstack.append((1, j))
        
        return True
        
    def check_ic(self, icexpr):
        for a, c in icexpr.terms:
            if a.isihc2():
                if not self.check_ic_mask(self.index.get_mask(a.x[0]), 
                                          self.index.get_mask(a.x[1]), 
                                          self.index.get_mask(a.z)):
                    return False
            elif a.ishc():
                if not self.check_hc_mask(self.index.get_mask(a.x[0]), 
                                          self.index.get_mask(a.z)):
                    return False
            else:
                return False
            
        return True
    
    def from_ic_inplace(self, icexpr, roots = None):
        n_root = 0
        if roots is not None:
            self.record(roots)
            n_root = self.index.comprv.size()
            
        ics = []
        for (a, c) in icexpr.terms:
            if a.isic2():
                self.record(a.x[0])
                self.record(a.x[1])
                self.record(a.z)
                x0 = self.index.get_mask(a.x[0])
                x1 = self.index.get_mask(a.x[1])
                z = self.index.get_mask(a.z)
                x0 &= ~z
                x1 &= ~z
                x0 &= ~x1
                ics.append((x0, x1, z))
            elif a.ishc():
                self.record(a.x[0])
                self.record(a.z)
                x0 = self.index.get_mask(a.x[0])
                z = self.index.get_mask(a.z)
                x0 &= ~z
                ics.append((x0, -1, z))
        
        
        n = self.index.comprv.size()
        ics = [(x0, x1 if x1 >= 0 else (1 << n) - 1 - x0 - z, z) for x0, x1, z in ics]
        
        numcond = [0] * n
        for x0, x1, z in ics:
            for i in range(n):
                if z & (1 << i):
                    numcond[i] += 1
        
        ilist = list(range(n_root, n))
        ilist.sort(key = lambda i: numcond[i])
        
        n2 = n - n_root
        
        xk = 0
        zk = 0
        vis = 0
        
        np2 = (1 << n2)
        dp = [1000000] * np2
        dpi = [-1] * np2
        dped = [-1] * np2
        dp[np2 - 1] = 0
        for tvis in range(np2 - 2, -1, -1):
            vis = (tvis << n_root) | ((1 << n_root) - 1)
            nvis = iutil.bitcount(vis)
            for i in ilist:
                if vis & (1 << i) == 0:
                    nedge = nvis * (10000 - numcond[i]) + dp[tvis | (1 << (i - n_root))]
                    # nedge = nvis + dp[tvis | (1 << (i - n_root))]
                    if nedge < dp[tvis]:
                        dp[tvis] = nedge
                        dpi[tvis] = i
                        dped[tvis] = vis
                    for (x0, x1, z) in ics:
                        if z & ~vis != 0:
                            continue
                        if x0 & (1 << i) != 0:
                            xk = x1
                            zk = (z + x0 - (1 << i)) & vis
                        elif x1 & (1 << i) != 0:
                            xk = x0
                            zk = (z + x1 - (1 << i)) & vis
                        else:
                            continue
                        if vis & ~(zk | xk) != 0:
                            continue
                        nedge = iutil.bitcount(zk) * (10000 - numcond[i]) + dp[tvis | (1 << (i - n_root))]
                        # nedge = iutil.bitcount(zk) + dp[tvis | (1 << (i - n_root))]
                        if nedge < dp[tvis]:
                            dp[tvis] = nedge
                            dpi[tvis] = i
                            dped[tvis] = zk
        
        #for vis in range(np2):
        #    print("{0:b}".format(vis) + " " + str(dp[vis]) + " " + str(dpi[vis]) + " " + "{0:b}".format(dped[vis]))
        cvis = 0
        for it in range(n2):
            i = dpi[cvis]
            ed = dped[cvis]
            for j in range(n):
                if ed & (1 << j) != 0:
                    self.add_edge_id(j, i)
            cvis |= (1 << (i - n_root))
            
    
    def from_ic(icexpr, roots = None):
        """Construct Bayesian network from the sum of conditional mutual 
        information terms (Expr).
        """
        r = BayesNet()
        r.from_ic_inplace(icexpr, roots)
        return r
    
    def from_ic_list(icexpr, roots = None):
        """Construct a list of Bayesian networks from the sum of conditional 
        mutual information terms (Expr).
        """
        r = []
        icexpr = icexpr.copy()
        while not icexpr.iszero():
            
            t = BayesNet.from_ic(icexpr, roots = roots).tsorted()
            olen = len(icexpr.terms)
            icexpr.terms = [(a, c) for a, c in icexpr.terms if not t.check_ic(Expr.fromterm(a))]
            icexpr.mhash = None
            
            if len(icexpr.terms) == olen:
                tl = BayesNet.from_ic_list(Expr.fromterm(icexpr.terms[0][0]), roots = roots)
                icexpr.terms = [(a, c) for a, c in icexpr.terms if not any(t.check_ic(Expr.fromterm(a)) for t in tl)]
                icexpr.mhash = None
                r += tl
                continue
            
            r.append(t)
        return r
    
    def get_markov(self):
        """Get Markov chains as a list of lists.
        """
            
        cs = self.tsorted()
        n = cs.index.comprv.size()
        r = []
        
        def parent_min(i):
            r = i
            for j in cs.parent[i]:
                r = min(r, parent_min(j))
            return r
        
        def parent_segment(i):
            if not cs.parent[i]:
                return -1
            m = min(cs.parent[i])
            if len(cs.parent[i]) != i - m:
                return -1
            
            for j in range(m + 1, i):
                if set(cs.parent[j]) != set(cs.parent[m]).union(range(m, j)):
                    return -1
            
            return m
        
        def node_segment(i):
            for j in range(i - 1, -1, -1):
                if set(cs.parent[i]) != set(cs.parent[j]).union(range(j, i)):
                    return j + 1
            return 0
        
        def recur(st, en):
            if st >= en:
                return
            
            i = en - 1
            
            cms = [en, parent_min(i)]
            if cms[-1] > st:
                while cms[-1] > st:
                    t = parent_min(cms[-1] - 1)
                    cms.append(t)
                tl = []
                for i in range(len(cms) - 1):
                    if i:
                        tl.append([])
                    tl.append(list(range(cms[i + 1], cms[i])))
                r.append(tl)
                for i in range(len(cms) - 1):
                    recur(cms[i + 1], cms[i])
                
                return
            
            if len(cs.parent[i]) >= i - st:
                recur(st, i)
                return
            
            m = node_segment(i)
            t = [list(range(m, i + 1))]
            i = m
            
            while i >= st:
                t.append(cs.parent[i])
                m = parent_segment(i)
                if m < 0:
                    break
                i = m
                
            t.append([j for j in range(st, i) if j not in cs.parent[i]])
            
            while not t[-1]:
                t.pop()
                
            if len(t) >= 3:
                r.append(t)
            
            recur(st, i)
        
        recur(0, n)
        
        return [[sum((cs.index.comprv[a] for a in b), Comp.empty()) for b in reversed(tl)]
                for tl in reversed(r)]
    
    
    def get_ic_sorted(self):
        n = self.index.comprv.size()
        r = Expr.zero()
        compvis = Comp.empty()
        for i in range(n):
            ps = Comp.empty()
            for j in self.parent[i]:
                ps += self.index.comprv[j]
            y = compvis - ps
            if y.size() > 0:
                r += Expr.Ic(self.index.comprv[i], y, ps)
            compvis += self.index.comprv[i]
        return r
            
    def get_ic(self):
        return self.tsorted().get_ic_sorted()
            
    def get_region(self):
        n = self.index.comprv.size()
        r = Expr.zero()
        for i in range(n):
            if self.fcn[i]:
                ps = Comp.empty()
                for j in self.parent[i]:
                    ps += self.index.comprv[j]
                r += Expr.Hc(self.index.comprv[i], ps)
        
        r += self.get_ic()
        return r == 0
        
    def tostring(self, tsort = True):
        if tsort:
            tself = self.tsorted()
            if tself is not None:
                return tself.tostring(tsort = False)
        
        n = self.index.comprv.size()
        r = ""
        for i in range(n):
            first = True
            for j in self.parent[i]:
                if not first:
                    r += ","
                r += self.index.comprv.varlist[j].tostring()
                first = False
            r += " -> " + self.index.comprv.varlist[i].tostring() + ("*" if self.fcn[i] else "") + "\n"
        return r
        
    def __str__(self):
        return self.tostring()
    
    def __repr__(self):
        return self.tostring()
    
    def _latex_(self):
        return self.tostring()
        
        
    def __hash__(self):
        return hash(self.tostring())
        
    def graph(self, tsort = True, shape = "plaintext", lr = True, groups = None, ortho = False):
        """Return the graphviz digraph of the network that can be displayed in the console.
        """
        if tsort:
            return self.tsorted().graph(tsort = False, shape = shape, lr = lr, groups = groups)
        
        n = self.index.comprv.size()
        r = graphviz.Digraph()
        if lr:
            r.graph_attr["rankdir"] = "LR"
        if ortho:
            r.graph_attr["splines"] = "ortho"

        if groups is None:
            groups = []
        
        remrv = self.index.comprv.copy()
        for gi, g in enumerate(groups):
            with r.subgraph(name = "cluster_" + str(gi)) as rs:
                rs.attr(color = "blue")
                for c in g:
                    if not remrv.ispresent(c):
                        continue
                    remrv -= c
                    i = self.index.get_index(c)
                    rs.node(self.index.comprv[i].get_name(), str(self.index.comprv[i])
                           + ("*" if self.fcn[i] else ""), shape = shape)
                
            
        for i in range(n):
            if not remrv.ispresent(self.index.comprv[i]):
                continue
            r.node(self.index.comprv[i].get_name(), str(self.index.comprv[i])
                   + ("*" if self.fcn[i] else ""), shape = shape)
        
        for i in range(n):
            for j in self.parent[i]:
                r.edge(self.index.comprv[j].get_name(), self.index.comprv[i].get_name())
        
        return r
    
    
class ValIndex:
    def __init__(self, v = None):
        if v is None:
            self.v = None
            self.vmap = {}
        else:
            self.v = v
            self.vmap = {}
            for i, x in enumerate(v):
                self.vmap[x] = i
    
    def get_index(self, x):
        return self.vmap.get(x, -1)
    
    
class ConcDist(IBaseObj):
    """Concrete distributions / conditional distributions of random variables."""
    
    def convert_shape(x):
        if x is None:
            return tuple()
        if isinstance(x, int):
            return (x,)
        if isinstance(x, Comp):
            return x.get_shape()
        if isinstance(x, ConcDist):
            return x.shape_out
        return tuple(x)
    
    def convert_shape_pair(x):
        if x is None:
            return (tuple(), tuple())
        if isinstance(x, int):
            return (tuple(), (x,))
        if isinstance(x, Term):
            return (x.z.get_shape(), sum((t.get_shape() for t in x.x), tuple()))
        if isinstance(x, Comp) or isinstance(x, ConcDist):
            return (tuple(), ConcDist.convert_shape(x))
        if isinstance(x, tuple) and (len(x) == 0 or isinstance(x[0], int)):
            return (tuple(), tuple(x))
        return (ConcDist.convert_shape(x[0]), ConcDist.convert_shape(x[1]))
        
        
    
    def __init__(self, p = None, num_in = None, shape = None, shape_in = None, shape_out = None, isvar = False, randomize = False, isfcn = False):
        self.isvar = isvar
        self.iscache = False
            
        self.v = None
        
        # if p is None and shape_in is None and shape_out is None:
        #     self.p = None
        #     return
        
        self.expr = None
        self.isfcn = isfcn
        
        if isinstance(p, list) and iutil.hasinstance(p, Expr):
            p = ExprArray(p)
        
        if isinstance(p, ExprArray):
            self.expr = p
            if num_in is None:
                if shape_in is None:
                    num_in = 0
                else:
                    num_in = len(shape_in)
            if num_in is not None:
                shape_in = p.shape[:num_in]
                shape_out = p.shape[num_in:]
            p = None
        
        if self.isvar and torch is None:
            raise ImportError("Requires pytorch. Please install it first.")
            
        if shape is not None:
            shape_in, shape_out = ConcDist.convert_shape_pair(shape)
        
        if isinstance(p, ConcDist):
            if num_in is None and shape is None and shape_in is None:
                num_in = p.get_num_in()
            p = p.p
        
        self.sublens = []
        
        if p is None:
            self.shape_in = ConcDist.convert_shape(shape_in)
            self.shape_out = ConcDist.convert_shape(shape_out)
            self.shape = self.shape_in + self.shape_out
            
            if randomize:
                self.randomize()
            else:
                self.set_uniform()
        else:
            if isinstance(p, list):
                p = numpy.array(p)
            self.p = p
            if num_in is None:
                if shape_in is not None:
                    num_in = len(shape_in)
                else:
                    num_in = 0
            tshape = p.shape
            self.shape_in = tshape[:num_in]
            self.shape_out = tshape[num_in:]
            self.shape = self.shape_in + self.shape_out
            
            if self.isvar:
                self.normalize()
        
        if isfcn:
            self.clamp_fcn()
            
    def clamp_fcn(self, randomize = False):
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            mzs = None
            m = -1.0
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                t = float(self.p[xs + zs])
                if randomize:
                    t /= numpy.random.exponential()
                self.p[xs + zs] = 0.0
                if t > m:
                    m = t
                    mzs = zs
            self.p[xs + mzs] = 1.0
        self.copy_torch()
        
        
    def istorch(self):
        return torch is not None and isinstance(self.p, torch.Tensor)
        
    def is_placeholder(self):
        return self.p is None
    
    def get_num_in(self):
        return len(self.shape_in)
    
    def card_out(self):
        r = 1
        for a in self.shape_out:
            r *= a
        return r
        
    def __getitem__(self, key):
        return self.p[key]
        
    def __setitem__(self, key, value):
        self.p[key] = value
    
    def copy_(self, other):
        """Copy content of other to self.
        """
        if isinstance(other, ConcDist):
            self.p = other.p
        else:
            self.p = other
        self.copy_torch()
    
    def flattened_sublen(self):
        shape_its = []
        shape_out_sub = []
        c = 0
        sublens = list(self.sublens) + [len(self.shape_out) - sum(self.sublens)]
        for l in sublens:
            shape_out_sub.append(iutil.product(self.shape_out[c:c+l]))
            c += l
        shape_out_sub = tuple(shape_out_sub)
        
        r = numpy.zeros(self.shape_in + shape_out_sub)
        if torch is not None and isinstance(self.p, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float64)
            
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                ids = []
                c = 0
                for l in sublens:
                    t = 0
                    for i in range(c, c + l):
                        t = t * self.shape_out[i] + zs[i]
                    ids.append(t)
                    c += l
                    
                r[xs + tuple(ids)] = self.p[xs + zs]
        
        r = ConcDist(r, num_in = self.get_num_in())
        r.sublens = [1] * (len(shape_out_sub) - 1)
        return r
        
    
    def calc_torch(self):
        if not self.isvar:
            return
        card_out = self.card_out()
        self.p = torch.zeros(self.shape_in + self.shape_out, dtype=torch.float64)
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            s = 0.0
            ci = 0
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                if ci < card_out - 1:
                    self.p[xs + zs] = self.v[xs + (ci,)]
                    s += self.v[xs + (ci,)]
                else:
                    self.p[xs + zs] = 1.0 - s
                ci += 1
    
    def copy_torch(self):
        if not self.isvar:
            self.v = None
            return
        card_out = self.card_out()
        # self.p = self.p.numpy()
        if self.v is None:
            self.v = numpy.zeros(self.shape_in + (card_out - 1,))
            self.v = torch.tensor(self.v, dtype=torch.float64, requires_grad = True)
            
        with torch.no_grad():
            for xs in itertools.product(*[range(x) for x in self.shape_in]):
                ci = 0
                for zs in itertools.product(*[range(z) for z in self.shape_out]):
                    if ci < card_out - 1:
                        self.v[xs + (ci,)] = float(self.p[xs + zs])
                    ci += 1
        self.calc_torch()
        
    def normalize(self):
        if self.isfcn:
            self.clamp_fcn()
            return
        
        ceps = PsiOpts.settings["eps"]
        ceps_d = PsiOpts.settings["opt_eps_denom"]
        
        card_out = self.card_out()
        
        if self.isvar and torch is not None and isinstance(self.p, torch.Tensor):
            self.p = self.p.detach().numpy()
        
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            s = 0.0
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                s += self.p[xs + zs]
            if s > ceps:
                for zs in itertools.product(*[range(z) for z in self.shape_out]):
                    self.p[xs + zs] /= s
            else:
                for zs in itertools.product(*[range(z) for z in self.shape_out]):
                    self.p[xs + zs] = 1.0 / card_out
            
        self.copy_torch()
    
    def clamp(self):
        if not self.isvar:
            return
        if self.isfcn:
            self.clamp_fcn(randomize = True)
            return
        
        ceps = PsiOpts.settings["eps"]
        ceps_d = PsiOpts.settings["opt_eps_denom"]
        
        card_out = self.card_out()
        
        vt = self.v.detach().numpy()
            
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for z in range(card_out - 1):
                t = vt[xs + (z,)]
                if numpy.isnan(t):
                    self.randomize()
                    return
                vt[xs + (z,)] = max(t, 0.0)
                
            if card_out > 2:
                while True:
                    s = 0.0
                    minpos = 1e20
                    numpos = 0
                    for z in range(card_out - 1):
                        if vt[xs + (z,)] > 0:
                            s += vt[xs + (z,)]
                            minpos = min(minpos, vt[xs + (z,)])
                            numpos += 1
                    if s <= 1.0 + ceps:
                        break
                    tored = (s - 1.0) / numpos
                    good = False
                    if tored <= minpos:
                        good = True
                    else:
                        tored = minpos
                        
                    for z in range(card_out - 1):
                        if vt[xs + (z,)] > 0:
                            vt[xs + (z,)] -= tored
                    
                    if good:
                        break
                
            for z in range(card_out - 1):
                vt[xs + (z,)] = min(vt[xs + (z,)], 1.0)
                
            # for z in range(card_out - 1):
            #     self.v[xs + (z,)] = vt[xs + (z,)]
        
        with torch.no_grad():
            self.v.copy_(torch.tensor(vt, dtype=torch.float64))
        # self.v.copy_(torch.tensor(vt))
            
        # with torch.no_grad():
        #     for xs in itertools.product(*[range(x) for x in self.shape_in]):
        #         for z in range(card_out - 1):
        #             self.v[xs + (z,)] = vt[xs + (z,)]
        # self.v = torch.tensor(self.v, requires_grad = True)
        self.calc_torch()
    
    def set_uniform(self):
        card_out = self.card_out()
            
        self.p = numpy.ones(self.shape_in + self.shape_out) * (1.0 / card_out)
        if self.isvar:
            self.normalize()
    
    def randomize(self):
        self.p = numpy.random.exponential(size = self.shape)
        self.normalize()
    
    def hop(self, prob):
        if torch is not None and isinstance(self.p, torch.Tensor):
            self.p = self.p.detach().numpy()
            
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            if numpy.random.uniform() >= prob:
                continue
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                self.p[xs + zs] = numpy.random.exponential()
                
        self.normalize()
    
    def get_p(self):
        return self.p
    
    def get_v(self):
        # if self.isfcn:
        #     return None
        return self.v
    
    def numpy(self):
        """Convert to numpy array."""
        if iutil.istorch(self.p):
            return self.p.detach().numpy()
        return self.p
    
    def torch(self):
        """Convert to torch.Tensor."""
        if iutil.istorch(self.p):
            return self.p
        return torch.tensor(self.p, dtype=torch.float64)
    
    
    def entropy(self):
        """Entropy of this distribution."""
        ceps = PsiOpts.settings["eps"]
        ceps_d = PsiOpts.settings["opt_eps_denom"]
        loge = PsiOpts.settings["ent_coeff"]
        istorch = torch is not None and isinstance(self.p, torch.Tensor)
        
        r = 0.0
        for xs in itertools.product(*[range(m) for m in self.shape]):
            c = self.p[xs]
            
            if istorch:
                r -= c * torch.log((c + ceps_d) / (1.0 + ceps_d)) * loge
            else:
                if c > ceps:
                    r -= c * numpy.log(c) * loge
        
        return ConcReal(r)
    
    def items(self):
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                yield self.p[xs + zs]
        
    
    def convert(x):
        if not isinstance(x, ConcDist):
            x = ConcDist(x)
        return x
    
    def __add__(self, other):
        other = ConcDist.convert(other)
        
        if (self.shape_in, self.shape_out) != (other.shape_in, other.shape_out):
            raise ValueError("Shape mismatch.")
            return
        
        return ConcDist(self.p + other.p, num_in = self.get_num_in())
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return ConcDist(self.p * float(other), num_in = self.get_num_in())
        
        other = ConcDist.convert(other)
        
        if self.shape_in != other.shape_in:
            raise ValueError("Shape mismatch.")
            return
        
        r = None
        if torch is not None and (isinstance(self.p, torch.Tensor) or isinstance(other.p, torch.Tensor)):
            r = torch.zeros(self.shape_in + self.shape_out + other.shape_out, dtype=torch.float64)
        else:
            r = numpy.zeros(self.shape_in + self.shape_out + other.shape_out)
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                for ws in itertools.product(*[range(w) for w in other.shape_out]):
                    r[xs + zs + ws] += self.p[xs + zs] * other.p[xs + ws]
        return ConcDist(r, num_in = self.get_num_in())
    
    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self * other
        other = ConcDist.convert(other)
        return other * self
    
    def __pow__(self, other):
        r = ConcDist(shape_in = self.shape_in, shape_out = tuple())
        for i in range(other):
            r = r * self
        return r
            
    
    def chan_product(self, other):
        """Product channel.
        """
        other = ConcDist.convert(other)
        
        r = None
        if torch is not None and (isinstance(self.p, torch.Tensor) or isinstance(other.p, torch.Tensor)):
            r = torch.zeros(self.shape_in + other.shape_in + self.shape_out + other.shape_out, dtype=torch.float64)
        else:
            r = numpy.zeros(self.shape_in + other.shape_in + self.shape_out + other.shape_out)
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for x2s in itertools.product(*[range(x2) for x2 in other.shape_in]):
                for zs in itertools.product(*[range(z) for z in self.shape_out]):
                    for ws in itertools.product(*[range(w) for w in other.shape_out]):
                        r[xs + x2s + zs + ws] += self.p[xs + zs] * other.p[x2s + ws]
        return ConcDist(r, num_in = self.get_num_in() + other.get_num_in())
    
    
    def chan_power(self, other):
        """n-product channel.
        """
        r = ConcDist(shape_in = tuple(), shape_out = tuple())
        for i in range(other):
            r = r.chan_product(self)
        return r
        
    def __matmul__(self, other):
        other = ConcDist.convert(other)
        
        if self.shape_out != other.shape_in:
            raise ValueError("Shape mismatch.")
            return
        
        r = None
        if torch is not None and (isinstance(self.p, torch.Tensor) or isinstance(other.p, torch.Tensor)):
            r = torch.zeros(self.shape_in + other.shape_out, dtype=torch.float64)
        else:
            r = numpy.zeros(self.shape_in + other.shape_out)
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                for ws in itertools.product(*[range(w) for w in other.shape_out]):
                    r[xs + ws] += self.p[xs + zs] * other.p[zs + ws]
        return ConcDist(r, num_in = self.get_num_in())
    
    
    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return ConcDist(self.p * float(1.0 / other), num_in = self.get_num_in())
        
        if self.shape_in != other.shape_in:
            raise ValueError("Shape mismatch.")
            return
        
        cshape_in = self.shape_out[:len(other.shape_out)]
        if cshape_in != other.shape_out:
            raise ValueError("Shape mismatch.")
            return
        cshape_out = self.shape_out[len(other.shape_out):]
        
        ceps = PsiOpts.settings["eps"]
        ceps_d = PsiOpts.settings["opt_eps_denom"]
        
        zsn = 1
        for k in cshape_out:
            zsn *= k
        cepsdzsn = ceps_d / zsn
        
        r = None
        if torch is not None and (isinstance(self.p, torch.Tensor) or isinstance(other.p, torch.Tensor)):
            r = torch.zeros(self.shape_in + cshape_in + cshape_out, dtype=torch.float64)
            for xs in itertools.product(*[range(x) for x in self.shape_in]):
                for zs in itertools.product(*[range(z) for z in cshape_in]):
                    for ws in itertools.product(*[range(w) for w in cshape_out]):
                        r[xs + zs + ws] = (self.p[xs + zs + ws] + cepsdzsn) / (other.p[xs + zs] + ceps_d)
        else:
            r = numpy.zeros(self.shape_in + cshape_in + cshape_out)
            for xs in itertools.product(*[range(x) for x in self.shape_in]):
                for zs in itertools.product(*[range(z) for z in cshape_in]):
                    if other.p[xs + zs] > ceps:
                        for ws in itertools.product(*[range(w) for w in cshape_out]):
                            r[xs + zs + ws] = self.p[xs + zs + ws] / other.p[xs + zs]
                    else:
                        for ws in itertools.product(*[range(w) for w in cshape_out]):
                            r[xs + zs + ws] = 1.0 / zsn
            
        return ConcDist(r, num_in = len(self.shape_in + cshape_in))
            
    
    def semidirect(self, other, ids = None):
        """Semidirect product.
        """
        if ids is None:
            ids = range(len(other.shape_in))
        r = None
        if torch is not None and (isinstance(self.p, torch.Tensor) or isinstance(other.p, torch.Tensor)):
            r = torch.zeros(self.shape_in + self.shape_out + other.shape_out, dtype=torch.float64)
        else:
            r = numpy.zeros(self.shape_in + self.shape_out + other.shape_out)
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                for ws in itertools.product(*[range(w) for w in other.shape_out]):
                    r[xs + zs + ws] = self.p[xs + zs] * other.p[tuple(zs[i] for i in ids) + ws]
        return ConcDist(r, num_in = self.get_num_in())
        
    
    def marginal(self, *args):
        """
        Marginal distribution.

        Parameters
        ----------
        *args : int
            Indices of the random variables of interest. E.g. for P(Y0,Y1,Y2|X),
            P.marginal(0,2) gives P(Y0,Y2|X)

        Returns
        -------
        ConcDist
            The marginal distribution.

        """
        ids = args
        if isinstance(ids, int):
            ids = [ids]
        r = None
        cshape = tuple(self.shape_out[i] for i in ids)
        if torch is not None and isinstance(self.p, torch.Tensor):
            r = torch.zeros(self.shape_in + cshape, dtype=torch.float64)
        else:
            r = numpy.zeros(self.shape_in + cshape)
        
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for zs in itertools.product(*[range(z) for z in cshape]):
                wrange = [range(w) for w in self.shape_out]
                for i in range(len(cshape)):
                    if len(wrange[ids[i]]) == 1 and wrange[ids[i]][0] != zs[i]:
                        break
                    wrange[ids[i]] = [zs[i]]
                else:
                    for ws in itertools.product(*wrange):
                        r[xs + zs] += self.p[xs + ws]
                        
        return ConcDist(r, num_in = self.get_num_in())
        
    
    def reorder(self, ids):
        r = None
        cshape = tuple(self.shape_out[i] for i in ids)
        if torch is not None and isinstance(self.p, torch.Tensor):
            r = torch.zeros(self.shape_in + cshape, dtype=torch.float64)
        else:
            r = numpy.zeros(self.shape_in + cshape)
        
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                r[xs + tuple(zs[ids[i]] for i in range(len(ids)))] = self.p[xs + zs]
                        
        return ConcDist(r, num_in = self.get_num_in())
        
    
    def given(self, *args):
        """
        For a conditional distribution P(Y|X), give the distribution P(Y|X=x).

        Parameters
        ----------
        *args : int or None.
            The values to substitute to X.
            Must have the same number of arguments as the number of random variables
            conditioned. Arguments are either int (value of RV) or None if the RV
            is not substituted.

        Returns
        -------
        ConcDist
            The distribution after substitution.

        """
        r = None
        cshape = tuple(self.shape_in[i] for i in range(len(self.shape_in)) if args[i] is None)
        if torch is not None and isinstance(self.p, torch.Tensor):
            r = torch.zeros(cshape + self.shape_out, dtype=torch.float64)
        else:
            r = numpy.zeros(cshape + self.shape_out)
        
        for xs in itertools.product(*[range(x) for x in cshape]):
            xs2 = [0] * len(self.shape_in)
            xsi = 0
            for i in range(len(self.shape_in)):
                if args[i] is None:
                    xs2[i] = xs[xsi]
                    xsi += 1
                else:
                    xs2[i] = args[i]
            xs2 = tuple(xs2)
            
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                r[xs + zs] = self.p[xs2 + zs]
                        
        return ConcDist(r, num_in = len(cshape))
        
        
    
    def mean(self, f = None):
        """
        Returns the expectation of the function f.

        Parameters
        ----------
        f : function, numpy.array or torch.Tensor
            If f is a function, the number of arguments must match the number of
            dimensions (random variables) of the joint distribution.
            If f is an array or tensor, shape must match the shape of the 
            distribution.
        Returns
        -------
        r : float or torch.Tensor
            The expectation. Type is torch.Tensor if self or f is torch.Tensor.

        """
        
        if f is None:
            f = lambda x: x
        
        r = None
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                if callable(f):
                    t = self.p[xs + zs] * f(*(xs + zs))
                else:
                    t = self.p[xs + zs] * f[xs + zs]
                if r is not None:
                    r += t
                else:
                    r = t
        return r
            
        
    def __str__(self):
        return str(self.p)
    
    def __repr__(self):
        r = ""
        r += "ConcDist("
        t = repr(self.p)
        if t.find("\n") >= 0:
            r += "\n"
        r += t
        if self.get_num_in() > 0:
            r += ", num_in=" + str(self.get_num_in())
        r += ")"
        return r
        
    
    def fcn(fcncall, shape):
        shape_in, shape_out = ConcDist.convert_shape_pair(shape)
        return ConcDist(ExprArray.fcn(fcncall, shape_in + shape_out), shape_in = shape_in, shape_out = shape_out)
        
    def det_fcn(fcncall, shape, isvar = False):
        shape_in, shape_out = ConcDist.convert_shape_pair(shape)
        p = numpy.zeros(shape_in + shape_out)
        for xs in itertools.product(*[range(x) for x in shape_in]):
            
            t = fcncall(*xs)
            
            # t = 0
            # if len(xs) == 1:
            #     t = fcncall(xs[0])
            # else:
            #     t = fcncall(xs)
            
            if isinstance(t, (bool, float)):
                t = int(t)
                
            if isinstance(t, int):
                t = (t,)
            p[xs + t] = 1.0
            
        return ConcDist(p, num_in = len(shape_in), isvar = isvar)
        
    
    def valid_region(self, skip_simplify = False):
        """For a symbolic distribution, returns the region where this is a
        valid distribution.
        """
        if self.expr is None:
            return Region.universe()
        r = Region.universe()
        for xs in itertools.product(*[range(x) for x in self.shape_in]):
            sumexpr = Expr.zero()
            for zs in itertools.product(*[range(z) for z in self.shape_out]):
                cexpr = self.expr[xs + zs]
                r.iand_norename(cexpr >= 0)
                sumexpr += cexpr
            r.iand_norename(sumexpr == 1)
        
        if not skip_simplify:
            return r.simplified()
        return r
        
    
    def uniform(n, isvar = False):
        """n-ary uniform distribution."""
        return ConcDist(numpy.ones(n) / n, isvar = isvar)
        
    def bit(isvar = False):
        """Fair bit."""
        return ConcDist.uniform(2, isvar = isvar)
        
    def bern(a, isvar = False):
        """Bernoulli distribution."""
        return ConcDist([1.0 - a, a], isvar = isvar)
    
    def random(n, isvar = False):
        """n-ary random distribution."""
        r = ConcDist(numpy.ones(n) / n, isvar = isvar)
        r.randomize()
        return r
        
    def symm_chan(n, crossover, isvar = False):
        """n-ary symmetric channel."""
        a = 1.0 - crossover
        b = crossover / (n - 1)
        # return ConcDist(numpy.ones((n, n)) * b + numpy.eye(n) * (a - b), num_in = 1, isvar = isvar)
        return ConcDist([[a if i == j else b for j in range(n)] for i in range(n)],
                        num_in = 1, isvar = isvar)
        
    def bsc(crossover, isvar = False):
        """Binary symmetric channel."""
        return ConcDist.symm_chan(2, crossover, isvar = isvar)
        
    def bin_chan(cross01, cross10, isvar = False):
        """Binary channel."""
        return ConcDist([[1.0 - cross01, cross01], [cross10, 1.0 - cross10]], num_in = 1, isvar = isvar)
    
    def erasure_chan(n, er_prob, isvar = False):
        """n-ary erasure channel."""
        # return ConcDist(numpy.hstack([numpy.eye(n) * (1.0 - er_prob), 
        #                               numpy.ones((n, 1)) * er_prob]), num_in = 1, isvar = isvar)
        return ConcDist([[(1.0 - er_prob) if i == j else 0.0 for j in range(n)] + [er_prob] for i in range(n)],
                        num_in = 1, isvar = isvar)
        
    def bec(er_prob, isvar = False):
        """Binary erasure channel."""
        return ConcDist.erasure_chan(2, er_prob, isvar = isvar)
        
    def flat(shape_in, isvar = False):
        """Transition probability for flattening a random vector into one random variable."""
        if isinstance(shape_in, int):
            shape_in = (shape_in,)
        nout = iutil.product(shape_in)
        p = numpy.zeros(shape_in + (nout,))
        for xs in itertools.product(*[range(x) for x in shape_in]):
            t = 0
            for a, b in zip(xs, shape_in):
                t = t * b + a
            p[xs + (t,)] = 1.0
        return ConcDist(p, num_in = len(shape_in), isvar = isvar)
    
    def add(shape_in, isvar = False):
        """Transition probability from several random variables to their sum."""
        if isinstance(shape_in, int):
            shape_in = (shape_in,)
        nout = sum(shape_in) - len(shape_in) + 1
        p = numpy.zeros(shape_in + (nout,))
        for xs in itertools.product(*[range(x) for x in shape_in]):
            p[xs + (sum(xs),)] = 1.0
        return ConcDist(p, num_in = len(shape_in), isvar = isvar)
        
    def gaussian(r, l, isvar = False):
        """Quantized standard Gaussian distribution in the range [-r, r],
        divided into l cells.
        """
        sqrt2 = numpy.sqrt(2.0)
        p = numpy.zeros(l)
        cdf = 0.0
        cdf0 = 0.0
        for i in range(l + 1):
            x = (i * 2.0 / l - 1.0) * r
            cdf2 = 0.5 * (1 + scipy.special.erf(x / sqrt2))
            if i > 0:
                p[i - 1] = cdf2 - cdf
            cdf = cdf2
            if i == 0:
                cdf0 = cdf
        for i in range(l):
            p[i] /= cdf - cdf0
        return ConcDist(p, num_in = 0, isvar = isvar)
        
    def convolve_kernel(shape_in, kernel, isvar = False):
        """Transition probability from X to X+Z, where X is a random vector with a
        pmf of shape shape_in, and Z is independent of X and follows the distribution
        given by kernel.
        """
        if isinstance(shape_in, int):
            shape_in = (shape_in,)
        if isinstance(kernel, ConcDist):
            kernel = kernel.p
        shape_out = tuple(a + b - 1 for a, b in zip(shape_in, kernel.shape))
        p = numpy.zeros(shape_in + shape_out)
        for xs in itertools.product(*[range(x) for x in shape_in]):
            for zs in itertools.product(*[range(x) for x in kernel.shape]):
                p[xs + tuple(x + z for x, z in zip(xs, zs))] = kernel[zs]
        return ConcDist(p, num_in = len(shape_in), isvar = isvar)
        
    
    
class ConcReal(IBaseObj):
    """Concrete real variable."""
    
    def __init__(self, x = None, lbound = None, ubound = None, scale = 1.0, isvar = False, isint = False, randomize = False):
        self.isvar = isvar
        self.isint = isint
        
        if self.isvar and torch is None:
            raise ImportError("Requires pytorch. Please install it first.")
        
        if x is None:
            x = 0.0
        if isinstance(x, int):
            x = float(x)
            
        if isinstance(x, ConcReal):
            x = x.x
            
        if isinstance(lbound, int):
            lbound = float(lbound)
        if isinstance(ubound, int):
            ubound = float(ubound)
        if isinstance(scale, int):
            scale = float(scale)
            
        self.x = x
        self.v = None
        self.lbound = lbound
        self.ubound = ubound
        self.scale = scale
        
        self.copy_torch()
        
        if randomize:
            self.randomize()
    
    def const(x = 0.0):
        """Constant.
        """
        return ConcReal(x, x, x)
    
    def convert(x):
        if isinstance(x, int) or isinstance(x, float):
            x = ConcReal.const(x)
        elif not isinstance(x, ConcReal):
            x = ConcReal(x)
        return x
    
    def calc_torch(self):
        if self.isvar:
            self.x = self.v
        
    def copy_torch(self):
        if self.isvar:
            if self.v is None:
                self.v = torch.tensor(self.x, dtype=torch.float64, requires_grad = True)
            with torch.no_grad():
                self.v.copy_(torch.tensor(self.x, dtype=torch.float64))
            self.x = self.v
        
    def copy_(self, other):
        """Copy content of other to self.
        """
        if isinstance(other, int):
            other = float(other)
        if isinstance(other, ConcReal):
            self.x = other.x
        else:
            self.x = other
        self.copy_torch()
            
    def __add__(self, other):
        other = ConcReal.convert(other)
        
        return ConcReal(self.x + other.x, 
                        None if self.lbound is None or other.lbound is None else self.lbound + other.lbound,
                        None if self.ubound is None or other.ubound is None else self.ubound + other.ubound,
                        self.scale + other.scale)
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = ConcReal.convert(other)
        lbound = None
        ubound = None
        for a in [self.lbound, self.ubound]:
            for b in [other.lbound, other.ubound]:
                if a is not None and b is not None:
                    t = a * b
                    if lbound is None or lbound > t:
                        lbound = t
                    if ubound is None or ubound < t:
                        ubound = t
                    
        return ConcReal(self.x * other.x, lbound, ubound, self.scale * other.scale)
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        return self + other * -1
    
    def __rsub__(self, other):
        return other + self * -1
    
    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other):
        return self * (1 / other)
    
    def __rtruediv__(self, other):
        if (isinstance(other, int) or isinstance(other, float)) and other > 0:
            other = float(other)
            lbound = None
            ubound = None
            if self.lbound is not None and self.lbound > 0:
                ubound = other / self.lbound
                if self.ubound is not None:
                    lbound = other / self.ubound
            if self.ubound is not None and self.ubound < 0:
                lbound = other / self.ubound
                if self.lbound is not None:
                    ubound = other / self.lbound
                    
            return ConcReal(other / self.x, lbound, ubound, other / self.scale)
        
        return other * (1 / self)
    
    def clamp(self):
        if not self.isvar:
            return
        
        vt = float(self.v.detach().numpy())
        
        if numpy.isnan(vt):
            self.randomize()
            return
        
        if self.lbound is not None and vt < self.lbound:
            vt = self.lbound
        if self.ubound is not None and vt > self.ubound:
            vt = self.ubound
        if self.isint:
            vt = round(vt)
            
        with torch.no_grad():
            self.v.copy_(torch.tensor(vt, dtype=torch.float64))
    
    def randomize(self):
        if self.lbound is None or self.ubound is None:
            self.x = numpy.random.exponential() * self.scale
            if self.ubound is not None:
                self.x = self.ubound - self.x
            elif self.lbound is not None:
                self.x = self.lbound + self.x
            else:
                if numpy.random.uniform() < 0.5:
                    self.x *= -1
        else:
            self.x = numpy.random.uniform(self.lbound, self.ubound)
            
        self.copy_torch()
        
    
    def hop(self, prob):
        if numpy.random.uniform() < prob:
            self.randomize()
    
    def get_x(self):
        return self.x
    
    def get_v(self):
        return self.v
    
    def __float__(self):
        return float(self.x)
    
    def __int__(self):
        return int(round(float(self.x)))
    
    def torch(self):
        """Convert to torch.Tensor."""
        if iutil.istorch(self.x):
            return self.x
        return torch.tensor(self.x, dtype=torch.float64)
    
    def __str__(self):
        return str(self.x)
    
    def __repr__(self):
        r = ""
        r += "ConcReal("
        r += str(self.x)
        if self.lbound is not None:
            r += ", lbound=" + str(self.lbound)
        if self.ubound is not None:
            r += ", ubound=" + str(self.ubound)
        r += ")"
        return r
    
    
class RealRV(IBaseObj):
    """Discrete real-valued random variable."""
    
    def __init__(self, x, fcn = None, supp = None):
        self.comp = x
        self.fcn = fcn
        self.supp = supp
    
    @property
    def x(self):
        return self.comp
    
    
class ConcModel(IBaseObj):
    """Concrete distributions of random variables and values of real variables."""
    
    def __init__(self, bnet = None, istorch = None):
        if istorch is None:
            self.istorch = PsiOpts.settings["istorch"]
        else:
            self.istorch = istorch
            
        if self.istorch and torch is None:
            raise ImportError("Requires pytorch. Please install it first.")
            
        if bnet is None:
            self.bnet = BayesNet()
        else:
            self.bnet = bnet
        v = self.bnet.index.comprv
        n = len(v)
        # self.ps = [None] * n
        
        self.psmap = {}
        self.psmap_cache = {}
        self.psv = [None] * n
        self.psmap_mask = {}
        
        self.card = [None] * n
        self.pt = None
        
        self.index_real = IVarIndex()
        self.reals = []
        self.opt_reg = None
        
        
    def copy_shallow(self):
        r = ConcModel()
        r.istorch = self.istorch
        r.bnet = self.bnet.copy()
        r.psmap = dict(self.psmap)
        r.psmap_cache = dict(self.psmap_cache)
        r.psv = list(self.psv)
        r.psmap_mask = dict(self.psmap_mask)
        r.card = list(self.card)
        r.index_real = self.index_real.copy()
        r.reals = list(self.reals)
        return r
        
    def set_real(self, x, v):
        self.clear_cache()
        if v == "var":
            self.set_real(x, ConcReal(isvar = True))
            return
        if v == "var,rand":
            self.set_real(x, ConcReal(isvar = True, randomize = True))
            return
        
        if isinstance(x, Expr):
            x = x.allcomp()
        x.record_to(self.index_real)
        i = self.index_real.get_index(x)
        
        while len(self.reals) < len(self.index_real.compreal):
            self.reals.append(None)
        self.reals[i] = v
        
    def get_real(self, x):
        if isinstance(x, Expr):
            x = x.allcomp()
        i = self.index_real.get_index(x)
        if i < 0:
            return None
        return self.reals[i]
        
    def clear_cache(self):
        self.pt = None
        # self.psmap = {key: item for key, item in self.psmap.items() if not item.iscache}
        self.psmap_mask = {}
        self.psmap_cache = {}
        
    def comp_to_tuple(self, x):
        r = []
        for a in x:
            t = self.bnet.index.get_index(a)
            if t < 0:
                return None
            r.append(t)
        return tuple(r)
        
    def comp_to_pair(self, x):
        if isinstance(x, Term):
            return (self.comp_to_tuple(x.z), 
                    sum((self.comp_to_tuple(t) for t in x.x), tuple()))
        elif isinstance(x, Comp):
            return (tuple(), self.comp_to_tuple(x))
        else:
            return (self.comp_to_tuple(x[0]), self.comp_to_tuple(x[1]))
        
        
    def comp_get_sublens(self, x):
        if isinstance(x, Term):
            r = [len(a) for a in x.x]
            r.pop()
            return r
        return []
        
    def set_prob(self, x, p):
        if isinstance(p, str):
            opt_split = p.split(",")
            opt = None
            randomize = False
            isvar = False
            isfcn = False
            mode = ""
            
            for copt in opt_split:
                if copt == "var":
                    isvar = True
                elif copt == "rand":
                    randomize = True
                elif copt == "fcn":
                    isfcn = True
                else:
                    mode = copt
            
            if mode == "flat":
                shape_in, shape_out = self.convert_shape_pair(x)
                self.set_prob(x, ConcDist.flat(shape_in, isvar = isvar))
            elif mode == "add":
                shape_in, shape_out = self.convert_shape_pair(x)
                self.set_prob(x, ConcDist.add(shape_in, isvar = isvar))
            else:
                shape_in, shape_out = self.convert_shape_pair(x)
                self.set_prob(x, ConcDist(shape_in = shape_in, shape_out = shape_out, 
                                          isvar = isvar, randomize = randomize, isfcn = isfcn))
            return
        
        
        if isinstance(p, collections.Callable) and not isinstance(p, (ConcDist, list, ExprArray)):
            dist = ConcDist.det_fcn(p, self.convert_shape_pair(x))
            self.set_prob(x, dist)
            return
        
        self.bnet += x
        cin, cout = self.comp_to_pair(x)
        
        if isinstance(p, list):
            if iutil.hasinstance(p, Expr):
                p = ExprArray(p)
            else:
                p = numpy.array(p)
            
        shape = p.shape
        if len(shape) != len(cin) + len(cout):
            raise ValueError("Number of dimensions of prob. table = " + str(len(shape)) 
                              + " does not match number of variables = " + str(len(cin) + len(cout)) + ".")
            return
        for j in range(len(cin + cout)):
            t = self.get_card_id((cin + cout)[j])
            if t is not None and t != shape[j]:
                raise ValueError("Length of dimension " + str(self.bnet.index.comprv[(cin + cout)[j]]) + " of prob. table = " + str(shape[j]) 
                                  + " does not match its cardinality = " + str(t) + ".")
                return
            
        while len(self.card) < len(self.bnet.index.comprv):
            self.card.append(None)
        while len(self.psv) < len(self.bnet.index.comprv):
            self.psv.append(None)
            
        for j in range(len(cin + cout)):
            self.card[(cin + cout)[j]] = shape[j]
        
        if not isinstance(p, ConcDist):
            p = ConcDist(p, num_in = len(cin))
        
        self.psmap[(cin, cout)] = p
        for k in cout:
            self.psv[k] = (cin, cout)
        self.clear_cache()
        
    def calc_dist(self, p):
        if p is None:
            return
        if p.expr is not None:
            p.p = self[p.expr]
        
    def get_prob_mask(self, mask):
        t = self.psmap_mask.get(mask, None)
        self.calc_dist(t)
        if t is not None:
            return t
        
        n = len(self.bnet.index.comprv)
        k = 0
        while (1 << (k + 1)) <= mask:
            k += 1
            
        if self.psv[k] is None:
            raise ValueError("Random variable " + str(self.bnet.index.comprv[k]) + " has unspecified distribution.")
            return
            
        tin, tout = self.psv[k]
        tp = self.psmap[(tin, tout)]
        self.calc_dist(tp)
        
        tin_mask = 0
        for a in tin:
            tin_mask |= 1 << a
        tout_mask = 0
        for a in tout:
            tout_mask |= 1 << a
        
        mask1 = (mask | tin_mask) & ~tout_mask
        
        p2 = None
        
        if mask1 > 0:
            p1 = self.get_prob_mask(mask1)
            p2 = p1.semidirect(tp, [iutil.bitcount(mask1 & ((1 << i) - 1)) for i in tin])
        else:
            p2 = tp
        
        idinv = [None] * n
        ci = 0
        for i in range(n):
            if mask1 & (1 << i):
                idinv[i] = ci
                ci += 1
        for i in tout:
            idinv[i] = ci
            ci += 1
        
        p3 = p2.marginal(*[idinv[i] for i in range(n) if mask & (1 << i)])
        self.psmap_mask[mask] = p3
        return p3
        
    def get_prob_pair(self, cin, cout):
        t = self.psmap.get((cin, cout), None)
        self.calc_dist(t)
        if t is not None:
            return t
        t = self.psmap_cache.get((cin, cout), None)
        self.calc_dist(t)
        if t is not None:
            return t
        
        istorch = self.istorch
        
        cinlen = [self.get_card_id(i) for i in cin]
        coutlen = [self.get_card_id(i) for i in cout]
        cin_mask = 0
        for a in cin:
            cin_mask |= 1 << a
        cout_mask = 0
        for a in cout:
            cout_mask |= 1 << a
        
        
        p1 = self.get_prob_mask(cin_mask | cout_mask)
        p1 = p1.reorder([iutil.bitcount((cin_mask | cout_mask) & ((1 << i) - 1)) for i in cin + cout])
        
        r = None
        if cin_mask != 0:
            p0 = self.get_prob_mask(cin_mask)
            p0 = p0.reorder([iutil.bitcount(cin_mask & ((1 << i) - 1)) for i in cin])
            r = p1 / p0
        else:
            r = p1
            
        self.psmap_cache[(cin, cout)] = r
        return r
        
        
    def get_prob(self, x):
        cin, cout = self.comp_to_pair(x)
        if cin is None or cout is None:
            raise ValueError("Some random variables are absent in the model.")
            return
        r = self.get_prob_pair(cin, cout)
        if isinstance(r, ConcDist):
            r.sublens = self.comp_get_sublens(x)
        return r
        
    
    def get_card_id(self, i):
        if i >= len(self.card):
            return None
        return self.card[i]
    
    def get_card(self, x):
        i = self.bnet.index.get_index(x)
        if i < 0:
            return None
        return self.get_card_id(i)
    
    def get_card_default(self, x):
        i = self.bnet.index.get_index(x)
        if i < 0:
            return x.get_card()
        return self.get_card_id(i)
        
    def convert_shape(self, x):
        if x is None:
            return tuple()
        if isinstance(x, int):
            return (x,)
        if isinstance(x, Comp):
            return tuple(self.get_card_default(a) for a in x)
        if isinstance(x, ConcDist):
            return x.shape_out
        return tuple(x)
    
    def convert_shape_pair(self, x):
        if x is None:
            return (tuple(), tuple())
        if isinstance(x, int):
            return (tuple(), (x,))
        if isinstance(x, Term):
            return (self.convert_shape(x.z), sum((self.convert_shape(t) for t in x.x), tuple()))
        if isinstance(x, Comp) or isinstance(x, ConcDist):
            return (tuple(), self.convert_shape(x))
        if isinstance(x, tuple) and (len(x) == 0 or isinstance(x[0], int)):
            return (tuple(), tuple(x))
        return (self.convert_shape(x[0]), self.convert_shape(x[1]))
        
                
    def get_H(self, x):
        p = self.get_prob(x)
        return p.entropy().x
        
    def get_ent_vector(self, x):
        n = len(x)
        r = []
        for mask in range(1 << n):
            r.append(self.get_H(x.from_mask(mask)))
        return r
    
    def discover(self, x, eps = None, skip_simplify = False):
        """Discover conditional independence among variables in x.
        """
        v = self.get_ent_vector(x)
        return Region.ent_vector_discover_ic(v, x, eps, skip_simplify)
    
    def get_region(self):
        return self.bnet.get_region()
        
    def allcomprv(self):
        return self.bnet.index.comprv.copy()
    
    def allcompreal(self):
        return self.index_real.compreal.copy()
    
    def allcomp(self):
        return self.allcomprv() + self.allcompreal()
    
    def add_reg(self, reg):
        varlist = []
        cons = Region.universe()
        
        if isinstance(reg, RegionOp):
            reg = reg.tosimple()
            if reg is None:
                raise ValueError("User-defined information quantities with RegionOp constraints cannot be optimized.")
                return None
            
        card0 = PsiOpts.settings["opt_aux_card"]
        
        reg.simplify_quick(zero_group = 0)
        
        regcom = reg.copy()
        regcom.iand_norename(regcom.completed_semigraphoid(max_iter = 10000))
        
        
        # print(reg)
        # print(regcom)
        
        tbnet = regcom.get_bayesnet(roots = self.bnet.index.comprv.inter(regcom.allcomprv()), 
                                 skip_simplify = True)
        # print(tbnet)
        
        fcnreg = Region.universe()
        
        for a in tbnet.index.comprv + reg.allcomprv():
            if self.bnet.index.get_index(a) >= 0:
                continue
            
            ccard = a.get_card()
            if ccard is None:
                ccard = card0
            
            pa = None
            if tbnet.index.get_index(a) >= 0:
                pa = tbnet.get_parents(a)
            else:
                pa = self.bnet.index.comprv
                
            # print(pa)
            
            isfcn = reg.copy_noaux().implies(Expr.Hc(a, pa) <= 0)
            if isfcn:
                fcnreg.iand_norename(Expr.Hc(a, pa) <= 0)
            # print(reg)
            # print(a)
            # print(pa)
            # print(isfcn)
            p = ConcDist(shape = (tuple(self.get_card(t) for t in pa), (ccard,)),
                         isvar = True, randomize = True, isfcn = isfcn)
            
            
            self[a | pa] = p
            varlist.append(p)
            
        for a in reg.allcomprealvar_exprlist():
            if self.get_real(a) is None:
                t = ConcReal(isvar = True)
                self[a] = t
                varlist.append(t)
            
        csreg = self.get_region() & cons & fcnreg
        for ineq in reg:
            if not csreg.implies(ineq):
                cons.iand_norename(ineq)
                csreg.iand_norename(ineq)
        
        return (varlist, cons)
    
        
    def get_val_regterm(self, term, esgn):
        
        if term.fcncall is not None:
            cargs = []
            for a in term.fcnargs:
                if isinstance(a, Term) or isinstance(a, Comp):
                    cargs.append(self.get_prob(a))
                elif isinstance(a, Expr):
                    cargs.append(self.get_val(a, 0))
                elif isinstance(a, Region):
                    cargs.append(1 if self[a] else 0)
                elif a == "model":
                    cargs.append(self)
                else:
                    cargs.append(a)
            return term.get_fcneval(cargs)
        
        
        t = term.get_reg_sgn_bds()
        
        if t is None:
            return None
        reg, sgn, bds = t
        
        if len(bds) == 0:
            return None
        
        if isinstance(reg, RegionOp):
            reg = reg.tosimple()
            if reg is None:
                raise ValueError("User-defined information quantities with RegionOp constraints cannot be optimized.")
                return None
        
        rcomp = reg.allcomprv()
        for b in bds:
            rcomp += b.allcomprv_shallow()
            
        if self.bnet.index.comprv.super_of(rcomp):
            r = numpy.inf
            for b in bds:
                t = self.get_val(b, esgn * sgn) * sgn
                if float(t) < float(r):
                    r = t
            return r * sgn
        
        card0 = PsiOpts.settings["opt_aux_card"]
        
        reg.simplify_quick(zero_group = 0)
        
        cs = self.copy_shallow()
        
        varlist, cons = cs.add_reg(reg)
        
        tvar = None
        if len(bds) == 1:
            tvar = bds[0]
        else:
            tvar = Expr.real("#TVAR_" + term.get_name())
            tvar_r = ConcReal(isvar = True, randomize = True)
            cs[tvar] = tvar_r
            varlist.append(tvar_r)
            for b in bds:
                cons &= tvar * sgn <= b * sgn
        
        retval = cs.optimize(tvar, varlist, cons, sgn = sgn)
        self.opt_reg = cs.opt_reg
        
        return retval
        
    
    def get_val(self, expr, esgn = 0):
        
        r = 0.0
        for (a, c) in expr.terms:
            
            termType = a.get_type()
            if termType == TermType.IC:
                k = len(a.x)
                for t in range(1 << k):
                    csgn = -1;
                    mask = self.bnet.index.get_mask(a.z)
                    for i in range(k):
                        if (t & (1 << i)) != 0:
                            csgn = -csgn
                            mask |= self.bnet.index.get_mask(a.x[i])
                    if mask != 0:
                        # r += self.get_H_mask(mask) * csgn * c
                        r += self.get_H(self.bnet.index.from_mask(mask)) * csgn * c
            elif termType == TermType.REAL or termType == TermType.REGION:
                if a.isone():
                    r += c
                else:
                    t = self.get_real(a.x[0])
                    
                    if t is None:
                        if termType == TermType.REGION:
                            t = self.get_val_regterm(a, esgn * (1 if c > 0 else -1))
                        if t is None:
                            return None
                    if isinstance(t, ConcReal):
                        t = t.x
                    r += t * c
        
        return ConcReal(r)
        
    
    def __call__(self, x):
        if isinstance(x, Expr):
            return self.get_val(x)
        
        elif isinstance(x, ExprArray):
            if self.istorch:
                # r = torch.tensor([self[a] for a in x], dtype=torch.float64)
                # return torch.reshape(r, x.shape)
                r = torch.hstack(tuple(iutil.ensure_torch(self[a]) for a in x))
                return torch.reshape(r, x.shape)
            else:
                r = numpy.array([float(self[a]) for a in x])
                return numpy.reshape(r, x.shape)
            
        elif isinstance(x, Region):
            if x.aux_present():
                x = x.simplified_quick()
            
            if x.aux_present():
                cs = self.copy_shallow()
                varlist, cons = cs.add_reg(x)
                
                retval = cs.optimize(Expr.zero(), varlist, cons, sgn = 1)
                
                self.opt_reg = cs.opt_reg
                return x.evalcheck(cs)
            
            else:
                return x.evalcheck(self)
        
        return None
    
    def __getitem__(self, x):
        if isinstance(x, tuple) or isinstance(x, Comp) or isinstance(x, Term):
            return self.get_prob(x)
        if isinstance(x, CompArray):
            return self.get_prob(x.get_term())
        if isinstance(x, Expr) and len(x) == 1:
            a, c = x.terms[0]
            if c == 1:
                termType = a.get_type()
                if termType == TermType.REAL or termType == TermType.REGION:
                    t = self.get_real(a.x[0])
                    if t is not None:
                        return t
        return self(x)
    
    def __setitem__(self, key, value):
        if isinstance(key, Expr):
            self.set_real(key.allcomp(), value)
        else:
            self.set_prob(key, value)
    
    def convert_torch_tensors(self, x):
        r = []
        r_dist = []
        if isinstance(x, list):
            for a in x:
                tr, tdist = self.convert_torch_tensors(a)
                r += tr
                r_dist += tdist
        elif torch is not None and isinstance(x, torch.Tensor):
            r.append(x)
        elif isinstance(x, Comp):
            for a in x:
                i = self.bnet.index.get_index(x)
                if i >= 0:
                    tr, tdist = self.convert_torch_tensors(self.ps[i])
                    r += tr
                    r_dist += tdist
        elif isinstance(x, Expr):
            t = self.get_real(x)
            if t is not None:
                tr, tdist = self.convert_torch_tensors(t)
                r += tr
                r_dist += tdist
        elif isinstance(x, ExprArray):
            for x2 in x:
                t = self.get_real(x2)
                if t is not None:
                    tr, tdist = self.convert_torch_tensors(t)
                    r += tr
                    r_dist += tdist
        elif isinstance(x, ConcDist) or isinstance(x, ConcReal):
            t = x.get_v()
            if t is not None and torch is not None and isinstance(t, torch.Tensor):
                r.append(t)
            
            if x.isvar:
                r_dist.append(x)
                
        return (r, r_dist)
    
    def tensor_copy_list(x, y):
        with torch.no_grad():
            for i in range(len(y)):
                if len(x) <= i:
                    x.append(y[i].detach().clone())
                else:
                    x[i].copy_(y[i])
                    
    def get_tensor(self, x):
        r = None
        if isinstance(x, Expr):
            r = self.get_val(x)
        else:
            r = x(self)
        if isinstance(r, ConcReal):
            r = r.x
        return r
    
    def tensor_list_to_array(varlist):
        r = []
        for v in varlist:
            k = functools.reduce(lambda x, y: x*y, v.shape, 1)
            r += list(numpy.reshape(v.detach().numpy(), (k,)))
            
        for i in range(len(r)):
            if numpy.isnan(r[i]):
                r[i] = 0.0
                
        return numpy.array(r)
    
    
    def tensor_list_grad_to_array(varlist):
        r = []
        for v in varlist:
            k = functools.reduce(lambda x, y: x*y, v.shape, 1)
            if v.grad is None:
                r += [0.0 for i in range(k)]
            else:
                r += list(numpy.reshape(v.grad.numpy(), (k,)))
            
        for i in range(len(r)):
            if numpy.isnan(r[i]):
                r[i] = 0.0
                
        return numpy.array(r)
    
    
    def tensor_list_from_array(varlist, a):
        c = 0
        for v in varlist:
            k = functools.reduce(lambda x, y: x*y, v.shape, 1)
            with torch.no_grad():
                v.copy_(torch.tensor(numpy.reshape(a[c:c+k], v.shape), dtype=torch.float64))
            if v.grad is not None:
                v.grad.data.zero_()
            c += k
    
    
    def tensor_list_get_bds(varlist, distlist, ismat):
        c = 0
        cons = []
        num_cons = 0
        cons_A_data = []
        cons_A_r = []
        cons_A_c = []
        xsize = 0
        for v in varlist:
            k = functools.reduce(lambda x, y: x*y, v.shape, 1)
            xsize += k
            
        bds = [(-numpy.inf, numpy.inf) for i in range(xsize)]
            
        for v in varlist:
            k = functools.reduce(lambda x, y: x*y, v.shape, 1)
            for d in distlist:
                if d.get_v() is v:
                    if isinstance(d, ConcDist):
                        
                        stride = functools.reduce(lambda x, y: x*y, d.shape_out, 1) - 1
                        
                        if stride >= 2:
                            for c2 in range(c, c+k, stride):
                                if ismat:
                                    cons_A_data += [1.0] * stride
                                    cons_A_r += [num_cons] * stride
                                    cons_A_c += list(range(c2, c2+stride))
                                    num_cons += 1
                                else:
                                    def get_fcn(i0, i1):
                                        def fcn(x):
                                            return 1.0 - sum(x[i0: i1])
                                        return fcn
                                    def get_jac(i0, i1):
                                        def fcn(x):
                                            r = numpy.zeros(xsize)
                                            r[i0:i1] = -numpy.ones(i1-i0)
                                            return r
                                        return fcn
                                    cons.append({
                                        "type": "ineq",
                                        "fun": get_fcn(c2, c2+stride),
                                        "jac": get_jac(c2, c2+stride)
                                        })
                            
                        for i in range(c, c+k):
                            bds[i] = (0.0, 1.0)
                            
                    elif isinstance(d, ConcReal):
                        if d.lbound is not None or d.ubound is not None:
                            bds[c] = (-np.inf if d.lbound is None else d.lbound,
                                      np.inf if d.ubound is None else d.ubound)
                            
                    break
                    
            c += k
        
        if ismat:
            if num_cons == 0:
                return (bds, [])
            # mat = scipy.sparse.csr_matrix((cons_A_data, (cons_A_r, cons_A_c)), shape = (num_cons, xsize))
            mat = scipy.sparse.coo_matrix((cons_A_data, (cons_A_r, cons_A_c)), shape = (num_cons, xsize)).toarray()
            lcons = scipy.optimize.LinearConstraint(mat, -numpy.inf, 1.0)
            return (bds, [lcons])
        else:
            return (bds, cons)
    
            
    
    def optimize(self, expr, vs, reg = None, sgn = 1, optimizer = None, learnrate = None, learnrate2 = None,
                 momentum = None, num_iter = None, num_iter2 = None, num_points = None, 
                 num_hop = None, hop_temp = None, hop_prob = None,
                 alm_rho = None, alm_rho_pow = None, alm_step = None, alm_penalty = None,
                 eps_converge = None, eps_tol = None):
        """
        Minimize/maximize expr with variables in the list vs, constrained in the region reg.
        Uses a combination of SLSQP, gradient descent, Adam (or any optimizer with the pytorch interface)
        and basin-hopping. Constraints are handled using augmented Lagrangian method.
        
        Kraft, D. A software package for sequential quadratic programming. 1988. Tech. Rep. DFVLR-FB 88-28, 
        DLR German Aerospace Center - Institute for Flight Mechanics, Koln, Germany.
        Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization."
        arXiv preprint arXiv:1412.6980 (2014).
        Wales, David J.; Doye, Jonathan P. K. (1997). "Global Optimization by Basin-Hopping
        and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms".
        The Journal of Physical Chemistry A. 101 (28): 5111-5116.
        Hestenes, M. R. (1969). "Multiplier and gradient methods". Journal of Optimization 
        Theory and Applications. 4 (5): 303-320.

        Parameters
        ----------
        expr : Expr
            The expression to be minimized. Can either be an Expr object or a 
            callable accepting ConcModel as argument.
        vs : list
            The variables to be optimized over (list of ConcDist and/or ConcReal).
        reg : Region, optional
            The region where the variables are constrained. The default is None.
        sgn : int, optional
            Set to 1 for maximization, -1 for minimization. The default is 1.
        optimizer : closure, optional
            The optimizer. Either "sgd", "adam" or any function that returns a pytorch
            optimizer given the list of variables as argument. The default is None.
        learnrate : float, optional
            Learning rate. The default is None.
        learnrate2 : float, optional
            Learning rate in phase 2. The default is None.
        momentum : float, optional
            Momentum. The default is None.
        num_iter : int, optional
            Number of iterations. The default is None.
        num_iter2 : int, optional
            Number of iterations in phase 2. The default is None.
        num_points : int, optional
            Number of random starting points. The default is None.
        num_hop : int, optional
            Number of hops for basin-hopping. The default is None.
        hop_temp : float, optional
            Temperature for basin-hopping. The default is None.
        hop_prob : float, optional
            Probability of hopping for basin-hopping. The default is None.
        alm_rho : float, optional
            Rho in augmented Lagrangian method. The default is None.
        alm_rho_pow : float, optional
            Multiply rho by this amount after each iteration. The default is None.
        eps_converge : float, optional
            Convergence is declared if the change is smaller than eps_converge. The default is None.
        eps_tol : float, optional
            Tolerance for equality and inequality constraints. The default is None.

        Returns
        -------
        r : float
            The minimum value.

        """
        
        verbose = PsiOpts.settings.get("verbose_opt", False)
        verbose_step = PsiOpts.settings.get("verbose_opt_step", False)
        verbose_step_var = PsiOpts.settings.get("verbose_opt_step_var", False)
        
        
        if optimizer is None:
            optimizer = PsiOpts.settings["opt_optimizer"]
        if learnrate is None:
            learnrate = PsiOpts.settings["opt_learnrate"]
        if learnrate2 is None:
            learnrate2 = PsiOpts.settings["opt_learnrate2"]
        if momentum is None:
            momentum = PsiOpts.settings["opt_momentum"]
        if num_iter is None:
            num_iter = PsiOpts.settings["opt_num_iter"]
        if num_iter2 is None:
            num_iter2 = PsiOpts.settings["opt_num_iter2"]
        if num_points is None:
            num_points = PsiOpts.settings["opt_num_points"]
        if num_hop is None:
            num_hop = PsiOpts.settings["opt_num_hop"]
        if hop_temp is None:
            hop_temp = PsiOpts.settings["opt_hop_temp"]
        if hop_prob is None:
            hop_prob = PsiOpts.settings["opt_hop_prob"]
        if alm_rho is None:
            alm_rho = PsiOpts.settings["opt_alm_rho"]
        if alm_rho_pow is None:
            alm_rho_pow = PsiOpts.settings["opt_alm_rho_pow"]
        if alm_step is None:
            alm_step = PsiOpts.settings["opt_alm_step"]
        if alm_penalty is None:
            alm_penalty = PsiOpts.settings["opt_alm_penalty"]
        if eps_converge is None:
            eps_converge = PsiOpts.settings["opt_eps_converge"]
        if eps_tol is None:
            eps_tol = PsiOpts.settings["opt_eps_tol"]
            
            
        truth = PsiOpts.settings["truth"]
        
        cs = self.copy_shallow()
        
        if isinstance(expr, (int, float)):
            expr = Expr.const(expr)
        
        # if reg is not None or expr.isregtermpresent():
        #     tvaropt = Expr.real("#TMPVAROPT")
        #     reg = reg & (tvaropt * sgn <= expr * sgn)
        #     reg = reg.
        
        if not isinstance(vs, list):
            vs = [vs]
            
        if reg is None:
            reg = []
        elif isinstance(reg, Region) or isinstance(reg, tuple):
            reg = [reg]
        elif not isinstance(reg, list):
            reg = [(reg, ">=")]
        
        for i in range(len(reg)):
            if isinstance(reg[i], RegionOp):
                reg[i] = reg[i].tosimple()
            if reg[i] is None:
                reg[i] = Region.universe()
            
        if truth is not None:
            truth_simple = truth.tosimple()
            if truth_simple is not None:
                reg.append(truth_simple)
        
        for i in range(len(reg)):
            if not cs.allcomprv().super_of(reg[i].allcomprv()):
                tvarlist, tcons = cs.add_reg(reg[i])
                reg[i] = tcons
                vs += tvarlist
            
            
        def str_tuple(x):
            if x[0] == 0:
                return iutil.tostr_verbose(x[1] * -sgn)
            return iutil.tostr_verbose((x[0], x[1] * -sgn))
            
        def tuple_copy1(x, y):
            for i in range(len(y)):
                if len(x) <= i:
                    x.append([None, y[i][1]])
                else:
                    x[i][1] = y[i][1]
            
        varlist, distlist = cs.convert_torch_tensors(vs)
        
        cons = []
        cons_init = []
        
        scipy_optimizer = None
        if optimizer == "sgd":
            pass
        elif optimizer == "adam":
            pass
        elif isinstance(optimizer, str):
            scipy_optimizer = optimizer
        
        for creg in reg:
            if isinstance(creg, Region):
                for a in creg.exprs_ge:
                    if a.isnonpos_ic2():
                        if self.bnet.check_ic(a):
                            continue
                    if scipy_optimizer is None:
                        slack = torch.tensor(0.0, dtype=torch.float64, requires_grad = True)
                        cons.append([a, 0.0, slack])
                        varlist.append(slack)
                    else:
                        cons.append([a, 0.0, True])
                
                for a in creg.exprs_eq:
                    if a.isnonpos_ic2() or a.isnonneg_ic2():
                        if self.bnet.check_ic(a):
                            continue
                    cons.append([a, 0.0])
                    
            elif isinstance(creg, tuple):
                if creg[1] == ">=":
                    if scipy_optimizer is None:
                        slack = torch.tensor(0.0, dtype=torch.float64, requires_grad = True)
                        cons.append([creg[0], 0.0, slack])
                        varlist.append(slack)
                    else:
                        cons.append([creg[0], 0.0, True])
                
        # if reg.isuniverse():
        #     reg = None
            
        
        ismat = False
        bds = None
        cons_list = None
        cons_list_fcn = None
        use_jac = False
        
        def get_fcn(cexpr, mul, has_f, has_d):
            def fcn(x):
                # print(str(cexpr) + "  " + str(x))
                cs.clear_cache()
                
                ConcModel.tensor_list_from_array(varlist, x)
                
                for d in distlist:
                    d.calc_torch()
                    
                cval = cs.get_tensor(cexpr)
                if mul != 1:
                    cval *= mul
                
                # print("   " + str(float(cval)))
                if not has_d:
                    return float(cval)
                
                if isinstance(cval, float):
                    if has_f:
                        return (float(cval), ConcModel.tensor_list_grad_to_array(varlist) * 0.0)
                    else:
                        return ConcModel.tensor_list_grad_to_array(varlist) * 0.0
                    
                cval.backward()
                
                if has_f:
                    return (float(cval), ConcModel.tensor_list_grad_to_array(varlist))
                else:
                    return ConcModel.tensor_list_grad_to_array(varlist)
            return fcn
        
        
        if scipy_optimizer is not None:
            
            ismat = (scipy_optimizer == "trust-constr")
            
            use_jac = True
            
            for cismat in ([False, True] if ismat else [False]):
                bds, cons_list = ConcModel.tensor_list_get_bds(varlist, distlist, ismat = cismat)
                # print(bds)
                
                for a in cons:
                    cons_dict = {}
                    if len(a) >= 3:
                        cons_dict["type"] = "ineq"
                    else:
                        cons_dict["type"] = "eq"
                        
                    cons_dict["fun"] = get_fcn(a[0], 1, True, False)
                    cons_dict["jac"] = get_fcn(a[0], 1, False, True)
                    
                    if cismat:
                        cons_list.append(scipy.optimize.NonlinearConstraint(
                            cons_dict["fun"], 0.0, numpy.inf, jac = cons_dict["jac"]
                            ))
                    else:
                        cons_list.append(cons_dict)
                
                if not cismat:
                    cons_list_fcn = cons_list
            
        
            
        
        tuple_copy1(cons_init, cons)
        
        int_big = 100000000
        
        r = (int_big, numpy.inf)
        r_v = []
        r_v_d = []
        t = (int_big, numpy.inf)
        
        num_iter_list = [num_iter]
        if num_iter2 > 0:
            num_iter_list.append(num_iter2)
        
        
        if verbose:
            print("========     model      ========")
            for a in cs.bnet.index.comprv:
                print(str(cs.bnet.get_parents(a)) + " -> " + str(a) + "  card=" + str(cs.get_card(a)))
                
            print("========     probs      ========")
            for (cin, cout), cp in cs.psmap.items():
                print(str(sum(cs.bnet.index.comprv[i] for i in cin)) + " -> " 
                      + str(sum(cs.bnet.index.comprv[i] for i in cout)) 
                      + (" var" if cp.isvar else "") + (" fcn" if cp.isfcn else ""))
            if sgn > 0:
                print("========    maximize    ========")
            else:
                print("========    minimize    ========")
            print(expr)
            
            print("========      over      ========")
            for d in distlist:
                if isinstance(d, ConcDist):
                    print("dist shape=" + str((d.shape_in, d.shape_out)))
                elif isinstance(d, ConcDist):
                    print("real=" + str(d.x))
            
            if len(cons):
                print("========  constraints   ========")
                for a in cons:
                    if len(a) >= 3:
                        print(str(a[0]) + ">=0")
                    else:
                        print(str(a[0]) + "==0")
                
                
        for cpass, cur_num_iter in enumerate(num_iter_list):
            cur_num_points = 1
            if cpass == 0:
                cur_num_points = num_points
                
            for ip in range(cur_num_points):
                
                if PsiOpts.is_timer_ended():
                    break
                
                if ip > 0:
                    for d in distlist:
                        d.randomize()
                    tuple_copy1(cons, cons_init)
                    cs.clear_cache()
                    
                cur_lr = learnrate
                cur_num_hop = num_hop
                if cpass > 0:
                    cur_lr = learnrate2
                    cur_num_hop = 1
                    
                for ih in range(cur_num_hop):
                
                    if PsiOpts.is_timer_ended():
                        break
                    
                    r_start = []
                    r_start_d = []
                    v_start = t
                    if ih > 0:
                        ConcModel.tensor_copy_list(r_start, varlist)
                        tuple_copy1(r_start_d, cons)
                        for d in distlist:
                            d.hop(hop_prob)
                        tuple_copy1(cons, cons_init)
                        cs.clear_cache()
                        
                    if scipy_optimizer is None:
                        if optimizer == "sgd":
                            cur_optimizer = torch.optim.SGD(varlist, lr = cur_lr, momentum = momentum)
                        elif optimizer == "adam":
                            cur_optimizer = torch.optim.Adam(varlist, lr = cur_lr)
                        else:
                            cur_optimizer = optimizer(varlist)
                    
                    # for a in con_eq:
                    #     a[1] = 0.0
                    
                    # for a in con_ge:
                    #     a[1] = 0.0
                    #     a[2].copy_(0.0)
                        
                    cr = (int_big, numpy.inf)
                    # lastval = (int_big, numpy.inf)
                    lastval = numpy.inf
                    
                    
                    
                    if scipy_optimizer is not None:
                        
                        res = None
                        resx = None
                        resfun = None
                        if len(varlist):
                            with warnings.catch_warnings():
                                
                                warnings.simplefilter("ignore")
                                
                                opts = {"maxiter": cur_num_iter, "disp": verbose_step}
                                if scipy_optimizer == "trust-constr":
                                    opts["initial_tr_radius"] = 0.001
                                    if verbose_step:
                                        opts["verbose"] = 3
                                
                                res = scipy.optimize.minimize(
                                    get_fcn(expr, -sgn, True, use_jac), 
                                    ConcModel.tensor_list_to_array(varlist),
                                    method = scipy_optimizer,
                                    jac = use_jac,
                                    bounds = bds,
                                    constraints = cons_list,
                                    tol = eps_tol,
                                    options = opts
                                    )
                                
                                resx = res.x
                                resfun = res.fun
                        else:
                            resx = numpy.array([])
                            resfun = get_fcn(expr, -sgn, True, use_jac)(numpy.array([]))[0]
                            
                        ConcModel.tensor_list_from_array(varlist, resx)
                        
                        bad = False
                        for d in distlist:
                            if isinstance(d, ConcDist) and d.isfcn:
                                d.clamp()
                                bad = True
                        if bad:
                            resx = ConcModel.tensor_list_to_array(varlist)
                            resfun = get_fcn(expr, -sgn, True, use_jac)(resx)[0]
                            
                        num_violate = 0
                        sum_violate = 0.0
                        for a in cons_list_fcn:
                            t = a["fun"](resx)
                            if numpy.isnan(t):
                                sum_violate = numpy.inf
                                num_violate += 1
                                continue
                            
                            if a["type"] == "eq":
                                if abs(t) > eps_tol:
                                    sum_violate += abs(t)
                                    num_violate += 1
                            else:
                                if t < -eps_tol:
                                    sum_violate += -t
                                    num_violate += 1
                                
                        penalty = sum_violate * alm_penalty
                                    
                        score = resfun + penalty
                        if numpy.isnan(score):
                            score = numpy.inf
                        t = (0, score)
                            
                            
                        if verbose_step:
                            print((("pass=" + str(cpass + 1) + " ") if cpass > 0 else "")
                                  + "#pt=" + str(ip)
                                  + " val=" + str(resfun)
                                  + " violate=" + str((num_violate, sum_violate)) + " opt=" + str_tuple(r))
                        
                    else:
                        for it in range(cur_num_iter + 1):
                
                            if PsiOpts.is_timer_ended():
                                break
                            
                            # cur_optimizer.zero_grad()
                            cval = cs.get_tensor(expr)
                            if -sgn != 1:
                                cval *= -sgn
                            
                            t_cval = float(cval)
                            if numpy.isnan(t_cval):
                                t_cval = numpy.inf
                            
                            crho = numpy.power(alm_rho_pow, it // alm_step) * alm_rho
                            crho0 = numpy.power(alm_rho_pow, (it - 1) // alm_step) * alm_rho
                            
                            num_violate = 0
                            sum_dual_step = 0.0
                            penalty = 0.0
                            
                            for a in cons:
                                con_val = cs.get_tensor(a[0])
                                p_con_val = float(con_val)
                                if len(a) >= 3:
                                    if p_con_val < -eps_tol:
                                        num_violate += 1
                                        penalty += -p_con_val * alm_penalty
                                else:
                                    if abs(p_con_val) > eps_tol:
                                        num_violate += 1
                                        penalty += abs(p_con_val) * alm_penalty
                                        
                                if len(a) >= 3:
                                    con_val -= a[2]
                                
                                t_con_val = float(con_val)
                                if numpy.isnan(t_con_val):
                                    t_con_val = 0.0
                                    
                                sum_dual_step += abs(t_con_val)
                                if it > 0 and it % alm_step == 0:
                                    t = a[1] + crho0 * t_con_val
                                    if not numpy.isnan(t):
                                        a[1] = t
                                cval += con_val * a[1] + torch.square(con_val) * crho * 0.5
                                
                                if verbose_step:
                                    if len(a) >= 3:
                                        # print(str(a[0]) + ">=0 : val=" + str(p_con_val) + " dual=" + str(a[1]))
                                        print(str(a[0]) + ">=0 : val=" + iutil.tostr_verbose(p_con_val) 
                                              + " dual=" + iutil.tostr_verbose(a[1]) + " slack=" + iutil.tostr_verbose(float(a[2])))
                                    else:
                                        print(str(a[0]) + "==0 : val=" + iutil.tostr_verbose(p_con_val) 
                                              + " dual=" + iutil.tostr_verbose(a[1]))
                                    
                            
                            t = (num_violate, t_cval + penalty)
                            cr = min(cr, t)
                            if not numpy.isnan(cr[1]) and cr < r:
                                r = cr
                                ConcModel.tensor_copy_list(r_v, varlist)
                                tuple_copy1(r_v_d, cons)
                            
                            if verbose_step:
                                print((("pass=" + str(cpass + 1) + " ") if cpass > 0 else "")
                                      + "#pt=" + str(ip) + " #iter=" + str(it) 
                                      + " val=" + str_tuple(t) + " opt=" + str_tuple(min(r, cr)))
                            
                            if verbose_step_var:
                                for d in distlist:
                                    print(d)
                                    
                            # if t[0] == lastval[0] and abs(t[1] - lastval[1]) <= eps_converge:
                            #     break
                            if abs(t_cval - lastval) + sum_dual_step <= eps_converge:
                                break
                            if it == cur_num_iter:
                                break
                            # lastval = t
                            lastval = t_cval
                            
                            cur_optimizer.zero_grad()
                            cval.backward()
                            cur_optimizer.step()
                            
                            for d in distlist:
                                d.clamp()
                            for a in cons:
                                if len(a) >= 3:
                                    if numpy.isnan(float(a[2])) or float(a[2]) < 0:
                                        with torch.no_grad():
                                            a[2].copy_(torch.tensor(0.0, dtype=torch.float64))
                                
                            cs.clear_cache()
                        
                    cr = min(cr, t)
                    if not numpy.isnan(cr[1]) and cr < r:
                        r = cr
                        ConcModel.tensor_copy_list(r_v, varlist)
                        tuple_copy1(r_v_d, cons)
                    
                    if ih > 0:
                        accept_prob = 1.0
                        if numpy.isnan(t[1]):
                            accept_prob = 0.0
                        elif t > v_start:
                            accept_prob = numpy.exp((v_start[1] - t[1]) / hop_temp)
                        if verbose_step:
                            print("HOP #" + str(ih) + " from=" + str_tuple(v_start) 
                                  + " to=" + str_tuple(t) + " prob=" + iutil.tostr_verbose(accept_prob))
                        if numpy.random.uniform() >= accept_prob:
                            if verbose_step:
                                print("HOP REJECT")
                            ConcModel.tensor_copy_list(varlist, r_start)
                            tuple_copy1(cons, r_start_d)
                            for d in distlist:
                                d.calc_torch()
                            t = v_start
                            cs.clear_cache()
                        else:
                            if verbose_step:
                                print("HOP ACCEPT")
                        
            if len(r_v):
                ConcModel.tensor_copy_list(varlist, r_v)
                tuple_copy1(cons, r_v_d)
                for d in distlist:
                    d.calc_torch()
                cs.clear_cache()
            
        self.opt_reg = cs
        self.clear_cache()
        return (r[1] if r[0] == 0 else numpy.inf) * -sgn

    def minimize(self, *args, **kwargs):
        """
        Maximize expr with variables in the list vs, constrained in the region reg.
        Refer to optimize for details.
        """
        return self.optimize(*args, sgn = -1, **kwargs)

    def maximize(self, *args, **kwargs):
        """
        Maximize expr with variables in the list vs, constrained in the region reg.
        Refer to optimize for details.
        """
        return self.optimize(*args, sgn = 1, **kwargs)
    
    def opt_model(self):
        return self.opt_reg
    
    def get_bayesnet(self):
        return self.bnet.copy()
    
    def table(self, *args, **kwargs):
        """Plot the information diagram as a Karnaugh map.
        """
        return universe().table(*args, self, **kwargs)
        
    def venn(self, *args, **kwargs):
        """Plot the information diagram as a Venn diagram.
        Can handle up to 5 random variables (uses Branko Grunbaum's Venn diagram for n=5).
        """
        return universe().venn(*args, self, **kwargs)
        
    def graph(self, **kwargs):
        """Return the Bayesian network among the random variables as a 
        graphviz digraph that can be displayed in the console.
        """
        return self.get_bayesnet().graph(**kwargs)
    
    
class SparseMat:
    """List of lists sparse matrix. Do NOT use directly"""
    
    def __init__(self, width):
        self.width = width
        self.x = []
    
    def from_row(row, width):
        r = SparseMat(width)
        r.x.append(list(row))
        return r
    
    def ratio(self, other):
        ceps = PsiOpts.settings["eps"]
        r = None
        for a, b in zip(self.x, other.x):
            if len(a) != len(b):
                return None
            for ax, bx in zip(a, b):
                if ax[0] != bx[0]:
                    return None
                if abs(bx[1]) <= ceps:
                    if abs(ax[1]) <= ceps:
                        continue
                    else:
                        return None
                t = ax[1] / bx[1]
                if r is not None and abs(t - r) > ceps:
                    return None
                r = t
        return r
    
    def addrow(self):
        self.x.append([])
        
    def poprow(self):
        self.x.pop()
        
    def add_last_row(self, j, c):
        self.x[len(self.x) - 1].append((j, c))
    
    def extend(self, other):
        self.width = max(self.width, other.width)
        self.x += other.x
    
    def simplify_row(self, i):
        ceps = PsiOpts.settings["eps"]
        self.x[i].sort()
        t = self.x[i]
        self.x[i] = []
        cj = -1
        cc = 0.0
        for (j, c) in t:
            if j == cj:
                cc += c
            else:
                if abs(cc) > ceps:
                    self.x[i].append((cj, cc))
                cj = j
                cc = c
        if abs(cc) > ceps:
            self.x[i].append((cj, cc))
                
    def simplify_last_row(self):
        self.simplify_row(len(self.x) - 1)
        
    def simplify(self):
        for i in range(len(self.x)):
            self.simplify_row(i)
        
    def last_row_isempty(self):
        return len(self.x[len(self.x) - 1]) == 0
    
    def unique_row(self):
        ir = 1.61803398875 - 0.1
        p = 10007
        rowmap = {}
        for i in range(len(self.x)):
            a = self.x[i]
            h = 0
            for (j, c) in a:
                h = h * p + hash(c + j * ir)
            if h in rowmap:
                if a == self.x[rowmap[h]]:
                    a[:] = []
            else:
                rowmap[h] = i
        self.x = [a for a in self.x if len(a) > 0]
        
    def row_dense(self, i):
        r = [0.0] * self.width
        for (j, c) in self.x[i]:
            r[j] += c
        return r
        
    def nonzero_cols(self):
        r = [False] * self.width
        for i in range(len(self.x)):
            for (j, c) in self.x[i]:
                r[j] = True
        return r
        
    def mapcol(self, m, allowmiss = False):
        for i in range(len(self.x)):
            for k in range(len(self.x[i])):
                j2 = m[self.x[i][k][0]]
                if not allowmiss and j2 < 0:
                    return False
                self.x[i][k] = (j2, self.x[i][k][1])
            if allowmiss:
                self.x[i] = [(j, c) for (j, c) in self.x[i] if j >= 0]
        return True
    
    def sumrows(self, m, remove = False):
        r = []
        for i2 in range(len(self.x)):
            a = self.x[i2]
            cr = 0.0
            did = False
            for i in range(len(a)):
                if a[i][0] in m:
                    cr += m[a[i][0]] * a[i][1]
                    if remove:
                        a[i] = (-1, a[i][1])
                        did = True
            if did:
                self.x[i2] = [(j, c) for (j, c) in self.x[i2] if j >= 0]
            r.append(cr)
        return r
    
    def tolil(self):
        r = scipy.sparse.lil_matrix((len(self.x), self.width))
        for i in range(len(self.x)):
            for (j, c) in self.x[i]:
                r[i, j] += c
        return r
        
    def tonumpyarray(self):
        r = numpy.zeros((len(self.x), self.width))
        for i in range(len(self.x)):
            for (j, c) in self.x[i]:
                r[i, j] += c
        return r

    
class LinearProg:
    """A linear programming instance. Do NOT use directly"""
    
    def __init__(self, index, lptype, bnet = None, lp_bounded = None, save_res = False, prereg = None, dual_enabled = None, val_enabled = None):
        self.index = index
        self.lptype = lptype
        self.nvar = 0
        self.nxvar = 0
        self.xvarid = []
        self.realshift = 0
        self.cellpos = []
        self.bnet = bnet
        self.pinfeas = False
        self.constmap = {}
        
        if lp_bounded is None:
            self.lp_bounded = PsiOpts.settings["lp_bounded"]
        else:
            self.lp_bounded = lp_bounded
        
        self.lp_ubound = PsiOpts.settings["lp_ubound"]
        self.lp_eps = PsiOpts.settings["lp_eps"]
        self.lp_eps_obj = PsiOpts.settings["lp_eps_obj"]
        self.zero_cutoff = PsiOpts.settings["lp_zero_cutoff"]
        self.eps_present = False
        self.affine_present = False
        
        self.fcn_mode = PsiOpts.settings["fcn_mode"]
        self.fcn_list = []
        
        self.save_res = save_res
        self.saved_var = []
        
        if prereg is not None:
            if self.fcn_mode >= 1:
                for x in prereg.exprs_gei:
                    self.addExpr_ge0_fcn(x)
                for x in prereg.exprs_eqi:
                    self.addExpr_ge0_fcn(x)
        
        if self.lptype == LinearProgType.H:
            self.nvar = (1 << self.index.num_rv()) - 1 + self.index.num_real()
            self.realshift = (1 << self.index.num_rv()) - 1
            
        elif self.lptype == LinearProgType.HC1BN:
            n = self.index.num_rv()
            nbnet = bnet.index.num_rv()
            
            self.cellpos = [-2] * (1 << n)
            cpos = 0
            for i in range(n):
                for mask in range(1 << i):
                    maski = mask + (1 << i)
                    
                    if self.fcn_mode >= 1:
                        if self.checkfcn_mask(1 << i, mask):
                            self.cellpos[maski] = -1
                            continue
                        
                        for j in range(i):
                            if mask & (1 << j) != 0:
                                if self.checkfcn_mask(1 << j, mask - (1 << j)):
                                    maskj = mask - (1 << j) + (1 << i)
                                    self.cellpos[maski] = self.cellpos[maskj]
                                    break
                                    
                    if self.cellpos[maski] != -2:
                        continue
                    
                    if i >= nbnet:
                        self.cellpos[maski] = cpos
                        cpos += 1
                        continue
                        
                    for j in range(i):
                        if mask & (1 << j) != 0:
                            if bnet.check_ic_mask(1 << i, 1 << j, mask - (1 << j)):
                                maskj = mask - (1 << j) + (1 << i)
                                self.cellpos[maski] = self.cellpos[maskj]
                                break
                    if self.cellpos[maski] == -2:
                        self.cellpos[maski] = cpos
                        cpos += 1
            self.realshift = cpos
            self.nvar = self.realshift + self.index.num_real()
            
        
        self.nxvar = self.nvar
        self.Au = SparseMat(self.nvar)
        self.Ae = SparseMat(self.nvar)
        
        self.solver = None
        if self.nvar <= PsiOpts.settings["solver_scipy_maxsize"]:
            self.solver = iutil.get_solver("scipy")
        else:
            self.solver = iutil.get_solver()
        
        self.solver_param = {}
        self.bu = []
        self.be = []
        
        self.icp = [[] for i in range(self.index.num_rv())]
        
        self.dual_u = None
        self.dual_e = None
        self.dual_pf = (PsiOpts.settings["proof_enabled"] 
                        and not PsiOpts.settings["proof_nowrite"])
        
        if dual_enabled is not None:
            self.dual_enabled = dual_enabled
        else:
            self.dual_enabled = PsiOpts.settings["proof_enabled"]
        
        self.val_x = None
        if val_enabled is not None:
            self.val_enabled = val_enabled
        else:
            self.val_enabled = False
        
        self.optval = None
        
    def get_optval(self):
        return self.optval
    
    def addreal_id(self, A, k, c):
        A.add_last_row(self.realshift + k, c)
    
    def addH_mask(self, A, mask, c):
        if self.lptype == LinearProgType.H:
            A.add_last_row(mask - 1, c)
        elif self.lptype == LinearProgType.HC1BN:
            n = self.index.num_rv()
            for i in range(n):
                if mask & (1 << i) != 0:
                    cp = self.cellpos[mask & ((1 << (i + 1)) - 1)]
                    if cp >= 0:
                        A.add_last_row(cp, c)
    
    def addIc_mask(self, A, x, y, zmask, c):
        if self.lptype == LinearProgType.H:
            if x == y:
                A.add_last_row(((1 << x) | zmask) - 1, c)
                if zmask != 0:
                    A.add_last_row(zmask - 1, -c)
            else:
                A.add_last_row(((1 << x) | zmask) - 1, c)
                A.add_last_row(((1 << y) | zmask) - 1, c)
                A.add_last_row(((1 << x) | (1 << y) | zmask) - 1, -c)
                if zmask != 0:
                    A.add_last_row(zmask - 1, -c)
        elif self.lptype == LinearProgType.HC1BN:
            if x == y:
                self.addH_mask(A, (1 << x) | zmask, c)
                if zmask != 0:
                    self.addH_mask(A, zmask, -c)
            else:
                self.addH_mask(A, (1 << x) | zmask, c)
                self.addH_mask(A, (1 << y) | zmask, c)
                self.addH_mask(A, (1 << x) | (1 << y) | zmask, -c)
                if zmask != 0:
                    self.addH_mask(A, zmask, -c)
    
    def addExpr(self, A, x):
        for (a, c) in x.terms:
            termType = a.get_type()
            if termType == TermType.IC:
                k = len(a.x)
                for t in range(1 << k):
                    csgn = -1;
                    mask = self.index.get_mask(a.z)
                    for i in range(k):
                        if (t & (1 << i)) != 0:
                            csgn = -csgn
                            mask |= self.index.get_mask(a.x[i])
                    if mask != 0:
                        self.addH_mask(A, mask, c * csgn)
            elif termType == TermType.REAL or termType == TermType.REGION:
                k = self.index.get_index(a.x[0].varlist[0])
                if k >= 0:
                    self.addreal_id(A, k, c)
                
    def addfcn(self, x):
        for (a, c) in x.terms:
            termType = a.get_type()
            if termType == TermType.IC and len(a.x) == 1:
                self.fcn_list.append((self.index.get_mask(a.x[0]), self.index.get_mask(a.z)))
        
    def checkfcn_mask(self, xmask, zmask):
        if xmask < 0:
            return False
        if zmask | xmask == zmask:
            return True
        did = True
        while did:
            did = False
            for cx, cz in self.fcn_list:
                if cz | zmask == zmask and cx | zmask != zmask:
                    zmask |= cx
                    if zmask | xmask == zmask:
                        return True
                    did = True
        return False
        
    def checkfcn(self, x, z):
        xmask = self.index.get_mask(x)
        if xmask < 0:
            return False
        zmask = self.index.get_mask(z.inter(self.index.comprv))
        return self.checkfcn_mask(xmask, zmask)
    
        
    def addExpr_ge0(self, x):
        if x.size() == 0:
            return
        self.Au.addrow()
        self.addExpr(self.Au, -x)
        self.bu.append(0.0)
        
    def addExpr_ge0_fcn(self, x):
        if x.size() == 0:
            return
        if self.fcn_mode >= 1:
            if x.isnonpos():
                self.addfcn(x)
        
    def addExpr_eq0(self, x):
        if x.size() == 0:
            return
        self.Ae.addrow()
        self.addExpr(self.Ae, x)
        self.be.append(0.0)
        
    def addExpr_eq0_fcn(self, x):
        if x.size() == 0:
            return
        if self.fcn_mode >= 1:
            if x.isnonpos() or x.isnonneg():
                self.addfcn(x)
        
        
    def add_ent_ineq(self):
        n = self.index.num_rv()
        npow = (1 << n)
        for x in range(n):
            self.Au.addrow()
            self.addIc_mask(self.Au, x, x, npow - 1 - (1 << x), -1.0)
            self.bu.append(0.0)
            
        for x in range(n):
            for y in range(x + 1, n):
                zmask = 0
                while zmask < npow:
                    self.Au.addrow()
                    self.addIc_mask(self.Au, x, y, zmask, -1.0)
                    if self.lptype == LinearProgType.HC1BN:
                        self.Au.simplify_last_row()
                        if self.Au.last_row_isempty():
                            self.Au.poprow()
                        else:
                            self.bu.append(0.0)
                    else:
                        self.bu.append(0.0)
                    zmask += 1
                    if (zmask & (1 << x)) != 0:
                        zmask += (1 << x)
                    if (zmask & (1 << y)) != 0:
                        zmask += (1 << y)
        
    def finish(self):
        ceps = PsiOpts.settings["eps"]
        
        self.add_ent_ineq()
        
        PsiRec.num_lpprob += 1
        
        #self.Au.simplify()
        #self.Ae.simplify()
        
        if self.lptype == LinearProgType.HC1BN:
            self.Au.unique_row()
            self.bu = [0.0] * len(self.Au.x)
        
        
        if False:
            k = self.index.get_index(IVar.one())
            if k >= 0:
                self.Ae.addrow()
                self.addreal_id(self.Ae, k, 1.0)
                self.be.append(1.0)
                self.affine_present = True
            
            k = self.index.get_index(IVar.eps())
            if k >= 0:
                self.Ae.addrow()
                self.addreal_id(self.Ae, k, 1.0)
                self.be.append(self.lp_eps)
                self.eps_present = True
                self.affine_present = True
            
            k = self.index.get_index(IVar.inf())
            if k >= 0:
                self.Ae.addrow()
                self.addreal_id(self.Ae, k, 1.0)
                self.be.append(self.lp_ubound)
                self.affine_present = True
        
        
        if True:
            id_one = self.index.get_index(IVar.one())
            if id_one >= 0:
                id_one += self.realshift
            id_eps = self.index.get_index(IVar.eps())
            if id_eps >= 0:
                id_eps += self.realshift
            id_inf = self.index.get_index(IVar.inf())
            if id_inf >= 0:
                id_inf += self.realshift
            
            if id_one >= 0 or id_eps >= 0 or id_inf >= 0:
                self.affine_present = True
                    
            if id_eps >= 0:
                self.eps_present = True
        
        
        
        if self.affine_present:
            self.lp_bounded = True
            
        if self.lp_bounded:
            if self.index.num_rv() > 0:
                self.Au.addrow()
                self.addH_mask(self.Au, (1 << self.index.num_rv()) - 1, 1.0)
                #for i in range(self.realshift):
                #    self.Au.add_last_row(i, 1.0)
                self.bu.append(self.lp_ubound)
            for i in range(self.index.num_real()):
                if Term.fromcomp(self.index.compreal[i]).isrealvar():
                    self.Au.addrow()
                    self.addreal_id(self.Au, i, 1.0)
                    self.bu.append(self.lp_ubound)
                    self.Au.addrow()
                    self.addreal_id(self.Au, i, -1.0)
                    self.bu.append(self.lp_ubound)
                
                    
        cols = self.Au.nonzero_cols()
        coles = self.Ae.nonzero_cols()
        cols = [a or b for a, b in zip(cols, coles)]
        
        #print(self.Au.x)
        
        if True:
            self.constmap = {}
            if id_one >= 0:
                cols[id_one] = False
                self.constmap[id_one] = 1.0
            if id_eps >= 0:
                cols[id_eps] = False
                self.constmap[id_eps] = self.lp_eps
            if id_inf >= 0:
                cols[id_inf] = False
                self.constmap[id_inf] = self.lp_ubound
            
            if len(self.constmap):
                self.bu = [b - a for a, b in zip(self.Au.sumrows(self.constmap, remove = True), self.bu)]
                self.be = [b - a for a, b in zip(self.Ae.sumrows(self.constmap, remove = True), self.be)]
        
        #print(self.Au.x)
        
        self.xvarid = [0] * self.nvar
        self.nxvar = 0
        for i in range(self.nvar):
            if cols[i]:
                self.xvarid[i] = self.nxvar
                self.nxvar += 1
            else:
                self.xvarid[i] = -1
                
        self.Au.mapcol(self.xvarid)
        self.Au.width = self.nxvar
        self.Ae.mapcol(self.xvarid)
        self.Ae.width = self.nxvar
        
        # print(self.xvarid)
        # print(self.Au.x)
        # print(self.bu)
        #print(self.Au.x)
        
        if True:
            for i in range(len(self.Au.x)):
                if len(self.Au.x[i]) == 0:
                    if self.bu[i] < -ceps:
                        self.pinfeas = True
                    self.bu[i] = None
            
            for i in range(len(self.Ae.x)):
                if len(self.Ae.x[i]) == 0:
                    if abs(self.be[i]) > ceps:
                        self.pinfeas = True
                    self.be[i] = None
            
            self.Au.x = [a for a in self.Au.x if len(a)]
            self.Ae.x = [a for a in self.Ae.x if len(a)]
            self.bu = [a for a in self.bu if a is not None]
            self.be = [a for a in self.be if a is not None]
        
        #print(self.Au.x)
        
        if self.solver == "scipy":
            self.solver_param["Aus"] = self.Au.tolil()
            self.solver_param["Aes"] = self.Ae.tolil()
            
        elif self.solver.startswith("pulp."):
            prob = pulp.LpProblem("lpentineq" + str(PsiRec.num_lpprob), pulp.LpMinimize)
            xvar = pulp.LpVariable.dicts("x", [str(i) for i in range(self.nxvar)])
            
            if True:
                for a, b in zip(self.Au.x, self.bu):
                    if len(a):
                        prob += pulp.LpConstraint(pulp.lpSum([xvar[str(j)] * c for (j, c) in a]), 
                                                  sense = -1, rhs = b)
                
                for a, b in zip(self.Ae.x, self.be):
                    if len(a):
                        prob += pulp.LpConstraint(pulp.lpSum([xvar[str(j)] * c for (j, c) in a]), 
                                                  sense = 0, rhs = b)
            
            if False:
                for a, b in zip(self.Au.x, self.bu):
                    if len(a):
                        #print(" $ ".join([str((j, c)) for (j, c) in a]))
                        prob += pulp.LpConstraint(pulp.LpAffineExpression([(xvar[str(j)], c) for (j, c) in a]), 
                                                  sense = -1, rhs = b)
                
                for a, b in zip(self.Ae.x, self.be):
                    if len(a):
                        prob += pulp.LpConstraint(pulp.LpAffineExpression([(xvar[str(j)], c) for (j, c) in a]), 
                                                  sense = 0, rhs = b)
                        
            if False:
                for i in range(len(self.Au.x)):
                    cexpr = None
                    for (j, c) in self.Au.x[i]:
                        if cexpr is None:
                            cexpr = c * xvar[str(j)]
                        else:
                            cexpr += c * xvar[str(j)]
                    if cexpr is not None:
                        prob += cexpr <= self.bu[i]
                
                for i in range(len(self.Ae.x)):
                    cexpr = None
                    for (j, c) in self.Ae.x[i]:
                        if cexpr is None:
                            cexpr = c * xvar[str(j)]
                        else:
                            cexpr += c * xvar[str(j)]
                    if cexpr is not None:
                        prob += cexpr == self.be[i]
                    
            #print(prob)
            
            self.solver_param["prob"] = prob
            self.solver_param["xvar"] = xvar
        
        elif self.solver.startswith("pyomo."):
            solver_opt = self.solver[self.solver.index(".") + 1 :]
            opt = SolverFactory(solver_opt)
            
            model = pyo.ConcreteModel()
            
            if self.dual_enabled:
                model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            
            model.n = pyo.Param(default=self.nxvar)
            model.x = pyo.Var(pyo.RangeSet(model.n), domain=pyo.Reals)
            
            model.c = pyo.ConstraintList()
            
            for i in range(len(self.Au.x)):
                cexpr = None
                for (j, c) in self.Au.x[i]:
                    if cexpr is None:
                        cexpr = c * model.x[j + 1]
                    else:
                        cexpr += c * model.x[j + 1]
                if cexpr is not None:
                    model.c.add(cexpr <= self.bu[i])
                    
            for i in range(len(self.Ae.x)):
                cexpr = None
                for (j, c) in self.Ae.x[i]:
                    if cexpr is None:
                        cexpr = c * model.x[j + 1]
                    else:
                        cexpr += c * model.x[j + 1]
                if cexpr is not None:
                    model.c.add(cexpr == self.be[i])
            
            
            self.solver_param["opt"] = opt
            self.solver_param["model"] = model
            
    def id_toexpr(self):
        n = self.index.num_rv()
        
        xvarinv = [0] * self.nxvar
        for i in range(self.nvar):
            if self.xvarid[i] >= 0:
                xvarinv[self.xvarid[i]] = i
        
        cellposinv = list(range(1, self.realshift + 1))
        if self.lptype == LinearProgType.HC1BN:
            for mask in range((1 << n) - 1, 0, -1):
                if self.cellpos[mask] >= 0:
                    cellposinv[self.cellpos[mask]] = mask
                
        def rf(j2):
            j = xvarinv[j2]
            
            if j >= self.realshift:
                return (Expr.real(self.index.compreal.varlist[j - self.realshift].name), 0)
            
            mask = cellposinv[j]
            term = None
            
            if self.lptype == LinearProgType.H:
                term = Term.H(Comp.empty())
                for i in range(n):
                    if mask & (1 << i) != 0:
                        term.x[0].varlist.append(self.index.comprv.varlist[i])
                    
            elif self.lptype == LinearProgType.HC1BN:
                term = Term.Hc(Comp.empty(), Comp.empty())
                for i in range(n):
                    if mask & (1 << i) != 0:
                        term.z.varlist.append(self.index.comprv.varlist[i])
                term.x[0].varlist.append(term.z.varlist.pop())
            
            return (Expr.fromterm(term), mask)
        
        return rf
    
    def row_toexpr(self):
        idt = self.id_toexpr()
        
        def rf(x):
            expr = Expr.zero()
            if len(x) == 0:
                return expr
            if isinstance(x[0], tuple):
                for (j2, c) in x:
                    te, mask = idt(j2)
                    #expr.terms.append((te, c))
                    expr += te * c
            else:
                ceps = PsiOpts.settings["eps"]
                for i in range(len(x)):
                    if abs(x[i]) > ceps:
                        te, mask = idt(i)
                        expr += te * x[i]
            return expr
        
        return rf
        
        
    def get_region(self, toreal = None, toreal_only = False, A = None, skip_simplify = False):
        if toreal is None:
            toreal = Comp.empty()
        
        torealmask = self.index.get_mask(toreal)
        idt = self.id_toexpr()
        
        if A is None:
            A = ([(a, 1, b) for a, b in zip(self.Au.x, self.bu)] + 
                [(a, 0, b) for a, b in zip(self.Ae.x, self.be)])
        else:
            A = [(a, 1, 0.0) for a in A.x]
            
        
        r = Region.universe()
        for x, sn, b in A:
            expr = Expr.zero()
            toreal_present = False
            
            for (j2, c) in x:
                te, mask = idt(j2)
                
                termreal = (mask & torealmask) != 0
                toreal_present |= termreal
                
                if termreal:
                    expr += Expr.real("R_" + str(te)) * c
                else:
                    expr += te * c
            
            if toreal_present or not toreal_only:
                if not skip_simplify:
                    expr.simplify()
                
                if sn == 1:
                    r &= (expr <= Expr.const(b))
                else:
                    r &= (expr == Expr.const(b))
        
        return r
    
    def get_dual_region(self):
        if self.dual_e is None or self.dual_u is None:
            return None
        rowt = self.row_toexpr()
        ceps = PsiOpts.settings["eps"]
        r = Region.universe()
        
        for x, d in zip(self.Au.x, self.dual_u):
            if abs(d) > ceps:
                r.exprs_ge.append(rowt(x).simplified() * d)
        
        for x, d in zip(self.Ae.x, self.dual_e):
            if abs(d) > ceps:
                r.exprs_eq.append(rowt(x).simplified() * d)
        
        return r
    
    def get_dual_sum_region(self, x):
        r = self.get_dual_region()
        with PsiOpts(proof_enabled = False):
            rsum = sum(r.exprs_ge + r.exprs_eq, Expr.zero())
            rdiff = (x - rsum).simplified()
            if not rdiff.iszero():
                rowt = self.row_toexpr()
                for t in rdiff:
                    v = self.get_vec(t)[0]
                    if v is not None:
                        tv = (t - rowt(v)).simplified()
                        if not tv.iszero():
                            r.exprs_eq.append(tv)
                
        return r
            
    
    def write_pf(self, x):
        r = self.get_dual_sum_region(x)
        if r is None:
            return
        xstr = str(x) + " >= 0"
        pf = None
        
        if PsiOpts.settings["proof_step_dualsum"]:
            rt = r.removed_trivial()
            pf = None
            if rt.isuniverse():
                pf = ProofObj.from_region(x >= 0, c = "Claim: ")
            else:
                pf = ProofObj.from_region(("implies", rt, x >= 0), c = "Claim: ")
            cadds, csums = r.get_sum_seq()
            for i, (cadd, csum) in enumerate(zip(cadds, csums)):
                if i == 0:
                    pf += ProofObj.from_region(csum >= 0, c = "Have:")
                else:
                    pf += ProofObj.from_region(csum >= 0, c = ["Add: ", cadd >= 0])
            
            # self.trytry()
            # r2 = Region.universe()
            # cur = Expr.zero()
            # for x in r.exprs_ge:
            #     cur = (cur + x).simplified()
            #     r2.exprs_ge.append(cur)
            # for x in r.exprs_eq:
            #     cur = (cur + x).simplified()
            #     r2.exprs_ge.append(cur)
            # pf += ProofObj.from_region(r2, c = "Steps for " + xstr)
        else:
            pf = ProofObj.from_region(r, c = ["Duals for ", x >= 0])
            
        PsiOpts.set_setting(proof_add = pf)
    
    def get_extreme_rays_vec(self, A = None):
        ma = None
        cn = 0
        if A is None:
            cn = self.Au.width
            ma = self.Ae.tonumpyarray()
            ma = numpy.vstack((numpy.zeros(cn), self.Au.tonumpyarray(), ma, -ma))
        else:
            cn = A.width
            ma = numpy.vstack((numpy.zeros(cn), A.tonumpyarray()))
            
        #print(ma)
        hull = scipy.spatial.ConvexHull(ma)
        #print("ConvexHull finished")
        r = []
        #tone = numpy.array(self.get_vec(Expr.H(self.index.comprv)))
        
        rset = set()
        ceps = PsiOpts.settings["eps"]
        
        for i in range(len(hull.simplices)):
            if abs(hull.equations[i,-1]) > ceps:
                continue
            t = hull.equations[i,:-1]
            
            vv = max(abs(t))
            if vv > ceps:
                t = t / vv
            
            ts = ",".join(iutil.float_tostr(x) for x in t)
            if ts not in rset:
                rset.add(ts)
                r.append(t)
        
        return r
        
    def get_region_elim_rays(self, aux = None, A = None, skip_simplify = False):
        if aux is None:
            aux = Comp.empty()
        ceps = PsiOpts.settings["eps"]
        
        auxmask = self.index.get_mask(aux)
        #print("Before get_extreme_rays_vec")
        vs = self.get_extreme_rays_vec(A)
        #print("After get_extreme_rays_vec")
        idt = self.id_toexpr()
        var_expr = [None] * self.nxvar
        var_id = [-1] * self.nxvar
        var_id_inv = [-1] * self.nxvar
        nleft = 0
        for j in range(self.nxvar):
            var_expr[j], mask = idt(j)
            if mask & auxmask == 0:
                var_id[j] = nleft
                var_id_inv[nleft] = j
                nleft += 1
                
        vset = set()
        vset.add(",".join(["0"] * nleft))
        ma = numpy.zeros((1, nleft))
        for v in vs:
            nvv = 0
            vv = numpy.zeros(nleft)
            for i in range(len(v)):
                if var_id[i] >= 0:
                    vv[var_id[i]] = v[i]
                    nvv += 1
            if nvv > 0:
                ts = ",".join(iutil.float_tostr(x) for x in vv)
                #print("ts = " + ts)
                if ts not in vset:
                    vset.add(ts)
                    ma = numpy.vstack((ma, vv))
        
        
        eig, ev = numpy.linalg.eig(ma.T.dot(ma))
        #print(nleft)
        #print(ma)
        #print(ma.T.dot(ma))
        #print(eig)
        #print(ev)
        ev0 = numpy.zeros((nleft, 0))
        ev1 = numpy.zeros((nleft, 0))
        for i in range(nleft):
            if abs(eig[i]) <= ceps:
                ev0 = numpy.hstack((ev0, ev[:,i:i+1]))
            else:
                ev1 = numpy.hstack((ev1, ev[:,i:i+1]))
                
        def expr_fromvec(vec):
            vv = max(abs(vec))
            if vv > ceps:
                vec = vec / vv
            cexpr = Expr.zero()
            for i in range(nleft):
                if abs(vec[i]) > ceps:
                    cexpr += var_expr[var_id_inv[i]] * vec[i]
            if not skip_simplify:
                cexpr.simplify()
            return cexpr
            
        
        mv = ma.dot(ev1)
        
        #print(ev0)
        #print(ev1)
        
        r = Region.universe()
        for i in range(ev0.shape[1]):
            expr = expr_fromvec(ev0[:,i])
            #print(expr)
            r.iand_norename(expr == 0)
        
        if ev1.shape[1] == 0:
            return r
        
        if ev1.shape[1] == 1:
            svis = [False, False]
            for i in range(len(mv)):
                if mv[i, 1] > ceps:
                    svis[1] = True
                if mv[i, 1] < -ceps:
                    svis[0] = True
                    
            if svis[0] and svis[1]:
                return r
            expr = expr_fromvec(ev1[:,0])
            if svis[0]:
                if not expr.isnonneg():
                    r.iand_norename(expr >= 0)
            elif svis[1]:
                if not expr.isnonpos():
                    r.iand_norename(expr <= 0)
            else:
                r.iand_norename(expr == 0)
            return r
        
            
        hull = scipy.spatial.ConvexHull(mv)
        
        #print(len(hull.simplices))
        #tone_o = numpy.array(self.get_vec(Expr.H(self.index.comprv - aux)))
        #tone = numpy.array([tone_o[var_id_inv[j]] for j in range(nleft)])
        
        rset = set()
        
        for i in range(len(hull.simplices)):
            
            if abs(hull.equations[i,-1]) > ceps:
                continue
            t = ev1.dot(hull.equations[i,:-1])
            
            vv = max(abs(t))
            #print(t)
            #print(vv)
            if vv > ceps:
                t = t / vv
            #print(t)
            
            ts = ",".join(iutil.float_tostr(x) for x in t)
            if ts not in rset:
                rset.add(ts)
                expr = expr_fromvec(t)
                if not expr.isnonpos():
                    r.iand_norename(expr <= 0)
        
        if not skip_simplify:
            r.simplify_quick()
        
        return r
        
    
    def get_extreme_rays(self, A = None):
        idt = self.id_toexpr()
        vs = self.get_extreme_rays_vec(A)
        r = RegionOp.union([])
        ceps = PsiOpts.settings["eps"]
        for v in vs:
            tr = Region.universe()
            for i in range(len(v)):
                if abs(v[i]) > ceps:
                    te, mask = idt(i)
                    tr &= te == v[i]
            r |= tr
        return r
    
    
    def get_vec(self, x, sparse = False):
        optobj = SparseMat(self.nvar)
        optobj.addrow()
        self.addExpr(optobj, x)
        optobj.simplify_last_row()
        
        c1 = 0.0
        if len(self.constmap):
            c1 = optobj.sumrows(self.constmap, remove = True)[0]
            
        if not optobj.mapcol(self.xvarid):
            return None, None
        
        optobj.width = self.nxvar
        
        if sparse:
            return (optobj, c1)
        else:
            return (optobj.row_dense(0), c1)
    
    def call_prog(self, c):
        
        ceps = PsiOpts.settings["eps"]
        if all(abs(x) <= ceps for x in c):
            return (None, None)
            
        if len(self.Au.x) == 0 and len(self.Ae.x) == 0:
            return (None, None)
        
        if self.solver == "scipy":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = scipy.optimize.linprog(c, self.solver_param["Aus"], self.bu, self.solver_param["Aes"], self.be, 
                                         bounds = (None, None), method = "interior-point", options={'sparse': True})
            
            
            if res.status == 0:
                return (float(res.fun), [float(res.x[i]) for i in range(self.nxvar)])
            return (None, None)
            
            
        elif self.solver.startswith("pulp."):
            prob = self.solver_param["prob"]
            xvar = self.solver_param["xvar"]
            
            cexpr = None
            for i in range(len(c)):
                if abs(c[i]) > PsiOpts.settings["eps"]:
                    if cexpr is None:
                        cexpr = c[i] * xvar[str(i)]
                    else:
                        cexpr += c[i] * xvar[str(i)]
                        
            if cexpr is not None:
                prob.setObjective(cexpr)
                try:
                    res = prob.solve(iutil.pulp_get_solver(self.solver))
                except Exception as err:
                    warnings.warn(str(err), RuntimeWarning)
                    res = 0
                
                if pulp.LpStatus[res] == "Optimal":
                    
                    return (prob.objective.value(), [xvar[str(i)].value() for i in range(self.nxvar)])
                
            return (None, None)
            
        
        elif self.solver.startswith("pyomo."):
            coptions = PsiOpts.get_pyomo_options()
            opt = self.solver_param["opt"]
            model = self.solver_param["model"]
            
            def o_rule(model):
                cexpr = None
                for i in range(len(c)):
                    if abs(c[i]) > PsiOpts.settings["eps"]:
                        if cexpr is None:
                            cexpr = c[i] * model.x[i + 1]
                        else:
                            cexpr += c[i] * model.x[i + 1]
                return cexpr
            
            model.del_component("o")
            model.o = pyo.Objective(rule=o_rule)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = opt.solve(model, options = coptions)
                except Exception as err:
                    warnings.warn(str(err), RuntimeWarning)
                    res = None

            if (res is not None and res.solver.status == pyo.SolverStatus.ok 
                and res.solver.termination_condition == pyo.TerminationCondition.optimal):
                    
                return (model.o(), [model.x[i + 1]() for i in range(self.nxvar)])
            
            return (None, None)
        
        
        
        
    def checkexpr_ge0(self, x, saved = False, optval = None):
        verbose = PsiOpts.settings.get("verbose_lp", False)
        
        if self.pinfeas:
            return True
        
        if self.eps_present:
            x = x.substituted(Expr.eps(), Expr.eps() * (self.lp_eps_obj / self.lp_eps))
        
        zero_cutoff = self.zero_cutoff
        if saved:
            
            if False:
                optobj = SparseMat(self.nvar)
                optobj.addrow()
                self.addExpr(optobj, x)
                optobj.simplify_last_row()
                
                if len(optobj.x[0]) == 0:
                    return True
                
                if not optobj.mapcol(self.xvarid):
                    return False
                
                optobj.width = self.nxvar
            
            
            optobj, optobjc1 = self.get_vec(x, sparse = True)
            if optobj is None:
                return False
            
            #cvec = numpy.array(c)
            for i in range(len(self.saved_var)):
                a = self.saved_var[i]
                #if sum(x * y for x, y in zip(c, a)) < zero_cutoff:
                #if numpy.dot(a, cvec) < zero_cutoff:
                if sum(ca * a[j] for j, ca in optobj.x[0]) + optobjc1 < zero_cutoff:
                    for j in range(i, 0, -1):
                        self.saved_var[j], self.saved_var[j - 1] = self.saved_var[j - 1], self.saved_var[j]
                    return False
            return True
        
        verbose_lp_cons = PsiOpts.settings.get("verbose_lp_cons", False)
        if verbose_lp_cons:
            print("============ LP constraints ============")
            print(self.get_region(skip_simplify = True))
            
            print("============  LP objective  ============")
            
            if False:
                optobj = SparseMat(self.nvar)
                optobj.addrow()
                self.addExpr(optobj, x)
                optobj.simplify_last_row()
                optobjc1 = 0.0
                if len(self.constmap):
                    optobjc1 = optobj.sumrows(self.constmap, remove = True)[0]
                if not optobj.mapcol(self.xvarid):
                    return False
                optobj.width = self.nxvar
            
            optobj, optobjc1 = self.get_vec(x, sparse = True)
            
            if optobj is None:
                print("objective contains new terms")
            else:
                optreg = self.get_region(A = optobj, skip_simplify = True)
                if len(optreg.exprs_ge) > 0:
                    print(optreg.exprs_ge[0] + optobjc1)
            print("========================================")
        
        if self.fcn_mode >= 1:
            if x.isnonpos_hc():
                fcn_res = True
                for (a, c) in x.terms:
                    if not self.checkfcn(a.x[0], a.z):
                        fcn_res = False
                        break
                #print("FCN " + str(x) + "  " + str(fcn_res))
                if fcn_res:
                    return True
                if self.fcn_mode >= 2:
                    return False
            
        if self.lptype == LinearProgType.HC1BN:
            if x.isnonpos_ic2():
                if self.bnet.check_ic(x):
                    if verbose:
                        print("LP True: bnet")
                    return True
                #return False
        res = None
        
        ceps = PsiOpts.settings["eps"]
        
        c, c1 = self.get_vec(x)
        if c is None:
            pass
        
        else:
            #if len([x for x in c if abs(x) > PsiOpts.settings["eps"]]) == 0:
            if all(abs(x) <= ceps for x in c):
                if c1 >= -ceps:
                    if verbose:
                        print("LP True: zero")
                    return True
                else:
                    c = None
            
        if c is None:
            c = [-ceps * 2] + [0.0] * (self.nxvar - 1)
            c1 = -1.0
            #return False
            
        if len(self.Au.x) == 0 and len(self.Ae.x) == 0:
            if verbose:
                print("LP False: no constraints")
            return False
            
        
        
        
        if verbose:
            print("LP nrv=" + str(self.index.num_rv()) + " nreal=" + str(self.index.num_real())
            + " nvar=" + str(self.Au.width) + "/" + str(self.nvar) + " nineq=" + str(len(self.Au.x))
            + " neq=" + str(len(self.Ae.x)) + " solver=" + self.solver)
        
        if self.solver == "scipy":
            rec_limit = 50
            if self.Au.width > rec_limit:
                warnings.warn("The scipy solver is not recommended for problems of size > " 
                              + str(rec_limit) + " (current " + str(self.Au.width) + "). "
                              + "Please switch to another solver. See "
                              + "https://github.com/cheuktingli/psitip#solver", RuntimeWarning)
            
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = scipy.optimize.linprog(c, self.solver_param["Aus"], self.bu, self.solver_param["Aes"], self.be, 
                                         bounds = (None, None), method = "interior-point", options={'sparse': True})
                    
                if verbose:
                    print("  status=" + str(res.status) + " optval=" + str(res.fun))
                    
            if self.affine_present and self.lp_bounded and res.status == 2:
                return True
            
            if res.status == 0:
                self.optval = float(res.fun) + c1
                if optval is not None:
                    optval.append(self.optval)
                    
                if self.val_enabled:
                    self.val_x = [0.0] * self.nxvar
                    for i in range(self.nxvar):
                        self.val_x[i] = float(res.x[i])
                        
                if self.optval >= zero_cutoff:
                    return True
            
            if res.status == 0 and self.save_res:
                self.saved_var.append(array.array("d", [float(a) for a in res.x]))
                #self.saved_var.append(numpy.array(list(res.x)))
                if verbose:
                    print("  added : " + str(len(self.saved_var)) + ", " + str(sum(self.saved_var[-1])))
            
            return False
        
        elif self.solver.startswith("pulp."):
            prob = self.solver_param["prob"]
            xvar = self.solver_param["xvar"]
            
            cexpr = None
            for i in range(len(c)):
                if abs(c[i]) > PsiOpts.settings["eps"]:
                    if cexpr is None:
                        cexpr = c[i] * xvar[str(i)]
                    else:
                        cexpr += c[i] * xvar[str(i)]
                        
            if cexpr is not None:
                prob.setObjective(cexpr)
                try:
                    res = prob.solve(iutil.pulp_get_solver(self.solver))
                except Exception as err:
                    warnings.warn(str(err), RuntimeWarning)
                    res = 0
                    
                if verbose:
                    print("  status=" + pulp.LpStatus[res] + " optval=" + str(prob.objective.value()))
                
                if self.affine_present and self.lp_bounded and (pulp.LpStatus[res] == "Infeasible" or pulp.LpStatus[res] == "Undefined"):
                    return True
                #if pulp.LpStatus[res] == "Infeasible":
                #    return True
                if pulp.LpStatus[res] == "Optimal":
                    
                    self.optval = prob.objective.value() + c1
                    
                    if optval is not None:
                        optval.append(self.optval)
                        
                    if self.dual_enabled:
                        self.dual_u = [0.0] * len(self.Au.x)
                        self.dual_e = [0.0] * len(self.Ae.x)
                        
                        for i, (name, c) in enumerate(prob.constraints.items()):
                            #print(str(i) + "  " + name + "  " + str(c.pi))
                            if c.pi is None:
                                self.dual_u = None
                                self.dual_e = None
                                break
                            
                            if i < len(self.Au.x):
                                self.dual_u[i] = c.pi
                            else:
                                self.dual_e[i - len(self.Au.x)] = c.pi
                                    
                        if self.dual_pf:
                            self.write_pf(x)
                    
                    if self.val_enabled:
                        self.val_x = [0.0] * self.nxvar
                        for i in range(self.nxvar):
                            self.val_x[i] = xvar[str(i)].value()
                                
                        
                    if self.optval >= zero_cutoff:
                        return True
            
                if pulp.LpStatus[res] == "Optimal" and self.save_res:
                    self.saved_var.append(array.array("d", [xvar[str(i)].value() for i in range(len(c))]))
                    #self.saved_var.append(numpy.array([xvar[str(i)].value() for i in range(len(c))]))
                    if verbose:
                        print("  added : " + str(len(self.saved_var)) + ", " + str(sum(self.saved_var[-1])))
                    
                return False
            else:
                return True
            
        
        elif self.solver.startswith("pyomo."):
            coptions = PsiOpts.get_pyomo_options()
            opt = self.solver_param["opt"]
            model = self.solver_param["model"]
            
            def o_rule(model):
                cexpr = None
                for i in range(len(c)):
                    if abs(c[i]) > PsiOpts.settings["eps"]:
                        if cexpr is None:
                            cexpr = c[i] * model.x[i + 1]
                        else:
                            cexpr += c[i] * model.x[i + 1]
                return cexpr
            
            model.del_component("o")
            model.o = pyo.Objective(rule=o_rule)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = opt.solve(model, options = coptions)
                except Exception as err:
                    warnings.warn(str(err), RuntimeWarning)
                    res = None

            if verbose and res is not None:
                print("  status=" + ("OK" if res.solver.status == pyo.SolverStatus.ok else "NO"))
            
            if res is not None and self.affine_present and self.lp_bounded and res.solver.termination_condition == pyo.TerminationCondition.infeasible:
                return True
            
            #print("save_res = " + str(self.save_res))
            if (res is not None and res.solver.status == pyo.SolverStatus.ok 
                and res.solver.termination_condition == pyo.TerminationCondition.optimal):
                    
                self.optval = model.o() + c1
                
                if optval is not None:
                    optval.append(self.optval)
                    
                if self.dual_enabled:
                    self.dual_u = [0.0] * len(self.Au.x)
                    self.dual_e = [0.0] * len(self.Ae.x)
                
                    for c in model.component_objects(pyo.Constraint, active=True):
                        for i, index in enumerate(c):
                            if i < len(self.Au.x):
                                self.dual_u[i] = model.dual[c[index]]
                            else:
                                self.dual_e[i - len(self.Au.x)] = model.dual[c[index]]
                                
                    if self.dual_pf:
                        self.write_pf(x)
                
                if self.val_enabled:
                    self.val_x = [0.0] * self.nxvar
                    for i in range(self.nxvar):
                        self.val_x[i] = model.x[i + 1]()
                        
                if self.optval >= zero_cutoff:
                    #print("RETURN TRUE")
                    return True
                
                if self.save_res:
                    self.saved_var.append(array.array("d", [model.x[i + 1]() for i in range(len(c))]))
                    #self.saved_var.append(numpy.array([model.x[i + 1]() for i in range(len(c))]))
                    if verbose:
                        print("  added : " + str(len(self.saved_var)) + ", " + str(sum(self.saved_var[-1])))
                    
            return False
        
        
    
    def checkexpr_eq0(self, x, saved = False):
        return self.checkexpr_ge0(x, saved) and self.checkexpr_ge0(-x, saved)
    
    
    def checkexpr(self, x, sg, saved = False):
        if sg == "==":
            return self.checkexpr_eq0(x, saved)
        elif sg == ">=":
            return self.checkexpr_ge0(x, saved)
        elif sg == "<=":
            return self.checkexpr_ge0(-x, saved)
        else:
            return False
    
    def evalexpr_ge0_saved(self, x):
        c, c1 = self.get_vec(x)
        if c is None:
            return -1e8
        
        r = 0.0
        for a in self.saved_var:
            r += min(sum([x * y for x, y in zip(c, a)]) + c1, 0.0)
        
        return r
        
    def get_val(self, expr):
        if self.val_x is None:
            return None
        c, c1 = self.get_vec(expr, sparse = True)
        r = c1
        if len(c.x) == 0:
            return r
        for j, a in c.x[0]:
            r += self.val_x[j] * a
        return r
    
    def __call__(self, x):
        if isinstance(x, Expr):
            return self.get_val(x)
        elif isinstance(x, Region):
            return x.evalcheck(lambda expr: self.get_val(expr))
        return None
    
    def __getitem__(self, x):
        return self(x)
    
    def get_dual_expr(self, x):
        if self.dual_u is None or self.dual_e is None:
            return None
        
        c, c1 = self.get_vec(x, sparse = True)
        c.simplify()
        #print(c)
        for a, d in itertools.chain(zip(self.Ae.x, self.dual_e), zip(self.Au.x, self.dual_u)):
            #print(a)
            a2 = SparseMat.from_row(a, self.Au.width)
            a2.simplify()
            t = a2.ratio(c)
            if t is not None:
                return d * t
        return None
        
    def get_dual(self, reg):
        if isinstance(reg, Expr):
            return self.get_dual_expr(reg)
        
        r = []
        for x in reg.exprs_ge:
            r.append(self.get_dual_expr(x))
        for x in reg.exprs_eq:
            r.append(self.get_dual_expr(x))
            
        if len(r) == 1:
            return r[0]
        else:
            return r
    
    
    def proj_hull(prog, n, init_pt = None, toexpr = None, iscone = False, isfrac = None):
        """Convex hull method for polyhedron projection.
        C. Lassez and J.-L. Lassez, Quantifier elimination for conjunctions of linear constraints via a
        convex hull algorithm, IBM Research Report, T.J. Watson Research Center, RC 16779 (1991)
        """
        
        if cdd is None:
            raise ImportError("Convex hull method requires pycddlib. Please install it first.")
        
        verbose = PsiOpts.settings.get("verbose_discover", False)
        verbose_outer = PsiOpts.settings.get("verbose_discover_outer", False)
        verbose_detail = PsiOpts.settings.get("verbose_discover_detail", False)
        verbose_terms = PsiOpts.settings.get("verbose_discover_terms", False)
        verbose_terms_inner = PsiOpts.settings.get("verbose_discover_terms_inner", False)
        verbose_terms_outer = PsiOpts.settings.get("verbose_discover_terms_outer", False)
        #max_denom = PsiOpts.settings.get("max_denom", 1000)
        
        if isfrac is None:
            isfrac = PsiOpts.settings.get("discover_hull_frac_enabled", False)
        frac_denom = PsiOpts.settings.get("discover_hull_frac_denom", -1)
        
        ceps = PsiOpts.settings["eps_lp"]
        
        if init_pt is None:
            if iscone:
                init_pt = [0] * n
            else:
                _, init_pt = prog([1] * n)
                if init_pt is None:
                    init_pt = [0] * n
            
        mat = cdd.Matrix([[1] + init_pt], number_type=("fraction" if isfrac else "float"))
        mat.rep_type = cdd.RepType.GENERATOR
        
        ineqs_tight = []
        did = True
        while did:
            
            if PsiOpts.is_timer_ended():
                break
            
            #print("MROWSIZE " + str(mat.row_size))
            did = False
            poly = cdd.Polyhedron(mat)
            # print("MAT:")
            # print(mat)
            ineqs = poly.get_inequalities()
            # print("INEQ:")
            # print(ineqs)
            lset = ineqs.lin_set
            
            if verbose_terms or verbose_terms_inner:
                print("INNER:")
                for i in range(ineqs.row_size):
                    y = ineqs[i]
                    print("  " + str(toexpr(y[1:n+1])) + (" == " if i in lset else " >= ") + iutil.float_tostr(-y[0]))
                    
            for i in range(ineqs.row_size):
                    
                if PsiOpts.is_timer_ended():
                    break
                
                for sgn in ([1, -1] if i in lset else [1]):
                    
                    if PsiOpts.is_timer_ended():
                        break
                    
                    #print("IROWSIZE " + str(ineqs.row_size))
                    x = [a * sgn for a in ineqs[i]]
                    xnorm = sum(abs(a) for a in x)
                    if xnorm <= ceps:
                        continue
                    x = [a / xnorm for a in x]
                    for y in ineqs_tight:
                        if sum(abs(a - b) for a, b in zip(x, y)) <= ceps:
                            break
                    else:
                        if verbose_detail:
                            print("MIN " + str(toexpr(x[1:n+1])))
                        # print("PROG " + str(x))
                        opt, v = prog(x[1:n+1])
                        # print("  VS " + str(opt) + "  " + str(-x[0]))
                        if opt is None or opt >= -x[0] - ceps:
                            ineqs_tight.append(list(x))
                            if verbose:
                                if opt is None:
                                    print("NONE")
                                if verbose_outer or abs(x[0]) <= 100:
                                    print("ADD " + str(toexpr(x[1:n+1])) + " >= " + iutil.float_tostr(-x[0]))
                                    if verbose_terms or verbose_terms_outer:
                                        print("OUTER:")
                                        for y in ineqs_tight:
                                            print("  " + str(toexpr(y[1:n+1])) + " >= " + iutil.float_tostr(-y[0]))
                            #print("TIGHT " + str(list(x)))
                            continue
                        
                        if isfrac:
                            if frac_denom > 0:
                                v = [fractions.Fraction(a).limit_denominator(frac_denom) for a in v]
                            else:
                                v = [fractions.Fraction(a) for a in v]
                                
                        if iscone:
                            vnorm = sum(abs(a) for a in v)
                            if vnorm > ceps:
                                v = [a / vnorm for a in v]
                            v = [0] + v
                        else:
                            v = [1] + v
                        for i2 in range(mat.row_size):
                            y = mat[i2]
                            if sum(abs(a - b) for a, b in zip(v, y)) <= ceps:
                                break
                        else:
                            if verbose_detail:
                                print("PT " + str(v))
                            mat.extend([v])
                            did = True
        
        return ineqs_tight
    
    
    def discover_hull(self, A, iscone = False):
        """Convex hull method for polyhedron projection.
        C. Lassez and J.-L. Lassez, Quantifier elimination for conjunctions of linear constraints via a
        convex hull algorithm, IBM Research Report, T.J. Watson Research Center, RC 16779 (1991)
        """
        
        verbose = PsiOpts.settings.get("verbose_discover", False)
        verbose_detail = PsiOpts.settings.get("verbose_discover_detail", False)
        verbose_terms = PsiOpts.settings.get("verbose_discover_terms", False)
        
            
        n = self.nxvar
        m = len(A.x)
        
        toexpr = None
        if verbose or verbose_detail or verbose_terms:
            itoexpr = self.row_toexpr()
            def ctoexpr(x):
                c = [0.0] * n
                for i in range(m):
                    for j, a in A.x[i]:
                        c[j] += a * x[i]
                return itoexpr(c)
            toexpr = ctoexpr
        
        #print(A.x)
        #print(self.Au.x)
        
        def cprog(x):
            c = [0.0] * n
            for i in range(m):
                for j, a in A.x[i]:
                    c[j] += a * x[i]
            
            opt, v = self.call_prog(c)
            if opt is None:
                return (None, None)
            r = [0.0] * m
            for i in range(m):
                for j, a in A.x[i]:
                    r[i] += a * v[j]
            return (opt, r)
        
        return LinearProg.proj_hull(cprog, m, toexpr = toexpr, iscone = iscone)
        
    
    def corners_value(self, ispolar = False, isfrac = None):
        if cdd is None:
            raise ImportError("Requires pycddlib. Please install it first.")
        
        
        if isfrac is None:
            isfrac = PsiOpts.settings.get("discover_hull_frac_enabled", False)
        frac_denom = PsiOpts.settings.get("discover_hull_frac_denom", -1)
        
        ceps = PsiOpts.settings["eps_lp"]
            
        n = self.nxvar
        # print(n)
        # print(self.Au.x)
        # print(self.Ae.x)
        
        mat = None
        
        if len(self.Au.x):
            ma = numpy.hstack([numpy.array([self.bu]).T, -self.Au.tonumpyarray()])
            if mat is None:
                mat = cdd.Matrix(ma, number_type=("fraction" if isfrac else "float"))
                mat.rep_type = cdd.RepType.INEQUALITY
            else:
                mat.extend(ma)
        
        if len(self.Ae.x):
            ma = numpy.hstack([numpy.array([self.be]).T, -self.Ae.tonumpyarray()])
            if mat is None:
                mat = cdd.Matrix(ma, linear = True, number_type=("fraction" if isfrac else "float"))
                mat.rep_type = cdd.RepType.INEQUALITY
            else:
                mat.extend(ma, linear = True)
        
        if mat is None:
            return None
        
        # print(mat)
        poly = cdd.Polyhedron(mat)
        
        gs = poly.get_generators()
        fs = poly.get_inequalities()
        # print(gs)
        gi = poly.get_incidence()
        fi = poly.get_input_incidence()
        #print(fs)
        
        if ispolar:
            gs, fs = fs, gs
            gi, fi = fi, gi
            
        
        ng = gs.row_size
        angles = [0.0] * ng
        
        if n >= 2:
            for i in range(ng):
                avg = None
                if ispolar:
                    avg = gs[i][1:]
                else:
                    avg = [0.0] * n
                    for k in gi[i]:
                        avg = [fs[k][j + 1] + avg[j] for j in range(n)]
                angles[i] = math.atan2(avg[1], avg[0])
                if len(gi[i]) == 0:
                    angles[i] = 1e20
        
        gsj = sorted([i for i in range(ng) if len(gi[i])], key = lambda k: angles[k])
        gsjinv = [0] * ng
        for i in range(len(gsj)):
            gsjinv[gsj[i]] = i
        
        gsr = [list(gs[gsj[i]]) for i in range(len(gsj))]
        #fir = [[gsjinv[a] for a in x] for x in fi if len(x) > 0 and len(x) < ng]
        fir = [[gsjinv[a] for a in x] for x in fi if len(x) > 0]
        
        return (gsr, fir)
    
    
    def table(self, *args, **kwargs):
        """Plot the information diagram as a Karnaugh map.
        """
        return universe().table(*args, self, **kwargs)
        
    def venn(self, *args, **kwargs):
        """Plot the information diagram as a Venn diagram.
        Can handle up to 5 random variables (uses Branko Grunbaum's Venn diagram for n=5).
        """
        return universe().venn(*args, self, **kwargs)
        
        
        
    
class RegionType:
    NIL = 0
    NORMAL = 1
    UNION = 2
    INTER = 3
    
class Region(IBaseObj):
    """A region consisting of equality and inequality constraints"""
    
    def __init__(self, exprs_ge, exprs_eq, aux, inp, oup, exprs_gei = None, exprs_eqi = None, auxi = None):
        self.exprs_ge = exprs_ge
        self.exprs_eq = exprs_eq
        self.aux = aux
        self.inp = inp
        self.oup = oup
        
        if exprs_gei is not None:
            self.exprs_gei = exprs_gei
        else:
            self.exprs_gei = []
            
        if exprs_eqi is not None:
            self.exprs_eqi = exprs_eqi
        else:
            self.exprs_eqi = []
            
        if auxi is not None:
            self.auxi = auxi
        else:
            self.auxi = Comp.empty()
        
    def get_type(self):
        return RegionType.NORMAL
    
    def isnormalcons(self):
        return not self.imp_present()
    
    def universe():
        return Region([], [], Comp.empty(), Comp.empty(), Comp.empty())
    
    def Ic(x, y, z = None):
        if z is None:
            z = Comp.empty()
        x = x - z
        y = y - z
        if x.isempty() or y.isempty():
            return Region.universe()
        return Region([], [Expr.Ic(x, y, z)], Comp.empty(), Comp.empty(), Comp.empty())
        
    
    def empty():
        return Region([-Expr.one()], [], Comp.empty(), Comp.empty(), Comp.empty())
    
    def from_bool(b):
        if b:
            return Region.universe()
        else:
            return Region.empty()
    
    def setuniverse(self):
        self.exprs_ge = []
        self.exprs_eq = []
        self.aux = Comp.empty()
        self.inp = Comp.empty()
        self.oup = Comp.empty()
        self.exprs_gei = []
        self.exprs_eqi = []
        self.auxi = Comp.empty()
        
    def isempty(self):
        if not (len(self.exprs_gei) == 0 and len(self.exprs_eqi) == 0):
            return False
        
        ceps = PsiOpts.settings["eps"]
        for x in self.exprs_ge:
            t = x.get_const()
            if t is not None and t < -ceps:
                return True
        for x in self.exprs_eq:
            t = x.get_const()
            if t is not None and abs(t) > ceps:
                return True
            
        return False
        
    def isuniverse(self, sgn = True, canon = False):
        if canon and (not self.aux.isempty() or not self.auxi.isempty()):
            return False
        if sgn:
            return len(self.exprs_ge) == 0 and len(self.exprs_eq) == 0 and len(self.exprs_gei) == 0 and len(self.exprs_eqi) == 0
        else:
            return self.isempty()
    
    def copy(self):
        return Region([x.copy() for x in self.exprs_ge],
                      [x.copy() for x in self.exprs_eq], 
                        self.aux.copy(), self.inp.copy(), self.oup.copy(),
                        [x.copy() for x in self.exprs_gei],
                        [x.copy() for x in self.exprs_eqi],
                        self.auxi.copy())
        
    
    def copy_noaux(self):
        return Region([x.copy() for x in self.exprs_ge],
                      [x.copy() for x in self.exprs_eq], 
                        Comp.empty(), self.inp.copy(), self.oup.copy(),
                        [x.copy() for x in self.exprs_gei],
                        [x.copy() for x in self.exprs_eqi],
                        Comp.empty())
        
    
    def copy_(self, other):
        self.exprs_ge = [x.copy() for x in other.exprs_ge]
        self.exprs_eq = [x.copy() for x in other.exprs_eq]
        self.aux = other.aux.copy()
        self.inp = other.inp.copy()
        self.oup = other.oup.copy()
        self.exprs_gei = [x.copy() for x in other.exprs_gei]
        self.exprs_eqi = [x.copy() for x in other.exprs_eqi]
        self.auxi = other.auxi.copy()
        
    def imp_intersection(self):
        return Region([x.copy() for x in self.exprs_ge] + [x.copy() for x in self.exprs_gei],
                      [x.copy() for x in self.exprs_eq] + [x.copy() for x in self.exprs_eqi], 
                        self.aux.copy() + self.auxi.copy(), self.inp.copy(), self.oup.copy())
        
    def imp_intersection_noaux(self):
        return Region([x.copy() for x in self.exprs_ge] + [x.copy() for x in self.exprs_gei],
                      [x.copy() for x in self.exprs_eq] + [x.copy() for x in self.exprs_eqi], 
                        Comp.empty(), Comp.empty(), Comp.empty())
        
    def imp_copy(self):
        return Region([],
                      [], 
                        Comp.empty(), Comp.empty(), Comp.empty(),
                        [x.copy() for x in self.exprs_gei],
                        [x.copy() for x in self.exprs_eqi],
                        self.auxi.copy())
        
    def imp_flipped(self):
        return Region([x.copy() for x in self.exprs_gei],
                        [x.copy() for x in self.exprs_eqi],
                        self.auxi.copy(), self.inp.copy(), self.oup.copy(),
                        [x.copy() for x in self.exprs_ge],
                      [x.copy() for x in self.exprs_eq], 
                        self.aux.copy())
        
    def consonly(self):
        return Region([x.copy() for x in self.exprs_ge],
                        [x.copy() for x in self.exprs_eq],
                        self.aux.copy(), self.inp.copy(), self.oup.copy())
        
    def imp_flippedonly(self):
        return Region([x.copy() for x in self.exprs_gei],
                        [x.copy() for x in self.exprs_eqi],
                        self.auxi.copy(), Comp.empty(), Comp.empty())
        
    def imp_flippedonly_noaux(self):
        return Region([x.copy() for x in self.exprs_gei],
                        [x.copy() for x in self.exprs_eqi],
                        Comp.empty(), Comp.empty(), Comp.empty())
    
    def imp_present(self):
        return len(self.exprs_gei) > 0 or len(self.exprs_eqi) > 0 or not self.auxi.isempty()
    
    def imp_flip(self):
        self.exprs_ge, self.exprs_gei = self.exprs_gei, self.exprs_ge
        self.exprs_eq, self.exprs_eqi = self.exprs_eqi, self.exprs_eq
        self.aux, self.auxi = self.auxi, self.aux
        return self
    
        
    def imp_only_copy_to(self, other):
        other.exprs_ge = []
        other.exprs_eq = []
        other.aux = Comp.empty()
        other.inp = Comp.empty()
        other.oup = Comp.empty()
        other.exprs_gei = [x.copy() for x in self.exprs_gei]
        other.exprs_eqi = [x.copy() for x in self.exprs_eqi]
        other.auxi = self.auxi.copy()
        
        
    def __len__(self):
        return len(self.exprs_ge) + len(self.exprs_eq)
    
    def __getitem__(self, key):
        t = [(a, False) for a in self.exprs_ge] + [(a, True) for a in self.exprs_eq]
        r = t[key]
        if not isinstance(r, list):
            r = [r]
        
        c = Region.universe()
        for a, eq in r:
            if eq:
                c.exprs_eq.append(a)
            else:
                c.exprs_ge.append(a)
        return c
    
        
    def sum_entrywise(self, other):
        return Region([x + y for (x, y) in zip(self.exprs_ge, other.exprs_ge)],
                      [x + y for (x, y) in zip(self.exprs_eq, other.exprs_eq)], 
                        self.aux.interleaved(other.aux), self.inp.interleaved(other.inp), self.oup.interleaved(other.oup),
                        [x + y for (x, y) in zip(self.exprs_gei, other.exprs_gei)],
                        [x + y for (x, y) in zip(self.exprs_eqi, other.exprs_eqi)],
                        self.auxi.interleaved(other.auxi))

    def ispresent(self, x):
        """Return whether any variable in x appears here."""
        for z in self.exprs_ge:
            if z.ispresent(x):
                return True
        for z in self.exprs_eq:
            if z.ispresent(x):
                return True
        if self.aux.ispresent(x):
            return True
        if self.inp.ispresent(x):
            return True
        if self.oup.ispresent(x):
            return True
        for z in self.exprs_gei:
            if z.ispresent(x):
                return True
        for z in self.exprs_eqi:
            if z.ispresent(x):
                return True
        if self.auxi.ispresent(x):
            return True
        return False
    
    def affine_present(self):
        """Return whether there are any affine constraint."""
        return self.ispresent((Expr.one() + Expr.eps() + Expr.inf()).allcomp())

    def imp_ispresent(self, x):
        for z in self.exprs_gei:
            if z.ispresent(x):
                return True
        for z in self.exprs_eqi:
            if z.ispresent(x):
                return True
        if self.auxi.ispresent(x):
            return True
        return False
        
    def rename_var(self, name0, name1):
        for x in self.exprs_ge:
            x.rename_var(name0, name1)
        for x in self.exprs_eq:
            x.rename_var(name0, name1)
        self.aux.rename_var(name0, name1)
        self.inp.rename_var(name0, name1)
        self.oup.rename_var(name0, name1)
        for x in self.exprs_gei:
            x.rename_var(name0, name1)
        for x in self.exprs_eqi:
            x.rename_var(name0, name1)
        self.auxi.rename_var(name0, name1)

        
    def rename_map(self, namemap):
        """Rename according to name map.
        """
        for x in self.exprs_ge:
            x.rename_map(namemap)
        for x in self.exprs_eq:
            x.rename_map(namemap)
        self.aux.rename_map(namemap)
        self.inp.rename_map(namemap)
        self.oup.rename_map(namemap)
        for x in self.exprs_gei:
            x.rename_map(namemap)
        for x in self.exprs_eqi:
            x.rename_map(namemap)
        self.auxi.rename_map(namemap)
        return self
    
    @fcn_substitute
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), in place"""
        for x in self.exprs_ge:
            x.substitute(v0, v1)
        for x in self.exprs_eq:
            x.substitute(v0, v1)
        for x in self.exprs_gei:
            x.substitute(v0, v1)
        for x in self.exprs_eqi:
            x.substitute(v0, v1)
            
        if not isinstance(v0, Expr):
            self.aux.substitute(v0, v1)
            self.inp.substitute(v0, v1)
            self.oup.substitute(v0, v1)
            self.auxi.substitute(v0, v1)
        return self
        
    def substituted(self, *args, **kwargs):
        """Substitute variable v0 by v1 (v1 can be compound), return result"""
        r = self.copy()
        r.substitute(*args, **kwargs)
        return r

    @fcn_substitute
    def substitute_aux(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), and remove auxiliary v0, in place"""
        for x in self.exprs_ge:
            x.substitute(v0, v1)
        for x in self.exprs_eq:
            x.substitute(v0, v1)
        for x in self.exprs_gei:
            x.substitute(v0, v1)
        for x in self.exprs_eqi:
            x.substitute(v0, v1)
            
        if not isinstance(v0, Expr):
            self.aux -= v0
            self.inp -= v0
            self.oup -= v0
            self.auxi -= v0
        return self
        
    def substituted_aux(self, *args, **kwargs):
        """Substitute variable v0 by v1 (v1 can be compound), and remove auxiliary v0, return result"""
        r = self.copy()
        r.substitute_aux(*args, **kwargs)
        return r

    def remove_present(self, v):
        self.exprs_ge = [x for x in self.exprs_ge if not x.ispresent(v)]
        self.exprs_eq = [x for x in self.exprs_eq if not x.ispresent(v)]
        self.exprs_gei = [x for x in self.exprs_gei if not x.ispresent(v)]
        self.exprs_eqi = [x for x in self.exprs_eqi if not x.ispresent(v)]
        
        if isinstance(v, Comp):
            self.aux -= v
            self.inp -= v
            self.oup -= v
            self.auxi -= v

    def remove_notpresent(self, v):
        self.exprs_ge = [x for x in self.exprs_ge if x.ispresent(v)]
        self.exprs_eq = [x for x in self.exprs_eq if x.ispresent(v)]
        self.exprs_gei = [x for x in self.exprs_gei if x.ispresent(v)]
        self.exprs_eqi = [x for x in self.exprs_eqi if x.ispresent(v)]
        
        if isinstance(v, Comp):
            self.aux = self.aux.inter(v)
            self.inp = self.inp.inter(v)
            self.oup = self.oup.inter(v)
            self.auxi = self.auxi.inter(v)
        
    
    def remove_notcontained(self, v):
        t = self.allcomp() - v
        if not t.isempty():
            self.remove_present(t)
        
    def condition(self, b):
        """Condition on random variable b, in place"""
        for x in self.exprs_ge:
            x.condition(b)
        for x in self.exprs_eq:
            x.condition(b)
        for x in self.exprs_gei:
            x.condition(b)
        for x in self.exprs_eqi:
            x.condition(b)
        return self

    def conditioned(self, b):
        """Condition on random variable b, return result"""
        r = self.copy()
        r.condition(b)
        return r
        
        
    def symm_sort(self, terms):
        """Sort the random variables in terms assuming symmetry among those terms."""
        
        for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi:
            x.symm_sort(terms)
            
    
    def find_name(self, *args):
        r = self.allcomp().find_name(*args)
        
    def placeholder(*args):
        r = Expr.zero()
        for a in args:
            if isinstance(a, Comp):
                r += Expr.H(a) * 0
            elif isinstance(a, Expr):
                r += a * 0
        return r >= 0
        
    def empty_placeholder(*args):
        r = Region.empty()
        r.iand_norename(Region.placeholder(*args))
        return r
        
    def record_to(self, index):
        for x in self.exprs_ge:
            x.record_to(index)
        for x in self.exprs_eq:
            x.record_to(index)
        index.record(self.aux)
        index.record(self.inp)
        index.record(self.oup)
        for x in self.exprs_gei:
            x.record_to(index)
        for x in self.exprs_eqi:
            x.record_to(index)
        index.record(self.auxi)
        
        
    def name_avoid(self, name0, regs = None):
        index = IVarIndex()
        self.record_to(index)
        if regs is not None:
            for r in regs:
                r.record_to(index)
        name1 = name0
        while index.get_index_name(name1) >= 0:
            name1 += PsiOpts.settings["rename_char"]
        return name1
        
    def rename_avoid(self, reg, name0):
        index = IVarIndex()
        reg.record_to(index)
        sindex = IVarIndex()
        self.record_to(sindex)
        
        name1 = name0
        while (index.get_index_name(name1) >= 0
        or (name1 != name0 and sindex.get_index_name(name1) >= 0)):
            name1 += PsiOpts.settings["rename_char"]
            
        if name1 != name0:
            self.rename_var(name0, name1)
    
    def aux_addprefix(self, pref = "@@"):
        for i in range(len(self.aux.varlist)):
            self.rename_var(self.aux.varlist[i].name,
                            pref + self.aux.varlist[i].name)
    
    def aux_present(self):
        return not self.getaux().isempty() or not self.getauxi().isempty()
    
    def aux_clear(self):
        self.aux = Comp.empty()
        self.auxi = Comp.empty()
        
    def getaux(self):
        return self.aux.copy()
    
    def getauxi(self):
        return self.auxi.copy()
    
    def getauxall(self):
        return self.aux + self.auxi
    
    def getauxs(self):
        r = []
        if not self.aux.isempty():
            r.append((self.aux.copy(), True))
        if not self.auxi.isempty():
            r.append((self.auxi.copy(), False))
        return r
    
    def aux_avoid(self, reg, samesuffix = True):
        if samesuffix:
            self.aux_avoid_from(reg.allcomprv_noaux(), samesuffix = True)
            reg.aux_avoid_from(self.allcomprv(), samesuffix = True)
        else:
            for a in reg.getauxi().varlist:
                reg.rename_avoid(self, a.name)
            for a in reg.getaux().varlist:
                reg.rename_avoid(self, a.name)
            for a in self.getauxi().varlist:
                self.rename_avoid(reg, a.name)
            for a in self.getaux().varlist:
                self.rename_avoid(reg, a.name)
    
    def aux_avoid_from(self, reg, samesuffix = True):
        if samesuffix:
            if isinstance(reg, Region):
                reg = reg.allcomprv()
            reg = reg + self.allcomprv_noaux()
            auxcomp = self.getauxall()
            if not reg.ispresent(auxcomp):
                return
            
            rename_char = PsiOpts.settings["rename_char"]
            for rep in ["set", "add", "suffix"]:
                for k in range(1, 20 if rep else 1000):
                    rdict = {}
                    rset = set()
                    bad = False
                    for a in auxcomp:
                        t = iutil.set_suffix_num(a.get_name(), k, rename_char, replace_mode = rep)
                        
                        if t in rset:
                            bad = True
                            break
                        rdict[a.get_name()] = t
                        rset.add(t)
                    if not bad and not reg.ispresent(sum(Comp.rv(a) for a in rset)):
                        self.rename_map(rdict)
                        return
                
        else:
            for a in self.getaux().varlist:
                self.rename_avoid(reg, a.name)
            for a in self.getauxi().varlist:
                self.rename_avoid(reg, a.name)
        
    def iand_norename(self, other):
        co = other
        self.exprs_ge += [x.copy() for x in co.exprs_ge]
        self.exprs_eq += [x.copy() for x in co.exprs_eq]
        self.exprs_gei += [x.copy() for x in co.exprs_gei]
        self.exprs_eqi += [x.copy() for x in co.exprs_eqi]
        self.aux += co.aux
        self.auxi += co.auxi
        
        return self
        
    def __iand__(self, other):
        if isinstance(other, bool):
            if not other:
                return Region.empty()
            return self
        
        if (isinstance(other, RegionOp) or self.imp_present() or other.imp_present() 
            or (not PsiOpts.settings["prefer_expand"] and (self.aux_present() or other.aux_present()))):
            return RegionOp.inter([self]) & other
            
        if not self.aux_present() and not other.aux_present():
            self.exprs_ge += [x.copy() for x in other.exprs_ge]
            self.exprs_eq += [x.copy() for x in other.exprs_eq]
            self.exprs_gei += [x.copy() for x in other.exprs_gei]
            self.exprs_eqi += [x.copy() for x in other.exprs_eqi]
            return self
        
        co = other.copy()
        self.aux_avoid(co)
            
        self.exprs_ge += [x.copy() for x in co.exprs_ge]
        self.exprs_eq += [x.copy() for x in co.exprs_eq]
        self.exprs_gei += [x.copy() for x in co.exprs_gei]
        self.exprs_eqi += [x.copy() for x in co.exprs_eqi]
        self.aux += co.aux
        self.auxi += co.auxi
        
        return self
        
        
    def __and__(self, other):
        r = self.copy()
        r &= other
        return r
        
    def __rand__(self, other):
        r = self.copy()
        r &= other
        return r
    
        
    def __pow__(self, other):
        if other <= 0:
            return Region.universe()
        r = self.copy()
        for i in range(other - 1):
            r &= self
        return r
        
    def __or__(self, other):
        if self.isempty():
            return other.copy()
        return RegionOp.union([self]) | other
        
    def __ror__(self, other):
        if self.isempty():
            return other.copy()
        return RegionOp.union([self]) | other
        
    def __ior__(self, other):
        if self.isempty():
            return other.copy()
        return RegionOp.union([self]) | other
    
    def implicate(self, other, skip_simplify = False):
        co = other.copy()
        if not skip_simplify and PsiOpts.settings["imp_simplify"]:
            if co.imp_present():
                co.simplify()
        self.aux_avoid(co)
        
        self.exprs_ge += [x.copy() for x in co.exprs_gei]
        self.exprs_eq += [x.copy() for x in co.exprs_eqi]
        self.exprs_gei += [x.copy() for x in co.exprs_ge]
        self.exprs_eqi += [x.copy() for x in co.exprs_eq]
        
        #self.aux += co.auxi
        #self.auxi = co.aux + self.auxi
        
        self.aux = co.auxi + self.aux
        self.auxi += co.aux
        
        return self
    
    def implicated(self, other, skip_simplify = False):
        if isinstance(other, RegionOp) or self.imp_present() or other.imp_present() or not other.aux.isempty():
            return RegionOp.union([self]).implicated(other, skip_simplify)
        
        r = self.copy()
        r.implicate(other, skip_simplify)
        return r
    
    def __le__(self, other):
        return other.implicated(self)
        
    def __ge__(self, other):
        return self.implicated(other)
        
    
    def __rshift__(self, other):
        return other.implicated(self)
    
    def __rrshift__(self, other):
        return self.implicated(other)
        
    def __lshift__(self, other):
        return self.implicated(other)
    
    def __eq__(self, other):
        #return self.implies(other) and other.implies(self)
        return RegionOp.inter([self.implicated(other), other.implicated(self)])
        
    def __ne__(self, other):
        return ~(RegionOp.inter([self.implicated(other), other.implicated(self)]))
        
    
    def relax_term(self, term, gap):
        self.simplify_quick()
                
        for x in self.exprs_eq:
            c = x.get_coeff(term)
            if c != 0.0:
                self.exprs_ge.append(x.copy())
                self.exprs_ge.append(-x)
                x.setzero()
        
        for x in self.exprs_ge:
            c = x.get_coeff(term)
            if c > 0.0:
                x.substitute(Expr.fromterm(term), Expr.fromterm(term) + gap)
            elif c < 0.0:
                x.substitute(Expr.fromterm(term), Expr.fromterm(term) - gap)
                
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
    
    def relax(self, w, gap):
        """Relax real variables in w by gap, in place"""
        for (a, c) in w.terms:
            if a.get_type() == TermType.REAL:
                self.relax_term(a, gap)
        return self
    
    def relaxed(self, w, gap):
        """Relax real variables in w by gap, return result"""
        r = self.copy()
        r.relax(w, gap)
        return r
    
    def one_flipped(self):
        if len(self.exprs_eq) > 0 or len(self.exprs_ge) != 1:
            return None
        return self.exprs_ge[0] <= 0
    
    def broken_present(self, w, flipped = True):
        """Convert region to intersection of individual constraints if they contain w"""
        if self.imp_present():
            return RegionOp.from_region(self).broken_present(w, flipped)
        
        r = RegionOp.inter([])
        cs = self.copy()
        for x in cs.exprs_eq:
            if x.ispresent(w):
                r.regs.append((x == 0, True))
                x.setzero()
        for x in cs.exprs_ge:
            if x.ispresent(w):
                if flipped:
                    r.regs.append((~(x <= 0), True))
                else:
                    r.regs.append((x >= 0, True))
                x.setzero()
        cs.exprs_eq = [x for x in cs.exprs_eq if not x.iszero()]
        cs.exprs_ge = [x for x in cs.exprs_ge if not x.iszero()]
        cs.aux = Comp.empty()
        cs.auxi = Comp.empty()
        r.regs.append((cs, True))
        
        return r.exists(self.aux).forall(self.auxi)
        
    
    def corners_optimum(self, w, sn):
        """Return union of regions corresponding to maximum/minimum of the real variable w"""
        
        for x in self.exprs_eq:
            if x.get_coeff(w.terms[0][0]) != 0:
                return self.copy()
        
        r = []
        
        for i in range(len(self.exprs_ge)):
            x = self.exprs_ge[i]
            if x.get_coeff(w.terms[0][0]) * sn < 0:
                cs2 = self.copy()
                cs2.exprs_ge.pop(i)
                cs2.exprs_eq.append(x.copy())
                cs2.aux = Comp.empty()
                cs2.auxi = Comp.empty()
                r.append(cs2)
        
        if len(r) == 0:
            return Region.universe()
        if len(r) == 1:
            return r[0].exists(self.aux).forall(self.auxi)
        return RegionOp.union(r).exists(self.aux).forall(self.auxi)
            
    
    def corners_optimum_eq(self, w, sn):
        """Return union of regions corresponding to maximum/minimum of the real variable w"""
        
        for x in self.exprs_eq:
            if x.get_coeff(w.terms[0][0]) != 0:
                return self.copy()
        
        r = []
        cs = self.copy()
        cs.remove_present(w.terms[0][0].x[0])
        
        for i in range(len(self.exprs_ge)):
            x = self.exprs_ge[i]
            if x.get_coeff(w.terms[0][0]) * sn < 0:
                cs2 = cs.copy()
                cs2.exprs_eqi.append(x.copy())
                r.append(cs2)
        
        if len(r) == 0:
            return Region.universe()
        if len(r) == 1:
            return r[0]
        return RegionOp.inter(r)
            
        
    
    def corners(self, w):
        """Return union of regions corresponding to corner points of the real variables in w"""
        terms = []
        if isinstance(w, Expr):
            for (a, c) in w.terms:
                if a.get_type() == TermType.REAL:
                    terms.append(a)
        else:
            for w2 in w:
                for (a, c) in w2.terms:
                    if a.get_type() == TermType.REAL:
                        terms.append(a)
        
        n = len(terms)
        cmat = []
        cmatall = []
        for x in self.exprs_eq:
            coeff = [x.get_coeff(term) for term in terms]
            cmat.append(coeff[:])
            cmatall.append(coeff[:])
        
        rank = numpy.linalg.matrix_rank(cmat)
        
        if rank >= n:
            return [self.copy()]
        
        cs = self.copy()
        ges = []
        gec = []
        for x in cs.exprs_ge:
            coeff = [x.get_coeff(term) for term in terms]
            cmatall.append(coeff[:])
            if len([x for x in coeff if abs(x) > PsiOpts.settings["eps"]]) > 0:
                ges.append(x.copy())
                gec.append(coeff)
                x.setzero()
        
        cs.exprs_ge = [x for x in cs.exprs_ge if not x.iszero()]
        
        rankall = numpy.linalg.matrix_rank(cmatall)
        
        r = []
        
        for comb in itertools.combinations(range(len(ges)), rankall - rank):
            mat = cmat[:]
            for i in comb:
                mat.append(gec[i])
            if numpy.linalg.matrix_rank(mat) >= rankall:
                cs2 = cs.copy()
                for i2 in range(len(ges)):
                    if i2 in comb:
                        cs2.exprs_eq.append(ges[i2].copy())
                    else:
                        cs2.exprs_ge.append(ges[i2].copy())
                r.append(cs2)
        
        return RegionOp.union(r)
        
    
    def corners_value(self, w, ispolar = False, skip_discover = False, inf_value = 1e6):
        """Return the vertices and the facet list of the polytope with coordinates in the list w."""
        
        if not skip_discover:
            t = real_array("#TMPVAR", 0, len(w))
            return self.discover([(a, b) for a, b in zip(t, w)]).corners_value(t, ispolar, True, inf_value)
        
        cindex = IVarIndex()
        self.record_to(cindex)
        for a in w:
            a.record_to(cindex)
        
        prog = self.imp_flipped().init_prog(index = cindex)
        
        g, f = prog.corners_value(ispolar = ispolar)
        
        A = SparseMat(0)
        Ab = []
        for a in w:
            c, c1 = prog.get_vec(a, sparse = True)
            A.extend(c)
            Ab.append(c1)
        
        r = []
        for b in g:
            v = [sum([b[j + 1] * c for j, c in row], 0.0) for row in A.x]
            if abs(b[0]) < 1e-11:
                maxv = max(abs(x) for x in v)
                v = [0.0] + [(x / maxv) * inf_value for x in v]
            else:
                v = [1.0 if b[0] > 0 else -1.0] + [x / abs(b[0]) + y for x, y in zip(v, Ab)]
            r.append(v)
        
        return (r, f)
    
    
    def sign_present(self, term):
        
        sn_present = [False] * 2
        
        for x in self.exprs_ge:
            c = x.get_coeff(term)
            if c > 0.0:
                sn_present[1] = True
            elif c < 0.0:
                sn_present[0] = True
                
        for x in self.exprs_eq:
            c = x.get_coeff(term)
            if c != 0.0:
                sn_present[0] = True
                sn_present[1] = True
                
        for x in self.exprs_gei:
            c = x.get_coeff(term)
            if c > 0.0:
                sn_present[0] = True
            elif c < 0.0:
                sn_present[1] = True
                
        for x in self.exprs_eqi:
            c = x.get_coeff(term)
            if c != 0.0:
                sn_present[0] = True
                sn_present[1] = True
        
        return sn_present
    
    
    def substitute_sign(self, v0, v1s):
        v0term = v0.terms[0][0]
        sn_present = [False] * 2
                
        for x in self.exprs_eq:
            if x.ispresent(v0):
                if v1s:
                    self.exprs_ge.append(x.copy())
                    self.exprs_ge.append(-x)
                    x.setzero()
                sn_present[0] = True
                sn_present[1] = True
                
        for x in self.exprs_eqi:
            if x.ispresent(v0):
                if v1s:
                    self.exprs_gei.append(x.copy())
                    self.exprs_gei.append(-x)
                    x.setzero()
                sn_present[0] = True
                sn_present[1] = True
        
        for x in self.exprs_ge:
            if x.ispresent(v0):
                c = x.get_coeff(v0term)
                if c > 0.0:
                    if v1s:
                        x.substitute(v0, v1s[1])
                    sn_present[1] = True
                else:
                    if v1s:
                        x.substitute(v0, v1s[0])
                    sn_present[0] = True
                
        for x in self.exprs_gei:
            if x.ispresent(v0):
                c = x.get_coeff(v0term)
                if c > 0.0:
                    if v1s:
                        x.substitute(v0, v1s[0])
                    sn_present[0] = True
                else:
                    if v1s:
                        x.substitute(v0, v1s[1])
                    sn_present[1] = True
             
        if v1s:
            self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
            self.exprs_eqi = [x for x in self.exprs_eqi if not x.iszero()]
        
        return sn_present
    
    def substitute_duplicate(self, v0, v1s):
        for l in [self.exprs_ge, self.exprs_eq, self.exprs_gei, self.exprs_eqi]:
            olen = len(l)
            for ix in range(olen):
                x = l[ix]
                if x.ispresent(v0):
                    for v1 in v1s:
                        l.append(x.substituted(v0, v1))
                    x.setzero()

        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        self.exprs_gei = [x for x in self.exprs_gei if not x.iszero()]
        self.exprs_eqi = [x for x in self.exprs_eqi if not x.iszero()]

    
    # def flattened_minmax(self, term, sgn, bds):
    #     sbds = self.get_lb_ub_eq(term)
    #     reg.eliminate_term(self)
        
    def lowest_present(self, v, sn):
        if not sn:
            return None
        if self.ispresent(v):
            return self
        return None
            
    def flatten_regterm(self, term):
        self.simplify_quick()
        sn = term.sn
        
        sn_present = [False] * 2
        
        for x in self.exprs_ge:
            c = x.get_coeff(term)
            if c * sn > 0.0:
                sn_present[1] = True
            elif c * sn < 0.0:
                sn_present[0] = True
                
        for x in self.exprs_eq:
            c = x.get_coeff(term)
            if c != 0.0:
                sn_present[0] = True
                sn_present[1] = True
                self.exprs_ge.append(x.copy())
                self.exprs_ge.append(-x)
                x.setzero()
                
        for x in self.exprs_gei:
            c = x.get_coeff(term)
            if c * sn > 0.0:
                sn_present[0] = True
            elif c * sn < 0.0:
                sn_present[1] = True
                
        for x in self.exprs_eqi:
            c = x.get_coeff(term)
            if c != 0.0:
                sn_present[0] = True
                sn_present[1] = True
                self.exprs_gei.append(x.copy())
                self.exprs_gei.append(-x)
                x.setzero()
        
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        self.exprs_eqi = [x for x in self.exprs_eqi if not x.iszero()]
        
        cs = self
        
        if sn == 0:
            sn_present[1] = False
            sn_present[0] = True
        
        if sn_present[1]:
            tmpvar = Expr.real("FLAT_TMP_" + str(term))
            for x in cs.exprs_ge:
                c = x.get_coeff(term)
                if c * sn > 0.0:
                    x.substitute(Expr.fromterm(term), tmpvar)
            for x in cs.exprs_gei:
                c = x.get_coeff(term)
                if c * sn < 0.0:
                    x.substitute(Expr.fromterm(term), tmpvar)
            reg2 = term.reg.copy()
            reg2.substitute(Expr.fromterm(term), tmpvar)
            cs.aux_avoid(reg2)
            newindep = Expr.Ic(reg2.getauxi(), cs.allcomprv() - cs.getaux() - reg2.allcomprv(), 
                                reg2.allcomprv_noaux()).simplified()
            
            if reg2.get_type() == RegionType.NORMAL:
                reg2 = reg2.corners_optimum_eq(tmpvar, sn)
            
            cs = reg2 & cs
            
            if not newindep.iszero():
                cs = cs.iand_norename((newindep == 0).imp_flipped())
            
            cs.eliminate_quick(tmpvar)
            
        if sn_present[0]:
            newvar = Expr.real(str(term))
            reg2 = term.reg.copy()
            cs.aux_avoid(reg2)
            newindep = Expr.Ic(reg2.getaux(), cs.allcomprv() - cs.getaux() - reg2.allcomprv(), 
                                reg2.allcomprv_noaux()).simplified()
            
            if reg2.get_type() == RegionType.NORMAL:
                reg2 = reg2.corners_optimum(Expr.fromterm(term), sn)
            
            cs = cs.implicated(reg2)
            #cs &= reg2.imp_flipped()
            
            if not newindep.iszero():
                cs = cs.iand_norename((newindep == 0).imp_flipped())
            cs.substitute(Expr.fromterm(term), newvar)
            
        return cs
    
    
    def flatten_ivar(self, ivar):
        cs = self
        newvar = Comp([ivar.copy_noreg()])
        reg2 = ivar.reg.copy()
        cs.aux_avoid(reg2)
        cs = cs.implicated(reg2, skip_simplify = True)
        cs.substitute(Comp([ivar]), newvar)
        if not ivar.reg_det:
            newindep = Expr.Ic(reg2.getaux() + newvar, cs.allcomprv() - cs.getaux() - reg2.allcomprv(), 
                                reg2.allcomprv_noaux() - newvar).simplified()
            if not newindep.iszero():
                cs = cs.iand_norename((newindep == 0).imp_flipped())
        return cs
    
    def isregtermpresent(self):
        for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi:
            if x.isregtermpresent():
                return True
        return False
        
    def numexpr(self):
        return len(self.exprs_ge) + len(self.exprs_eq) + len(self.exprs_gei) + len(self.exprs_eqi)
        
    def numterm(self):
        return sum([len(x.terms) for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi])
        
    def isplain(self):
        if not self.aux.isempty():
            return False
        if self.isregtermpresent():
            return False
        return True
    
    def regtermmap(self, cmap, recur):
        for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi + [Expr.H(self.aux + self.auxi)]:
            for (a, c) in x.terms:
                rvs = a.allcomprv_shallow()
                for b in rvs.varlist:
                    if b.reg is not None:
                        s = b.name
                        if not (s in cmap):
                            cmap[s] = b
                            if recur:
                                b.reg.regtermmap(cmap, recur)
                        
                if a.get_type() == TermType.REGION:
                    s = a.x[0].varlist[0].name
                    if not (s in cmap):
                        cmap[s] = a
                        if recur:
                            a.reg.regtermmap(cmap, recur)
        
    def flatten(self):
        
        verbose = PsiOpts.settings.get("verbose_flatten", False)
        write_pf_enabled = PsiOpts.settings.get("proof_enabled", False)
        
        cs = self
        
        did = False
        didall = False
        
        regterms = {}
        cs.regtermmap(regterms, False)
        regterms_in = {}
        for (name, term) in regterms.items():
            term.reg.regtermmap(regterms_in, True)
        for (name, term) in regterms.items():
            if not(name in regterms_in):
                if verbose:
                    print("=========  flatten   ========")
                    print(cs)
                    print("=========    term    ========")
                    print(term)
                    print("=========   region   ========")
                    print(term.reg)
                    
                if isinstance(term, IVar):
                    cs = cs.flatten_ivar(term)
                else:
                    cs = cs.flatten_regterm(term)
                did = True
                didall = True
                
                if verbose:
                    print("=========     to     ========")
                    print(cs)
                    
                break
        
        if write_pf_enabled:
            if didall:
                pf = ProofObj.from_region(self, c = "Expanded definitions to")
                PsiOpts.set_setting(proof_add = pf)
                
        if did:
            return cs.flatten()
        return cs
    
    def flatten_term(self, x, isimp = True):
        cs = self
        if isinstance(x, Comp):
            for y in x.varlist:
                cs = cs.flatten_ivar(y, isimp)
        else:
            for y, _ in x.terms:
                cs = cs.flatten_regterm(y, isimp)
        return cs
    
    def flattened(self, *args, minmax_elim = False):
        if not self.isregtermpresent() and len(args) == 0:
            return self.copy()
        
        cs = None
        if not isinstance(cs, RegionOp):
            cs = RegionOp.inter([self])
        else:
            cs = self.copy()
        cs.flatten(minmax_elim = minmax_elim)
        for x in args:
            cs.flatten_term(x)
        
        t = cs.tosimple()
        if t is not None:
            return t
        
        return cs
    
    def incorporate_tmp(self, x, tmplink):
        if isinstance(x, Comp):
            self &= (H(x) == tmplink)
        elif isinstance(x, Expr):
            self &= (x == tmplink)
        return self
    
    def incorporate_remove_tmp(self, tmplink):
        self.remove_present(tmplink)
    
    def to_cause_consequence(self):
        return [(self.imp_flippedonly_noaux(), self.consonly(), self.auxi.copy())]
    
    def and_cause_consequence(self, other, avoid = None, added_reg = None):
        cs = self.copy()
        
        other = other.copy()
        cs.aux_avoid(other)
        
        clist = other.to_cause_consequence()
        for imp, cons, auxi in clist:
            tcs = cs.tosimple()
            if tcs is None:
                return cs
            cs = tcs
            
            timp = imp.tosimple()
            if timp is None:
                continue
            
            tcs = imp.copy()
            tcs.implicate(cs)
            tcs = tcs.exists(auxi)
            # print(imp)
            # print(cons)
            # print(auxi)
            # print("TCS")
            # print(tcs)
            
            hint_aux_avoid = None
            if avoid is not None:
                hint_aux_avoid = []
                for a in auxi:
                    hint_aux_avoid.append((a, avoid.copy()))
                    
            for rr in tcs.check_getaux_inplace_gen(hint_aux_avoid = hint_aux_avoid):
                if iutil.signal_type(rr) == "":
                    #print(rr)
                    tcons = cons.copy()
                    Comp.substitute_list(tcons, rr)
                    cs &= tcons
                    if added_reg is not None:
                        added_reg &= tcons
                    
        return cs
        
    def toregionop(self):
        if not isinstance(self, RegionOp):
            return RegionOp.inter([self])
        else:
            return self.copy()
        
        
    def incorporated(self, *args):
        cs = self.toregionop()
        
        cargs = [a.copy() for a in args]
        
        to_remove_aux = []
        tmplink = Expr.real("#TMPLINK")
        
        for i in range(len(cargs)):
            x = cargs[i]
            if isinstance(x, Region):
                cs = cs.and_cause_consequence(x).toregionop()
            else:
                cs.flatten_term(x, isimp = False)
                x_noreg = x.copy_noreg()
                for j in range(i + 1, len(cargs)):
                    cargs[j].substitute(x, x_noreg)
                if not cs.ispresent(x_noreg):
                    #cs.eliminate(x_noreg, forall = True)
                    cs.incorporate_tmp(x_noreg, tmplink)
                    to_remove_aux.append(x_noreg)
                
        if len(to_remove_aux):
            cs.incorporate_remove_tmp(tmplink)
        
        # print("AFTER INCORPORATED")
        # print(cs)
        return cs.simplified_quick()
    
    def flattened_self(self):
        r = self.copy()
        r = r.flatten()
        return r.simplify_quick()
    
    def tosimple(self):
        if not self.auxi.isempty():
            return None
        return self.copy()
    
    def tosimple_noaux(self):
        if self.aux_present():
            return None
        return self.tosimple()
    
    def tosimple_safe(self):
        return self.tosimple()
    
    def commonpart_extend(self, v):
        ceps = PsiOpts.settings["eps"]
        tvar = Expr.real("#TVAR")
        did = False
        didpos = False
        toadd = []
        todel = set()
        
        for x in self.exprs_eq:
            c = x.commonpart_coeff(v)
            if c is None or numpy.isnan(c):
                return None
            if numpy.isinf(c):
                return None
            if abs(c) <= ceps:
                continue
            if x.isnonneg() and c > 0:
                return None
            did = True
            didpos = True
            toadd.append((x, c))
        
        for x in self.exprs_ge:
            c = x.commonpart_coeff(v)
            if c is None or numpy.isnan(c):
                return None
            if numpy.isinf(c):
                if c < 0:
                    return None
                todel.add(x)
                continue
            
            if abs(c) <= ceps:
                continue
            if x.isnonpos() and c < 0:
                return None
            did = True
            if c > 0:
                didpos = True
            toadd.append((x, c))
        
        for x in self.exprs_eqi:
            c = x.commonpart_coeff(v)
            if c is None or numpy.isnan(c):
                return None
            if numpy.isinf(c):
                return None
            if abs(c) <= ceps:
                continue
            did = True
            didpos = True
            toadd.append((x, c))
        
        for x in self.exprs_gei:
            c = x.commonpart_coeff(v)
            if c is None or numpy.isnan(c):
                return None
            if numpy.isinf(c):
                return None
            if abs(c) <= ceps:
                continue
            did = True
            didpos = True
            toadd.append((x, c))
        
        if not didpos:
            return None
        
        self.exprs_eq = [x for x in self.exprs_eq if x not in todel]
        self.exprs_ge = [x for x in self.exprs_ge if x not in todel]
        self.exprs_eqi = [x for x in self.exprs_eqi if x not in todel]
        self.exprs_gei = [x for x in self.exprs_gei if x not in todel]
        
        for x, c in toadd:
            x += tvar * c
            
        self.exprs_ge.append(tvar)
        self.eliminate(tvar)
        
        return self
    
    def var_neighbors(self, v):
        r = v.copy()
        for x in self.exprs_eq + self.exprs_ge + self.exprs_eqi + self.exprs_gei:
            r += x.var_neighbors(v)
        return r
    
    def aux_strengthen(self, addrv = None):
        
        if self.aux.isempty():
            return self
        
        if self.imp_present():
            return self
        
        allc = self.allcomprv()
        if addrv is not None:
            allc += addrv
        
        toadd = Region.universe()
        
        for a in self.aux:
            v = self.var_neighbors(a)
            b = allc - v
            if not b.isempty():
                toadd.exprs_eq.append(Expr.Ic(a, b, v - a))
        
        self.iand_norename(toadd)
        
    
    def simplify_aux_commonpart(self, reg = None, minlen = 1, maxlen = 1):
        
        if self.aux.isempty():
            return self
        
        if self.imp_present():
            return self
        
        if reg is None:
            reg = Region.universe()
            
        did = False
        
        taux = self.aux.copy()
        taux2 = taux.copy()
        tauxi = self.auxi.copy()
        self.aux = Comp.empty()
        self.auxi = Comp.empty()
        
        for clen in range(minlen, maxlen + 1):
            if clen == 2:
                continue
            if clen > len(taux):
                break
            for v in igen.subset(taux, minsize = clen):
                if PsiOpts.is_timer_ended():
                    break
                if clen == 1:
                    if self.commonpart_extend(v) is not None:
                        did = True
                else:
                    for v2 in igen.partition(v, clen):
                        if PsiOpts.is_timer_ended():
                            break
                        if self.commonpart_extend(v2) is not None:
                            did = True
                
        self.aux = taux2
        self.auxi = tauxi
        # if did:
        #     self.simplify_quick()
        
        return self
        
    def remove_realvar_sn(self, v, sn):
        # v = v.allcomp()
        v = v.terms[0][0]
        for x in self.exprs_ge:
            # if len(x) == 1 and x.terms[0][1] * sn > 0 and x.terms[0][0].get_type() == IVarType.REAL and x.allcomp() == v:
            if x.get_coeff(v) * sn > 0:
                x.setzero()

        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]

    def istight(self, canon = False):
        return all(x.istight(canon) for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi)

    def tighten(self):
        for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi:
            x.tighten()

    def optimum(self, v, b, sn, name = None, reg_outer = None, assume_feasible = True, allow_reuse = False, quick = None, quick_outer = None, tighten = False):
        """Return the variable obtained from maximizing (sn=1)
        or minimizing (sn=-1) the expression v over variables b (Comp, Expr or list)
        """

        if reg_outer is not None:
            if quick_outer is None:
                quick_outer = quick
            a0 = self.optimum(v, b, sn, name = name, reg_outer = None, assume_feasible = assume_feasible, allow_reuse = allow_reuse, quick = quick)
            a1 = reg_outer.optimum(v, b, sn, name = name, reg_outer = None, assume_feasible = assume_feasible, allow_reuse = allow_reuse, quick = quick_outer)
            a1.terms[0][0].substitute(a1.terms[0][0].x[0], a0.terms[0][0].x[0])
            a0.terms[0][0].reg_outer = a1.terms[0][0].reg
            if tighten:
                a0.tighten()
            return a0


        tmpstr = ""
        if name is not None:
            tmpstr = name
        else:
            if sn > 0:
                tmpstr = "max"
            else:
                tmpstr = "min"
            tmpstr += str(iutil.hash_short(self))
            tmpstr = tmpstr + "(" + str(v) + ")"
        tmpvar = Expr.real(tmpstr)

        if allow_reuse and v.size() == 1 and v.terms[0][0].get_type() == TermType.REAL:
            coeff = v.terms[0][1]
            if coeff < 0:
                sn *= -1
                
            cs = self.copy()
            
            if b is not None:
                b = Region.get_allcomp(b) - v.allcomp()
                if quick is False:
                    cs.eliminate(b)
                else:
                    cs.eliminate_quick(b)
            
            if assume_feasible:
                if cs.get_type() == RegionType.NORMAL:
                    cs.remove_realvar_sn(v, sn)
            
            if not allow_reuse:
                cs.substitute(Expr.fromterm(Term(v.terms[0][0].copy().x, Comp.empty())), tmpvar)
                v = tmpvar

            return Expr.fromterm(Term(v.terms[0][0].copy().x, Comp.empty(), cs, sn)) * coeff
        

        cs = self.copy()
        
        toadd = Region.universe()
        if sn > 0:
            toadd.exprs_ge.append(v - tmpvar)
        else:
            toadd.exprs_ge.append(tmpvar - v)
        
        toadd = toadd.flattened(minmax_elim = True)

        cs = cs & toadd
        # cs.iand_norename(toadd)
        
        if quick is None:
            quick = v.isrealvar() and v.allcomp().super_of(Region.get_allcomp(b))

        return cs.optimum(tmpvar, b, sn, name = name, reg_outer = reg_outer, assume_feasible = assume_feasible, allow_reuse = True, quick = quick)
    
    def maximum(self, expr, vs = None, reg_outer = None, **kwargs):
        """Return the variable obtained from maximizing the expression expr
        over variables vs (Comp, Expr or list)
        """
        return self.optimum(expr, vs, 1, reg_outer = reg_outer, **kwargs)
    
    def minimum(self, expr, vs = None, reg_outer = None, **kwargs):
        """Return the variable obtained from minimizing the expression expr
        over variables vs (Comp, Expr or list)
        """
        return self.optimum(expr, vs, -1, reg_outer = reg_outer, **kwargs)
    
    def init_prog(self, index = None, lptype = None, save_res = False, lp_bounded = None, dual_enabled = None, val_enabled = None):
        if index is None:
            index = IVarIndex()
            self.record_to(index)
            
        prog = None
        if lptype is None:
            lptype = PsiOpts.settings["lptype"]
            
        if lp_bounded is None:
            if save_res:
                lp_bounded = True
            
        if lptype == LinearProgType.H:
            prog = LinearProg(index, LinearProgType.H, lp_bounded = lp_bounded, 
                              save_res = save_res, prereg = self, dual_enabled = dual_enabled, val_enabled = val_enabled)
        elif lptype == LinearProgType.HC1BN:
            bnet = self.get_bayesnet_imp(skip_simplify = True)
            kindex = bnet.index.copy()
            kindex.add_varindex(index)
            prog = LinearProg(kindex, LinearProgType.HC1BN, bnet, lp_bounded = lp_bounded, 
                              save_res = save_res, prereg = self, dual_enabled = dual_enabled, val_enabled = val_enabled)
        
        for x in self.exprs_gei:
            prog.addExpr_ge0(x)
        for x in self.exprs_eqi:
            prog.addExpr_eq0(x)
            
        prog.finish()
        return prog
    
    def get_prog_region(self, toreal = None, toreal_only = False):
        cs = self.consonly().imp_flipped()
        index = IVarIndex()
        cs.record_to(index)
        
        r = cs.init_prog(index, lptype = LinearProgType.H).get_region(toreal, toreal_only)
        return r
    
    def get_extreme_rays(self):
        cs = self.consonly().imp_flipped()
        index = IVarIndex()
        cs.record_to(index)
        
        r = cs.init_prog(index, lptype = LinearProgType.H).get_extreme_rays()
        return r
        
    def implies_ineq_cons_hash(self, expr, sg):
        chash = hash(expr)
        
        if sg == ">=":
            for x in self.exprs_ge:
                if chash == hash(x):
                    return True
                
        for x in self.exprs_eq:
            if chash == hash(x):
                return True
        
        return False
    
    
    def implies_ineq_cons_quick(self, expr, sg):
        """Return whether self implies expr >= 0 or expr == 0, without linear programming"""
        
        if sg == "==" and expr.isnonneg():
            sg = ">="
            expr = -expr
            
        if sg == ">=":
            for x in self.exprs_ge:
                d = (expr - x).simplified()
                if d.isnonneg():
                    return True
            for x in self.exprs_eq:
                d = (expr - x).simplified()
                if d.isnonneg():
                    return True
                d = (expr + x).simplified()
                if d.isnonneg():
                    return True
            return False
        
        if sg == "==":
            for x in self.exprs_eq:
                d = (expr - x).simplified()
                if d.iszero():
                    return True
            return False
        
        return False
    
    def implies_ineq_quick(self, expr, sg):
        """Return whether self implies expr >= 0 or expr == 0, without linear programming"""
        
        if sg == "==" and expr.isnonneg():
            sg = ">="
            expr = -expr
            
        if sg == ">=":
            for x in self.exprs_gei:
                d = (expr - x).simplified()
                if d.isnonneg():
                    return True
            for x in self.exprs_eqi:
                d = (expr - x).simplified()
                if d.isnonneg():
                    return True
                d = (expr + x).simplified()
                if d.isnonneg():
                    return True
            return False
        
        if sg == "==":
            for x in self.exprs_eqi:
                d = (expr - x).simplified()
                if d.iszero():
                    return True
            return False
        
        return False
            
    def implies_ineq_prog(self, index, progs, expr, sg, save_res = False, saved = False):
        if saved == "both":
            return (self.implies_ineq_prog(index, progs, expr, sg, save_res, saved = True)
                    and self.implies_ineq_prog(index, progs, expr, sg, save_res, saved = False))
        
        #print("save_res = " + str(save_res))
        if not saved and self.implies_ineq_quick(expr, sg):
            return True
        if len(progs) == 0:
            progs.append(self.init_prog(index, save_res = save_res))
        if progs[0].checkexpr(expr, sg, saved = saved):
            return True
        return False
        
    def implies_impflipped_saved(self, other, index, progs):
            
        verbose_subset = PsiOpts.settings.get("verbose_subset", False)
            
        if verbose_subset:
            print(self)
            print(other)
        
        for x in other.exprs_ge:
            if not self.implies_ineq_prog(index, progs, x, ">=", save_res = True, saved = "both"):
                if verbose_subset:
                    print(str(x) + " >= 0 FAIL")
                return False
        for x in other.exprs_eq:
            if not self.implies_ineq_prog(index, progs, x, "==", save_res = True, saved = "both"):
                if verbose_subset:
                    print(str(x) + " == 0 FAIL")
                return False
        
        if verbose_subset:
            print("SUCCESS")
        return True
        
    def implies_saved(self, other, index, progs):
        self.imp_flip()
        r = self.implies_impflipped_saved(other, index, progs)
        self.imp_flip()
        return r
        
    
    def check_quick(self, skip_simplify = False):
        """Return whether implication is true"""
        verbose_subset = PsiOpts.settings.get("verbose_subset", False)
        
        cs = self
        if not skip_simplify:
            cs = self.simplified_quick(zero_group = 2)
            
        
        if verbose_subset:
            print(cs)
        
        for x in cs.exprs_ge:
            if not cs.implies_ineq_quick(x, ">="):
                if verbose_subset:
                    print(str(x) + " >= 0 FAIL")
                return False
        for x in cs.exprs_eq:
            if not cs.implies_ineq_quick(x, "=="):
                if verbose_subset:
                    print(str(x) + " == 0 FAIL")
                return False
        
        if verbose_subset:
            print("SUCCESS")
        return True
    
    
    def check_plain(self, skip_simplify = False):
        """Return whether implication is true"""
        verbose_subset = PsiOpts.settings.get("verbose_subset", False)
        
        cs = self
        if not skip_simplify:
            cs = self.simplified_quick(zero_group = 2)
            
        index = IVarIndex()
        
        cs.record_to(index)
        
        progs = []
        
        if verbose_subset:
            print(cs)
        
        for x in cs.exprs_ge:
            if not cs.implies_ineq_prog(index, progs, x, ">="):
                if verbose_subset:
                    print(str(x) + " >= 0 FAIL")
                return False
        for x in cs.exprs_eq:
            if not cs.implies_ineq_prog(index, progs, x, "=="):
                if verbose_subset:
                    print(str(x) + " == 0 FAIL")
                return False
        
        if verbose_subset:
            print("SUCCESS")
        return True
    
    def check_dual(self):
        verbose_subset = PsiOpts.settings.get("verbose_subset", False)
        ceps = PsiOpts.settings["eps_lp"]
        # cs = self.imp_flippedonly_noaux()
        cs = self
        r = Region.universe()
        
        index = IVarIndex()
        
        cs.record_to(index)
        
        progs = []
        progs.append(self.init_prog(index, dual_enabled = True))
        
        for x in self.exprs_ge + self.exprs_eq + [-y for y in self.exprs_eq]:
            if not cs.implies_ineq_prog(index, progs, x, ">="):
                if verbose_subset:
                    print(str(x) + " >= 0 FAIL")
                return None
            cdual = progs[0].get_dual_region()
            if cdual is not None:
                r.iand_norename(cdual)
            
            
            # v, prog = cs.minimum(x).solve_prog("dual")
            # if v is None or v < -ceps or prog is None:
            #     return None
            # cdual = prog.get_dual_region()
            # if cdual is not None:
            #     r &= cdual
        return r
    
    
    def ic_list(self, v):
        r = []
        for x in self.exprs_ge:
            if x.isnonpos():
                continue
            for (a, c) in x.terms:
                if not a.isic2():
                    continue
                if a.x[0].ispresent(v):
                    r.append((a.x[1].copy(), c))
                if a.x[1].ispresent(v):
                    r.append((a.x[0].copy(), c))
        return r
        
    def ic_list_similarity(self, vl, wl):
        r = 0
        for (va, vc) in vl:
            for (wa, wc) in wl:
                if vc * wc < 0:
                    continue
                if va == wa:
                    r += 2
                elif va.super_of(wa) or wa.super_of(va):
                    r += 1
        return r
    
    def get_hc(self):
        r = Expr.zero()
        for x in self.exprs_ge:
            if x.isnonpos():
                for a, c in x.terms:
                    if a.ishc():
                        cx = a.x[0] - a.z
                        if not cx.isempty():
                            r.terms.append((Term.Hc(cx, a.z), 1.0))
        for x in self.exprs_eq:
            if x.isnonpos() or x.isnonneg():
                for a, c in x.terms:
                    if a.ishc():
                        cx = a.x[0] - a.z
                        if not cx.isempty():
                            r.terms.append((Term.Hc(cx, a.z), 1.0))
        return r
    
    def get_var_avoid(self, a):
        r = None
        for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi:
            t = x.get_var_avoid(a)
            if r is None:
                r = t
            elif t is not None:
                r = r.inter(t)
        return r
    
    def get_aux_avoid_list(self):
        r = []
        caux = self.getaux()
        for a in caux:
            t = self.get_var_avoid(a)
            if t is not None:
                r.append((a, t))
        return r
    
    def check_getaux_inplace(self, must_include = None, single_include = None, hint_pair = None, hint_aux = None, hint_aux_avoid = None, max_iter = None, leaveone = None):
        """Return whether implication is true, with auxiliary search result"""
        if hint_aux_avoid is None:
            hint_aux_avoid = []
        hint_aux_avoid = hint_aux_avoid + self.get_aux_avoid_list()
        
        for rr in self.check_getaux_inplace_gen(must_include = must_include, 
                single_include = single_include, hint_pair = hint_pair,
                hint_aux = hint_aux, hint_aux_avoid = hint_aux_avoid,
                max_iter = max_iter, leaveone = leaveone):
            if iutil.signal_type(rr) == "":
                return rr
        return None
    
    def check_getaux_inplace_gen(self, must_include = None, single_include = None, hint_pair = None, hint_aux = None, hint_aux_avoid = None, max_iter = None, leaveone = None):
        """Generator that yields all auxiliary search result"""
        
        if leaveone is None:
            leaveone = PsiOpts.settings["auxsearch_leaveone"]
            
        leaveone_add_ineq = PsiOpts.settings["auxsearch_leaveone_add_ineq"]
        if self.aux.isempty():
            if len(self.exprs_eq) == 0 and len(self.exprs_ge) == 0:
                yield []
                return
            if not leaveone:
                if self.check_plain(skip_simplify = True):
                    yield []
                return
                
        write_pf_enabled = PsiOpts.settings.get("proof_enabled", False)
        
        
        verbose = PsiOpts.settings.get("verbose_auxsearch", False)
        verbose_step = PsiOpts.settings.get("verbose_auxsearch_step", False)
        verbose_result = PsiOpts.settings.get("verbose_auxsearch_result", False)
        verbose_cache = PsiOpts.settings.get("verbose_auxsearch_cache", False)
        verbose_step_cached = PsiOpts.settings.get("verbose_auxsearch_step_cached", False)
        
        as_generator = True
        
        if must_include is None:
            must_include = Comp.empty()
        
        noncircular = PsiOpts.settings["imp_noncircular"]
        noncircular_allaux = PsiOpts.settings["imp_noncircular_allaux"]
        #noncircular_skipfail = True
        
        forall_multiuse = PsiOpts.settings["forall_multiuse"]
        forall_multiuse_numsave = PsiOpts.settings["forall_multiuse_numsave"]
        auxsearch_local = PsiOpts.settings["auxsearch_local"]
        save_res = auxsearch_local
        if max_iter is None:
            max_iter = PsiOpts.settings["auxsearch_max_iter"]
        lpcost = 100
        maxcost = max_iter * lpcost
        curcost = [0]
        
        cs = self.copy()
        
        index = IVarIndex()
        
        #cs.record_to(index)
        
        progs = []
        
        #ccomp = must_include + cs.auxi + (index.comprv - cs.auxi - must_include)
        ccomp = must_include + cs.auxi + (cs.allcomprv() - cs.aux - cs.auxi - must_include)
        index.record(ccomp)
        index.record(cs.allcompreal())
        
        auxcomp = cs.aux.copy()
        auxcond = Comp.empty()
        for a in auxcomp.varlist:
            b = Comp([a])
            if self.imp_ispresent(b):
                auxcond += b
        auxcomp = auxcond + auxcomp
        
        n = auxcomp.size()
        n_cond = auxcond.size()
        m = ccomp.size()
        #m_flip = cs.auxi.size()
        
        clist = collections.deque()
        clist_hashset = set()
        
        
        flipflag = 0
        auxiclist = [cs.ic_list(a) for a in auxcomp]
        cs_flipped = cs.imp_flipped()
        ciclist = [cs_flipped.ic_list(a) for a in ccomp]
        cvisflag = 0
        for i in range(n):
            maxj = -1
            maxval = 0
            for j in range(m):
                t = cs.ic_list_similarity(auxiclist[i], ciclist[j])
                if t > maxval:
                    maxval = t
                    maxj = j
            if maxj >= 0:
                flipflag |= 1 << (m * i + maxj)
                cvisflag |= 1 << maxj
                
        for i in range(n):
            if (flipflag >> (m * i)) & ((1 << m) - 1) == 0:
                flipflag |= (((1 << (must_include + cs.auxi).size()) - 1) & ~cvisflag) << (m * i)
        
        mustflag = 0
        for i in range(n):
            #flipflag += ((1 << (must_include + cs.auxi).size()) - 1) << (m * i)
            mustflag += ((1 << must_include.size()) - 1) << (m * i)
        
        singleflag = 0
        if single_include is not None:
            for j in range(m):
                if single_include.ispresent(ccomp[j]):
                    singleflag += 1 << j
        #m_flip = 0
        # mp2 = (1 << m)
        
        comppair = [-1 for j in range(m)]
        auxpair = [-1 for i in range(n)]
        compside = [0 for j in range(m)]
        auxside = [0 for i in range(n)]
        
        if hint_pair is not None:
            for (a, b) in hint_pair:
                ai = -1
                bi = -1
                for i in range(n):
                    if auxcomp.varlist[i] == a.varlist[0]:
                        ai = i
                        auxside[i] = 1
                    elif auxcomp.varlist[i] == b.varlist[0]:
                        bi = i
                        auxside[i] = -1
                        
                if ai >= 0 and bi >= 0:
                    auxpair[ai] = bi
                    auxpair[bi] = ai
                    
                
                ai = -1
                bi = -1
                for i in range(m):
                    if ccomp.varlist[i] == a.varlist[0]:
                        ai = i
                        compside[i] = 1
                    elif ccomp.varlist[i] == b.varlist[0]:
                        bi = i
                        compside[i] = -1
                        
                if ai >= 0 and bi >= 0:
                    comppair[ai] = bi
                    comppair[bi] = ai
        
        setflag = 0
        if hint_aux is not None:
            for taux, tc in hint_aux:
                pair_allowed = taux.get_marker_key("incpair") is not None
                tcmask = 0
                tcmask_pair = 0
                for j in range(m):
                    if tc.ispresent(ccomp[j]):
                        tcmask |= 1 << j
                        if pair_allowed and comppair[j] < 0:
                            tcmask_pair |= 1 << j
                    elif pair_allowed and comppair[j] >= 0 and tc.ispresent(ccomp[comppair[j]]):
                        tcmask_pair |= 1 << j
                        
                for i in range(n):
                    if taux.ispresent(auxcomp[i]):
                        setflag |= tcmask << (m * i)
                    elif pair_allowed and auxpair[i] >= 0 and taux.ispresent(auxcomp[auxpair[i]]):
                        setflag |= tcmask_pair << (m * i)
        
        auxlist = [Comp.empty() for i in range(n)]
        auxflag = [0 for i in range(n)]
        
        eqs = []
        for x in cs.exprs_ge:
            eqs.append((x.copy(), ">="))
        for x in cs.exprs_eq:
            eqs.append((x.copy(), "=="))
        
        eqvs_range = n * 2 + 1
        eqvs = [[] for i in range(eqvs_range)]
        eqvsid = [[] for i in range(eqvs_range)]
        eqvpresflag = [[] for i in range(eqvs_range)]
        eqvleaveok = [[] for i in range(eqvs_range)]
        eqvs_emptyid = n * 2
        for (x, sg) in eqs:
            maxi = -1
            mini = 100000000
            presflag = 0
            for i in range(n):
                if x.ispresent(Comp([auxcomp.varlist[i]])):
                    maxi = max(maxi, i)
                    mini = min(mini, i)
                    presflag |= 1 << i
            ii = eqvs_emptyid
            if maxi >= 0:
                if mini == maxi:
                    ii = maxi * 2
                else:
                    ii = maxi * 2 + 1
            eqvsid[ii].append(len(eqvs[ii]))
            eqvs[ii].append((x, sg))
            eqvpresflag[ii].append(presflag)
            eqvleaveok[ii].append(sg == ">=" and not x.isnonpos() and not x.ispresent(auxcond))
        
        eqvsns = [[] for i in range(eqvs_range)]
        for i in range(eqvs_range):
            for j in range(len(eqvs[i])):
                x, sg = eqvs[i][j]
                if sg == "==":
                    eqvsns[i].append(0)
                else:
                    csn = None
                    for i2 in range(n):
                        if eqvpresflag[i][j] & (1 << i2) != 0:
                            tsn = x.get_sign(auxcomp.varlist[i2])
                            if tsn == 0 or ((csn is not None) and csn != tsn):
                                csn = 0
                                break
                            else:
                                csn = tsn
                    eqvsns[i].append(csn if csn is not None else 0)
        
        eqvsncache = [[] for i in range(n * 2 + 1)]
        eqvflagcache = [[] for i in range(n * 2 + 1)]
        for i in range(eqvs_range):
            eqvsncache[i] = [[[], []] for j in range(len(eqvs[i]))]
            eqvflagcache[i] = [{} for j in range(len(eqvs[i]))]
        
        auxsetflag = [(setflag >> (m * i)) & ((1 << m) - 1) for i in range(n)]
        
        
        auxavoidflag = [0 for i in range(n)]
        if hint_aux_avoid is not None:
            for taux, tc in hint_aux_avoid:
                pair_allowed = taux.get_marker_key("incpair") is not None
                tcmask = 0
                tcmask_pair = 0
                for j in range(m):
                    if tc.ispresent(ccomp[j]):
                        tcmask |= 1 << j
                        if pair_allowed and comppair[j] < 0:
                            tcmask_pair |= 1 << j
                    elif pair_allowed and comppair[j] >= 0 and tc.ispresent(ccomp[comppair[j]]):
                        tcmask_pair |= 1 << j
                        
                for i in range(n):
                    if taux.ispresent(auxcomp[i]):
                        auxavoidflag[i] |= tcmask
                    elif pair_allowed and auxpair[i] >= 0 and taux.ispresent(auxcomp[auxpair[i]]):
                        auxavoidflag[i] |= tcmask_pair
        
        if False:
            for i in range(n):
                t = self.get_var_avoid(auxcomp[i])
                if t is not None:
                    auxavoidflag[i] |= ccomp.get_mask(t)
            
            for i in range(n):
                auxavoidflag[i] &= ~auxsetflag[i]
        
        
        avoidflag = sum(auxavoidflag[i] << (m * i) for i in range(n))
        flipflag &= ~avoidflag
        
        disjoint_ids = [-1 for i in range(n)]
        symm_ids = [-1 for i in range(n)]
        nonsubset_ids = [-1 for i in range(n)]
        nonempty_is = [False for i in range(n)]
        symm_nonempty_ns = [0] * n
        for i in range(n):
            if auxcomp.varlist[i].markers is None:
                continue
            cdict = {v: w for v, w in auxcomp.varlist[i].markers}
            disjoint_ids[i] = cdict.get("disjoint", -1)
            symm_ids[i] = cdict.get("symm", -1)
            nonsubset_ids[i] = cdict.get("nonsubset", -1)
            nonempty_is[i] = "nonempty" in cdict
            symm_nonempty_ns[i] = cdict.get("symm_nonempty", 0)
            
        for i in range(n):
            if symm_nonempty_ns[i] > 0:
                nsymm = 0
                for i2 in range(i + 1, n):
                    if symm_ids[i] == symm_ids[i2]:
                        nsymm += 1
                if nsymm < symm_nonempty_ns[i]:
                    nonempty_is[i] = True
        
        #print("NONSUBSET  " + "; ".join(str(auxcomp[i]) for i in range(n) if nonsubset_ids[i] >= 0))
        
        
        fcns = cs_flipped.get_hc()
        fcns_mask = []
        for a, c in fcns.terms:
            fcns_mask.append((index.get_mask(a.x[0]), index.get_mask(a.z)))
        
        
        if verbose:
            print("========= aux search ========")
            print(cs.imp_flippedonly())
            print("========= subset of =========")
            #print(co)
            for i in range(n * 2 + 1):
                hs = ""
                if i == n * 2:
                    hs = "NONE"
                elif i % 2 == 0:
                    hs = str(auxcomp.varlist[i // 2])
                else:
                    hs = "<=" + str(auxcomp.varlist[i // 2])
                for j in range(len(eqvs[i])):
                    eqvsnstr = ""
                    if i < n * 2:
                        eqvsnstr = "  " + ("amb" if eqvsns[i][j] == 0
                                else "inc" if eqvsns[i][j] == 1 else "dec")
                    if leaveone and eqvleaveok[i][j]:
                        eqvsnstr += " " + "leaveok"
                    print(iutil.strpad(hs, 8, ": " + str(eqvs[i][j][0]) + " " + str(eqvs[i][j][1]) + " 0" + eqvsnstr))
                    
            print("========= variables =========")
            print(ccomp)
            print("========= auxiliary =========")
            #print(auxcomp)
            print(str(auxcond) + " ; " + str(auxcomp - auxcond))
            
            if len(fcns_mask) > 0:
                print("========= functions =========")
                for x, z in fcns_mask:
                    print(str(ccomp.from_mask(x)) + " <- " + str(ccomp.from_mask(z)))
            
            if hint_pair is not None:
                print("=========  pairing  =========")
                for i in range(n):
                    if auxpair[i] > i:
                        print(str(auxcomp.varlist[i]) + "  <->  " + str(auxcomp.varlist[auxpair[i]]))
                for i in range(m):
                    if comppair[i] > i:
                        print(str(ccomp.varlist[i]) + "  <->  " + str(ccomp.varlist[comppair[i]]))
                
            print("=========  initial  =========")
            for i in range(n):
                cflag = (flipflag >> (m * i)) & ((1 << m) - 1)
                csetflag = auxsetflag[i]
                cavoidflag = auxavoidflag[i]
                ccor = Comp.empty()
                cset = Comp.empty()
                cavoid = Comp.empty()
                for j in range(m):
                    if cflag & (1 << j) != 0:
                        ccor += ccomp[j]
                    if csetflag & (1 << j) != 0:
                        cset += ccomp[j]
                    if cavoidflag & (1 << j) != 0:
                        cavoid += ccomp[j]
                print(str(auxcomp.varlist[i]) + "  :  " + str(ccor) 
                    + ("  Fix: " + str(cset) if not cset.isempty() else "")
                    + ("  Avoid: " + str(cavoid) if not cavoid.isempty() else ""))
        
        
        
        if len(eqs) == 0:
            #print(bin(setflag))
            #print(bin(avoidflag))
            #print("m=" + str(m) + " n=" + str(n))
            #print("ccomp=" + str(ccomp) + " auxcomp=" + str(auxcomp))
            #for a in auxcomp.varlist:
            #    print(str(a) + " " + str(a.markers))
            
            mleft = [j for j in range(m * n) if setflag & (1 << j) == 0 and avoidflag & (1 << j) == 0]
            
            def check_nocond_recur(i, size, allflag):
                #print(str(i) + " " + str(size) + " " + bin(allflag))
                if i == n:
                    if mustflag != 0 and mustflag & allflag == 0:
                        return
                    rr = [(auxcomp[i2].copy(), auxlist[i2].copy()) for i2 in range(n)]
                    yield rr
                    return
                    
                mlefti = [j - m * i for j in mleft if j >= m * i and j < m * (i + 1)]
                symm_break = -1
                
                if symm_ids[i] >= 0:
                    for i2 in range(i):
                        if symm_ids[i] == symm_ids[i2]:
                            symm_break = max(symm_break, auxflag[i2])
                            
                if disjoint_ids[i] >= 0:
                    for i2 in range(i):
                        if disjoint_ids[i] == disjoint_ids[i2]:
                            mlefti = [j for j in mlefti if auxflag[i2] & (1 << j) == 0]
                            
                sizelb = max(size - sum(j >= m * (i + 1) for j in mleft), 0)
                sizeub = min(min(len(mlefti), m), size)
                if sizelb > sizeub:
                    return
                for tsize in range(sizelb, sizeub + 1):
                    for comb in itertools.combinations(list(reversed(mlefti)), tsize):
                        curcost[0] += 1
                        if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                            return
                        
                        auxflag[i] = sum(1 << j for j in comb) | auxsetflag[i]
                        if auxflag[i] < symm_break:
                            break
                        
                        if nonempty_is[i] and auxflag[i] == 0:
                            continue
                        
                        if any(auxflag[i] | z == auxflag[i] and auxflag[i] & x != 0 for x, z in fcns_mask):
                            continue
                                
                        if nonsubset_ids[i] >= 0:
                            tbad = False
                            for i2 in range(i):
                                if nonsubset_ids[i] == nonsubset_ids[i2]:
                                    if auxflag[i] | auxflag[i2] == auxflag[i] or auxflag[i] | auxflag[i2] == auxflag[i2]:
                                        tbad = True
                                        break
                            if tbad:
                                continue
                                    
                        
                        auxlist[i] = ccomp.from_mask(auxflag[i])
                        #print(str(i) + " " + str(m) + " " + str(ccomp) + " " + bin(auxflag[i]) + " " + str(auxlist[i]))
                        for rr in check_nocond_recur(i + 1, size - len(comb), allflag | (auxflag[i] << (m * i))):
                            yield rr
            
            #print("START")
            for tsize in range(len(mleft) + 1):
                for rr in check_nocond_recur(0, tsize, 0):
                    #print("; ".join(str(v) + ":" + str(w) for v, w in rr))
                    yield rr
            #print("END")
            return
            
        
        auxcache = [{} for i in range(n)]
        
        if verbose:
            print("========= progress: =========")
        
        leaveone_static = None
        
        
        # *********** Check conditions that does not depend on aux ***********
        
        if n_cond == 0:
            for (x, sg) in eqvs[eqvs_emptyid]:
                if not cs.implies_ineq_prog(index, progs, x, sg, save_res = save_res):
                    if leaveone and sg == ">=":
                        if verbose_step:
                            print("  F LO " + str(x) + " " + sg + " 0")
                        leaveone = False
                        leaveone_static = -x
                    else:
                        if verbose_step:
                            print("  F " + str(x) + " " + sg + " 0")
                        return None
        
        allflagcache = [{} for i in range(n + 1)]
        
        cursizepass = 0
        
        cs = Region.universe()
        
        cs_added = Region.universe()
        #self.imp_only_copy_to(cs_added)
        condflagadded = {}
        condflagadded_true = collections.deque()
        flagcache = [set() for i in range(n + 1)]
        
        maxprogress = [-1, 0, -1]
        flipflaglen = 0
        numfail = [0]
        numclear = [0]
        
        auxavoidflag_orig = list(auxavoidflag)
        
        def clear_cache(mini, maxi):
            if verbose_cache:
                print("========= cache clear: " + str(mini) + " - " + str(maxi) + " =========")
            progs[:] = []
            auxcache[mini:maxi] = [{} for i in range(mini, maxi)]
            for a in flagcache:
                a.clear()
                
            for i in range(mini, maxi):
                auxavoidflag[i] = auxavoidflag_orig[i]
                
            for i in range(mini * 2, eqvs_range):
                for j in range(len(eqvs[i])):
                    if eqvpresflag[i][j] & ((1 << maxi) - (1 << mini)) != 0:
                        eqvsncache[i][j] = [[], []]
                        eqvflagcache[i][j] = {}
                # eqvsncache[i] = [[[], []] for j in range(len(eqvs[i]))]
                # eqvflagcache[i] = [{} for j in range(len(eqvs[i]))]
                
        
        def build_region(i, allflag, allownew, cflag = 0, add_ineq = None):
            numclear[0] += 1
            
            cs_added_changed = False
            prev_csstr = cs.tostring(tosort = True)
            
                
            if add_ineq is not None:
                prev_csaddedstr = cs_added.tostring(tosort = True)
                cs_added.exprs_gei.append(add_ineq)
                cs_added.simplify_quick(zero_group = 2)
                #cs_added.split()
                if prev_csaddedstr != cs_added.tostring(tosort = True):
                    cs_added_changed = True
                    clist.appendleft(cflag)
                    while len(clist) > forall_multiuse_numsave:
                        clist.pop()
                    if verbose_step:
                        print("========= leave one added =========")
                        print(cs_added.imp_flipped())
                        print("==================")
                        
                cs_added.imp_only_copy_to(cs)
                
                
            elif i >= n_cond and (forall_multiuse or leaveone):
                csnew = Region.universe()
                if not (allflag in condflagadded):
                    self.imp_only_copy_to(csnew)
                    for i3 in range(i, n_cond):
                        csnew.remove_present(Comp([auxcomp.varlist[i3]]))
                    for i3 in range(i):
                        csnew.substitute(Comp([auxcomp.varlist[i3]]), auxlist[i3])
                    
                    if allownew:
                        condflagadded[allflag] = True
                        prev_csaddedstr = cs_added.tostring(tosort = True)
                        cs_added.iand_norename(csnew)
                        cs_added.simplify_quick(zero_group = 2)
                        #cs_added.split()
                        if prev_csaddedstr != cs_added.tostring(tosort = True):
                            cs_added_changed = True
                            condflagadded_true.appendleft(allflag)
                            while len(condflagadded_true) > forall_multiuse_numsave:
                                condflagadded_true.pop()
                            if verbose_step:
                                print("========= forall added =========")
                                print(cs_added.imp_flipped())
                                print("==================")
                    
                cs_added.imp_only_copy_to(cs)
                if not allownew:
                    cs.iand_norename(csnew)
                    
            else:
                self.imp_only_copy_to(cs)
                for i3 in range(i, n_cond):
                    cs.remove_present(Comp([auxcomp.varlist[i3]]))
                for i3 in range(i):
                    cs.substitute(Comp([auxcomp.varlist[i3]]), auxlist[i3])
            
            if cs_added_changed or prev_csstr != cs.tostring(tosort = True):
                if verbose_cache:
                    print("========= cleared =========")
                    print(prev_csstr)
                    print("========= to =========")
                    print(cs.tostring(tosort = True))
                maxi = n
                if forall_multiuse and not cs_added_changed:
                    maxi = n_cond
                if noncircular:
                    if noncircular_allaux:
                        clear_cache(n_cond, maxi)
                    else:
                        clear_cache(min(1, n_cond), maxi)
                else:
                    clear_cache(0, maxi)
            else:
                if verbose_cache:
                    print("========= not cleared =========")
                    print(prev_csstr)
                
        
        #build_region(0, 0, True)
        build_region(0, 0, leaveone)
        
        def is_marker_sat(i):
            if nonempty_is[i] and auxflag[i] == 0:
                return False
        
            if any(auxflag[i] | z == auxflag[i] and auxflag[i] & x != 0 for x, z in fcns_mask):
                return False
            
            if nonsubset_ids[i] >= 0:
                for i2 in range(i):
                    if nonsubset_ids[i] == nonsubset_ids[i2]:
                        if auxflag[i] | auxflag[i2] == auxflag[i] or auxflag[i] | auxflag[i2] == auxflag[i2]:
                            return False
                
            if symm_ids[i] >= 0:
                for i2 in range(i):
                    if symm_ids[i] == symm_ids[i2]:
                        if auxflag[i] < auxflag[i2]:
                            return False
                
            if disjoint_ids[i] >= 0:
                tbad = False
                for i2 in range(i):
                    if disjoint_ids[i] == disjoint_ids[i2]:
                        if auxflag[i] & auxflag[i2] != 0:
                            return False
            
            return True
        
        
        # *********** Sandwich procedure ***********
        
        dosandwich = PsiOpts.settings["auxsearch_sandwich"]
        dosandwich_inc = PsiOpts.settings["auxsearch_sandwich_inc"]
        
        if dosandwich:
            if verbose:
                print("========== sandwich =========")
                
            for i in range(n * 2 + 1):
                if i == eqvs_emptyid:
                    continue
                for j in range(len(eqvs[i])):
                    if leaveone and eqvleaveok[i][j]:
                        continue
                    
                    presflag = eqvpresflag[i][j]
                    for sg in ([1, -1] if eqvs[i][j][1] == "==" else [1]):
                        x = eqvs[i][j][0] * sg
                        sns = [0] * n
                        bad = False
                        for i2 in range(n):
                            if presflag & (1 << i2):
                                sns[i2] = x.get_sign(auxcomp.varlist[i2])
                                if sns[i2] == 0:
                                    bad = True
                                    break
                                if not dosandwich_inc and sns[i2] > 0:
                                    bad = True
                                    break
                        if bad:
                            continue
                        
                        for ix in range(-1, n):
                            if sns[ix] == 0:
                                continue

                            xt = x.copy()
                            for i2 in range(n):
                                if i2 == ix:
                                    continue
                                cmask = 0
                                if sns[i2] < 0:
                                    cmask = (setflag >> (m * i2)) & ((1 << m) - 1)
                                else:
                                    cmask = ~(avoidflag >> (m * i2)) & ((1 << m) - 1)
                                # print(avoidflag)
                                # print(cmask)
                                xt.substitute(Comp([auxcomp.varlist[i2]]), ccomp.from_mask(cmask))
                            
                            
                            if ix == -1:
                                tres = cs.implies_ineq_prog(index, progs, xt, ">=", save_res = save_res, saved = "both")
                                if verbose_step:
                                    print(str(x) + "  :  " + str(xt) + " >= 0 " + str(tres))
                                if not tres:
                                    return None
                                continue
                            
                            cmask0 = (setflag >> (m * ix)) & ((1 << m) - 1)
                            cmask1 = ~(avoidflag >> (m * ix)) & ((1 << m) - 1)
                            cmaskc = cmask1 & ~cmask0
                            for k in range(m):
                                if cmaskc & (1 << k):
                                    cmask = 0
                                    if sns[ix] < 0:
                                        cmask = cmask0 | (1 << k)
                                    else:
                                        cmask = cmask1 & ~(1 << k)
                                
                                    xt2 = xt.copy()
                                    xt2.substitute(Comp([auxcomp.varlist[ix]]), ccomp.from_mask(cmask))
                                    
                                    tres = cs.implies_ineq_prog(index, progs, xt2, ">=", save_res = save_res, saved = "both")
                                    if verbose_step:
                                        print(str(x) + "  :  " + str(xt2) + " >= 0 " + str(tres))
                                    if not tres:
                                        if sns[ix] < 0:
                                            avoidflag |= 1 << (m * ix + k)
                                            auxavoidflag[ix] |= 1 << k
                                        else:
                                            setflag |= 1 << (m * ix + k)
                                            auxsetflag[ix] |= 1 << k
                                    
                                    
            flipflag &= ~avoidflag
            auxavoidflag_orig = list(auxavoidflag)
            
            if verbose:
                print("======= after sandwich ======")
                for i in range(n):
                    cflag = (flipflag >> (m * i)) & ((1 << m) - 1)
                    csetflag = auxsetflag[i]
                    cavoidflag = auxavoidflag[i]
                    ccor = Comp.empty()
                    cset = Comp.empty()
                    cavoid = Comp.empty()
                    for j in range(m):
                        if cflag & (1 << j) != 0:
                            ccor += ccomp[j]
                        if csetflag & (1 << j) != 0:
                            cset += ccomp[j]
                        if cavoidflag & (1 << j) != 0:
                            cavoid += ccomp[j]
                    print(str(auxcomp.varlist[i]) + "  :  " + str(ccor) 
                        + ("  Fix: " + str(cset) if not cset.isempty() else "")
                        + ("  Avoid: " + str(cavoid) if not cavoid.isempty() else ""))
                print("=============================")
        
        
        
        def check_local(i0, allflag, leave_id = None):
            cflag = ((flipflag | setflag) >> (m * i0)) << (m * i0)
            mleft = [j for j in range(m * i0, m * n) if setflag & (1 << j) == 0]
            #mleft = mleft[::-1]
            
            
            while True:
                cleave_id = leave_id
                
                for i in range(i0, n):
                    auxflag[i] = (cflag >> (m * i)) & ((1 << m) - 1)
                    auxlist[i] = ccomp.from_mask(auxflag[i])
                
                cres = True
                
                bad = False
                for i in range(i0, n):
                    if not is_marker_sat(i):
                        bad = True
                        break
                            
                if bad:
                    cres = False
                    
                if cres:
                    for i2 in range(i0 * 2, n * 2):
                        i2r = -1
                        isone = False
                        if i2 != eqvs_emptyid:
                            i2r = i2 // 2
                            isone = (i2 % 2 == 0)
                        
                        auxflagi = auxflag[i2r]
                        if isone and (auxflagi in auxcache[i2r]):
                            cres = auxcache[i2r][auxflagi]
                        else:
                            for ieqid in range(len(eqvs[i2])):
                                ieq = eqvsid[i2][ieqid]
                                if cleave_id == (i2, ieq):
                                    continue
                                
                                (x, sg) = eqvs[i2][ieq]
                                x2 = x.copy()
                                if isone:
                                    x2.substitute(Comp([auxcomp.varlist[i2r]]), auxlist[i2r])
                                else:
                                    for i3 in range(i2r + 1):
                                        x2.substitute(Comp([auxcomp.varlist[i3]]), auxlist[i3])
                                
                                    
                                curcost[0] += lpcost
                                if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                                    return False
                                tres = cs.implies_ineq_prog(index, progs, x2, sg, save_res = save_res)
                                
                                #print(iutil.strpad("; ".join([str(auxlist[i3]) for i3 in range(i)]),
                                #   26, " LO#" + str(numfail[0]), 8, " " + str(x) + " " + sg + " 0 " + str(tres)))
                                   
                                if not tres:
                                    if isone and eqvsns[i2][ieq] < 0 and (not leaveone or not eqvleaveok[i2][ieq]) and iutil.bitcount(auxflagi & ~auxsetflag[i]) == 1:
                                        auxavoidflag[i2r] |= auxflagi & ~auxsetflag[i]
                                        if verbose_step:
                                            print(iutil.strpad("  L AVOID " + str(auxcomp.varlist[i2r]), 
                                                    12, " : " + str(ccomp.from_mask(auxavoidflag[i2r]))))
                                        
                                    if leaveone and cleave_id is None and eqvleaveok[i2][ieq]:
                                        cleave_id = (i2, ieq)
                                        if verbose_step:
                                            print(iutil.strpad("; ".join([str(auxlist[i3]) for i3 in range(n)]),
                                               26, " LSET#" + str(numfail[0]), 12, " " + str(x) + " " + sg + " 0"))
                                    else:
                                        if verbose_step:
                                            numfail[0] += 1
                                            print(iutil.strpad("; ".join([str(auxlist[i3]) for i3 in range(n)]),
                                               26, " LO#" + str(numfail[0]), 12, " " + str(x) + " " + sg + " 0"))
                                            
                                        eqvsid[i2].pop(ieqid)
                                        eqvsid[i2].insert(0, ieq)
                                        flagcache[i2r + 1].add(cflag & ((1 << ((i2r + 1) * m)) - 1))
                                        #print("FCA " + str(i2r + 1) + " " + bin(cflag & ((1 << ((i2r + 1) * m)) - 1)))
                                        cres = False
                                        break
                    
                            if isone and not leaveone:
                                auxcache[i2r][auxflagi] = cres
                        if not cres:
                            break
                        
                if cres:
                    if as_generator:
                        if leaveone and cleave_id is not None:
                            x2 = -eqvs[cleave_id[0]][cleave_id[1]][0]
                            for i3 in range(n):
                                x2.substitute(Comp([auxcomp.varlist[i3]]), auxlist[i3])
                            #print("BUILD " + str(x2))
                            
                            if leaveone_add_ineq:
                                build_region(n_cond, allflag, False, cflag = cflag, add_ineq = x2)
                            yield ("leaveone", x2.copy())
                        else:
                            yield allflag | cflag
                    else:
                        return True
                
                flagcache[n].add(cflag)
                
                def check_local_recur(i, kflag, sizelb, sizeub, kleave_id):
                    
                    csizelb = max(sizelb - sum(j >= m * (i + 1) for j in mleft), 0)
                    csizeub = sizeub
                    
                    #print(str(i) + " " + str(csizelb) + " " + str(csizeub))
                    
                    for tsize in range(csizelb, csizeub + 1):
                        mlefti = [j - m * i for j in mleft if j >= m * i and j < m * (i + 1)]
                        mlefti = [j for j in mlefti if auxavoidflag[i] & (1 << j) == 0]
                        cflagi = ((cflag >> (m * i)) & ((1 << m) - 1))
                        mleftimustflag = cflagi & auxavoidflag[i]
                        ttsize = tsize - iutil.bitcount(mleftimustflag)
                        if ttsize > len(mlefti):
                            break
                        if ttsize < 0:
                            continue
                        for comb in itertools.combinations(mlefti, ttsize):
                            
                            curcost[0] += 1
                            if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                                return False
                            
                            
                            auxflag[i] = sum(1 << j for j in comb) ^ cflagi
                            auxflag[i] &= ~auxavoidflag[i]
                            if not auxcache[i].get(auxflag[i], True):
                                continue
                            
                                
                            if not is_marker_sat(i):
                                continue
                            
                            auxlist[i] = ccomp.from_mask(auxflag[i])
                            
                            ckflag = kflag | (auxflag[i] << (m * i))
                            ckleave_id = kleave_id
                            
                            if i == n - 1 and mustflag != 0 and mustflag & (allflag | ckflag) == 0:
                                continue
                            if ckflag in flagcache[i + 1]:
                                continue
                            
                            cres = True
                            
                            for i2 in range(i * 2, (i + 1) * 2):
                                i2r = -1
                                isone = False
                                if i2 != eqvs_emptyid:
                                    i2r = i2 // 2
                                    isone = (i2 % 2 == 0)
                                
                                auxflagi = auxflag[i2r]
                                
                                for ieqid in range(len(eqvs[i2])):
                                    ieq = eqvsid[i2][ieqid]
                                    
                                    (x, sg) = eqvs[i2][ieq]
                                    x2 = x.copy()
                                    if isone:
                                        x2.substitute(Comp([auxcomp.varlist[i2r]]), auxlist[i2r])
                                    else:
                                        for i3 in range(i2r + 1):
                                            x2.substitute(Comp([auxcomp.varlist[i3]]), auxlist[i3])
                                    
                                    curcost[0] += lpcost
                                    if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                                        return False
                                    tres = cs.implies_ineq_prog(index, progs, x2, sg, save_res = save_res, saved = True)
                                    
                                    if not tres:
                                        if isone and eqvsns[i2][ieq] < 0 and (not leaveone or not eqvleaveok[i2][ieq]) and iutil.bitcount(auxflagi & ~auxsetflag[i]) == 1:
                                            auxavoidflag[i2r] |= auxflagi & ~auxsetflag[i]
                                            if verbose_step:
                                                print(iutil.strpad("  T AVOID " + str(auxcomp.varlist[i2r]), 
                                                        12, " : " + str(ccomp.from_mask(auxavoidflag[i2r]))))
                                            
                                        if leaveone and ckleave_id is None and eqvleaveok[i2][ieq]:
                                            #if verbose_step and verbose_step_cached:
                                            #    print(iutil.strpad("; ".join([str(auxlist[i3]) for i3 in range(i2r + 1)]),
                                            #       26, " TSET=" + str(tsize) + ",#" + str(numfail[0]), 12, " " + str(x) + " " + sg + " 0"))
                                            ckleave_id = (i2, ieq)
                                        else:
                                            if verbose_step and verbose_step_cached:
                                                numfail[0] += 1
                                                print(iutil.strpad("; ".join([str(auxlist[i3]) for i3 in range(i2r + 1)]),
                                                   26, " TO=" + str(tsize) + ",#" + str(numfail[0]), 12, " " + str(x) + " " + sg + " 0"))
                                            
                                            eqvsid[i2].pop(ieqid)
                                            eqvsid[i2].insert(0, ieq)
                                            
                                            flagcache[i2r + 1].add(ckflag)
                                            #print("FCB " + str(i2r + 1) + " " + bin(ckflag))
                                            cres = False
                                            break
                            
                                if isone and not cres and not leaveone:
                                    auxcache[i2r][auxflagi] = cres
                                if not cres:
                                    break
                                
                            if not cres:
                                continue
                            if i == n - 1:
                                return True
                            if check_local_recur(i + 1, ckflag, sizelb - len(comb), sizeub - len(comb), ckleave_id):
                                return True
                            if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                                return False
                            
                    return False
                
                sizeseq = [0, 1, 2, 4]
                sizeseq = [s for s in sizeseq if s < len(mleft)] + [len(mleft)]
                found = False
                for si in range(1, len(sizeseq)):
                    if si == 0:
                        tcflag = cflag
                        cflag = 0
                        if check_local_recur(i0, 0, 1, 1, None):
                            found = True
                            cflag = sum(auxflag[i] << (m * i) for i in range(i0, n))
                            break
                        cflag = tcflag
                    else:
                        if check_local_recur(i0, 0, sizeseq[si - 1] + 1, sizeseq[si], None):
                            found = True
                            cflag = sum(auxflag[i] << (m * i) for i in range(i0, n))
                            break
                
                if not found:
                    return False
        
        
        def check_recur(i, size, stepsize, allflag):
            if i == n and mustflag != 0 and mustflag & allflag == 0:
                return False
            
            if allflag in allflagcache[i]:
                return False
            
            cprogress = i * (n * 2 + 2)
            if cprogress > maxprogress[0]:
                maxprogress[0] = cprogress
                maxprogress[1] = allflag | (flipflag & ~((1 << (m * i)) - 1))
                maxprogress[2] = i
                    
            if not noncircular and i == n_cond:
                
                if verbose_cache:
                    print("========= cache clear: circ, suff # " + str(numclear[0]) + " =========")
                
                build_region(n_cond, allflag, False)
            
                
            i2lb = 0
            i2ub = 0
            if noncircular:
                if i > 0:
                    i2lb = (i - 1) * 2
                    i2ub = i * 2
            else:
                if i >= n_cond:
                    i2lb = 0
                    if i > n_cond:
                        i2lb = (i - 1) * 2
                    i2ub = i * 2
                    
            for i2 in range(i2lb, i2ub):
                cres = True
                i2r = -1
                isone = False
                if i2 != eqvs_emptyid:
                    i2r = i2 // 2
                    isone = (i2 % 2 == 0)
                
                auxflagi = auxflag[i2r]
                if isone and (auxflagi in auxcache[i2r]):
                    cres = auxcache[i2r][auxflagi]
                else:
                    
                    for ieqid in range(len(eqvs[i2])):
                        ieq = eqvsid[i2][ieqid]
                        
                        (x, sg) = eqvs[i2][ieq]
                        x2 = x.copy()
                        if isone:
                            x2.substitute(Comp([auxcomp.varlist[i2r]]), auxlist[i2r])
                        else:
                            for i3 in range(i2r + 1):
                                x2.substitute(Comp([auxcomp.varlist[i3]]), auxlist[i3])
                        
                        auxflagpres = 0
                        auxflagprescn = 0
                        for i3 in range(i2r + 1):
                            if eqvpresflag[i2][ieq] & (1 << i3) != 0:
                                auxflagpres |= auxflag[i3] << (m * auxflagprescn)
                                auxflagprescn += 1
                                
                        tres = None
                        computed = False
                        
                        eqvsn = eqvsns[i2][ieq]
                        if eqvsn == 0:
                            tres = eqvflagcache[i2][ieq].get(auxflagpres, None)
                        else:
                            eqvsn = (eqvsn + 1) // 2
                            for f in eqvsncache[i2][ieq][eqvsn]:
                                if auxflagpres == f | auxflagpres:
                                    tres = (eqvsn == 1)
                                    break
                            if tres is None:
                                for f in eqvsncache[i2][ieq][1 - eqvsn]:
                                    if f == f | auxflagpres:
                                        tres = (eqvsn == 0)
                                        break
                        
                        if tres is None:
                            
                            curcost[0] += lpcost
                            if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                                return False
                            tres = cs.implies_ineq_prog(index, progs, x2, sg, save_res = save_res)
                            computed = True
                            
                            eqvsn = eqvsns[i2][ieq]
                            if eqvsn == 0:
                                eqvflagcache[i2][ieq][auxflagpres] = tres
                            else:
                                eqvsncache[i2][ieq][1 if tres else 0].append(auxflagpres)
                        
                        if not tres:
                            if verbose_step and (verbose_step_cached or computed):
                                numfail[0] += 1
                                print(iutil.strpad("; ".join([str(auxlist[i3]) for i3 in range(i)]),
                                   26, " S=" + str(cursizepass) + ",T=" + str(stepsize)
                                   + ",L=" + str(flipflaglen) + ",#" + str(numfail[0]), 18, " " + str(x) + " " + sg + " 0"
                                   + ("" if computed else " (Ca)")))
                            eqvsid[i2].pop(ieqid)
                            eqvsid[i2].insert(0, ieq)
                            cres = False
                            break
                    if isone:
                        auxcache[i2r][auxflagi] = cres
                if not cres:
                    allflagcache[i][allflag] = True
                    return False
            
                cprogress = i * (n * 2 + 2) + i2 + 1
                if cprogress > maxprogress[0]:
                    maxprogress[0] = cprogress
                    maxprogress[1] = allflag | (flipflag & ~((1 << (m * i)) - 1))
                    maxprogress[2] = i
                    
                    
            if i == n_cond and n_cond > 0:
                
                if noncircular:
                    
                    if verbose_cache:
                        print("========= cache clear: nonc, checkempty # " + str(numclear[0]) + " =========")
                              
                build_region(i, allflag, True)
                
                for ieqid in range(len(eqvs[eqvs_emptyid])):
                    ieq = eqvsid[eqvs_emptyid][ieqid]
                    
                    (x, sg) = eqvs[eqvs_emptyid][ieq]
                    
                    curcost[0] += lpcost
                    if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                        return False
                    if not cs.implies_ineq_prog(index, progs, x, sg, save_res = save_res):
                        if verbose_step:
                            numfail[0] += 1
                            print(iutil.strpad("; ".join([str(auxlist[i3]) for i3 in range(i)]),
                               26, " S=" + str(cursizepass) + ",T=" + str(stepsize)
                               + ",L=" + str(flipflaglen) + ",#" + str(numfail[0]), 18, " " + str(x) + " " + sg + " 0"))
                        allflagcache[i][allflag] = True
                        
                        eqvsid[eqvs_emptyid].pop(ieqid)
                        eqvsid[eqvs_emptyid].insert(0, ieq)
                        return False
                
                    
            if i == n:
                if as_generator:
                    yield allflag
                    return
                else:
                    return True
            
            if i == n_cond and auxsearch_local:
                if as_generator:
                    for rr in check_local(i, allflag):
                        yield rr
                    return
                else:
                    return check_local(i, allflag)
                
            cflipflag = (flipflag >> (m * i)) & ((1 << m) - 1)
            if i >= flipflaglen - 1:
                i2 = auxpair[i]
                if i2 >= 0 and i2 < i:
                    cflipflag = auxflag[i2]
                    for j in range(m):
                        j2 = comppair[j]
                        if j2 >= 0:
                            if (auxflag[i2] & (1 << j) != 0) and (auxflag[i2] & (1 << j2) == 0):
                                cflipflag &= ~(1 << j)
                    
            csetflag = (setflag >> (m * i)) & ((1 << m) - 1)
            
            mleft = [j for j in range(m) if csetflag & (1 << j) == 0]
            
            #sizelb = max(0, size - m * (n - i - 1))
            sizelb = 0
            sizeub = min(min(size, len(mleft)), stepsize)
            
            pnumclear = -1
            
            for tsize in range(sizelb, sizeub + 1):
                
                for comb in itertools.combinations(mleft, tsize):
                    
                    curcost[0] += 1
                    if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                        return False
                    #print(tsize, comb)
                    tflag = 0
                    for j in comb:
                        tflag += (1 << j)
                    
                    auxlist[i] = Comp.empty()
                    auxflag[i] = 0
                    for j in range(m):
                        if (csetflag & (1 << j) != 0) or (cflipflag & (1 << j) != 0) != (tflag & (1 << j) != 0):
                            auxflag[i] += (1 << j)
                            auxlist[i].varlist.append(ccomp.varlist[j])
                    
                    if (auxflag[i] & singleflag) != 0 and iutil.bitcount(auxflag[i]) > 1:
                        continue
                    
                    if i < n_cond:
                        pass
                    else:
                        if (auxflag[i] in auxcache[i]) and not auxcache[i]:
                            continue
                        
                        
                    if noncircular and i < n_cond and numclear[0] != pnumclear:
                    
                        if verbose_cache:
                            print("========= cache clear: nonc, inc # " + str(numclear[0]) + " =========")
                                  
                        if noncircular_allaux:
                            build_region(0, 0, False)
                        else:
                            build_region(i, allflag, False)
                            
                        pnumclear = numclear[0]
                    
                    recur = check_recur(i+1, size - tsize, stepsize, allflag + (auxflag[i] << (m * i)))
                    if as_generator:
                        for rr in recur:
                            yield rr
                    else:
                        if recur:
                            return True
                 
                    if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
                        return False
            return False
        
        res_hashset = set()
        
        maxsize = m * n - iutil.bitcount(setflag)
        size = 0
        stepsize = 0
        while True:
            cursizepass = size
            prevprogress = maxprogress[0]
            recur = check_recur(0, size, stepsize, 0)
            if not as_generator:
                recur = [True] if recur else []
            for rr in recur:
                
                if verbose or verbose_result:
                    print("========= success cost " + str(curcost[0]) + "/" + str(maxcost) + " =========")
                    #print("========= final region =========")
                    #print(cs.imp_flipped())
                    print("========== aux  =========")
                    for i in range(n):
                        print(iutil.strpad(str(auxcomp.varlist[i]), 6, ": " + str(auxlist[i])))
                        
                namelist = [auxcomp.varlist[i].name for i in range(n)]
                res = []
                for i in range(n):
                    i2 = namelist.index(self.aux.varlist[i].name)
                    cval = auxlist[i2].copy()
                    if forall_multiuse and i2 < n_cond and len(condflagadded_true) > 0:
                        cval = [ccomp.from_mask(x >> (m * i2)).copy() for x in condflagadded_true]
                        if len(cval) == 1:
                            cval = cval[0]
                    
                    if i2 >= n_cond and len(clist) > 0:
                        cval = [cval] + [ccomp.from_mask(x >> (m * i2)).copy() for x in clist]
                        if len(cval) == 1:
                            cval = cval[0]
                        
                    res.append((Comp([self.aux.varlist[i].copy()]), cval))
                
                if as_generator:
                    res_hash = hash(iutil.list_tostr_std(res))
                    if not (res_hash in res_hashset):
                        if iutil.signal_type(rr) == "leaveone":
                            yield ("leaveone", res, rr[1])
                        elif leaveone_static is not None:
                            yield ("leaveone", res, leaveone_static.copy())
                        else:
                            yield res
                        res_hashset.add(res_hash)
                else:
                    return res
            
            if size >= maxsize:
                break
            
            if n_cond == 0 and auxsearch_local:
                break
            
            flipflag = maxprogress[1]
            flipflaglen = maxprogress[2]
            if prevprogress != maxprogress[0]:
                size = 1
            else:
                if size < 2:
                    size += 1
                else:
                    size *= 2
                if size >= maxsize:
                    size = maxsize
                    stepsize = m
                else:
                    clen = max(flipflaglen, 1)
                    stepsize = (size + clen - 1) // clen
            
        if PsiOpts.is_timer_ended() or (maxcost > 0 and curcost[0] >= maxcost):
            yield ("max_iter_reached", )
        return None
    
        
    def add_sfrl_imp(self, x, y, gap = None, noaux = True, name = None):
        ccomp = self.allcomprv() - self.aux
        
        if name is None:
            name = self.name_avoid(y.tostring(add_bracket = True) + "%" + x.tostring(add_bracket = True))
        newvar = Comp.rv(name)
        self.exprs_gei.append(-Expr.I(x, newvar))
        self.exprs_gei.append(-Expr.Hc(y, x + newvar))
        others = ccomp - x - y
        if not others.isempty():
            self.exprs_gei.append(-Expr.Ic(newvar, others, x + y))
        if gap is not None:
            if not isinstance(gap, Expr):
                gap = Expr.const(gap)
            self.exprs_gei.append(gap.copy() - Expr.Ic(x, newvar, y))
            
        if not noaux:
            self.auxi += newvar
        
        return newvar
    
    @fcn_list_to_list
    def add_sfrl(self, x, y, gap = None, noaux = True, name = None):
        self.imp_flip()
        r = self.add_sfrl_imp(x, y, gap, noaux, name)
        self.imp_flip()
        return r
        
        
    def add_esfrl_imp(self, x, y, gap = None, noaux = True):
        if x.super_of(y):
            return Comp.empty(), y.copy()
        if x.isempty():
            return y.copy(), Comp.empty()
        
        ccomp = self.allcomprv() - self.aux
        
        newvar = Comp.rv(self.name_avoid(y.tostring(add_bracket = True) + "%" + x.tostring(add_bracket = True)))
        newvark = Comp.rv(self.name_avoid(y.tostring(add_bracket = True) + "%" + x.tostring(add_bracket = True) + "_K"))
        self.exprs_gei.append(-Expr.I(x, newvar))
        self.exprs_gei.append(-Expr.Hc(newvark, x + newvar))
        self.exprs_gei.append(-Expr.Hc(y, newvar + newvark))
        others = ccomp - x - y
        if not others.isempty():
            self.exprs_gei.append(-Expr.Ic(newvar + newvark, others, x + y))
        if gap is not None:
            if not isinstance(gap, Expr):
                gap = Expr.const(gap)
            self.exprs_gei.append(gap.copy() + Expr.I(x, y) - Expr.H(newvark))
        
        if not noaux:
            self.auxi += newvar + newvark
        
        return newvar, newvark
    
    def add_esfrl(self, x, y, gap = None, noaux = True):
        self.imp_flip()
        r = self.add_esfrl_imp(x, y, gap, noaux)
        self.imp_flip()
        return r
        
    def check_getaux_sfrl(self, sfrl_level = None, sfrl_minsize = 0, sfrl_maxsize = None, sfrl_gap = None, hint_pair = None, hint_aux = None):
        """Return whether implication is true, with auxiliary search results."""
        verbose_sfrl = PsiOpts.settings.get("verbose_sfrl", False)
        
        if sfrl_level is None:
            sfrl_level = PsiOpts.settings["sfrl_level"]
        if sfrl_maxsize is None:
            sfrl_maxsize = PsiOpts.settings["sfrl_maxsize"]
        if sfrl_gap is None:
            sfrl_gap = PsiOpts.settings["sfrl_gap"]
        
        gap = None
        gappresent = False
        if sfrl_gap == "zero":
            gap = Expr.zero()
        elif sfrl_gap != "":
            gap = Expr.real(sfrl_gap)
            gappresent = True
        
        enable_multiple = (sfrl_level >= PsiOpts.SFRL_LEVEL_MULTIPLE)
        
        n = sfrl_maxsize
        
        cs = self
        
        
        ccomp = cs.auxi + (cs.allcomprv() - cs.aux - cs.auxi)
        m = ccomp.size()
        
        sfrlcomp = [[0, 0] for i in range(n)]
        
        tres = None
        
        def check_getaux_sfrl_recur(i):
            if i >= sfrl_minsize:
                cs = self.copy()
                csfrl = Comp.empty()
                for i2 in range(i):
                    sfrlx = sum([ccomp[j] for j in range(m) if sfrlcomp[i2][0] & (1 << j) != 0])
                    sfrly = sum([ccomp[j] for j in range(m) if sfrlcomp[i2][1] & (1 << j) != 0])
                    csfrl += cs.add_sfrl_imp(sfrlx, sfrly, gap, noaux = False)
                
                if verbose_sfrl:
                    print("==========   SFRL   ========= =========")
                    print(csfrl)
                    
                if enable_multiple:
                    tres = cs.check_getaux_inplace(must_include = csfrl, 
                                                   hint_pair = hint_pair, hint_aux = hint_aux)
                else:
                    tres = cs.check_getaux_inplace(must_include = csfrl, single_include = csfrl, 
                                                   hint_pair = hint_pair, hint_aux = hint_aux)
                    
                if tres is not None:
                    return tres
            if i == n:
                return None
            
            for size in range(2, m + 1):
                for xsize in range(1, size):
                    for xtuple in itertools.combinations(range(m), xsize):
                        xmask = sum([1 << x for x in xtuple])
                        yset = [i for i in range(m) if not (i in xtuple)]
                        for ytuple in itertools.combinations(yset, size - xsize):
                            ymask = sum([1 << y for y in ytuple])
                            sfrlcomp[i][0] = xmask
                            sfrlcomp[i][1] = ymask
                            tres = check_getaux_sfrl_recur(i + 1)
                            if tres is not None:
                                return tres
                            
            if False:
                for xmask in range(1, (1 << m) - 1):
                    ymask = (1 << m) - 1 - xmask
                    while ymask != 0:
                        sfrlcomp[i][0] = xmask
                        sfrlcomp[i][1] = ymask
                        tres = check_getaux_sfrl_recur(i + 1)
                        if tres is not None:
                            return tres
                        ymask = (ymask - 1) & ~xmask
        
        
        return check_getaux_sfrl_recur(0)
    
    
    def check_getaux(self, hint_pair = None, hint_aux = None):
        """Return whether implication is true, with auxiliary search results."""
        
        write_pf_enabled = PsiOpts.settings.get("proof_enabled", False)
    
        for rr in self.check_getaux_gen(hint_pair, hint_aux):
            if iutil.signal_type(rr) == "":
                if write_pf_enabled:
                    if len(rr) > 0:
                        pf = ProofObj.from_region(self, c = "Claim:")
                        PsiOpts.set_setting(proof_step_in = pf)
                        
                        pf = ProofObj.from_region(None, c = "Substitute:\n" + iutil.list_tostr_std(rr))
                        PsiOpts.set_setting(proof_add = pf)
                        
                        cs = self.copy()
                        Comp.substitute_list(cs, rr, isaux = True)
                        if cs.getaux().isempty():
                            with PsiOpts(proof_enabled = True):
                                cs.check_plain()
                                
                        PsiOpts.set_setting(proof_step_out = True)
                        
                return rr
        return None
    
    def check_getaux_dict(self, **kwargs):
        r = self.check_getaux(**kwargs)
        if r is None:
            return None
        return Comp.substitute_list_to_dict(r)
    
    def check_getaux_array(self, **kwargs):
        r = self.check_getaux(**kwargs)
        if r is None:
            return None
        return CompArray(Comp.substitute_list_to_dict(r))
    
    def check_getaux_gen(self, hint_pair = None, hint_aux = None):
        """Generator that yields all auxiliary search results."""
        truth = PsiOpts.settings["truth"]
        if truth is not None:
            with PsiOpts(truth = None):
                for rr in (truth >> self).check_getaux_gen(hint_pair, hint_aux):
                    yield rr
                return
        
        if self.isregtermpresent():
            cs = RegionOp.inter([self])
            for rr in cs.check_getaux_gen(hint_pair, hint_aux):
                yield rr
            return
        
        write_pf_enabled = PsiOpts.settings.get("proof_enabled", False)
        
        cs = self.copy()
        cs.simplify_quick(zero_group = 2)
        cs.split()
        
        PsiOpts.settings["proof_enabled"] = False
        
        res = None
        
        sfrl_level = PsiOpts.settings["sfrl_level"]
        
        hint_aux_avoid = self.get_aux_avoid_list()
        
        for rr in cs.check_getaux_inplace_gen(hint_pair = hint_pair, hint_aux = hint_aux, hint_aux_avoid = hint_aux_avoid):
            PsiOpts.settings["proof_enabled"] = write_pf_enabled
            yield rr
            PsiOpts.settings["proof_enabled"] = False
        
        if sfrl_level > 0:
            res = cs.check_getaux_sfrl(sfrl_minsize = 1, hint_pair = hint_pair, hint_aux = hint_aux)
            if res is not None:
                PsiOpts.settings["proof_enabled"] = write_pf_enabled
                yield res
                PsiOpts.settings["proof_enabled"] = False
            
        PsiOpts.settings["proof_enabled"] = write_pf_enabled
    
        
    def check(self):
        """Return whether implication is true"""
        truth = PsiOpts.settings["truth"]
        if truth is not None:
            with PsiOpts(truth = None):
                return (truth >> self).check()
        
        if self.isplain():
            return self.check_plain()
    
        return self.check_getaux() is not None
    
    def evalcheck(self, f):
        truth = PsiOpts.settings["truth"]
        if truth is not None:
            with PsiOpts(truth = None):
                return (truth >> self).evalcheck(f)
        
        ceps = PsiOpts.settings["eps_check"]
        for x in self.exprs_gei:
            if not float(f(x)) >= -ceps:
                return True
        for x in self.exprs_eqi:
            if not abs(float(f(x))) <= ceps:
                return True
        for x in self.exprs_ge:
            if not float(f(x)) >= -ceps:
                return False
        for x in self.exprs_eq:
            if not abs(float(f(x))) <= ceps:
                return False
        return True
        
    def eval_max_violate(self, f):
        truth = PsiOpts.settings["truth"]
        if truth is not None:
            with PsiOpts(truth = None):
                return (truth >> self).eval_max_violate(f)
        
        ceps = PsiOpts.settings["eps_check"]
        for x in self.exprs_gei:
            t = float(f(x))
            if not numpy.isnan(t) and not t >= -ceps:
                return 0.0
        for x in self.exprs_eqi:
            t = float(f(x))
            if not numpy.isnan(t) and not abs(t) <= ceps:
                return 0.0
            
        r = 0.0
        for x in self.exprs_ge:
            t = float(f(x))
            if numpy.isnan(t):
                return numpy.inf
            r = max(r, -t)
        for x in self.exprs_eq:
            t = float(f(x))
            if numpy.isnan(t):
                return numpy.inf
            r = max(r, abs(t))
        return r
    
    def example(self):
        cs = self.exists(self.allcomprv() - self.getaux() - self.getauxi())
        P = ConcModel()
        if not P[cs]:
            return None
        return P.opt_model()
    
    def implies(self, other, **kwargs):
        """Whether self implies other"""
        if kwargs:
            r = None
            with PsiOpts(**{"simplify_" + key: val for key, val in kwargs.items()}):
                other = other.simplified()
            return self.implies(other)

        return (self <= other).check()
        
    def implies_getaux(self, other, hint_pair = None, hint_aux = None):
        """Whether self implies other, with auxiliary search result"""
        res = (self <= other).check_getaux(hint_pair, hint_aux)
        if res is None:
            return None
        return res
        #auxlist = other.aux.varlist + self.auxi.varlist
        #return [(Comp([auxlist[i]]), res[i][1]) for i in range(len(res))]
        
    def implies_getaux_gen(self, other, hint_pair = None, hint_aux = None):
        """Whether self implies other, yield all auxiliary search result"""
        for rr in (self <= other).check_getaux_gen(hint_pair, hint_aux):
            yield rr
        
    
    def equiv(self, other, **kwargs):
        """Whether self is equivalent to other"""
        return self.implies(other, **kwargs) and other.implies(self, **kwargs)
    
    def allcomp(self):
        index = IVarIndex()
        self.record_to(index)
        return index.comprv + index.compreal
        
    def allcomprv(self):
        index = IVarIndex()
        self.record_to(index)
        return index.comprv
        
    def allcompreal(self):
        index = IVarIndex()
        self.record_to(index)
        return index.compreal
        
    def allcomprealvar(self):
        r = Comp.empty()
        for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi:
            r += x.allcomprealvar()
        return r
    
        # index = IVarIndex()
        # self.record_to(index)
        # return index.compreal - Comp([IVar.eps(), IVar.one()])
        
    def allcomprealvar_exprlist(self):
        t = self.allcomprealvar()
        return ExprArray.make([Expr.real(v) for v in t])
    
    def allcomprv_noaux(self):
        return self.allcomprv() - self.getauxall()
    
    def aux_remove(self):
        self.aux = Comp.empty()
        self.auxi = Comp.empty()
        return self
        
    def completed_semigraphoid_ic(self, max_iter = None):
        verbose = PsiOpts.settings.get("verbose_semigraphoid", False)
        
        def mask_impl(a, b):
            a0, a1, az = a
            b0, b1, bz = b
            
            a0x = a0 & (bz & ~az)
            a0 &= ~a0x
            az |= a0x
            a1x = a1 & (bz & ~az)
            a1 &= ~a1x
            az |= a1x
            
            if az != bz:
                return False
            return a0 | b0 == a0 and a1 | b1 == a1
        
        
        icexpr = self.get_ic()
        index = IVarIndex()
        icexpr.record_to(index)
        icl = set()
        
        if verbose:
            print("==========   SEMIGRAPHOID   =========")
            print(icexpr)
            print("=====================================")
        
        for a, c in icexpr.terms:
            if len(a.x) == 1:
                mz = index.get_mask(a.z)
                m0 = index.get_mask(a.x[0]) & ~mz
                icl.add((m0, m0, mz))
            elif len(a.x) == 2:
                mz = index.get_mask(a.z)
                m0 = index.get_mask(a.x[0]) & ~mz
                m1 = index.get_mask(a.x[1]) & ~mz
                if m0 > m1:
                    m0, m1 = m1, m0
                icl.add((m0, m1, mz))
        
        #for a0, a1, az in icl:
        #    print(str(Expr.Ic(index.from_mask(a0), index.from_mask(a1), index.from_mask(az))))
                            
        
        citer = 0
        iclw = icl.copy()
        did = True
        while did:
            did = False
            icl2 = icl.copy()
            for a0k, a1k, azk in icl:
                if max_iter is not None and citer > max_iter:
                    break
                for b0k, b1k, bzk in icl:
                    if max_iter is not None and citer > max_iter:
                        break
                    if (a0k, a1k, azk) == (b0k, b1k, bzk):
                        continue
                    if azk & ~(b0k | b1k | bzk) != 0:
                        continue
                    if bzk & ~(a0k | a1k | azk) != 0:
                        continue
                    
                    for aj in range(2):
                        for bj in range(2):
                            citer += 1
                            if max_iter is not None and citer > max_iter:
                                break
                            
                            a0, a1, az = a0k, a1k, azk
                            b0, b1, bz = b0k, b1k, bzk
                            if aj != 0:
                                a0, a1 = a1, a0
                            a0o, a1o, azo = a0, a1, az
                            if bj != 0:
                                b0, b1 = b1, b0
                            b0o, b1o, bzo = b0, b1, bz
                            
                            if a0 & b0 == 0:
                                continue
                            
                            b0x = b0 & (az & ~bz)
                            b0 &= ~b0x
                            bz |= b0x
                            
                            b1z = b1 | bz
                            a0x = a0 & (b1z & ~az)
                            a0 &= ~a0x
                            az |= a0x
                            a1x = a1 & (b1z & ~az)
                            a1 &= ~a1x
                            az |= a1x
                            
                            b1 &= az
                            
                            a0 &= b0 & ~bz
                            a1 &= ~bz
                            b1 &= ~bz
                            
                            # if verbose:
                            #     print(str(Expr.Ic(index.from_mask(a0o), index.from_mask(a1o), index.from_mask(azo)))
                            #           + " - " + str(Expr.Ic(index.from_mask(b0o), index.from_mask(b1o), index.from_mask(bzo)))
                            #           + " : " + str(Expr.Ic(index.from_mask(a0), index.from_mask(a1 | b1), index.from_mask(bz)))
                            #           + " / " + str(index.from_mask(az)) + "=" + str(index.from_mask(b1 | bz))
                            #           )
                            
                            if az != b1 | bz:
                                continue
                            
                            if a0 == 0 or a1 | b1 == 0:
                                continue
                            
                            t = (a0, a1 | b1, bz)
                            if mask_impl((a0o, a1o, azo), t) or mask_impl((b0o, b1o, bzo), t):
                                continue
                            
                            if verbose:
                                print(str(citer) + ": " + str(Expr.Ic(index.from_mask(a0o), index.from_mask(a1o), index.from_mask(azo)))
                                      + " & " + str(Expr.Ic(index.from_mask(b0o), index.from_mask(b1o), index.from_mask(bzo)))
                                      + " -> " + str(Expr.Ic(index.from_mask(a0), index.from_mask(a1 | b1), index.from_mask(bz)))
                                      )
                                
                            if t[0] > t[1]:
                                t = (t[1], t[0], t[2])
                                
                            if t in iclw:
                                continue
                            
                            #print(str(Expr.Ic(index.from_mask(t[0]), index.from_mask(t[1]), index.from_mask(t[2]))))
                            icl2.add(t)
                            iclw.add(t)
                            did = True
                            
            icl.clear()
            for a0, a1, az in icl2:
                for b0, b1, bz in icl2:
                    if (a0, a1, az) == (b0, b1, bz):
                        continue
                    if (mask_impl((b0, b1, bz), (a0, a1, az)) 
                        or mask_impl((b0, b1, bz), (a1, a0, az))):
                        break
                else:
                    icl.add((a0, a1, az))
        
        r = Expr.zero()
        for a0, a1, az in icl:
            r += Expr.Ic(index.from_mask(a0), index.from_mask(a1), index.from_mask(az))
        return r
               
    
    def completed_semigraphoid(self, max_iter = None):
        """ Use semi-graphoid axioms to deduce more conditional independence.
        Judea Pearl and Azaria Paz, "Graphoids: a graph-based logic for reasoning 
        about relevance relations", Advances in Artificial Intelligence (1987), pp. 357--363.
        """ 
        
        return self.completed_semigraphoid_ic(max_iter = max_iter) <= 0
    
    
    def eliminated_ic(self, w):
        
        icexpr = self.get_ic()
        index = IVarIndex()
        icexpr.record_to(index)
        icl = set()
        
        r = Expr.zero()
        
        for a, c in icexpr.terms:
            if a.z.ispresent(w):
                continue
            b = a.copy()
            for i in range(len(b.x)):
                b.x[i] -= w
            if not b.iszero():
                r += Expr.fromterm(b)
        
        return r <= 0
    
        
    def completed_sfrl(self, gap = None, max_iter = None):
        index = IVarIndex()
        self.record_to(index)
        n = len(index.comprv)
        cs = self.copy()
        tmpvar = Comp.empty()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                tmpvar += cs.add_sfrl(index.comprv[i], index.comprv[j], gap, noaux = False)
        
        cs2 = cs.completed_semigraphoid(max_iter = max_iter)
        cs3 = cs2.eliminated_ic(tmpvar)
        return cs3
            
        
    def convexified(self, q = None, forall = False):
        """Convexify by a time sharing RV q, return result"""
        r = self.copy()
        
        qname = "Q"
        if q is not None:
            qname = str(q)
        qname = r.name_avoid(qname)
        q = Comp.rv(qname)
        
        allcomp = r.allcomprv()
        
        r.condition(q)
        
        if forall:
            r |= ~(Expr.Ic(q, allcomp - r.getauxall() - r.inp - r.oup, r.inp) == 0)
            return r.forall(q)
        else:
            r &= Expr.Ic(q, allcomp - r.getauxall() - r.inp - r.oup, r.inp) == 0
            return r.exists(q)
        
            
    def tounion(self):
        return RegionOp.pack_type(self, RegionType.UNION)
    
    def convexified_diag(self, v = None, cross_only = True, skip_simplify = False):
        """Convexify with respect to the real variables in v along the diagonal, return result"""
        if v is None:
            v = self.allcomprealvar_exprlist()
            
        index = IVarIndex()
        self.record_to(index)
        for x in v:
            x.record_to(index)
        namemap = [dict(), dict()]
        v_new = [[], []]
        toelim = Expr.zero()
        for x in v:
            cname = x.get_name()
            for it in range(2):
                nname = index.name_avoid(cname)
                namemap[it][cname] = nname
                v_new[it].append(Expr.real(nname))
                Expr.real(nname).record_to(index)
                toelim += Expr.real(nname)
                
            
        r = RegionOp.empty()
        ru = self.tounion()
        
        if cross_only:
            r = ru.copy()
            
        for i, j in itertools.combinations(range(len(v)), 2):
            v0j = v[i] + v[j] - v_new[0][i]
            v1j = v[i] + v[j] - v_new[1][i]
            for k0, a0 in enumerate(ru):
                a0t = a0.copy()
                a0t.substitute(v[i], v_new[0][i])
                a0t.substitute(v[j], v0j)
                for k1, a1 in enumerate(ru):
                    if cross_only and k0 == k1:
                        continue
                    a1t = a1.copy()
                    a1t.substitute(v[i], v_new[1][i])
                    a1t.substitute(v[j], v1j)
                    t = a0t & a1t
                    t &= (v[i] >= v_new[0][i]) & (v[i] <= v_new[1][i])
                    
                    #print(str(i) + " , " + str(j))
                    r |= t.exists(v_new[0][i]+v_new[1][i])
                    
        if not skip_simplify:
            return r.simplified()
        else:
            return r
    
    
    def isconvex(self):
        """Check whether region is convex
        False return value does NOT necessarily mean region is not convex
        """
        return self.convexified().implies(self)
        #return ((self + self) / 2).implies(self)
        
    def simplify_quick(self, reg = None, zero_group = 0):
        """Simplify a region in place, without linear programming
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """
        
        if not PsiOpts.settings.get("simplify_enabled", False):
            return self
        
        write_pf_enabled = (PsiOpts.settings.get("proof_enabled", False) 
                            and PsiOpts.settings.get("proof_step_simplify", False))
        if write_pf_enabled:
            prevself = self.copy()
        
        #self.remove_missing_aux()
        
        if reg is None:
            reg = Region.universe()
        for x in self.exprs_ge:
            x.simplify(reg)
        for x in self.exprs_eq:
            x.simplify(reg)
        
        index = IVarIndex()
        self.record_to(index)
        gemask = [index.get_mask(x.allcomprv_shallow()) for x in self.exprs_ge]
        eqmask = [index.get_mask(x.allcomprv_shallow()) for x in self.exprs_eq]
        
        did = True
        if True:
            did = False
            for i in range(len(self.exprs_ge)):
                if not self.exprs_ge[i].iszero():
                    for j in range(i):
                        if not self.exprs_ge[j].iszero() and gemask[i] == gemask[j]:
                            ratio = self.exprs_ge[i].get_ratio(self.exprs_ge[j], skip_simplify = True)
                            if ratio is None:
                                continue
                            if ratio > PsiOpts.settings["eps"]:
                                self.exprs_ge[i] = Expr.zero()
                                gemask[i] = 0
                                did = True
                                break
                            elif ratio < -PsiOpts.settings["eps"]:
                                self.exprs_eq.append(self.exprs_ge[i])
                                eqmask.append(gemask[i])
                                self.exprs_ge[i] = Expr.zero()
                                gemask[i] = 0
                                self.exprs_ge[j] = Expr.zero()
                                gemask[j] = 0
                                did = True
                                break
                            
            
            for i in range(len(self.exprs_ge)):
                if not self.exprs_ge[i].iszero():
                    for j in range(len(self.exprs_eq)):
                        if not self.exprs_eq[j].iszero() and gemask[i] == eqmask[j]:
                            ratio = self.exprs_ge[i].get_ratio(self.exprs_eq[j], skip_simplify = True)
                            if ratio is None:
                                continue
                            self.exprs_ge[i] = Expr.zero()
                            gemask[i] = 0
                            did = True
                            break
            
            
            for i in range(len(self.exprs_eq)):
                if not self.exprs_eq[i].iszero():
                    for j in range(i):
                        if not self.exprs_eq[j].iszero() and eqmask[i] == eqmask[j]:
                            ratio = self.exprs_eq[i].get_ratio(self.exprs_eq[j], skip_simplify = True)
                            if ratio is None:
                                continue
                            self.exprs_eq[i] = Expr.zero()
                            eqmask[i] = 0
                            did = True
                            break
                            
            self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
            self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        
        for i in range(len(self.exprs_ge)):
            if self.exprs_ge[i].isnonneg():
                self.exprs_ge[i] = Expr.zero()
        
        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        
        if True:
            allzero = Expr.zero()
            for i in range(len(self.exprs_ge)):
                if self.exprs_ge[i].isnonpos():
                    for (a, c) in self.exprs_ge[i].terms:
                        allzero -= Expr.fromterm(a)
                    self.exprs_ge[i] = Expr.zero()
                    
            for i in range(len(self.exprs_eq)):
                if self.exprs_eq[i].isnonpos() or self.exprs_eq[i].isnonneg():
                    for (a, c) in self.exprs_eq[i].terms:
                        allzero -= Expr.fromterm(a)
                    self.exprs_eq[i] = Expr.zero()
                    
            if not allzero.iszero():
                allzero.simplify(reg)
                allzero.sortIc()
                #self.exprs_ge.append(allzero)
                self.exprs_ge.insert(0, allzero)
             
        if zero_group == 2:
            pass
        else:
            for i in range(len(self.exprs_ge)):
                if self.exprs_ge[i].isnonpos():
                    for (a, c) in self.exprs_ge[i].terms:
                        if zero_group == 1:
                            self.exprs_ge.append(-Expr.fromterm(a))
                        else:
                            self.exprs_eq.append(Expr.fromterm(a))
                    self.exprs_ge[i] = Expr.zero()
                    
            for i in range(len(self.exprs_eq)):
                if self.exprs_eq[i].isnonpos() or self.exprs_eq[i].isnonneg():
                    for (a, c) in self.exprs_eq[i].terms:
                        if zero_group == 1:
                            self.exprs_ge.append(-Expr.fromterm(a))
                        else:
                            self.exprs_eq.append(Expr.fromterm(a))
                    self.exprs_eq[i] = Expr.zero()
                
        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        for x in self.exprs_ge:
            x.simplify_mul(1)
        for x in self.exprs_eq:
            x.simplify_mul(2)
                
        if self.imp_present():
            t = self.imp_flippedonly()
            t.simplify_quick(reg, zero_group)
            self.exprs_gei = t.exprs_ge
            self.exprs_eqi = t.exprs_eq
            
            for x in self.exprs_ge:
                if self.implies_ineq_quick(x, ">="):
                    x.setzero()
            
            for x in self.exprs_eq:
                if self.implies_ineq_quick(x, "=="):
                    x.setzero()
                    
            self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
            self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        
        if write_pf_enabled:
            if self.tostring() != prevself.tostring():
                # pf = ProofObj.from_region(prevself, c = "Simplify")
                # pf += ProofObj.from_region(self, c = "Simplified as")
                pf = ProofObj.from_region(("equiv", prevself, self), c = "Simplify:")
                PsiOpts.set_setting(proof_add = pf)
            
        return self
            
    
    def iand_simplify_quick(self, other, skip_simplify = True):
        did = False
        
        for x in other.exprs_ge:
            if not self.implies_ineq_cons_hash(x, ">="):
                self.exprs_ge.append(x)
                did = True
        
        for x in other.exprs_eq:
            if not self.implies_ineq_cons_hash(x, "=="):
                self.exprs_eq.append(x)
                did = True
            
        if not skip_simplify and did:
            self.simplify_quick(zero_group = 1)
            
        return self
    
        
    def split_ic2(self):
        ge_insert = []
        
        for i in range(len(self.exprs_ge)):
            if self.exprs_ge[i].isnonpos():
                t = Expr.zero()
                for (a, c) in self.exprs_ge[i].terms:
                    if a.isic2():
                        ge_insert.append(-Expr.fromterm(a))
                    else:
                        t += Expr.fromterm(a) * c
                self.exprs_ge[i] = t
                
        for i in range(len(self.exprs_eq)):
            if self.exprs_eq[i].isnonpos() or self.exprs_eq[i].isnonneg():
                t = Expr.zero()
                for (a, c) in self.exprs_eq[i].terms:
                    if a.isic2():
                        ge_insert.append(-Expr.fromterm(a))
                    else:
                        t += Expr.fromterm(a) * c
                self.exprs_eq[i] = t
        
        self.exprs_ge = ge_insert + self.exprs_ge
        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        
    def split(self):
        ge_insert = []
        
        for i in range(len(self.exprs_ge)):
            if self.exprs_ge[i].isnonpos():
                for (a, c) in self.exprs_ge[i].terms:
                    ge_insert.append(-Expr.fromterm(a))
                self.exprs_ge[i] = Expr.zero()
                
        for i in range(len(self.exprs_eq)):
            if self.exprs_eq[i].isnonpos() or self.exprs_eq[i].isnonneg():
                for (a, c) in self.exprs_eq[i].terms:
                    ge_insert.append(-Expr.fromterm(a))
                self.exprs_eq[i] = Expr.zero()
        
        self.exprs_ge = ge_insert + self.exprs_ge
        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        if self.imp_present():
            t = self.imp_flippedonly()
            t.split()
            self.exprs_gei = t.exprs_ge
            self.exprs_eqi = t.exprs_eq
        
    def symmetrized(self, symm_set, union = False, convexify = False, skip_simplify = False):
        if symm_set is None:
            return self
        
        r = Region.universe()
        if union:
            r = RegionOp.empty()
            
        cs = self.copy()
        n = len(symm_set)
        m = min(len(a) for a in symm_set)
        tmpvar = [[] for i in range(n)]
        
        for i in range(n):
            for j in range(m):
                if isinstance(symm_set[i][j], Comp):
                    tmpvar[i].append(Comp.rv("#TMPVAR" + str(i) + "_" + str(j)))
                else:
                    tmpvar[i].append(Expr.real("#TMPVAR" + str(i) + "_" + str(j)))
                cs.substitute(symm_set[i][j], tmpvar[i][j])
                
        
        for p in itertools.permutations(range(n)):
            tcs = cs.copy()
            for i in range(n):
                for j in range(m):
                    tcs.substitute(tmpvar[i][j], symm_set[p[i]][j])
            if union:
                r |= tcs
            else:
                r &= tcs
        if convexify:
            r = r.convexified_diag(skip_simplify = True)
        
        if skip_simplify:
            return r
        else:
            return r.simplified()
        
    def simplify_bayesnet(self, reg = None, reduce_ic = False):
        
        if isinstance(reg, RegionOp):
            reg = reg.tosimple_noaux()
        if reg is None:
            reg = Region.universe()
            
        icexpr = Expr.zero()
        for x in self.exprs_ge:
            if x.isnonpos():
                icexpr += x
        for x in self.exprs_eq:
            if x.isnonpos():
                icexpr += x
            elif x.isnonneg():
                icexpr -= x
        
        if reg.isuniverse() and icexpr.iszero():
            return
        
        bnet = (reg & (icexpr >= 0)).get_bayesnet(skip_simplify = True)
        
        
        for x in self.exprs_ge:
            
            # Prevent circular simplification
            if not x.isnonpos():
                x.simplify(bnet = bnet)
        
        for x in self.exprs_eq:
            
            # Prevent circular simplification
            if not (x.isnonpos() or x.isnonneg()):
                x.simplify(bnet = bnet)
        
        if reduce_ic:
            for x in self.exprs_ge:
                if x.isnonpos():
                    if bnet.check_ic(-x):
                        x.setzero()
                        
            for x in self.exprs_eq:
                if x.isnonpos():
                    if bnet.check_ic(-x):
                        x.setzero()
                elif x.isnonneg():
                    if bnet.check_ic(x):
                        x.setzero()
            
            self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
            self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
            
            self.iand_norename(bnet.get_region())
        
        return self
        
    def zero_exprs(self, avoid = None):
        for x in self.exprs_ge:
            if x is avoid:
                continue
            if x.isnonpos():
                for y in x:
                    yield y
        
        for x in self.exprs_eq:
            if x is avoid:
                continue
            if x.isnonpos() or x.isnonneg():
                for y in x:
                    yield y
            else:
                yield x
    
    def empty_rvs(self):
        r = Comp.empty()
        for x in self.zero_exprs():
            if len(x) == 1 and x.terms[0][1] != 0 and x.terms[0][0].ish():
                r += x.terms[0][0].x[0]
        return r
        
    def simplify_pair(self, reg = None):
        did = False
        
        if isinstance(reg, RegionOp):
            reg = reg.tosimple_noaux()
        if reg is None:
            reg = Region.universe()
    
        for x, xs in [(a, ">=") for a in self.exprs_ge] + [(a, "==") for a in self.exprs_eq]:
            xcomp = x.complexity()
            for y in igen.pm(itertools.chain(self.zero_exprs(avoid = x), reg.zero_exprs())):
                x2 = (x + y).simplified()
                x2comp = x2.complexity()
                if x2comp < xcomp:
                    did = True
                    x.copy_(x2)
                    xcomp = x2comp
        
        for z in reg.empty_rvs():
            did = True
            self.substitute(z, Comp.empty())
        
        for z in self.empty_rvs():
            did = True
            self.substitute(z, Comp.empty())
            self.exprs_eq.append(Expr.H(z))
        
        if did:
            self.simplify_quick()
        
        return self
                    
     
    def simplify_redundant(self, reg = None, proc = None, full = True):
        write_pf_enabled = (PsiOpts.settings.get("proof_enabled", False) 
                            and PsiOpts.settings.get("proof_step_simplify", False))
        aux_relax = PsiOpts.settings.get("simplify_aux_relax", False)

        if write_pf_enabled:
            prevself = self.copy()
            red_reg = Region.universe()
        
                
        if reg is None:
            reg = Region.universe()
        
        #if self.isregtermpresent():
        #    return self
        
        allcompreal = self.allcompreal() + reg.allcompreal()
        
        aux = self.aux

        def preprocess(r):
            if proc is not None:
                r = proc(r)
            if aux_relax:
                r.aux += aux
                r.aux_strengthen()
                r.aux = Comp.empty()
            return r

        for i in range(len(self.exprs_ge) - 1, -1, -1):
            if PsiOpts.is_timer_ended():
                break
            t = self.exprs_ge[i]
            self.exprs_ge[i] = Expr.zero()
            cs = self.imp_intersection_noaux() & reg
            if not full:
                cs.remove_notcontained(t.allcomp() + allcompreal)
            cs = preprocess(cs)
            if not (cs <= (t >= 0)).check_plain():
                self.exprs_ge[i] = t
            elif write_pf_enabled:
                red_reg.exprs_ge.append(t)
        
        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        
        for i in range(len(self.exprs_eq) - 1, -1, -1):
            if PsiOpts.is_timer_ended():
                break
            t = self.exprs_eq[i]
            self.exprs_eq[i] = Expr.zero()
            cs = self.imp_intersection_noaux() & reg
            if not full:
                cs.remove_notcontained(t.allcomp() + allcompreal)
            cs = preprocess(cs)
            if not (cs <= (t == 0)).check_plain():
                self.exprs_eq[i] = t
            elif write_pf_enabled:
                red_reg.exprs_eq.append(t)
        
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        for i in range(len(self.exprs_gei) - 1, -1, -1):
            if PsiOpts.is_timer_ended():
                break
            t = self.exprs_gei[i]
            self.exprs_gei[i] = Expr.zero()
            cs = self.imp_flippedonly_noaux() & reg
            if not full:
                cs.remove_notcontained(t.allcomp() + allcompreal)
            cs = preprocess(cs)
            if not (cs <= (t >= 0)).check_plain():
                self.exprs_gei[i] = t
            elif write_pf_enabled:
                red_reg.exprs_gei.append(t)
        
        self.exprs_gei = [x for x in self.exprs_gei if not x.iszero()]
        
        for i in range(len(self.exprs_eqi) - 1, -1, -1):
            if PsiOpts.is_timer_ended():
                break
            t = self.exprs_eqi[i]
            self.exprs_eqi[i] = Expr.zero()
            cs = self.imp_flippedonly_noaux() & reg
            if not full:
                cs.remove_notcontained(t.allcomp() + allcompreal)
            cs = preprocess(cs)
            if not (cs <= (t == 0)).check_plain():
                self.exprs_eqi[i] = t
            elif write_pf_enabled:
                red_reg.exprs_eqi.append(t)
        
        self.exprs_eqi = [x for x in self.exprs_eqi if not x.iszero()]
        
        if False:
            if self.imp_present():
                t = self.imp_flippedonly()
                t.simplify_redundant(reg)
                self.exprs_gei = t.exprs_ge
                self.exprs_eqi = t.exprs_eq
            
        if write_pf_enabled:
            if not red_reg.isuniverse():
                pf = ProofObj.from_region(red_reg, c = "Remove redundant constraints")
                pf = ProofObj.from_region(self, c = "Result")
                PsiOpts.set_setting(proof_add = pf)
            
        return self
    
        
    def simplify_symm(self, symm_set, quick = False):
        """Simplify a region, assuming symmetry among variables in symm_set.
        """
        self.symm_sort(symm_set)
        self.simplify_quick()
        if not quick:
            self.simplify_redundant(proc = lambda t: t.symmetrized(symm_set, skip_simplify = True))
        
    def var_mi_only(self, v):
        return all(a.var_mi_only(v) for a in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi)

    def aux_hull_towards(self, v, tgt, reg = None):
        if self.imp_present():
            return False
        if not self.var_mi_only(v):
            return False

        if isinstance(reg, RegionOp):
            reg = reg.tosimple()
        if reg is None:
            reg = Region.universe()

        ege = [a for a in self.exprs_ge if a.ispresent(v)]
        eeq = [a for a in self.exprs_eq if a.ispresent(v)]

        eget = [a.substituted(v, tgt) for a in ege]
        eeqt = [a.substituted(v, tgt) for a in eeq]

        for e0, e1 in zip(eeq, eeqt):
            if not reg.implies(e0 == e1):
                return False
        
        sns = [[False] * len(ege), [False] * len(ege)]
        for i, (e0, e1) in enumerate(zip(ege, eget)):
            sns[0][i] = reg.implies(e0 <= e1)
            sns[1][i] = reg.implies(e0 >= e1)
        
        for i, (e0, e1, s0, s1) in enumerate(zip(ege, eget, sns[0], sns[1])):
            if s0 and s1:
                continue
            if not (s0 or s1):
                continue
            did = False
            lvls = [0] * len(ege)
            inc = Expr.zero()
            if s0:
                inc = e1 - e0
                lvls[i] = 1
                did = True
            else:
                inc = e0 - e1
                lvls[i] = -1
            inc.simplify()

            # print("TRY " + str(e0) + " " + str(e1) + " " + str(inc))
            bad = False
            for iz, (ez0, ez1, sz0, sz1) in enumerate(zip(ege, eget, sns[0], sns[1])):
                if iz == i:
                    continue
                if sz0 and sz1:
                    continue
                ezdiff = (ez1 - ez0).simplified()
                if sz0:
                    if reg.implies(ezdiff >= inc):
                        lvls[iz] = 1
                        did = True
                    else:
                        lvls[iz] = 0
                else:
                    if reg.implies(ezdiff >= -inc):
                        lvls[iz] = -1
                    else:
                        bad = True
                        break
                # print("  TO " + str(ez0) + " " + str(ez1) + " " + str(lvls[iz]))

            if bad or not did:
                continue
            
            t = Expr.real("#TMPVAR")
            for iz, (ez0, ez1, sz0, sz1) in enumerate(zip(ege, eget, sns[0], sns[1])):
                ez0 += lvls[iz] * t

            self.iand_norename(t >= 0)
            self.iand_norename(t <= inc)
            # print(self)
            with PsiOpts(simplify_aux_hull = False):
                self.eliminate(t)
            return True

        return False

    def simplify_aux_hull(self, reg = None):
        if self.aux.isempty():
            return self
        
        if self.imp_present():
            return self

        if isinstance(reg, RegionOp):
            reg = reg.tosimple()
        if reg is None:
            reg = Region.universe()

        did = True
        for it in range(100):
            did = False
            for a in (self.aux if it % 2 else reversed(self.aux)):
                ane = self.var_neighbors(a)
                for tgt in igen.subset(ane):
                    if self.aux_hull_towards(a, tgt, reg = reg):
                        did = True
                        break
                if did:
                    break
            if not did:
                break
        return self


    def simplify_aux_combine(self, reg = None):
        
        if self.aux.isempty():
            return self
        
        if self.imp_present():
            return self
            
        if reg is None:
            reg = Region.universe()
            
        did = True
        
        while did:
            if self.aux.isempty():
                break
            
            selfplain = self.copy()
            selfplain.aux = Comp.empty()
            selfplain &= reg

            did = False

            for i in range(len(self.aux)):
                for j in range(i + 1, len(self.aux)):
                    cs = self.copy()
                    cs.aux = Comp.empty()
                    cs.substitute(self.aux[i], self.aux[i] + self.aux[j])
                    cs.substitute(self.aux[j], self.aux[i] + self.aux[j])
                    if selfplain.implies(cs):
                        self.substitute(self.aux[j], self.aux[i])
                        did = True
                        break
                if did:
                    break
            
            if did:
                continue

            for i in range(len(self.aux)):
                for j in range(len(self.aux)):
                    if i == j:
                        continue
                    cs = self.copy()
                    cs.aux = Comp.empty()
                    cs.substitute(self.aux[j], Comp.empty())
                    cs.substitute(self.aux[i], self.aux[i] + self.aux[j])
                    if selfplain.implies(cs):
                        self.substitute(self.aux[j], Comp.empty())
                        did = True
                        break
                if did:
                    break
        
        return self
        

    def simplify_aux(self, reg = None, cases = None):
        
        if self.aux.isempty():
            return self
        
        if self.imp_present():
            return self
        
        leaveone = None
        if cases is not None:
            leaveone = True
            
        if reg is None:
            reg = Region.universe()
            
        did = True
        
        while did:
            if self.aux.isempty():
                break
            
            did = False
        
            index = IVarIndex()
            self.record_to(index)
            reg.record_to(index)
            
            taux = self.aux.copy()
            taux2 = taux.copy()
            tauxi = self.auxi.copy()
            self.aux = Comp.empty()
            self.auxi = Comp.empty()
            
            # print(self)
            
            
            for a in taux:
                if PsiOpts.is_timer_ended():
                    break
                a2 = Comp.rv(index.name_avoid(a.get_name()))
                cs = (self.consonly().substituted(a, a2) & reg) >> self.exists(a)
                #hint_aux_avoid = self.get_aux_avoid_list()
                hint_aux_avoid = [(a, a2)]
                # print(cs)
                for rr in cs.check_getaux_inplace_gen(hint_aux_avoid = hint_aux_avoid, leaveone = leaveone):
                    stype = iutil.signal_type(rr)
                    if stype == "":
                        # print(rr)
                        ar = a.copy()
                        Comp.substitute_list(ar, rr)
                        if ar == a:
                            continue
                        # print(self)
                        self.substitute(a, ar)
                        # print(self)
                        taux2 -= a
                        did = True
                        break
                    elif stype == "leaveone" and cases is not None:
                        ar = a.copy()
                        Comp.substitute_list(ar, rr[1])
                        if ar == a:
                            continue
                        tr = self.copy()
                        # tr.iand_norename(rr[2] <= 0)
                        tr.substitute(a, ar)
                        
                        tr.aux = taux2 - a
                        tr.auxi = tauxi.copy()
                        tr.simplify_quick()
                        tr.simplify_aux(reg, cases)
                        
                        cases.append(tr)
                        self.iand_norename(rr[2] >= 0)
                        did = True
                    
            self.aux = taux2
            self.auxi = tauxi
            
            if did:
                self.simplify_quick()
        
        return self
    
        
    def simplify_imp(self, reg = None):
        if not self.imp_present():
            return
    
    def simplify_expr_exhaust(self):
        for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi:
            x.simplify_exhaust()
        return self

    def expanded_cases_reduce(self, reg = None, skip_simplify = False):
        r = self.copy()
        cases = []
        r.simplify_aux(reg, cases)
        if len(cases) == 0:
            return r
        else:
            r2 = RegionOp.union([r] + cases)
            if not skip_simplify:
                return r2.simplified()
            else:
                return r2
        
    def expanded_cases(self, reg = None, leaveone = True, skip_simplify = False):
        
        if self.aux.isempty():
            return None
        
        if self.imp_present():
            return None
        
        cases = []
            
        if isinstance(reg, RegionOp):
            reg = reg.tosimple()
        if reg is None:
            reg = Region.universe()
            
        did = False
        index = IVarIndex()
        self.record_to(index)
        reg.record_to(index)
        
        # taux = self.aux.copy()
        # taux2 = taux.copy()
        # tauxi = self.auxi.copy()
        # self.aux = Comp.empty()
        # self.auxi = Comp.empty()
        
        rmap = index.calc_rename_map(self.aux)
        cs2 = self.copy()
        cs2.rename_map(rmap)
    
        s2 = self.consonly()
        s2.aux = Comp.empty()
        cs = (s2 & reg) >> cs2
        hint_aux_avoid = cs.get_aux_avoid_list()
        
        with PsiOpts(forall_multiuse_numsave = 0):
            for rr in cs.check_getaux_inplace_gen(hint_aux_avoid = hint_aux_avoid, leaveone = leaveone):
                stype = iutil.signal_type(rr)
                if stype == "":
                    continue
                
                    print(rr)
                    cs3 = cs2.copy()
                    Comp.substitute_list(cs3, rr, isaux = True)
                    cs3.eliminate(self.aux.inter(cs3.allcomp()))
                    cases.append(cs3)
                    
                elif stype == "leaveone":
                    print(str(rr[1]) + "  " + str(rr[2]))
                    cs3 = cs2.copy()
                    Comp.substitute_list(cs3, rr[1], isaux = True)
                    cs3.eliminate(self.aux.inter(cs3.allcomp()))
                    cases.append(cs3)
                    
                    cs2.iand_norename(rr[2] >= 0)
                    
                if PsiOpts.is_timer_ended():
                    break
        
        cs3 = cs2.copy()
        cs3.rename_map({b: a for a, b in rmap.items()})
        cases.append(cs3)
        
        
        if len(cases) == 1:
            return cases[0]
        else:
            r2 = RegionOp.union()
            for a in cases:
                r2 |= a
            if not skip_simplify:
                return r2.simplified()
            else:
                return r2
                
            
    def simplify_aux_empty(self, reg = None, skip_simplify = False):
        
        if self.aux.isempty():
            return self
        
        if self.imp_present():
            return self
        
        if isinstance(reg, RegionOp):
            reg = reg.tosimple()
        if reg is None:
            reg = Region.universe()
        
        cs = self.copy()
        cs.aux = Comp.empty()
        cs = cs & reg
        
        index_self = IVarIndex()
        cs.record_to(index_self)
        progs = []
        
        for taux in igen.subset(self.aux, minsize = 2):
            cs2 = self.copy()
            cs2.aux = Comp.empty()
            cs2.substitute(taux, Comp.empty())
            # if cs.implies(cs2):
            if cs.implies_saved(cs2, index_self, progs):
                self.substitute_aux(taux, Comp.empty())
                if not skip_simplify:
                    self.simplify_quick()
                self.simplify_aux_empty(reg, skip_simplify)
                return self
        
        return self
    
            
    def simplify_aux_recombine(self, reg = None, skip_simplify = False):
        
        if self.aux.isempty():
            return self
        
        if self.imp_present():
            return self
        
        if isinstance(reg, RegionOp):
            reg = reg.tosimple()
        if reg is None:
            reg = Region.universe()
        
        did = False
        index = IVarIndex()
        self.record_to(index)
        reg.record_to(index)
        
        
        rmap = index.calc_rename_map(self.aux)
        cs2 = self.copy()
        cs2.rename_map(rmap)
    
        s2 = self.consonly()
        s2.aux = Comp.empty()
        cs = (s2 & reg) >> cs2
        hint_aux_avoid = cs.get_aux_avoid_list()
        
        cmin = (len(self.aux), -len(self.aux))
        minlist = None
        
        with PsiOpts(auxsearch_leaveone_add_ineq = False):
            for rr in cs.check_getaux_inplace_gen(hint_aux_avoid = hint_aux_avoid):
                stype = iutil.signal_type(rr)
                if stype == "":
                    # print(rr)
                    
                    cnaux = 0
                    csize = 0
                    clist = []
                    cvar = Comp.empty()
                    for a in cs2.aux:
                        b = a.copy()
                        Comp.substitute_list(b, rr)
                        csize += len(b)
                        cvar += b
                        clist.append(b)
                    
                    for a in self.aux:
                        if cvar.ispresent(a):
                            cnaux += 1
                    
                    t = (cnaux, -csize)
                    if t < cmin:
                        cmin = t
                        minlist = clist
                    
                if PsiOpts.is_timer_ended():
                    break
                
                    
        if minlist is None:
            return self
        
        prevaux = self.aux.copy()
        self.rename_map(rmap)
        selfaux = self.aux.copy()
        for a, b in zip(selfaux, minlist):
            self.substitute_aux(a, b)
        self.eliminate(prevaux.inter(self.allcomp()))
        
        if not skip_simplify:
            self.simplify_quick()
            
        return self
    
    def sort(self):
        self.exprs_ge.sort(key = lambda x: x.sorting_priority())
        self.exprs_eq.sort(key = lambda x: x.sorting_priority())
        self.exprs_gei.sort(key = lambda x: x.sorting_priority())
        self.exprs_eqi.sort(key = lambda x: x.sorting_priority())
        
    def simplify(self, reg = None, zero_group = 0, **kwargs):
        """Simplify a region in place
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """

        if kwargs:
            r = None
            with PsiOpts(**{"simplify_" + key: val for key, val in kwargs.items()}):
                r = self.simplify(reg, zero_group)
            return r

        if not PsiOpts.settings.get("simplify_enabled", False):
            return self
        
        if reg is None:
            reg = Region.universe()
            
        nit = PsiOpts.settings.get("simplify_num_iter", 1)

        for it in range(nit):
            self.simplify_quick(reg, zero_group)
            
            if not PsiOpts.settings.get("simplify_quick", False):
                if PsiOpts.settings.get("simplify_remove_missing_aux", False):
                    self.remove_missing_aux()
                if PsiOpts.settings.get("simplify_aux_commonpart", False):
                    self.simplify_aux_commonpart(reg, 
                            maxlen = PsiOpts.settings.get("simplify_aux_xor_len", False))
                
                
                if PsiOpts.settings.get("simplify_aux_empty", False):
                    self.simplify_aux_empty()
                
                if PsiOpts.settings.get("simplify_aux_combine", False):
                    self.simplify_aux_combine(reg)
                
                if PsiOpts.settings.get("simplify_aux_recombine", False):
                    self.simplify_aux_recombine()
                elif PsiOpts.settings.get("simplify_aux", False):
                    self.simplify_aux()
                    
                if PsiOpts.settings.get("simplify_redundant", False):
                    self.simplify_redundant(reg, full = PsiOpts.settings.get("simplify_redundant_full", False))
                    
                if PsiOpts.settings.get("simplify_aux_hull", False):
                    self.simplify_aux_hull(reg)
                    
                if PsiOpts.settings.get("simplify_bayesnet", False):
                    self.simplify_bayesnet(reg)
                    
                if PsiOpts.settings.get("simplify_expr_exhaust", False):
                    self.simplify_expr_exhaust()
                    
                if PsiOpts.settings.get("simplify_pair", False):
                    self.simplify_pair(reg)
                    
                if PsiOpts.settings.get("simplify_remove_missing_aux", False):
                    self.remove_missing_aux()
        
        if PsiOpts.settings.get("simplify_sort", False):
            self.sort()
        
        return self
    
    
    def simplify_union(self, reg = None):
        self.simplify(reg)
        return self
    
    
    def simplified_quick(self, reg = None, zero_group = 0):
        """Returns the simplified region
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        r = self.copy()
        r.simplify_quick(reg, zero_group)
        return r
    
    def simplified(self, reg = None, zero_group = 0, **kwargs):
        """Returns the simplified region
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        r = self.copy()
        r.simplify(reg, zero_group, **kwargs)
        return r
    
    def remove_trivial(self):
        self.exprs_gei = [x for x in self.exprs_gei if not x.isnonneg()]
        self.exprs_eqi = [x for x in self.exprs_eqi if not (x.isnonneg() and x.isnonpos())]
        self.exprs_ge = [x for x in self.exprs_ge if not x.isnonneg()]
        self.exprs_eq = [x for x in self.exprs_eq if not (x.isnonneg() and x.isnonpos())]
    
    def removed_trivial(self):
        r = self.copy()
        r.remove_trivial()
        return r
    
    def get_sum_seq(self):
        exprs = list(self.exprs_ge + self.exprs_eq)
        c = Expr.zero()
        r0 = []
        r1 = []
        while exprs:
            minc = None
            minx = None
            mincomp = 1e20
            mini = 0
            for i, x in enumerate(exprs):
                t = None
                with PsiOpts(proof_enabled = False):
                    t = (c + x).simplified()
                tcomp = t.complexity()
                if tcomp < mincomp:
                    mincomp = tcomp
                    minc = t
                    minx = x
                    mini = i
                
            r0.append(minx)
            r1.append(minc)
            c = minc
            exprs.pop(mini)
            
            # print(i)
            # print(exprs)
            # print(minx)
            # print(minc)
        
        return (r0, r1)
        
    
    def get_ic(self, skip_simplify = False):
        cs = self
        if not skip_simplify:
            cs = self.simplified_quick(zero_group = 2)
            
        # icexpr = Expr.zero()
        # for x in cs.exprs_ge:
        #     if x.isnonpos():
        #         icexpr += x
        # for x in cs.exprs_eq:
        #     if x.isnonpos():
        #         icexpr += x
        #     elif x.isnonneg():
        #         icexpr -= x
        # return icexpr
            
        exprs = []
        r = Expr.zero()
        for x in cs.exprs_ge:
            if x.isnonpos():
                exprs.append(x)
                
        for x in cs.exprs_eq:
            if x.isnonpos() or x.isnonneg():
                exprs.append(x)
                
        for x in exprs:
            for a, c in x.terms:
                if a.isic2():
                    r += Expr.fromterm(a)
        return r
    
    
    def remove_ic(self):
        exprs = []
        r = Expr.zero()
        for x in self.exprs_ge:
            if x.isnonpos():
                exprs.append(x)
                
        for x in self.exprs_eq:
            if x.isnonpos() or x.isnonneg():
                exprs.append(x)
                
        for x in exprs:
            tterms = []
            for a, c in x.terms:
                if a.isic2():
                    r += Expr.fromterm(a)
                else:
                    tterms.append((a, c))
            x.terms = tterms
            x.mhash = None
            
                
        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        return r
    
    
    def get_bayesnet(self, roots = None, semigraphoid_iter = None, get_list = False, skip_simplify = False):
        """Return a Bayesian network containing the conditional independence
        conditions in this region
        """
        
        if semigraphoid_iter is None:
            semigraphoid_iter = PsiOpts.settings["bayesnet_semigraphoid_iter"]
        
        icexpr = self.get_ic(skip_simplify)
        
        if semigraphoid_iter > 0:
            icexpr += self.completed_semigraphoid_ic(max_iter = semigraphoid_iter)
            
        if get_list:
            return BayesNet.from_ic_list(icexpr, roots = roots)
        else:
            return BayesNet.from_ic(icexpr, roots = roots).tsorted()
    
    def graph(self, **kwargs):
        """Return the Bayesian network among the random variables as a 
        graphviz digraph that can be displayed in the console.
        """
        return self.get_bayesnet().graph(**kwargs)
    
    def get_bayesnet_imp(self, skip_simplify = False):
        """Return a Bayesian network containing the conditional independence
        conditions in this region
        """
        cs = self
        if not skip_simplify:
            cs = self.simplified_quick(zero_group = 2)
        icexpr = Expr.zero()
        for x in cs.exprs_gei:
            if x.isnonpos():
                icexpr += x
        for x in cs.exprs_eqi:
            if x.isnonpos():
                icexpr += x
            elif x.isnonneg():
                icexpr -= x
        return BayesNet.from_ic(icexpr).tsorted()
    
    def get_markov(self):
        """Get Markov chains as a list of lists.
        """
        bnets = self.get_bayesnet(get_list = True)
        r = []
        for bnet in bnets:
            r += bnet.get_markov()
        return r
        
    def table(self, *args, skip_cons = False, plot = True, use_latex = None, **kwargs):
        """Plot the information diagram as a Karnaugh map.
        """
        
        if use_latex is None:
            use_latex = PsiOpts.settings["venn_latex"]
        
        imp_r = self.imp_flippedonly_noaux()
        if not imp_r.isuniverse():
            return imp_r.table(self.consonly(), *args, skip_cons, plot, **kwargs)
        
        ceps = PsiOpts.settings["eps"]
        index = IVarIndex()
        
        cmodel = None
        for a in args:
            if isinstance(a, ConcModel) or isinstance(a, LinearProg):
                cmodel = a
        
        for a in args:
            if isinstance(a, Comp):
                a.record_to(index)
                
        self.get_bayesnet().allcomp().record_to(index)
        if cmodel is not None:
            if isinstance(cmodel, ConcModel):
                cmodel.get_bayesnet().allcomp().record_to(index)
        
        for a in args:
            if isinstance(a, Expr) or isinstance(a, Region):
                a.record_to(index)
        
        cs = self
        if cmodel is not None:
            if isinstance(cmodel, ConcModel):
                cs = cs & cmodel.get_bayesnet().get_region()
        cs = cs.imp_flipped()
        
        cs.record_to(index)
        comprv = index.comprv
        
        r = CellTable(comprv)
        progs = []
        # progs.append(self.init_prog(index, lptype = LinearProgType.H))
        progs.append(cs.init_prog(index))
        
        creg = Region.universe()
        
        for mask in range(1, 1 << len(comprv)):
            # ch = H(comprv.from_mask(mask) | comprv.from_mask(((1 << len(comprv)) - 1) ^ mask))
            ch = I(alland(comprv[i] for i in range(len(comprv)) if mask & (1 << i)) 
                   | comprv.from_mask(((1 << len(comprv)) - 1) ^ mask))
            ispos = ch.isnonneg() or cs.implies_ineq_prog(index, progs, ch, ">=")
            isneg = cs.implies_ineq_prog(index, progs, -ch, ">=")
            # print(str(ch) + "  " + str(ispos) + "  " + str(isneg))
            if ispos and isneg:
                r.set_attr(mask, "enabled", False)
                if not skip_cons:
                    creg.iand_norename(ch == 0)
            elif ispos:
                if cmodel is None:
                    r.set_attr(mask, "ispos", True)
                if not skip_cons and not ch.isnonneg():
                    creg.iand_norename(ch >= 0)
            elif isneg:
                if cmodel is None:
                    r.set_attr(mask, "isneg", True)
                if not skip_cons:
                    creg.iand_norename(ch <= 0)
                    
            if cmodel is not None:
                r.set_attr(mask, "cval", cmodel[ch])
                
        
        cnexpr = 0
        
        cargs = []
        if not skip_cons:
            for cexpr in self.exprs_ge:
                if not creg.implies(cexpr >= 0):
                    cargs.append(cexpr >= 0)
            for cexpr in self.exprs_eq:
                if not creg.implies(cexpr == 0):
                    cargs.append(cexpr == 0)
        
        cargs += args
        
        for a in cargs:
            exprlist = []
            if isinstance(a, Expr):
                exprlist = [(a, None)]
            elif isinstance(a, Region):
                exprlist = [(b, ">=") for b in a.exprs_ge] + [(b, "==") for b in a.exprs_eq]
            for b, sn in exprlist:
                if sn == ">=":
                    r.add_expr(b >= 0, None if cmodel is None else cmodel[b >= 0])
                elif sn == "==":
                    r.add_expr(b == 0, None if cmodel is None else cmodel[b == 0])
                else:
                    r.add_expr(b, None if cmodel is None else cmodel[b])
                maskvals = [0.0 for mask in range(1 << len(comprv))]
                for v, c in b.terms:
                    if v.get_type() == TermType.IC:
                        xmasks = [comprv.get_mask(x) for x in v.x]
                        zmask = comprv.get_mask(v.z)
                        for mask in range(1, 1 << len(comprv)):
                            if mask & zmask:
                                continue
                            if any(not(mask & xmask) for xmask in xmasks):
                                continue
                            maskvals[mask] += c
                            # print(mask)
                            
                for mask in range(1, 1 << len(comprv)):
                    if abs(maskvals[mask]) > ceps:
                        r.set_expr_val(mask, maskvals[mask])
                        # r.set_attr(mask, "val_" + str(cnexpr), maskvals[mask])
                cnexpr += 1
                
        # r.nexpr = cnexpr
        
        if plot:
            r.plot(use_latex = use_latex, **kwargs)
        else:
            return r
    
    
    def venn(self, *args, style = None, **kwargs):
        """Plot the information diagram as a Venn diagram.
        Can handle up to 5 random variables (uses Branko Grunbaum's Venn diagram for n=5).
        """
        if style is None:
            style = ""
        style = "venn,blend," + style
        return self.table(*args, style = style, **kwargs)
    
    
    def eliminate_term_eq(self, w):
        
        ee = None
        
        if ee is None:
            for x in self.exprs_eqi:
                c = x.get_coeff(w)
                if abs(c) > PsiOpts.settings["eps"]:
                    ee = (x * (-1.0 / c)).removed_term(w)
                    x.setzero()
                    break
        if ee is None:
            for x in self.exprs_eq:
                c = x.get_coeff(w)
                if abs(c) > PsiOpts.settings["eps"]:
                    ee = (x * (-1.0 / c)).removed_term(w)
                    x.setzero()
                    break
                
        
        if ee is not None:
            self.substitute(Expr.fromterm(w), ee)
            self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
            self.exprs_eqi = [x for x in self.exprs_eqi if not x.iszero()]
            return True
        
        return False
        
        
    def get_lb_ub_eq(self, w):
        """
        Get lower bounds, upper bounds and equality constraints about w.

        Parameters
        ----------
        w : Expr
            The real variable of interest.

        Returns
        -------
        tuple
            A tuple of 3 lists of Expr: Lower bounds, upper bounds and 
            equality constraints about w.

        """
        
        if isinstance(w, Expr):
            w = Term.fromcomp(w.allcomp())
        
        el = []
        er = []
        ee = []
        
        for x in self.exprs_ge:
            c = x.get_coeff(w)
            if abs(c) <= PsiOpts.settings["eps"]:
                pass
            elif c > 0:
                er.append((x * (-1.0 / c)).removed_term(w))
            else:
                el.append((x * (-1.0 / c)).removed_term(w))
                
        for x in self.exprs_eq:
            c = x.get_coeff(w)
            if abs(c) <= PsiOpts.settings["eps"]:
                pass
            else:
                ee.append((x * (-1.0 / c)).removed_term(w))
        
        return (er, el, ee)
        
        
        
    def eliminate_term(self, w, forall = False):
        verbose = PsiOpts.settings.get("verbose_eliminate", False)
        
        el = []
        er = []
        ee = []
        
        if verbose:
            print("=========== Eliminate ===========")
            print(w)
        
        for x in self.exprs_gei:
            c = x.get_coeff(w)
            if abs(c) <= PsiOpts.settings["eps"]:
                x.remove_term(w)
            elif c > 0:
                er.append((x * (-1.0 / c)).removed_term(w))
                x.setzero()
            else:
                el.append((x * (-1.0 / c)).removed_term(w))
                x.setzero()
                
        for x in self.exprs_eqi:
            c = x.get_coeff(w)
            if abs(c) <= PsiOpts.settings["eps"]:
                x.remove_term(w)
            else:
                ee.append((x * (-1.0 / c)).removed_term(w))
                x.setzero()
        
        elni = len(el)
        erni = len(er)
        eeni = len(ee)
        
        if not forall and elni + erni + eeni > 0:
            self.setuniverse()
            return self
        
        for x in self.exprs_ge:
            c = x.get_coeff(w)
            if abs(c) <= PsiOpts.settings["eps"]:
                x.remove_term(w)
            elif c > 0:
                er.append((x * (-1.0 / c)).removed_term(w))
                x.setzero()
            else:
                el.append((x * (-1.0 / c)).removed_term(w))
                x.setzero()
                
        for x in self.exprs_eq:
            c = x.get_coeff(w)
            if abs(c) <= PsiOpts.settings["eps"]:
                x.remove_term(w)
            else:
                ee.append((x * (-1.0 / c)).removed_term(w))
                x.setzero()
        
        
        if len(ee) > 0:
            
            if eeni == 0:
                for i in range(elni):
                    x = el[i]
                    for j in range(erni):
                        y = er[j]
                        self.exprs_gei.append(x - y)
            
            for i in range(len(el)):
                x = el[i]
                if i < elni and 0 < eeni:
                    self.exprs_gei.append(x - ee[0])
                else:
                    self.exprs_ge.append(x - ee[0])
                    
            for j in range(len(er)):
                y = er[j]
                if j < erni and 0 < eeni:
                    self.exprs_gei.append(ee[0] - y)
                else:
                    self.exprs_ge.append(ee[0] - y)
                    
            for i in range(1, len(ee)):
                x = ee[i]
                if i < eeni:
                    self.exprs_eqi.append(x - ee[0])
                else:
                    self.exprs_eq.append(x - ee[0])
                   
            
        else:
            for i in range(len(el)):
                x = el[i]
                for j in range(len(er)):
                    y = er[j]
                    if i < elni and j < erni:
                        self.exprs_gei.append(x - y)
                    else:
                        self.exprs_ge.append(x - y)
            
        if verbose:
            print("===========    To     ===========")
            print(self)
            
        return self
            
        
    def eliminate_toreal(self, w, forall = False):
        verbose = PsiOpts.settings.get("verbose_eliminate_toreal", False)
        
        if verbose:
            print("========== elim real ========")
            print(self)
            print("==========   var    =========")
            print(w)
            
        reals = self.allcompreal()
        self.iand_norename(self.get_prog_region(toreal = w, toreal_only = True))
        self.imp_flip()
        self.iand_norename(self.get_prog_region(toreal = w, toreal_only = True))
        self.imp_flip()
        self.remove_present(w)
        reals = self.allcompreal() - reals
        
        if verbose:
            print("==========    to    =========")
            print(self)
            
        for a in reals.varlist:
            self.eliminate_term(Term.fromcomp(Comp([a])), forall = forall)
            self.simplify_quick()
            if verbose:
                print("========== elim " + str(a) + " =========")
                print(self)
            
        return self
        
    
    def eliminate_toreal_rays(self, w):
        
        cs = self.consonly().imp_flipped()
        index = IVarIndex()
        cs.record_to(index)
        
        r = cs.init_prog(index, lptype = LinearProgType.H).get_region_elim_rays(w)
        self.exprs_ge = r.exprs_ge
        self.exprs_eq = r.exprs_eq
        return self
        
    def remove_aux(self, w):
        self.aux -= w
        self.auxi -= w
        
    def remove_missing_aux(self):
        #return
        taux = self.aux
        self.aux = Comp.empty()
        tauxi = self.auxi
        self.auxi = Comp.empty()
        allcomp = self.allcomprv()
        self.aux = taux.inter(allcomp)
        self.auxi = tauxi.inter(allcomp)
        
    @staticmethod
    def get_allcomp(w):
        if isinstance(w, CompArray) or isinstance(w, ExprArray):
            w = w.allcomp()
        
        if isinstance(w, list):
            w = sum((a.allcomp() for a in w), Comp.empty())
        
        if isinstance(w, Expr):
            w = w.allcomp()
        
        return w

    def eliminate(self, w, reg = None, toreal = False, forall = False, quick = False):
        """Fourier-Motzkin elimination, in place. 
        w is the Expr object with the real variables to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        
        w = Region.get_allcomp(w)
        
        # if isinstance(w, Comp):
        #     w = Expr.H(w)
        
        if not toreal and not forall and not self.auxi.isempty() and any(v.get_type() == IVarType.RV for v in w.allcomp()):
            return RegionOp.inter([self]).eliminate(w, reg, toreal, forall)
        
        #self.simplify_quick(reg)
        
        if toreal and PsiOpts.settings["eliminate_rays"]:
            w2 = w
            w = Comp.empty()
            toelim = Comp.empty()
            
            
            for v in w2.allcomp():
                if toreal or v.get_type() == IVarType.REAL:
                    w += v
                elif v.get_type() == IVarType.RV:
                    toelim += v
                    
            if not toelim.isempty():
                self.eliminate_toreal_rays(toelim)
        
        toelim = Comp.empty()
        
        simplify_needed = False
        for v in w.allcomp():
            if v.get_type() == IVarType.REAL or toreal:
                if simplify_needed:
                    if quick:
                        self.simplify_quick(reg)
                    else:
                        with PsiOpts(simplify_redundant_full = False, 
                                     simplify_aux_commonpart = False, 
                                     simplify_aux = False,
                                     simplify_bayesnet = False):
                            self.simplify(reg)
                            
                if v.get_type() == IVarType.REAL:
                    self.eliminate_term(Term.fromcomp(v), forall = forall)
                else:
                    self.eliminate_toreal(v, forall = forall)
                    
                simplify_needed = True
                
            else:
                if forall:
                    self.auxi += v
                else:
                    self.aux += v
        
        
        if simplify_needed:
            if quick:
                self.simplify_quick(reg)
            else:
                self.simplify(reg)
        
        
        return self
        
    def eliminate_quick(self, w, reg = None, toreal = False, forall = False):
        """Fourier-Motzkin elimination, in place. 
        w is the Expr object with the real variables to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        
        return self.eliminate(w, reg = reg, toreal = toreal, forall = forall, quick = True)
        
    def eliminated(self, w, reg = None, toreal = False, forall = False):
        """Fourier-Motzkin elimination, return region after elimination. 
        w is the Expr object with the real variable to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        r = self.copy()
        r.eliminate(w, reg, toreal, forall)
        return r
        
    def eliminated_quick(self, w, reg = None, toreal = False, forall = False):
        """Fourier-Motzkin elimination, return region after elimination. 
        w is the Expr object with the real variable to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        r = self.copy()
        r.eliminate_quick(w, reg, toreal, forall)
        return r
        
    def exists(self, w = None, reg = None, toreal = False):
        """Alias of eliminated.
        """
        if w is None:
            w = self.allcomprv_noaux()
        r = self.copy()
        r = r.eliminate(w, reg, toreal, forall = False)
        return r
        
    def exists_quick(self, w = None, reg = None, toreal = False):
        """Alias of eliminated_quick.
        """
        if w is None:
            w = self.allcomprv_noaux()
        r = self.copy()
        r = r.eliminate_quick(w, reg, toreal, forall = False)
        return r
        
    def forall(self, w = None, reg = None, toreal = False):
        """Region of intersection for all variable w.
        """
        if w is None:
            w = self.allcomprv_noaux()
        r = self.copy()
        r = r.eliminate(w, reg, toreal, forall = True)
        return r
        
    def forall_quick(self, w = None, reg = None, toreal = False):
        """Region of intersection for all variable w.
        """
        if w is None:
            w = self.allcomprv_noaux()
        r = self.copy()
        r = r.eliminate_quick(w, reg, toreal, forall = True)
        return r
        
    def projected(self, w = None, reg = None, quick = False):
        """Project the region to real variables in the list w by eliminating all
        other real variables. E.g. for a Region r with 3 real variables R1,R2,R3,
        r.projected([R1, R2]) keeps R1,R2 and eliminates R3, and 
        r.projected(S == R1+R2+R3) projects the region to the diagonal S == R1+R2+R3
        (i.e., introduces S and eliminates R1,R2,R3).
        """
        if w is None:
            w = []
        if isinstance(w, Expr):
            w = list(w)
        if isinstance(w, Region):
            w = [w]
        compreal = self.allcompreal()
        
        r = self.copy()
        for a in w:
            if isinstance(a, Expr):
                compreal -= a.allcomp()
            elif isinstance(a, Region):
                a2 = a
                if a2.isregtermpresent():
                    a2 = a2.flattened(minmax_elim = True)
                    
                if r.get_type() == RegionType.NORMAL and a2.get_type() == RegionType.NORMAL:
                    r.iand_norename(a2)
                else:
                    r &= a2
        
        # print(r)
        
        if quick:
            r = r.eliminate_quick(compreal, reg)
        else:
            r = r.eliminate(compreal, reg)
        return r
        

    def splice_rate(self, w0, w1):
        """Allow decreasing the real variable w0 (Expr) and increasing w1 (Expr) by the same ammount.
        """
        if isinstance(w0, Expr):
            w0 = [w0]
        if isinstance(w1, Expr):
            w1 = [w1]
        t = Expr.real("#TMPVAR")
        for a0 in w0:
            self.substitute(a0, a0 + t)
        for a1 in w1:
            self.substitute(a1, a1 - t)
        for a0 in w0:
            self &= a0 >= 0
        self &= t >= 0
        self.eliminate(t)
        return self

    def spliced_rate(self, w0, w1):
        """Allow decreasing the real variable w0 (Expr) and increasing w1 (Expr) by the same ammount.
        """
        r = self.copy()
        r.splice_rate(w0, w1)
        return r


    def marginal_eliminate(self, w):
        """Set input, in place. 
        Denote the RV's in w (type Comp) as input variables, and the region
        is the union over distributions of w.
        """
        self.inp += w
        return self
        
    def marginal_exists(self, w):
        """Set input, return result. 
        Denote the RV's in w (type Comp) as input variables, and the region
        is the union over distributions of w.
        """
        r = self.copy()
        r.marginal_eliminate(w)
        return r
        
    def marginal_forall(self, w):
        """Set input, return result. 
        Currently we do not differentiate between input distribution union
        and intersection. May change in the future.
        """
        r = self.copy()
        r.marginal_eliminate(w)
        return r
        
    def kernel_eliminate(self, w):
        """Set output, in place. 
        Denote the RV's in w (type Comp) as output variables, and the region
        is the union over channels leading to w with same marginal on w.
        """
        self.oup += w
        return self
        
    def kernel_exists(self, w):
        """Set output, return result. 
        Denote the RV's in w (type Comp) as output variables, and the region
        is the union over channels leading to w with same marginal on w.
        """
        r = self.copy()
        r.kernel_eliminate(w)
        return r
        
    def kernel_forall(self, w):
        """Set output, return result. 
        Currently we do not differentiate between channel union
        and intersection. May change in the future.
        """
        r = self.copy()
        r.kernel_eliminate(w)
        return r
    
        
    def issymmetric(self, w, quick = False):
        """Check whether region is symmetric with respect to the variables in w
        """
        csstr = ""
        if quick:
            csstr = self.tostring(tosort = True)
        for i in range(1, len(w)):
            t = self.copy()
            tvar = Comp.rv("SYMM_TMP")
            t.substitute(w[i], tvar)
            t.substitute(w[0], w[i])
            t.substitute(tvar, w[0])
            if quick:
                if t.tostring(tosort = True) != csstr:
                    return False
            else:
                if not t.implies(self):
                    return False
                if not self.implies(t):
                    return False
        return True
    
    
    def __imul__(self, other):
        compreal = self.allcompreal()
        for a in compreal.varlist:
            self.substitute(Expr.fromcomp(Comp([a])), Expr.fromcomp(Comp([a])) / other)
        self.simplify_quick()
        return self
        
    def __mul__(self, other):
        r = self.copy()
        r *= other
        return r
        
    def __rmul__(self, other):
        r = self.copy()
        r *= other
        return r
        
    def __itruediv__(self, other):
        self *= 1.0 / other
        return self
        
    def __truediv__(self, other):
        r = self.copy()
        r *= 1.0 / other
        return r
    
    def distribute(self):
        return self
    
    def sum_minkowski(self, other):
        """Minkowski sum of two regions with respect to their real variables.
        """
        if other.get_type() != RegionType.NORMAL:
            return other.sum_minkowski(self)
        
        cs = self.copy()
        co = other.copy()
        param_real = cs.allcomprealvar().inter(co.allcomprealvar())
        param_real_expr = Expr.zero()
        
        for a in param_real.varlist:
            newname = "TENSOR_TMP_" + a.name
            cs.substitute(Expr.real(a.name), Expr.real(a.name) - Expr.real(newname))
            co.substitute(Expr.real(a.name), Expr.real(newname))
            param_real_expr += Expr.real(newname)
        
        if PsiOpts.settings["tensorize_simplify"]:
            return (cs & co).eliminated(param_real_expr)
        else:
            return (cs & co).eliminated_quick(param_real_expr)
        
    def __add__(self, other):
        return self.sum_minkowski(other)
        
    def __iadd__(self, other):
        return self + other
        
    
    def negate(self):
        return ~RegionOp.union([self])
        
    def __invert__(self):
        return ~RegionOp.union([self])
    
    def copy_rename(self):
        """Return a copy with renamed variables, together with map from old name to new.
        """
        namemap = {}
        r = self.copy()
        
        index = IVarIndex()
        self.record_to(index)
        
        param_rv = index.comprv.copy()
        param_real = index.compreal.copy()
        
        for a in param_rv.varlist:
            name1 = index.name_avoid(a.name)
            index.record(Comp.rv(name1))
            namemap[a.name] = name1
            r.rename_var(a.name, name1)
            
        for a in param_real.varlist:
            name1 = index.name_avoid(a.name)
            index.record(Comp.real(name1))
            namemap[a.name] = name1
            r.rename_var(a.name, name1)
            
        return (r, namemap)
        
    def tensorize(self, reg_subset = None, chan_cond = None, nature = None, timeshare = False, hint_aux = None, same_dist = False):
        """Check whether region tensorizes, return auxiliary RVs if tensorizes. 
        chan_cond : The condition on the channel (e.g. degraded broadcast channel)
        """
        for rr in self.tensorize_gen(reg_subset = reg_subset, chan_cond = chan_cond,
                        nature = nature, timeshare = timeshare, hint_aux = hint_aux, same_dist = same_dist):
            if iutil.signal_type(rr) == "":
                return rr
        return None
        
    def tensorize_gen(self, reg_subset = None, chan_cond = None, nature = None, timeshare = False, hint_aux = None, same_dist = False):
        """Check whether region tensorizes, yield all auxiliary RVs if tensorizes. 
        chan_cond : The condition on the channel (e.g. degraded broadcast channel)
        """
        r2, namemap = self.copy_rename()
        rx = None
        
        if reg_subset is None:
            rx = self.copy()
        else:
            rx = reg_subset.copy()
        
        if chan_cond is None:
            chan_cond = Region.universe()
        chan_cond2 = chan_cond.copy()
        chan_cond2.rename_map(namemap)
        
        if nature is None:
            nature = Comp.empty()
        nature2 = nature.copy()
        nature2.rename_map(namemap)
        
        index = IVarIndex()
        self.record_to(index)
        #rx.record_to(index)
        
        param_rv = index.comprv - self.getauxall()
        param_real = index.compreal
        param_rv_map = Comp.empty()
        param_real_expr = Expr.zero()
        
        for a in param_rv.varlist:
            rx.substitute(Comp.rv(a.name), Comp.rv(a.name) + Comp.rv(namemap[a.name]))
            param_rv_map += Comp.rv(namemap[a.name])
        
        rsum = None
        
        if timeshare:
            rsum = self.sum_entrywise(r2)
            for a in param_real.varlist:
                rx.substitute(Expr.real(a.name), Expr.real(namemap[a.name]))
                rsum.substitute(Expr.real(a.name), Expr.zero())
            
        else:
            for a in param_real.varlist:
                r2.substitute(Expr.real(namemap[a.name]), Expr.real(namemap[a.name]) - Expr.real(a.name))
                rx.substitute(Expr.real(a.name), Expr.real(namemap[a.name]))
                param_real_expr += Expr.real(a.name)
            
            rsum = self.copy()
            rsum.iand_norename(r2)
            if PsiOpts.settings["tensorize_simplify"]:
                rsum = rsum.eliminated(param_real_expr)
            else:
                rsum = rsum.eliminated_quick(param_real_expr)
            
        
        if rsum.get_type() == RegionType.NORMAL:
            rsum.aux = self.aux.interleaved(r2.aux)
                
        for a in param_real.varlist:
            rx.substitute(Expr.real(namemap[a.name]), Expr.real(a.name))
            rsum.substitute(Expr.real(namemap[a.name]), Expr.real(a.name))
            
        #rx.rename_avoid(chan_cond)
        
        chan_cond.aux_avoid_from(rx)
        rx &= chan_cond
        chan_cond2.aux_avoid_from(rx)
        rx &= chan_cond2
        
        
        chan_cond_comp = chan_cond.allcomprv()
        chan_cond2_comp = chan_cond2.allcomprv()
        for i in range(chan_cond_comp.size()):
            namemap[chan_cond_comp.varlist[i].name] = chan_cond2_comp.varlist[i].name
        
        rx.iand_norename(Expr.Ic(self.allcomprv() - self.inp - self.oup - self.aux - self.auxi,
                      r2.allcomprv() - r2.oup - r2.aux - r2.auxi,
                      self.inp) == 0)
        rx.iand_norename(Expr.Ic(self.inp,
                      r2.allcomprv() - r2.inp - r2.oup - r2.aux - r2.auxi,
                      r2.inp) == 0)
        
        if not nature.isempty():
            rx.iand_norename(Expr.I(nature, nature2) == 0)
        
        hint_pair = []
        for (key, value) in namemap.items():
            hint_pair.append((Comp.rv(key), Comp.rv(value)))
        
        if same_dist:
            rx.iand_norename(eqdist(param_rv, param_rv_map))
        
        for rr in rx.implies_getaux_gen(rsum, hint_pair = hint_pair, hint_aux = hint_aux):
            yield rr
        
        
    def check_converse(self, reg_subset = None, chan_cond = None, nature = None, hint_aux = None):
        """Check whether self is the capacity region of the operational region. 
        reg_subset, return auxiliary RVs if true. 
        chan_cond : The condition on the channel (e.g. degraded broadcast channel).
        """
        return self.tensorize(reg_subset, chan_cond, nature, True, hint_aux = hint_aux)
        
    def check_converse_gen(self, reg_subset = None, chan_cond = None, nature = None, hint_aux = None):
        """Check whether self is the capacity region of the operational region. 
        reg_subset, yield all auxiliary RVs if true. 
        chan_cond : The condition on the channel (e.g. degraded broadcast channel).
        """
        for rr in self.tensorize_gen(reg_subset, chan_cond, nature, True, hint_aux = hint_aux):
            yield rr
        
    def __xor__(self, other):
        return self.eliminated(other)
        
    def __ixor__(self, other):
        return self.eliminate(other)
        
        
    def isfeasible(self):
        """Whether this region is feasible.
        """
        return not self.implies(Expr.one() <= 0)
        
    def __bool__(self):
        return self.check()
        
    def __call__(self):
        return self.check()
    
    def assume(self):
        """Assume this region is true in the current context.
        """
        PsiOpts.set_setting(truth_add = self)
    
    def assumed(self):
        """Create a context where this region is assumed to be true.
        Use "with region.assumed(): ..."
        """
        return PsiOpts(truth_add = self)
    
    def bound(self, expr, var, sgn = 0, minsize = 1, maxsize = 3, coeffmode = 1, skip_simplify = False):
        """Automatically discover bounds on expr in terms of variables in var. 
        Parameters:
            sgn       : Set to 1 for upper bound, -1 for lower bound, 0 for both.
            minsize   : Minimum number of terms in bound.
            maxsize   : Maximum number of terms in bound.
            coeffmode : Set to 0 to only allow positive terms.
                        Set to 1 to allow positive/negative terms, but not all negative.
                        Set to 2 to allow positive/negative terms.
            skip_simplify : Set to True to skip final simplification.
        """
        
        if sgn == 0:
            r = (self.bound(expr, var, sgn = 1, minsize = minsize, maxsize = maxsize, 
                              coeffmode = coeffmode, skip_simplify = skip_simplify)
                    & self.bound(expr, var, sgn = -1, minsize = minsize, maxsize = maxsize, 
                              coeffmode = coeffmode, skip_simplify = skip_simplify))
            
            if not skip_simplify:
                r.simplify_quick()
                
            return r
        
        varreal = []
        varrv = []
        
        if isinstance(var, list):
            for v in var:
                if isinstance(v, Expr):
                    varreal.append(v)
                else:
                    varrv.append(v)
        else:
            for v in var.allcomp():
                if v.get_type() == IVarType.REAL:
                    varreal.append(Expr.fromcomp(v))
                elif v.get_type() == IVarType.RV:
                    varrv.append(v)
        
        fcn = None
        if sgn > 0:
            fcn = lambda x: self.implies(expr <= x)
        elif sgn < 0:
            fcn = lambda x: self.implies(expr >= x)
            
        s = None
        for _, st in igen.test(igen.subset(itertools.chain(igen.sI(varrv), varreal), minsize = minsize, 
                                       maxsize = maxsize, coeffmode = coeffmode), 
                      fcn, sgn = sgn, yield_set = True):
            s = st
        
        if s is None:
            return Region.universe()
        
        r = Region.universe()
        for x in s:
            if sgn > 0:
                r.exprs_ge.append(x - expr)
            elif sgn < 0:
                r.exprs_ge.append(expr - x)
        
        if not skip_simplify:
            r.simplify()
            
        return r
    
    class SearchEntry:
        def __init__(self, x, x_reg = None, cmin = -1, cmax = 1):
            self.x = x
            if x_reg == None:
                self.x_reg = x.copy()
            else:
                self.x_reg = x_reg
            self.cmin = cmin
            self.cmax = cmax
    
    
    def discover(self, entries, method = "hull_auto", minsize = 1, maxsize = 2, skip_simplify = False, reg_init = None, skipto_ex = None, toreal_prefix = None):
        """Automatically discover inequalities between entries. 
        Parameters:
            entries   : List of variables of interest.
            minsize   : Minimum number of terms in bound.
            maxsize   : Maximum number of terms in bound.
            skip_simplify : Set to True to skip final simplification.
        """
        
        ceps = PsiOpts.settings["eps"]
        truth = PsiOpts.settings["truth"]
        if truth is not None:
            with PsiOpts(truth = None):
                return (self & truth).discover(entries, method, minsize, maxsize, skip_simplify, reg_init, skipto_ex, toreal_prefix)
            
        verbose = PsiOpts.settings.get("verbose_discover", False)
        verbose_detail = PsiOpts.settings.get("verbose_discover_detail", False)
        verbose_terms = PsiOpts.settings.get("verbose_discover_terms", False)
        verbose_terms_inner = PsiOpts.settings.get("verbose_discover_terms_inner", False)
        verbose_terms_outer = PsiOpts.settings.get("verbose_discover_terms_outer", False)
        
        simp_step = 10
        
        varreal = []
        varrv = []
        maxlen = 1
        
        cs = self.simplified_quick()

        #plain = cs.isplain()
        plain = True
        
        selfif = None
        progs_self = []
        progs_r = []
        index_self = IVarIndex()
        cs.record_to(index_self)
        index_r = IVarIndex()
        isaffine = cs.affine_present()
        
        if isinstance(entries, Expr):
            entries = entries.allcomp()
            
        for a in entries:
            a2 = a
            if not isinstance(a, tuple):
                if isinstance(a, Expr) and toreal_prefix is not None:
                    a2 = (a, Expr.real(toreal_prefix + str(a)))
                else:
                    a2 = (a, None)
                    
            if isinstance(a2[1], ExprArray) or isinstance(a2[1], CompArray):
                maxlen = max(maxlen, len(a2[1]))
            
            a2[0].record_to(index_r)
            
            aself = None
            if a2[1] is None:
                aself = a2[0]
            else:
                aself = a2[1]
            aself.record_to(index_self)
            
            if isinstance(aself, Expr) and aself.affine_present():
                isaffine = True
                
            plain = plain and not aself.isregtermpresent()
                
            if isinstance(a2[0], Expr):
                varreal.append(a2[0])
            else:
                varrv.append(a2[0])
                

        if cs.get_type() == RegionType.NORMAL and not cs.aux.isempty() and not cs.imp_present():
            cs.aux_strengthen(index_self.comprv)
            cs.aux = Comp.empty()
        
        plain = plain and cs.isplain()

        r = Region.universe()
        if reg_init is not None:
            r = reg_init.copy()
        
        #print(skipto_ex)
        if skipto_ex is not None:
            if not isinstance(skipto_ex, str):
                skipto_ex = skipto_ex.tostring()
            
        nadd = 0
        
        if plain:
            selfif = cs.imp_flipped()
        
        if method == "hull_auto":
            if isaffine:
                method = "hull"
            else:
                method = "hull_cone"
                
        if (method == "hull" or method == "hull_cone") and not plain:
            method = "guess"
            
        #print(index_r.comprv)
        #print(index_self.comprv)
        
        vis = set()
        
        terms = []
        if method == "hull" or method == "hull_cone":
            terms = list(itertools.chain(varreal, ent_vector(*varrv)))
        else:
            terms = list(itertools.chain(varreal, igen.sI(varrv)))

        lastres = [False]
        
        
        def expr_tr(ex):
            exsum = Expr.zero()
            for i in range(maxlen):
                tex = ex.copy()
                for a in entries:
                    if isinstance(a, tuple):
                        if isinstance(a[1], ExprArray) or isinstance(a[1], CompArray):
                            if i < len(a[1]):
                                tex.substitute(a[0], a[1][i])
                            else:
                                if isinstance(a[0], Expr):
                                    tex.substitute(a[0], Expr.zero())
                                else:
                                    tex.substitute(a[0], Comp.empty())
                        else:
                            tex.substitute(a[0], a[1])
                exsum += tex
            return exsum
        
        
        #print(method)
        
        if method == "hull" or method == "hull_cone":
            prog = cs.imp_flipped().init_prog(index = index_self, lp_bounded = True)
            A = SparseMat(0)
            for a in terms:
                A.extend(prog.get_vec(expr_tr(a), sparse = True)[0])
            rt = prog.discover_hull(A, iscone = (method == "hull_cone"))
            inf_thres = prog.lp_ubound / 5.0
            for x in rt:
                if abs(x[0]) > inf_thres:
                    continue
                expr = Expr.zero()
                for i in range(len(terms)):
                    if abs(x[i + 1]) > ceps:
                        expr += terms[i] * x[i + 1]
                r.iand_norename(expr >= -x[0])
            if not skip_simplify:
                r.simplify()
            return r
            
        def exgen():
            for size in range(minsize, maxsize + 1):
                if size == 1:
                    for i in range(len(terms)):
                        yield terms[i]
                        res0 = lastres[0]
                        yield -terms[i]
                        res1 = lastres[0]
                        if res0 and res1:
                            terms[i] = None
                    terms[:] = [a for a in terms if a is not None]
                elif size == 2:
                    for i in range(len(terms)):
                        for j in range(i):
                            if terms[j] is not None:
                                yield terms[i] - terms[j]
                                res0 = lastres[0]
                                yield terms[j] - terms[i]
                                res1 = lastres[0]
                                if res0 and res1:
                                    terms[i] = None
                                    break
                    terms[:] = [a for a in terms if a is not None]
                else:
                    for ex in igen.subset(terms, minsize = size, 
                                       maxsize = maxsize, coeffmode = -1, replacement = True):
                        yield ex
                    break
                            
        
        for ex in exgen():
            if PsiOpts.is_timer_ended():
                break
            
            if skipto_ex is not None:
                
                if skipto_ex != ex.tostring():
                    lastres[0] = False
                    continue
                skipto_ex = None
                
            if verbose_terms:
                print(str(ex) + " <= 0")
                
            ex = ex.simplified()
            if ex.isnonpos():
                lastres[0] = True
                continue
            
            exhash = hash(ex)
            if exhash in vis:
                lastres[0] = False
                continue
            vis.add(exhash)
            
            r2 = ex <= 0
            
            if r.implies_saved(r2, index_r, progs_r):
                lastres[0] = True
                continue
            
            exsum = expr_tr(ex)
            
            if ((plain and selfif.implies_impflipped_saved(exsum <= 0, index_self, progs_self)) 
            or (not plain and cs.implies(exsum <= 0))):
                if verbose:
                    print("ADD " + str(ex) + " <= 0")
                r &= r2
                nadd += 1
                if not skip_simplify and nadd % simp_step == 0:
                    r.simplify()
                progs_r = []
                #print("OKAY " + str(r2))
                if verbose_detail:
                    print(str(r))
                lastres[0] = True
            else:
                #print("FAIL " + str(r2) + "  " + str(plain) + "  " + str(saved_info))
                lastres[0] = False
                
        if not skip_simplify:
            r.simplify()
            
        return r
    
    
    def tostring(self, style = 0, tosort = False, lhsvar = "real", inden = 0, add_bracket = False,
                 small = False, skip_outer_exists = False):
        """Convert to string. 
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """

        style = iutil.convert_str_style(style)
        r = ""
        
        if isinstance(lhsvar, str) and lhsvar == "real":
            lhsvar = self.allcomprealvar()
        
        nlstr = "\n"
        if style & PsiOpts.STR_STYLE_LATEX:
            nlstr = "\\\\\n"
        
        spacestr = " "
        if style & PsiOpts.STR_STYLE_LATEX:
            spacestr = "\\;"
        
        if skip_outer_exists and not self.aux.isempty() and self.auxi.isempty() and self.isuniverse():
            return spacestr * inden + self.aux.tostring(style = style, tosort = tosort)
        
        imp_pres = False
        if self.exprs_gei or self.exprs_eqi:
            imp_pres = True
            inden_inner = inden
            inden_inner1 = inden + 1
            inden_inner2 = inden + 3
            add_bracket_inner = bool(style & PsiOpts.STR_STYLE_PSITIP)
            r += spacestr * inden
            if style & PsiOpts.STR_STYLE_LATEX:
                if add_bracket:
                    r += "\\left\\{"
                r += "\\begin{array}{l}\n"
                if not small:
                    r += "\displaystyle"
                r += " "
                inden_inner = 0
                inden_inner1 = 0
                inden_inner2 = 0
            else:
                r += "("
            r += self.imp_flippedonly_noaux().tostring(style = style, tosort = tosort, 
                                                       lhsvar = lhsvar, inden = inden_inner1,
                                                       add_bracket = add_bracket_inner).lstrip()
            if style & PsiOpts.STR_STYLE_PSITIP:
                r += nlstr + spacestr * inden_inner + ">> "
            elif style & PsiOpts.STR_STYLE_LATEX:
                r += (nlstr + ("\displaystyle " if not small else " ") + spacestr * inden_inner
                      + PsiOpts.settings["latex_matimplies"] + " " + spacestr + " ")
            else:
                r += nlstr + spacestr * inden_inner + "=> "
            
            cs = self.copy()
            cs.exprs_gei = []
            cs.exprs_eqi = []
            r += cs.tostring(style = style, tosort = tosort, 
                             lhsvar = lhsvar, inden = inden_inner2,
                             add_bracket = add_bracket_inner).lstrip()
            
            if style & PsiOpts.STR_STYLE_LATEX:
                r += nlstr + "\\end{array}"
                if add_bracket:
                    r += "\\right\\}"
            else:
                r += ")"
            return r
            
        
        if not self.auxi.isempty():
            if style & PsiOpts.STR_STYLE_PSITIP:
                pass
            else:
                if style & PsiOpts.STR_STYLE_LATEX:
                    if not style & PsiOpts.STR_STYLE_LATEX_QUANTAFTER:
                        r += PsiOpts.settings["latex_forall"] + " "
                        r += self.auxi.tostring(style = style, tosort = tosort)
                        r += PsiOpts.settings["latex_quantifier_sep"] + " "
            
        if not self.aux.isempty():
            if style & PsiOpts.STR_STYLE_PSITIP:
                pass
            else:
                if style & PsiOpts.STR_STYLE_LATEX:
                    if not style & PsiOpts.STR_STYLE_LATEX_QUANTAFTER:
                        if not skip_outer_exists:
                            r += PsiOpts.settings["latex_exists"] + " "
                        r += self.aux.tostring(style = style, tosort = tosort)
                        r += PsiOpts.settings["latex_quantifier_sep"] + " "
                
        cs = self
        bnets = None
        
        if style & PsiOpts.STR_STYLE_MARKOV:
            cs = cs.copy()
            tic = cs.remove_ic()
            bnets = BayesNet.from_ic_list(tic)
            
        eqnlist = ([x.tostring_eqn(">=", style = style, tosort = tosort, lhsvar = lhsvar) for x in cs.exprs_ge]
        + [x.tostring_eqn("==", style = style, tosort = tosort, lhsvar = lhsvar) for x in cs.exprs_eq])
        if tosort:
            eqnlist = zip(eqnlist, [lhsvar is not None and any(x.ispresent(t) for t in lhsvar) for x in cs.exprs_ge]
                          + [lhsvar is not None and any(x.ispresent(t) for t in lhsvar) for x in cs.exprs_eq])
            eqnlist = sorted(eqnlist, key=lambda a: (not a[1], len(a[0]), a[0]))
            eqnlist = [x for x, t in eqnlist]
        
        if style & PsiOpts.STR_STYLE_MARKOV:
            eqnlist2 = []
            for bnet in bnets:
                ms = bnet.get_markov()
                for cm in ms:
                    if len(cm) % 2 == 1 and all(cm[i].isempty() for i in range(1, len(cm), 2)):
                        tlist = [cm[i].tostring(style = style, tosort = tosort, 
                                                add_bracket = not (style & PsiOpts.STR_STYLE_PSITIP)) for i in range(0, len(cm), 2)]
                        
                        if style & PsiOpts.STR_STYLE_LATEX:
                            eqnlist2.append((" " + PsiOpts.settings["latex_indep"] + " ").join(tlist))
                        else:
                            eqnlist2.append("indep(" + ", ".join(tlist) + ")")
                    else:
                        tlist = [cm[i].tostring(style = style, tosort = tosort, 
                                                add_bracket = not (style & PsiOpts.STR_STYLE_PSITIP)) for i in range(len(cm))]
                        
                        if style & PsiOpts.STR_STYLE_LATEX:
                            eqnlist2.append((" " + PsiOpts.settings["latex_markov"] + " ").join(tlist))
                        else:
                            eqnlist2.append("markov(" + ", ".join(tlist) + ")")
                    
            if tosort:
                eqnlist2 = sorted(eqnlist2, key = lambda a: len(a))
            eqnlist += eqnlist2
            
        first = True

        use_array = style & PsiOpts.STR_STYLE_LATEX_ARRAY and len(eqnlist) > 1
        isplu = True
        if style & PsiOpts.STR_STYLE_LATEX:
            isplu = len(eqnlist) > 1 and use_array
        else:
            isplu = not self.aux.isempty() or len(eqnlist) > 1
        
        use_bracket = add_bracket or isplu
        
            
        if style & PsiOpts.STR_STYLE_PSITIP:
            r += spacestr * inden
            if use_bracket:
                r += "("
        elif style & PsiOpts.STR_STYLE_LATEX:
            r += spacestr * inden
            if use_bracket:
                r += "\\left\\{"
            if use_array:
                r += "\\begin{array}{l}\n"
        else:
            r += spacestr * inden
            if use_bracket:
                r += "{"
        
        for x in eqnlist:
            if style & PsiOpts.STR_STYLE_PSITIP:
                if first:
                    if use_bracket:
                        r += " "
                else:
                    r += nlstr + spacestr * inden + " &"
                if isplu:
                    r += "("
                    
                if use_bracket:
                    r += " "
                    
            elif style & PsiOpts.STR_STYLE_LATEX:
                if use_array:
                    if first:
                        r += spacestr * inden + "  "
                    else:
                        r += "," + nlstr + spacestr * inden + "  "
                    if small:
                        # r += "{\\scriptsize "
                        pass
                else:
                    if first:
                        r += " "
                    else:
                        r += "," + spacestr + "  "
            else:
                if first:
                    if use_bracket:
                        r += " "
                else:
                    r += "," + nlstr + spacestr * inden + "  "
            
            r += x
            
            if style & PsiOpts.STR_STYLE_PSITIP:
                r += " "
                if isplu:
                    r += ")"
            elif style & PsiOpts.STR_STYLE_LATEX:
                if use_array:
                    if small:
                        # r += "}"
                        pass
                
            first = False
            
        if len(eqnlist) == 0:
            if style & PsiOpts.STR_STYLE_PSITIP:
                r += " universe()"
            elif style & PsiOpts.STR_STYLE_LATEX:
                r += " " + PsiOpts.settings["latex_region_universe"]
        
            
        if style & PsiOpts.STR_STYLE_PSITIP:
            if use_bracket:
                r += " )"
        elif style & PsiOpts.STR_STYLE_LATEX:
            if use_array:
                r += nlstr + spacestr * inden + "\\end{array}"
            if use_bracket:
                r += " \\right\\}"
        else:
            if use_bracket:
                r += " }"
            
            
        if not self.aux.isempty():
            if style & PsiOpts.STR_STYLE_PSITIP:
                pass
            else:
                if style & PsiOpts.STR_STYLE_LATEX:
                    if style & PsiOpts.STR_STYLE_LATEX_QUANTAFTER:
                        r += " , " + PsiOpts.settings["latex_exists"] + " "
                        r += self.aux.tostring(style = style, tosort = tosort)
                else:
                    r += " , exists "
                    r += self.aux.tostring(style = style, tosort = tosort)
        
        if not self.auxi.isempty():
            if style & PsiOpts.STR_STYLE_PSITIP:
                pass
            else:
                if style & PsiOpts.STR_STYLE_LATEX:
                    if style & PsiOpts.STR_STYLE_LATEX_QUANTAFTER:
                        r += " , " + PsiOpts.settings["latex_forall"] + " "
                        r += self.auxi.tostring(style = style, tosort = tosort)
                else:
                    r += " , forall "
                    r += self.auxi.tostring(style = style, tosort = tosort)
                
        if imp_pres:
            r += ")"
            
        
        if not self.aux.isempty():
            if style & PsiOpts.STR_STYLE_PSITIP:
                r += ".exists(" + self.aux.tostring(style = style, tosort = tosort) + ")"
        
        if not self.auxi.isempty():
            if style & PsiOpts.STR_STYLE_PSITIP:
                r += ".forall(" + self.auxi.tostring(style = style, tosort = tosort) + ")"
            
            
        return r
    
        
    def __str__(self):
        lhsvar = None
        if PsiOpts.settings.get("str_lhsreal", False):
            lhsvar = "real"
        return self.tostring(PsiOpts.settings["str_style"], 
                             tosort = PsiOpts.settings["str_tosort"], lhsvar = lhsvar)
    
    def tostring_repr(self, style):
        if PsiOpts.settings.get("repr_check", False):
            #return str(self.check())
            if self.check():
                return str(True)
        
        lhsvar = None
        if PsiOpts.settings.get("str_lhsreal", False):
            lhsvar = "real"
            
        if PsiOpts.settings.get("repr_simplify", False):
            return self.simplified_quick().tostring(style, 
                                                    tosort = PsiOpts.settings["str_tosort"], lhsvar = lhsvar)
        
        return self.tostring(style, 
                             tosort = PsiOpts.settings["str_tosort"], lhsvar = lhsvar)
    
    
    def __repr__(self):
        return self.tostring_repr(PsiOpts.settings["str_style_repr"])
    
    def _latex_(self):
        return self.tostring_repr(iutil.convert_str_style("latex"))
        
        
    def __hash__(self):
        #return hash(self.tostring(tosort = True))
        
        return hash((
            hash(frozenset(hash(x) for x in self.exprs_ge)),
            hash(frozenset(hash(x) for x in self.exprs_eq)),
            hash(frozenset(hash(x) for x in self.exprs_gei)),
            hash(frozenset(hash(x) for x in self.exprs_eqi)),
            hash(self.aux), hash(self.inp), hash(self.oup), hash(self.auxi)
            ))
        
    def ent_vector_discover_ic(v, x, eps = None, skip_simplify = False):
        if eps is None:
            eps = PsiOpts.settings["eps_check"]
            
        n = len(x)
        mask_all = (1 << n) - 1
        
        r = Region.universe()
        
        for mask in range(1 << n):
            for i in range(n):
                if not (1 << i) & mask:
                    if v[mask | (1 << i)] - v[mask] <= eps:
                        r.exprs_eq.append(Expr.Hc(x[i], x.from_mask(mask)))
                        
        
        for mask in range(1, 1 << n):
            bins = dict()
            for xmask in igen.subset_mask(mask_all - mask):
                t = v[mask | xmask] - v[xmask]
                if t <= eps:
                    continue
                
                tbin0 = int(t / eps + 0.5)
                
                for tbin in range(max(tbin0 - 1, 0), tbin0 + 2):
                    if tbin in bins:
                        for ymask in bins[tbin]:
                            if ymask & xmask == ymask and xmask - ymask < mask:
                                r.exprs_eq.append(Expr.Ic(x.from_mask(mask), 
                                                          x.from_mask(xmask - ymask), 
                                                          x.from_mask(ymask)))
                        if tbin == tbin0:
                            bins[tbin].append(xmask)
                    else:
                        if tbin == tbin0:
                            bins[tbin] = [xmask]
        
        if not skip_simplify:
            r.simplify_bayesnet(reduce_ic = True)
            r.simplify()
        
        return r
    
        

class RegionOp(Region):
    """A region which is the union/intersection of a list of regions."""
    
    def __init__(self, rtype, regs, auxs, inp = None, oup = None):
        self.rtype = rtype
        self.regs = regs
        self.auxs = auxs
        
        self.inp = Comp.empty() if inp is None else inp
        self.oup = Comp.empty() if oup is None else oup
    
    def get_type(self):
        return self.rtype
    
    def isnormalcons(self):
        return False
    
    def isuniverse(self, sgn = True, canon = False):
        if canon:
            if sgn:
                return self.get_type() == RegionType.INTER and len(self.regs) == 0 and len(self.auxs) == 0
            else:
                return self.get_type() == RegionType.UNION and len(self.regs) == 0 and len(self.auxs) == 0
            
        #return False
        isunion = (self.get_type() == RegionType.UNION)
        for x, c in self.regs:
            if isunion ^ sgn ^ x.isuniverse(not c ^ sgn):
                return not isunion ^ sgn
        return isunion ^ sgn
        
    def isempty(self):
        return self.isuniverse(False)
    
    def isplain(self):
        return False
        
    def copy(self):
        return RegionOp(self.rtype, [(x.copy(), c) for x, c in self.regs],
                        [(x.copy(), c) for x, c in self.auxs], self.inp.copy(), self.oup.copy())
    
    def copy_noaux(self):
        return RegionOp(self.rtype, [(x.copy_noaux(), c) for x, c in self.regs],
                        [], self.inp.copy(), self.oup.copy())
        
    def copy_(self, other):
        self.rtype = other.rtype
        self.regs = [(x.copy(), c) for x, c in other.regs]
        self.auxs = [(x.copy(), c) for x, c in other.auxs]
        self.inp = other.inp.copy()
        self.oup = other.oup.copy()
    
    def imp_flip(self):
        if self.get_type() == RegionType.INTER:
            self.rtype = RegionType.UNION
        elif self.get_type() == RegionType.UNION:
            self.rtype = RegionType.INTER
        for x, c in self.regs:
            x.imp_flip()
        return self
    
    def imp_flipped(self):
        r = self.copy()
        r.imp_flip()
        return r
    
    def universe_type(rtype):
        return RegionOp(rtype, [(Region.universe(), True)], [])
    
    def union(xs = None):
        if xs is None:
            xs = []
        return RegionOp(RegionType.UNION, [(x.copy(), True) for x in xs], [])
    
    def inter(xs = None):
        if xs is None:
            xs = []
        return RegionOp(RegionType.INTER, [(x.copy(), True) for x in xs], [])
    
        
    def __len__(self):
        return len(self.regs)
    
    def __getitem__(self, key):
        r = self.regs[key]
        if isinstance(r, list):
            return RegionOp(self.rtype, r, [])
        else:
            if r[1]:
                return r[0]
            else:
                return RegionOp(self.rtype, [r], [])
    
    
    def ispresent(self, x):
        """Return whether any variable in x appears here."""
        for z, c in self.regs:
            if z.ispresent(x):
                return True
        for z, c in self.auxs:
            if z.ispresent(x):
                return True
        return False
    
        
    def allcomprealvar(self):
        r = Comp.empty()
        for z, c in self.regs:
            r += z.allcomprealvar()
        return r
    
    def rename_var(self, name0, name1):
        for x, c in self.regs:
            x.rename_var(name0, name1)
        for x, c in self.auxs:
            x.rename_var(name0, name1)
    
    def rename_map(self, namemap):
        for x, c in self.regs:
            x.rename_map(namemap)
        for x, c in self.auxs:
            x.rename_map(namemap)

        
    def getaux(self):
        r = Comp.empty()
        for x, c in self.auxs:
            if c:
                r += x
        for x, c in self.regs:
            if c:
                r += x.getaux()
            else:
                r += x.getauxi()
        return r
    
    def getauxi(self):
        r = Comp.empty()
        for x, c in self.auxs:
            if not c:
                r += x
        for x, c in self.regs:
            if c:
                r += x.getauxi()
            else:
                r += x.getaux()
        return r
    
    def getauxall(self):
        r = Comp.empty()
        for x, c in self.auxs:
            r += x
        for x, c in self.regs:
            r += x.getauxall()
        return r
    
    def getauxs(self):
        return [(x.copy(), c) for x, c in self.auxs]
    
    @fcn_substitute
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), in place."""
        for x, c in self.regs:
            x.substitute(v0, v1)
        if not isinstance(v0, Expr):
            for x, c in self.auxs:
                x.substitute(v0, v1)
        return self
    
    @fcn_substitute
    def substitute_aux(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), and remove auxiliary v0, in place."""
        for x, c in self.regs:
            x.substitute_aux(v0, v1)
        self.auxs = [(x - v0, c) for x, c in self.auxs if not (x - v0).isempty()]
        return self

    def remove_present(self, v):
        for x, c in self.regs:
            x.remove_present(v)
        self.auxs = [(x, c) for x, c in self.auxs if not x.ispresent(v)]
    
    def remove_notpresent(self, v):
        for x, c in self.regs:
            x.remove_notpresent(v)
        self.auxs = [(x, c) for x, c in self.auxs if x.ispresent(v)]
        
    def condition(self, b):
        """Condition on random variable b, in place."""
        for x, c in self.regs:
            x.condition(b)
        return self
        
    def symm_sort(self, terms):
        """Sort the random variables in terms assuming symmetry among those terms."""
        
        for x, c in self.regs:
            x.symm_sort(terms)
            
    
    def relax(self, w, gap):
        """Relax real variables in w by gap, in place"""
        for x, c in self.regs:
            if c:
                x.relax(w, gap)
            else:
                x.relax(w, -gap)
        return self
    
    def record_to(self, index):
        for x, c in self.regs:
            x.record_to(index)
        for x, c in self.auxs:
            x.record_to(index)

    def pack_type_self(self, totype):
        if len(self.auxs) == 0:
            ctype = self.get_type()
            if ctype == totype:
                return self
            if ctype == RegionType.UNION or ctype == RegionType.INTER:
                if len(self.regs) == 1:
                    self.rtype = totype
                    return self
        self.regs = [(self.copy(), True)]
        self.auxs = []
        self.rtype = totype
        return self
    
    def pack_type(x, totype):
        if len(x.getauxs()) == 0:
            ctype = x.get_type()
            if ctype == totype:
                return x.copy()
            if ctype == RegionType.UNION or ctype == RegionType.INTER:
                if len(x.regs) == 1:
                    r = x.copy()
                    r.rtype = totype
                    return r
        return RegionOp(totype, [(x.copy(), True)], [])
        
    def iand_norename(self, other):
        self.pack_type_self(RegionType.INTER)
        other = RegionOp.pack_type(other, RegionType.INTER)
        
        self.regs += other.regs
        self.auxs += other.auxs
        self.inter_compress()
        return self
    
    def inter_compress(self):
        if self.get_type() != RegionType.INTER and self.get_type() != RegionType.UNION:
            return
        curc = self.get_type() == RegionType.INTER
        cons = Region.universe()
        for x, c in self.regs:
            if c == curc and x.isnormalcons() and x.getauxall().isempty():
                cons &= x
                x.setuniverse()
        self.regs = [(cons, curc)] + self.regs
        self.regs = [(x, c) for x, c in self.regs if c != curc or not x.isuniverse()]
        
    
    def normalcons_sort(self):
        if self.get_type() != RegionType.INTER and self.get_type() != RegionType.UNION:
            return
        curc = self.get_type() == RegionType.INTER
        
        regsf = [(x, c) for x, c in self.regs if c == curc and x.isnormalcons()]
        regsb = [(x, c) for x, c in self.regs if c != curc and x.isnormalcons()]
        self.regs = regsf + [(x, c) for x, c in self.regs if not x.isnormalcons()] + regsb
        
    def __iand__(self, other):
        if isinstance(other, bool):
            if not other:
                return Region.empty()
            return self
        
        if other.isuniverse(canon = True):
            return self
        
        self.pack_type_self(RegionType.INTER)
        other = RegionOp.pack_type(other, RegionType.INTER)
        self.aux_avoid(other)
        
        self.regs += other.regs
        self.auxs += other.auxs
        self.inter_compress()
        return self
        
        
    def __and__(self, other):
        r = self.copy()
        r &= other
        return r
        
    def __rand__(self, other):
        r = self.copy()
        r &= other
        return r
    
        
    def __ior__(self, other):
        if isinstance(other, bool):
            if other:
                return Region.universe()
            return self
        
        if other.isuniverse(sgn = False, canon = True):
            return self
        
        other = RegionOp.pack_type(other, RegionType.UNION)
        self.pack_type_self(RegionType.UNION)
        self.aux_avoid(other)
        
        self.regs += other.regs
        self.auxs += other.auxs
        return self
        
        
    def ior_norename(self, other):
        other = RegionOp.pack_type(other, RegionType.UNION)
        self.pack_type_self(RegionType.UNION)
        
        self.regs += other.regs
        self.auxs += other.auxs
        return self
        
    def rior(self, other):
        other = RegionOp.pack_type(other, RegionType.UNION)
        self.pack_type_self(RegionType.UNION)
        self.aux_avoid(other)
        
        self.regs = other.regs + self.regs
        self.auxs = other.auxs + self.auxs
        return self
        
        
    def __or__(self, other):
        r = self.copy()
        r |= other
        return r
        
    def __ror__(self, other):
        r = self.copy()
        r |= other
        return r
    
    def append_avoid(self, x, c = True):
        y = x.copy()
        self.aux_avoid(y)
        self.regs.append((y, c))
    
    
    def __imul__(self, other):
        for i in range(len(self.regs)):
            self.regs[i][0] *= other
        return self
    
    def negate(self):
        if self.get_type() == RegionType.UNION:
            self.rtype = RegionType.INTER
        elif self.get_type() == RegionType.INTER:
            self.rtype = RegionType.UNION
        self.regs = [(x, not c) for x, c in self.regs]
        self.auxs = [(x, not c) for x, c in self.auxs]
        
        return self
        
    def __invert__(self):
        r = self.copy()
        r = r.negate()
        return r
        
    def negateornot(x, c):
        if c:
            return x.copy()
        else:
            return ~x
    
    def setuniverse(self):
        self.rtype = RegionType.INTER
        self.regs = []
    
    def setempty(self):
        self.rtype = RegionType.UNION
        self.regs = []
    
    def universe():
        return RegionOp(RegionType.INTER, [], [])
    
    def empty():
        return RegionOp(RegionType.UNION, [], [])
        #return ~Region.universe()
    
    def sum_minkowski(self, other):
        cs = self.copy()
        cs.distribute()
        other = other.copy()
        other.distribute()
        
        # warnings.warn("Minkowski sum of union or intersection regions is unsupported.", RuntimeWarning)
        
        auxs = [(x.copy(), c) for x, c in cs.getauxs() + other.getauxs()]
        if cs.get_type() == RegionType.UNION:
            return RegionOp(RegionType.UNION, [(x.sum_minkowski(other), c) for x, c in cs.regs], auxs)
        if other.get_type() == RegionType.UNION:
            return RegionOp(RegionType.UNION, [(cs.sum_minkowski(x), c) for x, c in other.regs], auxs)
        
        # The following are technically wrong
        if cs.get_type() == RegionType.INTER:
            return RegionOp(RegionType.INTER, [(x.sum_minkowski(other), c) for x, c in cs.regs], auxs)
        if other.get_type() == RegionType.INTER:
            return RegionOp(RegionType.INTER, [(cs.sum_minkowski(x), c) for x, c in other.regs], auxs)
        return cs.copy()
    
    
    def implicate(self, other, skip_simplify = False):
        cs = self
        cs |= ~other
        return cs
    
    def implicate_norename(self, other, skip_simplify = False):
        cs = self
        cs.ior_norename(~other)
        return cs
    
    def implicated(self, other, skip_simplify = False):
        r = self.copy()
        r = r.implicate(other, skip_simplify)
        return r
    
    def sum_entrywise(self, other):
        r = RegionOp(self.rtype, [], [])
        for ((x1, c1), (x2, c2)) in zip(self.regs, other.regs):
            r.regs.append((x1.sum_entrywise(x2), c1))
        for ((x1, c1), (x2, c2)) in zip(self.auxs, other.auxs):
            r.regs.append((x1.interleaved(x2), c1))
        return r
        
    def corners_optimum(self, w, sn):
        if self.get_type() == RegionType.UNION:
            r = self.copy()
            r.regs = []
            did = False
            for x, c in self.regs:
                if not c:
                    r.append_avoid(x.copy(), c)
                    continue
                t = x.corners_optimum(w, sn)
                if t.isuniverse():
                    r.append_avoid(x.copy(), c)
                else:
                    r.append_avoid(t, c)
                    did = True
            if did:
                return r
            return Region.universe()
        
        elif self.get_type() == RegionType.INTER:
            r = RegionOp.union([])
            r.auxs = [(x.copy(), c) for x, c in self.auxs]
            
            for x, c in self.regs:
                if not c:
                    continue
                t = x.corners_optimum(w, sn)
                if t.isuniverse():
                    continue
                else:
                    tt = self.copy()
                    for i in range(len(tt.regs)):
                        if tt.regs[i] is x:
                            tt.regs[i] = t
                            break
                    r.append_avoid(tt)
            
            if len(r.regs) > 0:
                return r
            return Region.universe()
        return Region.universe()
    
    def sign_present(self, term):
        r = [False] * 2
        for x, c in self.regs:
            t = x.sign_present(term)
            if not c:
                t.reverse()
            r[0] |= t[0]
            r[1] |= t[1]
            if r[0] and r[1]:
                break
        return r
    
    def substitute_sign(self, v0, v1s):
        r = [False] * 2
        t = [False] * 2
        for x, c in self.regs:
            if c:
                t = x.substitute_sign(v0, v1s)
            else:
                t = x.substitute_sign(v0, list(reversed(v1s)) if v1s else v1s)
                t.reverse()
            r[0] |= t[0]
            r[1] |= t[1]
        return r
    
    def substitute_duplicate(self, v0, v1s):
        for x, c in self.regs:
            x.substitute_duplicate(v0, v1s)
    
    def flatten_minmax(self, term, sn, bds):
        # for x, c in self.regs:
        #     x.flatten_minmax(term, sgn, bds)
        
        v1s = [Expr.real(self.name_avoid(str(term) + "_L")), 
               Expr.real(self.name_avoid(str(term) + "_U"))]
        sn_present = self.substitute_sign(Expr.fromterm(term), v1s)
        if sn < 0:
            sn_present.reverse()
            v1s.reverse()
        
        if sn_present[1]:
            treg = Region.universe()
            for b in bds:
                treg &= v1s[1] * sn <= b * sn
            self &= treg
            # print(self)
            # print(v1s[1])
            self.eliminate(v1s[1])
        
        if sn_present[0]:
            tself = RegionOp.pack_type(self.copy(), RegionType.UNION)
            if any(not c for x, c in tself.regs):
                tself = RegionOp.union([tself])
            self.copy_(RegionOp.union([]))
            for t2 in tself:
                if t2.ispresent(v1s[0]):
                    for b in bds:
                        t3 = t2.copy()
                        t3.substitute(v1s[0], b)
                        self |= t3
                else:
                    self |= t2
        
        # print("AFTER FLATTEN REGTERM")
        # print(self)
        # print(sn_present)
        # print()
        return self
        
    def lowest_present(self, v, sn):
        ps = []
        cpres = False
        for x, c in self.regs:
            if x.ispresent(v):
                ps.append(x)
                cpres = c
        
        if len(ps) == 0:
            return None
        if len(ps) == 1:
            t = ps[0].lowest_present(v, sn ^ (not cpres))
            if t is not None:
                return t
        
        if sn:
            return self
        return None

    def term_sn_present(self, term, sn):
        sn *= term.sn
        sn_present = self.substitute_sign(Expr.fromterm(term), None)
        if sn < 0:
            sn_present.reverse()
        return sn_present[0]

        
    def term_sn_present_both(self, term):
        sn_present = self.substitute_sign(Expr.fromterm(term), None)
        return sn_present[0] and sn_present[1]

    def flatten_regterm(self, term, isimp = True, minmax_elim = False):
        if term.reg is None:
            return
        
        # print("FLAT  " + str(minmax_elim))

        if minmax_elim:
            treg, tsgn, tbds = term.get_reg_sgn_bds()
            
            if treg.isuniverse() and tsgn != 0:
                self.flatten_minmax(term, tsgn, tbds)
                return
        
        self.simplify_quick()
        sn = term.sn
        
        v1s = [Expr.real(self.name_avoid(str(term) + "_L")), 
               Expr.real(self.name_avoid(str(term) + "_U"))]
        sn_present = self.substitute_sign(Expr.fromterm(term), v1s)
        if sn < 0:
            sn_present.reverse()
            v1s.reverse()
        
        if sn == 0:
            sn_present[1] = False
            sn_present[0] = True
            
        
        if sn_present[1]:
            term2 = term.copy()
            self.aux_avoid(term2.reg)

            rsb = term2.get_reg_sgn_bds()
            if (PsiOpts.settings["flatten_distribute"] and rsb is not None and not rsb[0].imp_present() and isimp
                and (PsiOpts.settings["flatten_distribute_multi"] or len(rsb[2]) == 1)):

                treg, tsgn, tbds = rsb

                lpres = self.lowest_present(v1s[1], True)
                lpres.substitute_duplicate(v1s[1], tbds)
                taux = treg.aux.copy()
                treg.aux = Comp.empty()
                lpres &= treg
                lpres.eliminate(taux)
                
            else:
                reg2 = term.reg.broken_present(Expr.fromterm(term), flipped = True)
                reg2.substitute(Expr.fromterm(term), v1s[1])
                if isimp:
                    self |= reg2
                else:
                    self &= ~reg2
        
        if sn_present[0]:
            reg2 = term.reg.copy()
            reg2.substitute(Expr.fromterm(term), v1s[0])
            if isimp:
                self.implicate(reg2)
            else:
                self &= reg2
        
        # print("AFTER FLATTEN REGTERM")
        # print(self)
        # print(sn_present)
        # print()

        return self
    
    
    def flatten_ivar(self, ivar, isimp = True):
        if ivar.reg is None:
            return
        
        newvar = Comp([ivar.copy_noreg()])
        reg2 = ivar.reg.copy()
        self.aux_avoid(reg2)
        
        if not ivar.reg_det:
            newindep = Expr.Ic(reg2.getaux() + newvar, 
                        self.allcomprv().reg_excluded() - self.getaux() - reg2.allcomprv() - newvar, 
                        reg2.allcomprv_noaux() - newvar).simplified()
            if not newindep.iszero():
                reg2.iand_norename(newindep == 0)
                #self.implicate_norename(newindep == 0)
                
        if isimp:
            #self.implicate_norename(reg2.exists(newvar), skip_simplify = True)
            self.implicate_norename(reg2, skip_simplify = True)
        else:
            self &= reg2
            
        self.substitute(Comp([ivar]), newvar)
        
        return self
    
    def isregtermpresent(self):
        for x, c in self.regs:
            if x.isregtermpresent():
                return True
        return False
        
    
    def regtermmap(self, cmap, recur):
        for x, c in self.regs:
            x.regtermmap(cmap, recur)
        
        aux = Comp.empty()
        for x, c in self.auxs:
            aux += x
        (H(aux) >= 0).regtermmap(cmap, recur)
        

    def regterm_split(self, term):
        if not (self.term_sn_present_both(term) or term.reg_outer is not None):
            return False

        # sn = term.sn
        # terml = term.copy()
        # terml.substitute(term.x[0], Comp.real(self.name_avoid(str(term) + "_L0")))
        # if terml.reg_outer is not None:
        #     terml.reg = terml.reg_outer
        #     terml.reg_outer = None

        # termr = term.copy()
        # termr.substitute(term.x[0], Comp.real(self.name_avoid(str(term) + "_R0")))
        # termr.reg_outer = None

        # v1s = [Expr.fromterm(terml), Expr.fromterm(termr)]
        
        # if sn < 0:
        #     v1s.reverse()

        v1s = [Expr.fromterm(term.upper_bound(name = self.name_avoid(str(term) + "_L0"))),
            Expr.fromterm(term.lower_bound(name = self.name_avoid(str(term) + "_R0")))]

        self.substitute_sign(Expr.fromterm(term), v1s)

        return True
        

    def flatten(self, minmax_elim = False):
        
        verbose = PsiOpts.settings.get("verbose_flatten", False)
        
        write_pf_enabled = PsiOpts.settings.get("proof_enabled", False)
        
        if write_pf_enabled:
            prevself = self.copy()
        
        did = True
        didall = False
        
        while did:
            did = False
            regterms = {}
            
            # self.regtermmap(regterms, True)
            # for (name, term) in regterms.items():
            #     regterms_in = {}
            #     term.reg.regtermmap(regterms_in, False)
            #     if not regterms_in:
            
            regterms_exc = {}
            self.regtermmap(regterms, False)
            for (name, term) in regterms.items():
                term.reg.regtermmap(regterms_exc, True)
            
            for (name, term) in regterms.items():
                if name not in regterms_exc:
                    if isinstance(term, IVar):
                        pass
                    else:
                        if self.regterm_split(term):
                            did = True
                            break
            if did:
                continue
            
            for cpass, (name, term) in itertools.product(range(2), regterms.items()):
                if name not in regterms_exc:
                    if isinstance(term, IVar):
                        if cpass == 1:
                            continue
                    else:
                        if cpass == 0 and self.term_sn_present(term, 1):
                            continue

                    if verbose:
                        print("========= flatten op ========")
                        print(self)
                        print("=========    term    ========")
                        print(term)
                        print("=========   region   ========")
                        print(term.reg)
                        
                    if isinstance(term, IVar):
                        self.flatten_ivar(term)
                    else:
                        self.flatten_regterm(term, minmax_elim = minmax_elim)
                    did = True
                    didall = True
                    
                    if verbose:
                        print("=========     to     ========")
                        print(self)
                        
                    break
        
        if write_pf_enabled:
            if didall:
                # pf = ProofObj.from_region(prevself, c = "Expand definitions")
                # pf += ProofObj.from_region(self, c = "Expanded definitions to")
                pf = ProofObj.from_region(("equiv", prevself, self), c = "Expand definitions:")
                PsiOpts.set_setting(proof_add = pf)
            
        return self
        
        
    
    def tosimple(self):
        r = Region.universe()
        if self.get_type() == RegionType.UNION:
            if len(self.regs) == 0:
                r = Region.empty()
            elif len(self.regs) == 1 and self.regs[0][1]:
                r = self.regs[0][0].tosimple()
                if r is None:
                    return None
            else:
                return None
        elif self.get_type() == RegionType.INTER:
            for x, c in self.regs:
                if not c:
                    return None
                t = x.tosimple()
                if t is None or t.imp_present():
                    return None
                r.iand_norename(t)
            
        for x, c in self.auxs:
            if not c:
                return None
            r.eliminate(x)
            
        return r
    
    
    def tosimple_safe(self):
        if not self.getauxi().isempty():
            return None
        for x, c in self.regs:
            if x.aux_present():
                return None
        return self.tosimple()
    
    
    
    def var_neighbors(self, v):
        r = v.copy()
        for x, c in self.regs:
            r += x.var_neighbors(v)
        return r
    
    def one_flipped(self):
        return None
        
    def distribute(self):
        """Expand to a single union layer.
        """
        if self.get_type() == RegionType.UNION:
            tregs = []
            for x, c in self.regs:
                if isinstance(x, RegionOp):
                    if not c:
                        x.negate()
                        c = True
                    x.distribute()
                if c and x.get_type() == RegionType.UNION:
                    tregs += x.regs
                    self.auxs = x.getauxs() + self.auxs
                else:
                    tregs.append((x, c))
            self.regs = tregs
            return self
        
        if self.get_type() == RegionType.INTER:
            tregs = [(Region.universe(), True)]
            self.rtype = RegionType.UNION
            for x, c in self.regs:
                if isinstance(x, RegionOp):
                    if not c:
                        x.negate()
                        c = True
                    x.distribute()
                if c and x.get_type() == RegionType.UNION:
                    tregs2 = []
                    for y, cy in x.regs:
                        for a, ca in tregs:
                            tregs2.append((RegionOp.negateornot(a, ca) & RegionOp.negateornot(y, cy), True))
                    tregs = tregs2
                    self.auxs = x.getauxs() + self.auxs
                else:
                    tregs = [(RegionOp.negateornot(a, ca) & RegionOp.negateornot(x, c), True) for a, ca in tregs]
                    self.auxs = x.getauxs() + self.auxs
            self.regs = tregs
            return self
        
        return self
    
    def tounion(self):
        r = self.copy()
        r = r.distribute()
        return RegionOp.pack_type(r, RegionType.UNION)
    
    def aux_appearance(self, curc):
        allcomprv = self.allcomprv() - self.getauxall()
        r = []
        for x, c in self.regs:
            if isinstance(x, RegionOp):
                r += x.aux_appearance(curc ^ (not c))
            else:
                xallcomprv = x.allcomprv() - x.getauxall()
                if not x.auxi.isempty():
                    r.append((x.auxi.copy(), curc ^ (not c) ^ True, xallcomprv.copy()))
                if not x.aux.isempty():
                    r.append((x.aux.copy(), curc ^ (not c), xallcomprv.copy()))
        
        rt = []
        for x, c in self.auxs[::-1]:
            rt.append((x.copy(), curc ^ (not c), allcomprv.copy()))
            allcomprv += x
        r += rt[::-1]
        return r
    
    def aux_remove(self):
        self.auxs = []
        for x, c in self.regs:
            if isinstance(x, RegionOp):
                x.aux_remove()
            else:
                x.aux = Comp.empty()
                x.auxi = Comp.empty()
        return self
    
    def aux_collect(self):
        """Collect auxiliaries to outermost layer.
        """
        iaux = []
        for x, c in self.regs:
            if isinstance(x, RegionOp):
                x.aux_collect()
                iaux += [(x2, c2 ^ (not c)) for x2, c2 in x.auxs]
                x.auxs = []
            else:
                iaux += [(x2, c2 ^ (not c)) for x2, c2 in x.getauxs()]
                x.aux = Comp.empty()
                x.auxi = Comp.empty()
                
        self.auxs = iaux + self.auxs
    
    
    def break_imp(self):
        for i in range(len(self.regs)):
            if self.regs[i][0].get_type() == RegionType.NORMAL and self.regs[i][0].imp_present():
                auxs = self.regs[i][0].getauxs()
                rc = self.regs[i][0].consonly()
                rc.aux = Comp.empty()
                ri = self.regs[i][0].imp_flippedonly()
                ri.aux = Comp.empty()
                r = RegionOp.union([rc])
                if not ri.isuniverse():
                    r = r.implicated(ri)
                r.auxs = auxs
                self.regs[i] = (r, self.regs[i][1])
            if isinstance(self.regs[i][0], RegionOp):
                self.regs[i][0].break_imp()
    
    def from_region(x):
        if isinstance(x, RegionOp):
            return x.copy()
        r = RegionOp.union([x])
        r.break_imp()
        if len(r.regs) == 1 and r.regs[0][1] and isinstance(r.regs[0][0], RegionOp):
            return r.regs[0][0]
        return r
    
    def break_present(self, w, flipped = True):
        for i in range(len(self.regs)):
            x, c = self.regs[i]
            if isinstance(x, RegionOp):
                x.break_present(w, flipped)
            else:
                self.regs[i] = (x.broken_present(w, flipped), c)
        return self
    
    def broken_present(self, w, flipped = True):
        r = self.copy()
        r = r.break_present(w, flipped)
        return r
    
    def aux_clean(self):
        auxs = self.auxs
        self.auxs = []
        
        for x, c in auxs:
            if len(self.auxs) > 0 and self.auxs[-1][1] == c:
                self.auxs[-1]= (self.auxs[-1][0] + x, self.auxs[-1][1])
            else:
                self.auxs.append((x, c))
        
    def emptycons_present(self):
        
        for x, c in self.regs:
            cons_present = False
            
            if x.get_type() == RegionType.INTER:
                for y, cy in x.regs:
                    if not cy:
                        cons_present = True
                        break
            elif x.get_type() == RegionType.NORMAL:
                if not c:
                    cons_present = True
            
            if not cons_present:
                return True
        
        return False
            
            
    def presolve(self):
    
            
        self.break_imp()
        self.simplify_quick(zero_group = 1)
        self.flatten(minmax_elim = PsiOpts.settings["flatten_minmax_elim"])
        self.break_imp()
        aux_ap = self.aux_appearance(True)
        self.aux_remove()
        self.distribute()
        
        if False:
            t = Comp.empty()
            for x, c, ccomp in aux_ap:
                t += x
            t = self.allcomprv() - t
            if not t.isempty():
                #aux_ap = [(t, False, Comp.empty())] + aux_ap
                aux_ap.append((t, False, Comp.empty()))
        
        
        #self.inter_compress()
        self.normalcons_sort()
        for x, c in self.regs:
            if isinstance(x, RegionOp):
                x.inter_compress()
        
        self.auxs = [(x, c) for x, c, ccomp in aux_ap]
        self.aux_clean()
        
        if not self.emptycons_present():
            self.regs.append((Region.empty(), True))
        
        return self
            
    
    def to_cause_consequence(self):
        cs = ~self
        cs.presolve()
        r = []
        #allauxs = cs.auxs
        auxi = cs.getaux()
        
        for x, c in cs.regs:
            #allcomprv = x.allcomprv()
            #cur_auxs_incomp = RegionOp.auxs_incomp(allauxs, allcomprv)
            
            req = Region.universe()
            cons = []
            
            if x.get_type() == RegionType.INTER:
                for y, cy in x.regs:
                    if cy:
                        req &= y
                    else:
                        cons.append(y)
            elif x.get_type() == RegionType.NORMAL:
                if c:
                    if x.isempty():
                        continue
                    req &= x
                else:
                    cons.append(x)
            
            ccons = None
            if len(cons) == 1:
                ccons = cons[0]
            else:
                ccons = RegionOp.union(cons)
            
            r.append((req, ccons, auxi.inter(x.allcomp())))
            
        return r
    
    def get_var_avoid(self, a):
        r = None
        for x, c in self.regs:
            t = x.get_var_avoid(a)
            if r is None:
                r = t
            elif t is not None:
                r = r.inter(t)
        return r
    
    def auxs_icreg(auxs, othercomp, exclcomp):
        r = Region.universe()
        ccomp = Comp.empty()
        for a, c in reversed(auxs):
            if not c:
                r.iand_norename(Region.Ic(a - exclcomp, othercomp, ccomp))
            ccomp += a
        return r
            
    def auxs_incomp(auxs, x):
        r = [(a.inter(x), c) for a, c in auxs]
        return [(a, c) for a, c in r if not a.isempty()]
#        r2 = []
#        for a, c in r:
#            if not a.isempty():
#                if len(r2) == 0 or r2[-1][1] != c:
#                    r2.append((a, c))
#                else:
#                    r2[-1] = (r2[-1][0] + a, c)
#        return r2
    
    def check_getaux_inplace(self, must_include = None, single_include = None, hint_pair = None, hint_aux = None, hint_aux_avoid = None, max_iter = None, leaveone = None):

        verbose = PsiOpts.settings.get("verbose_auxsearch", False)
        verbose_step = PsiOpts.settings.get("verbose_auxsearch_step", False)
        verbose_op = PsiOpts.settings.get("verbose_auxsearch_op", False)
        verbose_op_step = PsiOpts.settings.get("verbose_auxsearch_op_step", False)
        verbose_op_detail = PsiOpts.settings.get("verbose_auxsearch_op_detail", False)
        verbose_op_detail2 = PsiOpts.settings.get("verbose_auxsearch_op_detail2", False)
        
        ignore_must = PsiOpts.settings["ignore_must"]
        forall_multiuse = PsiOpts.settings["forall_multiuse"]
        forall_multiuse_numsave = PsiOpts.settings["forall_multiuse_numsave"]
        auxsearch_local = PsiOpts.settings["auxsearch_local"]
        init_leaveone = PsiOpts.settings["init_leaveone"]
        
        auxsearch_aux_strengthen = PsiOpts.settings["auxsearch_aux_strengthen"]

        if leaveone is None:
            leaveone = PsiOpts.settings["auxsearch_leaveone"]
        save_res = auxsearch_local
        if max_iter is None:
            max_iter = PsiOpts.settings["auxsearch_max_iter"]
        if hint_aux is None:
            hint_aux = []
        if hint_aux_avoid is None:
            hint_aux_avoid = []
        if must_include is None:
            must_include = Comp.empty()
        
        casesteplimit = PsiOpts.settings["auxsearch_op_casesteplimit"]
        caselimit = PsiOpts.settings["auxsearch_op_caselimit"]
        
        
        
        write_pf_enabled = PsiOpts.settings.get("proof_enabled", False)
        
        
        if verbose_op:
            print("========= aux search op ========")
            print(self)
        
        self.presolve()
        
        
        if verbose_op:
            print("=========   expanded    ========")
            print(self)
        
        csallcomprv = self.allcomprv()
        csaux = Comp.empty()
        csauxi = Comp.empty()
        for a, ca in self.auxs:
            if not ca:
                csauxi += a
            else:
                csaux += a
        
        write_pf_twopass = write_pf_enabled and not csaux.isempty()
        if write_pf_twopass:
            PsiOpts.settings["proof_enabled"] = False
        
        allauxs = self.auxs
        csnonaux = csallcomprv - csaux - csauxi
        if not csnonaux.isempty():
            allauxs.append((csnonaux, False))
        csauxiall = csauxi + csnonaux
        csallcomprv = csaux + csauxiall
        
        csaux_id = IVarIndex()
        csaux_id.record(csaux)
        csauxiall_id = IVarIndex()
        csauxiall_id.record(csauxiall)
        csall_id = IVarIndex()
        csall_id.record(csallcomprv)
        
        csaux_is = [csall_id.get_index(a) for a in csaux.varlist]
        csauxiall_is = [csall_id.get_index(a) for a in csauxiall.varlist]
        
        csauxidep = Comp.empty()
        
        nvar = len(csallcomprv)
        n = len(self.regs)
        nreqcheck = 0
        nfinal = 0
        
        depgraph = [[False] * nvar for j in range(nvar)]
        
        xallcomprv = []
        xconscomprv = []
        xreq = []
        xcons = []
        xmultiuse = []
        xleaveone = []
        xaux = []
        xauxi = []
        xaux_avoid = []
        xoneuse_aux = []
        xcforall = []
        xauxs_incomp = []
        xreqcomprv = []
        xconsonly = []
        xconsonly_init = []
        
        init_reg = Region.universe()
        
        for i in range(n):
            x, c = self.regs[i]
            allcomprv = x.allcomprv()
            cur_auxs_incomp = RegionOp.auxs_incomp(allauxs, allcomprv)
            
            req = Region.universe()
            cons = []
            
            if x.get_type() == RegionType.INTER:
                for y, cy in x.regs:
                    if cy:
                        req &= y
                    else:
                        cons.append(y)
            elif x.get_type() == RegionType.NORMAL:
                if c:
                    req &= x
                else:
                    cons.append(x)
            
            conscomprv = Comp.empty()
            for con in cons:
                conscomprv += con.allcomprv()
            
            cur_multiuse = forall_multiuse
            aux = Comp.empty()
            #auxi = csauxi.inter(conscomprv)
            auxi = csauxi.inter(allcomprv)
            aux_avoid = []
            cavoid = Comp.empty()
            cavoidmask = 0
            cforall_c = Comp.empty()
            cforall = Comp.empty()
            for a, ca in allauxs:
                b = a.inter(allcomprv)
                bmask = csall_id.get_mask(b)
                if not b.isempty():
                    if ca:
                        aux += b
                        aux_avoid.append((b.copy(), cavoid.copy()))
                        for j in range(nvar):
                            if bmask & (1 << j) != 0:
                                for j2 in range(nvar):
                                    if cavoidmask & (1 << j2) != 0:
                                        depgraph[j2][j] = True
                        
                        if not cforall_c.isempty():
                            cur_multiuse = False
                            csauxidep += cforall_c
                    else:
                        cforall_c += b.inter(auxi)
                cavoid += b
                cavoidmask |= bmask
                #cavoid += a
            
            if len(cons) == 0:
                cur_multiuse = True
                
            cforall_c = Comp.empty()
            for a, ca in allauxs[::-1]:
                if ca:
                    if not a.inter(allcomprv).isempty():
                        cforall = cforall_c.copy()
                else:
                    cforall_c += a
                
            cur_leaveone = leaveone and cur_multiuse and len(cons) == 0
            

            innermost_auxi = Comp.empty()
            if auxsearch_aux_strengthen and len(cur_auxs_incomp) and not cur_auxs_incomp[0][1]:
                innermost_auxi = cur_auxs_incomp[0][0].copy()

            cons2 = cons
            cons = []
            for x in cons2:
                y = x.copy()
                
                if not innermost_auxi.isempty():
                    innermost_auxi_int = innermost_auxi.inter(y.allcomprv())
                    if not innermost_auxi_int.isempty():
                        y.aux += innermost_auxi_int
                        y.aux_strengthen(req.allcomprv())
                        y.aux -= innermost_auxi_int

                for v in aux:
                    y.substitute(v, Comp.rv(v.get_name() + "_R" + str(i)))
                cons.append(y)
            
            oneuse_aux = MHashSet()
            oneuse_aux.add(None)
            
            
            if req.isuniverse() and aux.isempty():
                pass
            else:
                nreqcheck += 1
            
            if len(cons) == 0:
                nfinal += 1
                
                if init_leaveone and aux.isempty() and csnonaux.super_of(auxi):
                    ofl = req.one_flipped()
                    if ofl is not None:
                        init_reg &= ofl
            
            cur_consonly = req.isuniverse() and aux.isempty()
            cur_consonly_init = cur_consonly and len(cons) == 1 and csnonaux.super_of(auxi)
            # if cur_consonly_init:
            #     init_reg &= cons[0]
                
            xallcomprv.append(allcomprv)
            xconscomprv.append(conscomprv)
            xreq.append(req)
            xcons.append(cons)
            xmultiuse.append(cur_multiuse)
            xleaveone.append(cur_leaveone)
            xaux.append(aux)
            xauxi.append(auxi)
            xaux_avoid.append(aux_avoid)
            xoneuse_aux.append(oneuse_aux)
            xcforall.append(cforall)
            xauxs_incomp.append(cur_auxs_incomp)
            xreqcomprv.append(req.allcomprv())
            xconsonly.append(cur_consonly)
            xconsonly_init.append(cur_consonly_init)
            
            if verbose_op:
                print("========= #" + iutil.strpad(str(i), 3, " requires ========"))
                print(req)
                if len(cons) > 0:
                    print("========= consequences  ========")
                    print("\nOR\n".join(str(con) for con in cons))
                if not aux.isempty() or not auxi.isempty():
                    print("=========   auxiliary   ========")
                    print(" ".join(("|" if c else "&") + str(a) for a, c in cur_auxs_incomp))
                print("Multiuse = " + str(cur_multiuse), ", Leave one = " + str(cur_leaveone))
                print("Cons only = " + str(cur_consonly))
                
                
        hint_aux_avoid = hint_aux_avoid + self.get_aux_avoid_list()
        
        mustcomp = Comp.empty()
        
        if not ignore_must:
            for a in csauxi:
                cmarkers = a.get_markers()
                cdict = {v: w for v, w in cmarkers}
                if cdict.get("mustuse", False):
                    mustcomp += a
        
        rcases = MHashSet()
        rcases.add((init_reg, [False] * n))
        
        oneuse_added = [False] * n
                
        res = MHashSet()
        
        rcases_hashset = set()
        
        #rcases_hashset.add(hash(rcases))
        
        oneuse_set = MHashSet()
        oneuse_set.add(([None] * len(csaux), []))
        
        max_iter_pow = 4
        #cur_max_iter = 200
        cur_max_iter = 800
        
        max_yield_pow = 3
        #cur_max_yield = 5
        cur_max_yield = 16
        
        if nreqcheck <= 1:
            cur_max_iter *= 1000000000
            cur_max_yield *= 1000000000
        
        did = True
        prev_did = True
        
        caselimit_warned = False
        caselimit_reached = False
        
        if verbose_op:
            print("=========    markers    ========")
            for a in csallcomprv:
                cmarkers = a.get_markers()
                if len(cmarkers) > 0:
                    print(iutil.strpad(str(a), 6, " : " + str(cmarkers)))
            print("Must use:  " + str(mustcomp))
            print("csnonaux:  " + str(csnonaux))
            print("csauxidep: " + str(csauxidep))
            print("Max iter: " + str(cur_max_iter) + "  Max yield: " + str(cur_max_yield))
            print("=========  init region  ========")
            print(init_reg)
            print("========= begin search  ========")
        
        cnstep = 0
        
        #while did and (max_iter <= 0 or cur_max_iter < max_iter):
        while (did or cnstep <= 1) and (max_iter <= 0 or cur_max_iter < max_iter):
            if PsiOpts.is_timer_ended():
                break
            cnstep += 1
            prev_did = did
            did = False
            #rcases3 = rcases
            nonsimple_did = False
            
            for i in range(n):
                if len(rcases) == 0:
                    break
                
                cur_consonly = xconsonly[i]
                cur_consonly_init = xconsonly_init[i]
                
                if cnstep > 1 and cur_consonly:
                    continue
                if cnstep == 1 and not cur_consonly:
                    continue
                
                x, c = self.regs[i]
                
                allcomprv = xallcomprv[i]
                conscomprv = xconscomprv[i]
                req = xreq[i]
                cons = xcons[i]
                cur_multiuse = xmultiuse[i]
                cur_leaveone = xleaveone[i]
                aux = xaux[i]
                auxi = xauxi[i]
                aux_avoid = xaux_avoid[i]
                oneuse_aux = xoneuse_aux[i]
                cforall = xcforall[i]
                cur_auxs_incomp = xauxs_incomp[i]
                reqcomprv = xreqcomprv[i]
                
                
                if not cur_consonly:
                    if not aux.isempty() or len(cons) > 0:
                        nonsimple_did = True
                
                if len(rcases) > caselimit:
                    caselimit_reached = True
                    if not caselimit_warned:
                        caselimit_warned = True
                        warnings.warn("Max number of cases " + str(caselimit) + " reached. May give false reject.", RuntimeWarning)
                        
                if len(cons) >= 2 and caselimit_reached:
                    continue
                
                rcases2 = rcases
                rcases = MHashSet()
                
                for rcase_tuple in rcases2:
                    rcase = rcase_tuple[0]
                    rcase_vis = rcase_tuple[1]
                    
                    rcur = None
                    
                    if aux.isempty():
                        rcur = req.implicated(rcase)
                    else:
                        #rcur = req.implicated(rcase).exists(aux).forall(csnonaux)
                        rcur = req.implicated(rcase).exists(aux).forall(csauxiall)
                    
                    rcases_toadd = MHashSet()
                    rcases_toadd.add((rcase.copy(), rcase_vis[:]))
                    
                    cur_yield = 0
                    
                    for oneaux in reversed(oneuse_set):
                        auxvis = oneaux[1]
                        if i in auxvis:
                            continue
                        
                        auxmasks = oneaux[0]
                        auxlist = [(None if a is None else csauxiall.from_mask(a)) for a in auxmasks]
                        
                        mustleft = Comp.empty()
                        if nfinal == 1 and len(cons) == 0 and not mustcomp.isempty():
                            cmustcomp = Comp.empty()
                            for i2 in auxvis:
                                cmustcomp += xauxi[i2].inter(mustcomp)
                            if not cmustcomp.isempty():
                                auxlistall = sum((a for a in auxlist if a is not None), Comp.empty())
                                mustleft = cmustcomp - auxlistall
                                if aux.isempty() and not mustleft.isempty():
                                    #print("MUST NOT")
                                    #print(str(mustcomp) + "  " + str(cmustcomp) + "  " + str(auxlistall))
                                    #print(rcur)
                                    continue
                                
                        
                        
                        rcur2 = rcur.copy()
                        if verbose_op_detail:
                            print("========= #" + iutil.strpad(str(i), 3, " step     ========"))
                            print("SUB " + " ".join(str(csaux[j]) + ":" + str(auxlist[j]) for j in range(len(csaux)) if auxlist[j] is not None))
                            print("DEP " + " ".join(str(i2) for i2 in auxvis))
                            
                        #print("=====================")
                        #print(rcur2)
                        
                        
                        if verbose_op_detail2:
                            print("========= #" + iutil.strpad(str(i), 3, " before indep ===="))
                            print(rcur2)
                        
                        #clcomp = csnonaux.copy()
                        clcomp = csauxiall - csauxidep
                        #clcomp = rcur2.imp_flippedonly().allcomprv() + csnonaux
                        #for i2 in auxvis:
                        #    clcomp -= xallcomprv[i2]
                        
                        #for i2 in reversed(tsorted):
                        #    if i2 != i and oneauxs[i2] is None:
                        #        clcomp += xallcomprv[i2]
                        for i2 in auxvis:
                            #tauxi = xauxi[i2] - clcomp
                            #tcond = xallcomprv[i2] - xauxi[i2]
                            #rcur2.iand_norename(Region.Ic(tauxi, clcomp, tcond).imp_flipped())
                            #print("TREG")
                            #print(Expr.Ic(tauxi, clcomp, tcond))
                            rcur2.iand_norename(RegionOp.auxs_icreg(xauxs_incomp[i2], clcomp - xallcomprv[i2], clcomp + xreqcomprv[i2]).imp_flipped())
                        
                            if verbose_op_detail2:
                                treg2 = RegionOp.auxs_icreg(xauxs_incomp[i2], clcomp - xallcomprv[i2], clcomp)
                                if not treg2.isuniverse():
                                    print("========= #" + iutil.strpad(str(i), 3, "  indep " + str(i2) + " ====="))
                                    print(treg2)
                                    print("clcomp=" + str(clcomp))
                                
                            clcomp += xallcomprv[i2] - csaux
                            
                        for i2 in range(n):
                            if i2 not in auxvis:
                                if rcase_vis[i2]:
                                    rcur2.remove_present(xaux[i2].added_suffix("_R" + str(i2)))
                            else:
                                for v in xaux[i2]:
                                    w = auxlist[csaux.varlist.index(v.varlist[0])]
                                    if w is not None:
                                        rcur2.substitute_aux(Comp.rv(v.get_name() + "_R" + str(i2)), w)
                        
                        for j in range(len(csaux)):
                            if auxlist[j] is not None:
                                rcur2.substitute_aux(csaux[j], auxlist[j])
                                #print("SUB  " + "; ".join([str(v) + ":" + str(w) for v, w in oneaux]))
                            
                        if verbose_op_detail2:
                            print("========= #" + iutil.strpad(str(i), 3, " after rename ===="))
                            print(rcur2)
                            
                        #print(rcur2.getaux())
                        #print(rcur2.getaux().get_markers())
                        #if i == 4:
                        #    return None
                            
                        hint_aux_add = [(csaux[i], auxlist[i]) for i in range(len(csaux)) if auxlist[i] is not None]
                        hint_aux_avoid_add = [(csaux[i], csallcomprv - auxlist[i]) for i in range(len(csaux)) if auxlist[i] is not None]
                        #print(rcur2)
                        
                        cdepgraph = [a[:] for a in depgraph]
                        for j in range(len(csaux)):
                            mask = auxmasks[j]
                            if mask is not None:
                                for j2 in range(len(csauxiall_is)):
                                    if mask & (1 << j2) != 0:
                                        cdepgraph[j][csauxiall_is[j2]] = True
                                        
                        for j in range(len(csaux)):
                            if auxlist[j] is None and aux.ispresent(csaux[j]):
                                cmask = 0
                                for j2 in range(len(csauxiall_is)):
                                    tdepgraph = [a[:] for a in cdepgraph]
                                    tdepgraph[j][csauxiall_is[j2]] = True
                                    if iutil.iscyclic(tdepgraph):
                                        cmask |= 1 << j2
                                if cmask != 0:
                                    hint_aux_avoid_add.append((csaux[j], csauxiall.from_mask(cmask)))
                                    if verbose_op_detail2:
                                        print("AVOID " + str(csaux[j]) + " : " + str(csauxiall.from_mask(cmask)))
                        
                        #rcaseallcomprv = rcase.allcomprv() + csnonaux
                        rcaseallcomprv = rcur2.imp_flippedonly().allcomprv() + csnonaux
                        #rcaseallcomprv = csnonaux
                        
                        creg_indep = None
                        
                        for rr in rcur2.check_getaux_inplace_gen(must_include = must_include + mustleft, 
                                single_include = single_include, hint_pair = hint_pair,
                                hint_aux = hint_aux + hint_aux_add, 
                                hint_aux_avoid = hint_aux_avoid + hint_aux_avoid_add,
                                max_iter = cur_max_iter, leaveone = cur_leaveone):
                            
                            cur_yield += 1
                            if cur_yield > cur_max_yield:
                                did = True
                                break
                            
                            stype = iutil.signal_type(rr)
                            if stype == "":
                                if len(cons) > 0 and iutil.list_iscomplex(rr):
                                    continue
                                
                                t_multiuse = cur_multiuse
                                if t_multiuse and not csauxidep.isempty() and any(csauxidep.ispresent(w) for v, w in rr):
                                    t_multiuse = False
                                    
                                if t_multiuse:
                                    if creg_indep is None:
                                        creg_indep = RegionOp.auxs_icreg(cur_auxs_incomp, rcaseallcomprv - allcomprv, rcaseallcomprv + reqcomprv)
                                        #creg_indep.simplify()
                                        #creg_indep = RegionOp.auxs_icreg(cur_auxs_incomp, clcomp - allcomprv, clcomp)
                                        #cauxi = auxi - rcaseallcomprv
                                        #ccond = allcomprv - auxi
                                        #ccompleft = rcaseallcomprv - cauxi - ccond
                                        #creg_indep = Region.Ic(cauxi, ccompleft, ccond)
                                        #for v in aux:
                                        #    creg_indep.substitute(v, Comp.rv(str(v) + "_R" + str(i)))
                                        #print("CREG")
                                        #print(rcur2)
                                        #print(creg_indep)
                                        
                                    rcases_toadd2 = rcases_toadd
                                    rcases_toadd = MHashSet()
                                    for rcase_toadd_tuple in rcases_toadd2:
                                        rcase_toadd = rcase_toadd_tuple[0]
                                        rcase_toadd_vis = rcase_toadd_tuple[1]
                                        for con in cons:
                                            ccon = con.copy()
                                            ccon.iand_norename(creg_indep)
                                            Comp.substitute_list(ccon, rr, suffix = "_R" + str(i))
                                            crcase = rcase_toadd.copy()
                                            #crcase.iand_norename(ccon)
                                            #crcase.simplify_quick(zero_group = 1)
                                            crcase.iand_simplify_quick(ccon)
                                            rcases_toadd.add((crcase, rcase_toadd_vis[:]))
                                    if rcases_toadd != rcases_toadd2:
                                        tauxlist = [(None if w is None else w.copy()) for w in auxlist]
                                        for v, w in rr:
                                            
                                            #print(">>>>" + str(csaux) + "  " + str(v) + "  " + str(w))
                                            
                                            tauxlist[csaux.varlist.index(v.varlist[0])] = w.copy()
                                        rr2 = [(csaux[j], tauxlist[j]) for j in range(len(csaux)) if tauxlist[j] is not None]
                                        if len(rr2) > 0:
                                            res.add(rr2)
                                        if verbose_op_step:
                                            print("ADD  " + " ".join([str(v) + ":" + str(w) for v, w in rr2]))
                                else:
                                    oneuse_added[i] = True
                                    #oneuse_aux.clear()
                                    
                                    rcases_toadd2 = rcases_toadd
                                    rcases_toadd = MHashSet()
                                    for rcase_toadd_tuple in rcases_toadd2:
                                        rcase_toadd = rcase_toadd_tuple[0]
                                        rcase_toadd_vis = rcase_toadd_tuple[1]
                                        if rcase_toadd_vis[i]:
                                            rcases_toadd.add(rcase_toadd_tuple)
                                        else:
                                            for con in cons:
                                                crcase = rcase_toadd.copy()
                                                #crcase.iand_norename(con)
                                                #crcase.simplify_quick(zero_group = 1)
                                                crcase.iand_simplify_quick(con)
                                                rcases_toadd.add((crcase, [rcase_toadd_vis[i2] or i2 == i for i2 in range(n)]))
                                             
                                    tauxmasks = auxmasks[:]
                                    
                                    for v, w in rr:
                                        tauxmasks[csaux.varlist.index(v.varlist[0])] = csauxiall_id.get_mask(w)
                                    
                                    #print("; ".join(str(v) + ":" + str(w) for v, w in rr))
                                    #print(cdepends)
                                    
                                    if oneuse_set.add((tauxmasks, auxvis + [i])):
                                        did = True
                                        if verbose_op_step:
                                            tauxlist = [(None if w is None else w.copy()) for w in auxlist]
                                            for v, w in rr:
                                                tauxlist[csaux.varlist.index(v.varlist[0])] = w.copy()
                                            rr2 = [(csaux[j], tauxlist[j]) for j in range(len(csaux)) if tauxlist[j] is not None]
                                            print("ONE  " + " ".join([str(v) + ":" + str(w) for v, w in rr2]))
                                    
                            elif stype == "leaveone":
                                if len(cons) == 0:
                                    rcases_toadd2 = rcases_toadd
                                    rcases_toadd = MHashSet()
                                    for rcase_toadd_tuple in rcases_toadd2:
                                        rcase_toadd = rcase_toadd_tuple[0]
                                        rcase_toadd_vis = rcase_toadd_tuple[1]
                                        #crcase = rcase_toadd & (rr[2] >= 0)
                                        #crcase.simplify_quick(zero_group = 1)
                                        crcase = rcase_toadd.copy()
                                        crcase.iand_simplify_quick(rr[2] >= 0)
                                        rcases_toadd.add((crcase, rcase_toadd_vis))
                                        
                                    if rcases_toadd != rcases_toadd2:
                                        tauxlist = [(None if w is None else w.copy()) for w in auxlist]
                                        for v, w in rr[1]:
                                            tauxlist[csaux.varlist.index(v.varlist[0])] = w.copy()
                                        rr2 = [(csaux[j], tauxlist[j]) for j in range(len(csaux)) if tauxlist[j] is not None]
                                        if len(rr2) > 0:
                                            res.add(rr2)
                                        if verbose_op_step:
                                            print("LVO  " + " ".join([str(v) + ":" + str(w) for v, w in rr2]))
                                
                            elif stype == "max_iter_reached":
                                did = True
                            
                            if len(rcases_toadd) > casesteplimit:
                                break
                            if len(rcases_toadd) == 0:
                                break
                            
                        if len(rcases_toadd) > casesteplimit:
                            break
                        if len(rcases_toadd) == 0:
                            break
                            
                        if cur_yield > cur_max_yield:
                            did = True
                            break
                                
                    rcases += rcases_toadd
                
                if verbose_op_detail and i < n - 1:
                    print("=========     cases     ========")
                    print("\nOR\n".join(str(rcase[0]) for rcase in rcases))
                
            if cnstep != 1 and not nonsimple_did:
                break
            # if not nonsimple_did:
            #     break
            
            rcases_hash = hash(rcases)
            if rcases_hash not in rcases_hashset:
                did = True
                rcases_hashset.add(rcases_hash)
                    
                
            if verbose_op_step:
                print("=========     cases     ========")
                print("\nOR\n".join(str(rcase[0]) for rcase in rcases))
                
            if len(rcases) == 0:
                break
            cur_max_iter = int(cur_max_iter * max_iter_pow)
            cur_max_yield = int(cur_max_yield * max_yield_pow)
            
            
        PsiOpts.settings["proof_enabled"] = write_pf_enabled
        
        if len(rcases) == 0:
                    
            if verbose_op:
                print("=========    success    ========")
                print(iutil.list_tostr_std(res.x))
            
            resrr = None
            
            #resrr = res.x
            
            if len(res.x) == 1:
                resrr = res.x[0]
            else:
                resrr = res.x
            
            if write_pf_enabled:
                if write_pf_twopass:
                    
                    pf = ProofObj.from_region(self, c = "Claim:")
                    PsiOpts.set_setting(proof_step_in = pf)
                    
                    pf = ProofObj.from_region(None, c = "Substitute:\n" + iutil.list_tostr_std(resrr))
                    PsiOpts.set_setting(proof_add = pf)
                    
                    cs = self.copy()
                    Comp.substitute_list(cs, resrr, isaux = True)
                    if cs.getaux().isempty():
                        with PsiOpts(proof_enabled = True):
                            cs.check()
                    
                    PsiOpts.set_setting(proof_step_out = True)
            return resrr
        
        return None
    
    
    def check_getaux_op_inplace(self, hint_pair = None, hint_aux = None):
        """Return whether implication is true, with auxiliary search result."""
        r = []
        #print("")
        #print(self)
        for x in self.regs:
            #print(x)
            if x.get_type() == RegionType.NORMAL:
                t = x.check_getaux(hint_pair, hint_aux)
            else:
                t = x.check_getaux_op_inplace(hint_pair, hint_aux)
                
            if self.get_type() == RegionType.UNION and t is not None:
                return t
            if self.get_type() == RegionType.INTER and t is None:
                return None
            r.append(t)
        
        if self.get_type() == RegionType.INTER:
            return r
        
        return None
    
        
    def check_getaux(self, hint_pair = None, hint_aux = None):
        """Return whether implication is true, with auxiliary search result."""
        truth = PsiOpts.settings["truth"]
        if truth is not None:
            with PsiOpts(truth = None):
                return (truth >> self).check_getaux(hint_pair, hint_aux)
            
        cs = self.copy()
        return cs.check_getaux_inplace(hint_pair = hint_pair, hint_aux = hint_aux)
        
    def check_getaux_gen(self, hint_pair = None, hint_aux = None):
        """Return whether implication is true, with auxiliary search result."""
        rr = self.check_getaux(hint_pair = hint_pair, hint_aux = hint_aux)
        if rr is not None:
            yield rr
        
    def check(self):
        """Return whether implication is true."""
        return self.check_getaux() is not None
    
    
    def evalcheck(self, f):
        truth = PsiOpts.settings["truth"]
        if truth is not None:
            with PsiOpts(truth = None):
                return (truth >> self).evalcheck(f)
        
        ceps = PsiOpts.settings["eps_check"]
        
        isunion = (self.get_type() == RegionType.UNION)
        for x, c in self.regs:
            if isunion ^ c ^ x.evalcheck(f):
                return isunion
        return not isunion
    
    def eval_max_violate(self, f):
        truth = PsiOpts.settings["truth"]
        if truth is not None:
            with PsiOpts(truth = None):
                return (truth >> self).eval_max_violate(f)
        
        ceps = PsiOpts.settings["eps_check"]
        
        if self.get_type() == RegionType.INTER:
            r = 0.0
            for x, c in self.regs:
                t = x.eval_max_violate(f)
                
                if c:
                    r = max(r, t)
                else:
                    if t <= ceps:
                        return numpy.inf
            return r
        
        elif self.get_type() == RegionType.UNION:
            r = numpy.inf
            for x, c in self.regs:
                t = x.eval_max_violate(f)
                
                if c:
                    r = min(r, t)
                else:
                    if t > ceps:
                        return 0.0
            return r
        
        return 0.0
    
    def implies_getaux(self, other, hint_pair = None, hint_aux = None):
        """Whether self implies other, with auxiliary search result."""
        return (self <= other).check_getaux(hint_pair, hint_aux)
        

    def istight(self, canon = False):
        return all(x.istight(canon) for x, c in self.regs)

    def tighten(self):
        for x, c in self.regs:
            x.tighten()
        
    def simplify_op(self):
        if self.isuniverse():
            self.setuniverse()
            return self
        if self.isempty():
            self.setempty()
            return self
        
        if self.get_type() == RegionType.INTER or self.get_type() == RegionType.UNION:
            if len(self.auxs) == 0:
                tregs = []
                for x, c in self.regs:
                    if x.isuniverse(c ^ (self.get_type() == RegionType.UNION)):
                        continue
                    
                    if c and x.get_type() == self.get_type() and len(x.auxs) == 0:
                        tregs += x.regs
                        self.auxs = x.auxs + self.auxs
                    else:
                        tregs.append((x, c))
                self.regs = tregs
        
        self.aux_clean()
        
        return self
        
        
    def simplify_quick(self, reg = None, zero_group = 0):
        """Simplify a region in place, without linear programming. 
        Optional argument reg with constraints assumed to be true. 
        zero_group = 2: group all nonnegative terms as a single inequality.
        """
        
        if not PsiOpts.settings.get("simplify_enabled", False):
            return self
        
        #self.distribute()
        #self.remove_missing_aux()
        
        for x, c in self.regs:
            x.simplify_quick(reg, zero_group)
        
        self.simplify_op()
        return self
    
    def aux_push(self):
        for i in range(len(self.regs)):
            for x, c in self.auxs:
                if c:
                    self.regs[i] = (self.regs[i][0].exists(x), self.regs[i][1])
                else:
                    self.regs[i] = (self.regs[i][0].forall(x), self.regs[i][1])
        self.auxs = []
        return self
    
    def simplify_union(self, reg = None):
        """Simplify a union region in place. May take much longer than Region.simplify().
        Optional argument reg with constraints assumed to be true. 
        """
        if reg is None:
            reg = Region.universe()
            
        # print("simplify_union pre distribute")
        # print(self)
        
        self.distribute()
        
        # print("simplify_union post distribute")
        # print(self)
        
        if not self.get_type() == RegionType.UNION:
            return
        if any(not c for x, c in self.auxs):
            return
        #self.aux_push()
        aux = Comp.empty()
        for x, c in self.auxs:
            aux += x
        
        regc = [i for i in range(len(self.regs)) if self.regs[i][1]]
        for i in regc:
            self.regs[i][0].eliminate(aux)
            self.regs[i][0].simplify()
            self.regs[i][0].remove_aux(aux)
            
        
        regs_rem = [False for x, c in self.regs]
        for i, j in itertools.permutations(regc, 2):
            if regs_rem[i] or regs_rem[j]:
                continue
            #print("###")
            if (self.regs[i][0].exists(aux.inter(self.regs[i][0].allcomprv())) 
                & reg).implies(self.regs[j][0].exists(aux.inter(self.regs[j][0].allcomprv()))):
                regs_rem[i] = True
        self.regs = [(x, c) for i, (x, c) in enumerate(self.regs) if not regs_rem[i]]
        
        # print("simplify_union output")
        # print(self)
        
        return self
        
        
    def imp_present(self):
        return True

    def var_mi_only(self, v):
        return all(x.var_mi_only(v) for x, c in self.regs)

    def sort(self):
        pass
        
    def simplify(self, reg = None, zero_group = 0, **kwargs):
        """Simplify a region in place. 
        Optional argument reg with constraints assumed to be true. 
        zero_group = 2: group all nonnegative terms as a single inequality.
        """
        
        if kwargs:
            r = None
            with PsiOpts(**{"simplify_" + key: val for key, val in kwargs.items()}):
                r = self.simplify(reg, zero_group)
            return r

        
        if not PsiOpts.settings.get("simplify_enabled", False):
            return self
        
        simplify_redundant_op = (PsiOpts.settings.get("simplify_redundant_op", False) 
                                 and not PsiOpts.settings.get("simplify_quick", False))
                
        #self.distribute()
        self.remove_missing_aux()
        
        r_assumed = None
        if reg is None:
            r_assumed = Region.universe()
        else:
            r_assumed = reg.copy()
            
        if self.get_type() == RegionType.INTER or self.get_type() == RegionType.UNION:
            isunion = (self.get_type() == RegionType.UNION)
            for c_pass in [not isunion, isunion]:
                for x, c in self.regs:
                    if c == c_pass:
                        x.simplify(r_assumed, zero_group)
                        if c ^ isunion:
                            t = x.tosimple_noaux()
                            if t is not None:
                                r_assumed &= t
                                
            if simplify_redundant_op:
                regs_s = [(x.tosimple_noaux() if not c ^ isunion else None) for x, c in self.regs]
                regs_rem = [False for x, c in self.regs]
                for i, j in itertools.permutations([i for i in range(len(self.regs)) if regs_s[i] is not None], 2):
                    
                    if PsiOpts.is_timer_ended():
                        break
                    if regs_rem[i] or regs_rem[j]:
                        continue
                    #print("###")
                    
                    t_assumed = r_assumed.copy()
                    for k in range(len(self.regs)):
                        if k != i and not regs_rem[k] and not self.regs[k][1] ^ isunion:
                            tf = self.regs[k][0].one_flipped()
                            if tf is not None:
                                t_assumed &= tf
                                
                    if (regs_s[i] & t_assumed).implies(regs_s[j]):
                        regs_rem[i] = True
                self.regs = [(x, c) for i, (x, c) in enumerate(self.regs) if not regs_rem[i]]
            
        
        self.simplify_op()
        
        if PsiOpts.settings.get("simplify_union", False):
            self.simplify_union(reg)
            
        return self
    
    
    def simplified_quick(self, reg = None, zero_group = 0):
        """Returns the simplified region
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        r = self.copy()
        r.simplify_quick(reg, zero_group)
        t = r.tosimple_safe()
        if t is not None:
            return t
        return r
    
    def simplified(self, reg = None, zero_group = 0, **kwargs):
        """Returns the simplified region
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        r = self.copy()
        r.simplify(reg, zero_group, **kwargs)
        t = r.tosimple_safe()
        if t is not None:
            return t
        return r
    
        
    def add_aux(self, aux, c):
        if len(self.auxs) > 0 and self.auxs[-1][1] == c:
            self.auxs[-1]= (self.auxs[-1][0] + aux, self.auxs[-1][1])
        else:
            self.auxs.append((aux.copy(), c))
        
        
    def remove_aux(self, w):
        t = self.auxs
        self.auxs = []
        for x, c in t:
            y = x - w
            if not y.isempty():
                self.auxs.append((y, c))
        
        
    def remove_missing_aux(self):
        #return
        t = self.auxs
        self.auxs = []
        allcomp = self.allcomprv()
        for x, c in t:
            y = x.inter(allcomp)
            if not y.isempty():
                self.auxs.append((y, c))
        
    def eliminate(self, w, reg = None, toreal = False, forall = False, quick = False):
        
        w = Region.get_allcomp(w)
            
        toelim = Comp.empty()
        for v in w.allcomp():
            if toreal or v.get_type() == IVarType.REAL:
                toelim += v
            elif v.get_type() == IVarType.RV:
                self.add_aux(v, not forall)
        
        if not toelim.isempty():
            if forall:
                self.negate()
            # print(self)
            self.distribute()
            # print(self)
            for x, c in self.regs:
                x.simplify_quick()
                if not c and x.ispresent(toelim):
                    self.setuniverse()
                    return self
            for x, c in self.regs:
                if x.ispresent(toelim):
                    x.eliminate(toelim, reg = reg, toreal = toreal, forall = not c, quick = quick)
                
            if forall:
                self.negate()
            
        
        return self
        
    def eliminate_quick(self, w, reg = None, toreal = False, forall = False):
        return self.eliminate(w, reg = reg, toreal = toreal, forall = forall, quick = True)
        
    def marginal_eliminate(self, w):
        for x in self.regs:
            x.marginal_eliminate(w)
        
    def kernel_eliminate(self, w):
        for x in self.regs:
            x.kernel_eliminate(w)
          
    def tostring(self, style = 0, tosort = False, lhsvar = "real", inden = 0, add_bracket = False,
                 small = False, skip_outer_exists = False):
        """Convert to string. 
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        
        style = iutil.convert_str_style(style)
        
        if isinstance(lhsvar, str) and lhsvar == "real":
            lhsvar = self.allcomprealvar()
            
        r = ""
        interstr = ""
        nlstr = "\n"
        notstr = "NOT"
        spacestr = " "
        if style & PsiOpts.STR_STYLE_PSITIP:
            notstr = "~"
        elif style & PsiOpts.STR_STYLE_LATEX:
            notstr = "\\lnot"
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                nlstr = "\\\\\n"
            spacestr = "\\;"
            
        if self.get_type() == RegionType.UNION:
            if style & PsiOpts.STR_STYLE_PSITIP:
                interstr = "|"
            elif style & PsiOpts.STR_STYLE_LATEX:
                interstr = PsiOpts.settings["latex_or"]
            else:
                interstr = "OR"
        if self.get_type() == RegionType.INTER:
            if style & PsiOpts.STR_STYLE_PSITIP:
                interstr = "&"
            elif style & PsiOpts.STR_STYLE_LATEX:
                interstr = PsiOpts.settings["latex_and"]
            else:
                interstr = "AND"
        
        if self.isuniverse(sgn = True, canon = True):
            if style & PsiOpts.STR_STYLE_PSITIP:
                return spacestr * inden + "RegionOp.universe()"
            elif style & PsiOpts.STR_STYLE_LATEX:
                return spacestr * inden + PsiOpts.settings["latex_region_universe"]
            else:
                return spacestr * inden + "Universe"
        
        if self.isuniverse(sgn = False, canon = True):
            if style & PsiOpts.STR_STYLE_PSITIP:
                return spacestr * inden + "RegionOp.empty()"
            elif style & PsiOpts.STR_STYLE_LATEX:
                return spacestr * inden + PsiOpts.settings["latex_region_empty"]
            else:
                return spacestr * inden + "{}"
        
        for x, c in reversed(self.auxs):
            if c:
                if style & PsiOpts.STR_STYLE_PSITIP:
                    pass
                elif style & PsiOpts.STR_STYLE_LATEX:
                    if not style & PsiOpts.STR_STYLE_LATEX_QUANTAFTER:
                        r += PsiOpts.settings["latex_exists"] + " "
                        r += x.tostring(style = style, tosort = tosort)
                        r += PsiOpts.settings["latex_quantifier_sep"] + " "
            else:
                if style & PsiOpts.STR_STYLE_PSITIP:
                    pass
                elif style & PsiOpts.STR_STYLE_LATEX:
                    if not style & PsiOpts.STR_STYLE_LATEX_QUANTAFTER:
                        r += PsiOpts.settings["latex_forall"] + " "
                        r += x.tostring(style = style, tosort = tosort)
                        r += PsiOpts.settings["latex_quantifier_sep"] + " "
            
                
        inden_inner = inden
        inden_inner1 = inden + 2
        
        if style & PsiOpts.STR_STYLE_PSITIP:
            r += spacestr * inden + "(" + nlstr
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += spacestr * inden + "\\left\\{\\begin{array}{l}\n"
                inden_inner = 0
                inden_inner1 = 0
            else:
                r += spacestr * inden + "\\{" + nlstr
        else:
            r += spacestr * inden + "{" + nlstr
        
        rlist = [spacestr * inden_inner1 + ("" if c else " " + notstr) + 
                x.tostring(style = style, tosort = tosort, lhsvar = lhsvar, inden = inden_inner1, 
                           add_bracket = True, small = small).lstrip() 
                for x, c in self.regs]
        if tosort:
            rlist = zip(rlist, [any(x.ispresent(t) for t in lhsvar) for x, c in self.regs])
            rlist = sorted(rlist, key=lambda a: (not a[1], len(a[0]), a[0]))
            rlist = [x for x, t in rlist]
            
        r += (nlstr + spacestr * inden_inner + " " + interstr + nlstr).join(rlist)
        
                
        r += nlstr + spacestr * inden_inner
        if style & PsiOpts.STR_STYLE_PSITIP:
            r += ")"
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += "\\end{array}\\right\\}"
            else:
                r += "\\}"
        else:
            r += "}"
            
        for x, c in self.auxs:
            if c:
                if style & PsiOpts.STR_STYLE_PSITIP:
                    r += ".exists("
                elif style & PsiOpts.STR_STYLE_LATEX:
                    if style & PsiOpts.STR_STYLE_LATEX_QUANTAFTER:
                        r += " , " + PsiOpts.settings["latex_exists"] + " "
                    else:
                        continue
                else:
                    r += " , exists "
            else:
                if style & PsiOpts.STR_STYLE_PSITIP:
                    r += ".forall("
                elif style & PsiOpts.STR_STYLE_LATEX:
                    if style & PsiOpts.STR_STYLE_LATEX_QUANTAFTER:
                        r += " , " + PsiOpts.settings["latex_forall"] + " "
                    else:
                        continue
                else:
                    r += " , forall "
            r += x.tostring(style = style, tosort = tosort)
            
            if style & PsiOpts.STR_STYLE_PSITIP:
                r += ")"
            
        return r
            
        
    def __hash__(self):
        #return hash(self.tostring(tosort = True))
        
        return hash((self.rtype,
            hash(frozenset((hash(x), c) for x, c in self.regs)),
            hash(tuple((hash(x), c) for x, c in self.auxs)),
            hash(self.inp), hash(self.oup)
            ))
        
         
    
class MonotoneSet:
    
    def __init__(self, sgn = 1):
        self.sgn = sgn
        self.cache = []
        
    def add(self, x):
        if self.sgn > 0:
            self.cache = [y for y in self.cache if not y >= x]
        elif self.sgn < 0:
            self.cache = [y for y in self.cache if not y <= x]
        self.cache.append(x)
    
    def __contains__(self, x):
        for i in range(len(self.cache)):
            if (self.sgn > 0 and x >= self.cache[i]) or (self.sgn < 0 and x <= self.cache[i]):
                for j in range(i, 0, -1):
                    self.cache[j - 1], self.cache[j] = self.cache[j], self.cache[j - 1]
                return True
        return False
        
    
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, key):
        return self.cache[key]
        
    
    
class IBaseArray(IBaseObj):
    def __init__(self, x = None, shape = None):
        if x is None:
            x = []
        if isinstance(x, dict):
            x = list(x.items())

        cshape = None
        if isinstance(x, type(self).entry_cls):
            self.x = [x]
            cshape = tuple()
        elif iutil.istensor(x):
            cshape = tuple(x.shape)
            for xs in itertools.product(*[range(t) for t in cshape]):
                self.x.append(self.entry_convert(x[xs]))
        else:
            
            self.x = []
            tshape = []
            def recur(i, y):
                if not isinstance(y, (list, tuple)):
                    self.x.append(self.entry_convert(y))
                    return
                if len(tshape) <= i:
                    tshape.append(len(y))
                else:
                    if len(y) != tshape[i]:
                        raise ValueError("Shape mismatch.")
                        return
                for z in y:
                    recur(i + 1, z)
                
            recur(0, x)
            cshape = tuple(tshape)
            
        if shape is not None:
            self.shape = shape
        else:
            self.shape = cshape

    @property
    def shape(self):
        return self._shape
        # if len(self._shape) == 0:
        #     return self._shape
        # return self._shape + (len(self.x) // iutil.product(self._shape),)
    
    @shape.setter
    def shape(self, value):
        if isinstance(value, int):
            value = (value,)
        for i in range(len(value)):
            if value[i] < 0:
                tvalue = list(value)
                tvalue[i] = len(self.x) // (iutil.product(value[:i]) * iutil.product(value[i+1:]))
                value = tuple(tvalue)
                break
        self._shape = tuple(value)
    
    def reshaped(self, newshape):
        r = self.copy()
        r.shape = newshape
        return r
    
    def copy(self):
        return type(self)([a.copy() for a in self.x], shape = self.shape)
    
    @classmethod
    def empty(cls, shape = 0):
        if isinstance(shape, int):
            shape = (shape,)
        n = iutil.product(shape)
        return cls([cls.entry_cls_zero() for i in range(n)], shape = shape)
    
    @classmethod
    def zeros(cls, shape = 0):
        return cls.empty(shape = shape)
    
    @classmethod
    def ones(cls, shape = 0, x = None):
        if isinstance(shape, int):
            shape = (shape,)
        n = iutil.product(shape)
        if x is None:
            return cls([cls.entry_cls_one() for i in range(n)], shape = shape)
        else:
            return cls([x.copy() for i in range(n)], shape = shape)
    
    @classmethod
    def eye(cls, n, x = None):
        r = cls.zeros((n, n))
        for i in range(n):
            if x is None:
                r[i, i] = cls.entry_cls_one()
            else:
                r[i, i] = x.copy()
        return r
    
    @classmethod
    def make(cls, *args):
        r = cls.empty()
        for a in args:
            t = cls.entry_convert(a)
            if t is not None:
                r.append(t)
            else:
                for b in a:
                    r.append(b.copy())
        return r
    
    @classmethod
    def isthis(cls, x):
        if isinstance(x, cls):
            return True
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], cls.entry_cls):
            return True
        return False
    
    
    def to_dict(self):
        shape = self.shape
        if len(shape) < 2:
            return dict()
        r = dict()
        for xs in itertools.product(*[range(t) for t in shape[:-1]]):
            r[self[xs + (0,)]] = self[xs + (1,)]
        return r

    
    def allcomp(self):
        return sum([a.allcomp() for a in self.x], Comp.empty())
    
    
    def find_name(self, *args):
        return self.allcomp().find_name(*args)
    
    def from_mask(self, mask):
        """Return subset using bit mask."""
        r = type(self).entry_cls_zero()
        for i in range(len(self.x)):
            if mask & (1 << i) != 0:
                r += self.x[i]
        return r
        
    
    def append(self, a):
        if len(self.shape) != 1:
            raise ValueError("Can only append 1D array.")
            return
        self.x.append(a)
        self.shape = (self.shape[0] + 1,)
    
    def swapped_id(self, i, j):
        if i >= len(self.x) or j >= len(self.x):
            return self.copy()
        r = self.copy()
        r.x[i], r.x[j] = r.x[j], r.x[i]
        return r
    
    def transpose(self):
        r = type(self).zeros(tuple(reversed(self.shape)))
        for xs in itertools.product(*[range(t) for t in self.shape]):
            r[tuple(reversed(xs))] = self[xs]
        return r
    
    
    @fcn_substitute
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), in place."""
        for i in range(len(self.x)):
            self.x[i].substitute(v0, v1)
        return self
    
    @fcn_substitute
    def substitute_aux(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), and remove auxiliary v0, in place."""
        for i in range(len(self.x)):
            self.x[i].substitute_aux(v0, v1)
        return self
    
    def substituted(self, *args, **kwargs):
        """Substitute variable v0 by v1 (v1 can be compound), return result"""
        r = self.copy()
        r.substitute(*args, **kwargs)
        return r
        
    def substituted_aux(self, *args, **kwargs):
        """Substitute variable v0 by v1 (v1 can be compound), and remove auxiliary v0, return result"""
        r = self.copy()
        r.substitute_aux(*args, **kwargs)
        return r
    
    def set_len(self, n):
        if n < len(self.x):
            self.x = self.x[:n]
            return
        while n > len(self.x):
            self.x.append(type(self).entry_cls_zero())
    
    def __neg__(self):
        return type(self)([-a for a in self.x])
    
    def __iadd__(self, other):
        if iutil.istensor(other):
            if self.shape != other.shape:
                raise ValueError("Shape mismatch.")
                return
            for xs in itertools.product(*[range(t) for t in self.shape]):
                self[xs] += other[xs]
            return self
            
        # if isinstance(other, IBaseArray):
        #     r = []
        #     for i in range(len(other.x)):
        #         if i < len(self.x):
        #             self.x[i] += other.x[i]
        #         else:
        #             self.x.append(other.x[i].copy())
        #     return self
        
        for i in range(len(self.x)):
            self.x[i] += other
        return self
    
    def __imul__(self, other):
        if iutil.istensor(other):
            if self.shape != other.shape:
                raise ValueError("Shape mismatch.")
                return
            for xs in itertools.product(*[range(t) for t in self.shape]):
                self[xs] *= other[xs]
            return self
        
        for i in range(len(self.x)):
            self.x[i] *= other
        return self
    
    def __itruediv__(self, other):
        if iutil.istensor(other):
            if self.shape != other.shape:
                raise ValueError("Shape mismatch.")
                return
            for xs in itertools.product(*[range(t) for t in self.shape]):
                self[xs] /= other[xs]
            return self
        
        for i in range(len(self.x)):
            self.x[i] /= other
        return self
    
    def __ipow__(self, other):
        if iutil.istensor(other):
            if self.shape != other.shape:
                raise ValueError("Shape mismatch.")
                return
            for xs in itertools.product(*[range(t) for t in self.shape]):
                self[xs] **= other[xs]
            return self
        
        for i in range(len(self.x)):
            self.x[i] **= other
        return self
    
    
    def __mul__(self, other):
        r = self.copy()
        r *= other
        return r
    
    def __rmul__(self, other):
        r = self.copy()
        r *= other
        return r
    
    def __truediv__(self, other):
        r = self.copy()
        r /= other
        return r
    
    def __rtruediv__(self, other):
        r = type(self).ones(len(self.x))
        r *= other
        r /= self
        return r
    
    def __pow__(self, other):
        r = self.copy()
        r **= other
        return r
    
    def __rpow__(self, other):
        r = type(self).ones(len(self.x))
        r *= other
        r **= self
        return r
    
    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self.copy()
        r = self.copy()
        r += other
        return r
    
    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self.copy()
        r = self.copy()
        r += other
        return r
    
    def __isub__(self, other):
        self += -other
        return self
    
    def __sub__(self, other):
        r = self.copy()
        r += -other
        return r
    
    def __rsub__(self, other):
        r = -self
        r += other
        return r
    
    def __len__(self):
        return len(self.x)
    
    def key_to_id(self, key):
        if isinstance(key, int):
            key = (key,)
        if isinstance(key, tuple) and all(isinstance(xs, int) for xs in key):
            shape = self.shape
            if len(key) != len(shape):
                raise IndexError("Dimension mismatch.")
                return
            i = 0
            for s, k in zip(shape, key):
                if k < 0 or k >= s:
                    raise IndexError("Index out of bound.")
                    return
                i = i * s + k
            return i
        return 0
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.x[key]
        if isinstance(key, slice):
            r = self.x[key]
            if isinstance(r, list):
                return type(self)(r)
            return r
        return self.x[self.key_to_id(key)]
    
    def __setitem__(self, key, item):
        if isinstance(key, int):
            self.x[key] = self.entry_convert(item)
        self.x[self.key_to_id(key)] = self.entry_convert(item)
    
    
    def dot(self, other):
        """ Dot product like numpy.dot.
        """
        if isinstance(other, list):
            other = type(self)(other)
        if not iutil.istensor(other):
            return self * other
        selfshape = self.shape
        othershape = other.shape
        self_np = max(len(selfshape) - 1, 0)
        other_np = max(len(othershape) - 2, 0)
        if selfshape[self_np] != othershape[other_np]:
            raise ValueError("Shape mismatch.")
            return
        
        r = type(self).zeros(selfshape[:self_np] + othershape[:other_np] + othershape[other_np+1:])
        for xs in itertools.product(*[range(t) for t in selfshape[:self_np]]):
            for ys in itertools.product(*[range(t) for t in othershape[:other_np]]):
                for i in range(selfshape[self_np]):
                    for zs in itertools.product(*[range(t) for t in othershape[other_np+1:]]):
                        r[xs + ys + zs] += self[xs + (i,)] * other[ys + (i,) + zs]
                        
        if len(r.shape) == 0:
            return r[tuple()]
        return r
    
    def __matmul__(self, other):
        return self.dot(other)
    
    def __imatmul__(self, other):
        return (type(self)(other)).dot(self)
                    
    def trace_mat(self):
        """Trace of matrix.
        """
        selfshape = self.shape
        r = type(self).zeros(selfshape[2:])
        for i in range(min(selfshape[0], selfshape[1])):
            for xs in itertools.product(*[range(t) for t in selfshape[2:]]):
                r[xs] += self[(i, i) + xs]
                
        if len(r.shape) == 0:
            return r[tuple()]
        return r
    
    def trace(self):
        """Trace.
        """
        selfshape = self.shape
        n = min(selfshape)
        return sum((self[(i,) * len(selfshape)] for i in range(n)), type(self).entry_cls_zero())
    
    def diag(self):
        """Return diagonal.
        """
        selfshape = self.shape
        n = min(selfshape)
        r = []
        for i in range(n):
            r.append(self[(i,) * len(selfshape)])
        return type(self)(r)
    
    def record_to(self, index):
        for a in self.x:
            a.record_to(index)
            
    def isregtermpresent(self):
        for a in self.x:
            if a.isregtermpresent():
                return True
        return False
    
    
    def get_sum(self):
        """Sum of all entries.
        """
        return sum(self.x, type(self).entry_cls_zero())
    
    def avg(self):
        """Average of all entries.
        """
        return self.get_sum() / len(self)
    
    def tostring(self, style = 0, tosort = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        if len(self.shape) == 0:
            return ""

        style = iutil.convert_str_style(style)
        nlstr = "\n"
        if style & PsiOpts.STR_STYLE_LATEX:
            nlstr = "\\\\\n"
            
        shape = self.shape
        r = ""
        add_bracket = True
        list_bracket0 = "["
        list_bracket1 = "]"
        if style & PsiOpts.STR_STYLE_LATEX:
            # list_bracket0 = ""
            # list_bracket1 = ""
            list_bracket0 = PsiOpts.settings["latex_list_bracket_l"]
            list_bracket1 = PsiOpts.settings["latex_list_bracket_r"]
        
        if style & PsiOpts.STR_STYLE_PSITIP:
            r += type(self).cls_name + "("
            if len(shape) > 1:
                r += nlstr
            add_bracket = False
        
        if style & PsiOpts.STR_STYLE_LATEX:
            r += "\\left" + list_bracket0 + " "
            r += "\\begin{array}{" + "c" * self.shape[-1] + "}\n"

        # if style & PsiOpts.STR_STYLE_PSITIP:
        #     r += type(self).cls_name + "([ "
        #     add_bracket = False
        # elif style & PsiOpts.STR_STYLE_LATEX:
        #     if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
        #         r += "\\left\\[\\begin{array}{l}\n"
        #         add_bracket = False
        #     else:
        #         r += "\\[ "
        # else:
        #     r += "[ "
        
        for xs in itertools.product(*[range(t) for t in shape]):
            si0 = len(shape)
            while si0 > 0 and xs[si0 - 1] == 0:
                si0 -= 1
            si1 = len(shape)
            while si1 > 0 and xs[si1 - 1] == shape[si1 - 1] - 1:
                si1 -= 1
            if si0 < len(shape) and not style & PsiOpts.STR_STYLE_LATEX:
                r += " " * si0 + list_bracket0 * (len(shape) - si0)
            
            if type(self).tostring_bracket_needed:
                r += self[xs].tostring(style = style, tosort = tosort, add_bracket = add_bracket)
            else:
                r += self[xs].tostring(style = style, tosort = tosort)
            
            if si1 < len(shape) and not style & PsiOpts.STR_STYLE_LATEX:
                r += list_bracket1 * (len(shape) - si1)
            
            if style & PsiOpts.STR_STYLE_LATEX:
                if si1 > 0:
                    if si1 == len(shape):
                        r += " & "
                    else:
                        r += nlstr * (len(shape) - si1)

            else:
                if si1 > 0:
                    r += ","
                    if si1 == len(shape):
                        r += " "
                    else:
                        r += nlstr * (len(shape) - si1)
            
            
        if style & PsiOpts.STR_STYLE_PSITIP:
            r += ")"
        
        if style & PsiOpts.STR_STYLE_LATEX:
            r += "\\end{array}"
            r += "\\right" + list_bracket1
        
        return r
    
    
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"], PsiOpts.settings["str_tosort"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.settings["str_style_repr"])
    
    def _latex_(self):
        return self.tostring(iutil.convert_str_style("latex"))
        
    

class CompArray(IBaseArray):
    cls_name = "CompArray"
    entry_cls = Comp
    entry_cls_zero = Comp.empty
    entry_cls_one = None
    tostring_bracket_needed = True
    
    @staticmethod
    def entry_convert(a):
        if isinstance(a, Comp):
            return a.copy()
        return None
    
    def arg_convert(b):
        if isinstance(b, list) or isinstance(b, Comp):
            return CompArray.make(*b)
        return b
    
    
    def get_comp(self):
        return sum(self.x, Comp.empty())
    
    def get_term(self):
        return Term([a.copy() for a in self.x])
    
    
    
    def series(self, vdir):
        """Get past or future sequence.
        Parameters:
            vdir  : Direction, 1: future non-strict, 2: future strict,
                    -1: past non-strict, -2: past strict
        """
        if vdir == 1:
            return CompArray([sum(self.x[i:], Comp.empty()) for i in range(len(self.x))])
        elif vdir == 2:
            return CompArray([sum(self.x[i+1:], Comp.empty()) for i in range(len(self.x))])
        elif vdir == -1:
            return CompArray([sum(self.x[:i+1], Comp.empty()) for i in range(len(self.x))])
        elif vdir == -2:
            return CompArray([sum(self.x[:i], Comp.empty()) for i in range(len(self.x))])
        return self.copy()
        
    def past_ns(self):
        return self.series(-1)
        
    def past(self):
        return self.series(-2)
    
    def future_ns(self):
        return self.series(1)
    
    def future(self):
        return self.series(2)
    
    def series_list(self, name = None, suf0 = "Q", sufp = "P", suff = "F"):
        if name is None:
            if len(self.x) == 0:
                name = ""
            else:
                name = self.x[0].get_name()
                for a in self.x[1:]:
                    tname = a.get_name()
                    while not tname.startswith(name):
                        name = name[:-1]
        
        r = []
        if suf0 is not None:
            r.append((Comp.rv(name + suf0), self.copy()))
        if sufp is not None:
            r.append((Comp.rv(name + sufp), self.past()))
        if suff is not None:
            r.append((Comp.rv(name + suff), self.future()))
        return r
    
    @staticmethod
    def series_sym(x, sufp = "P", suff = "F"):
        if isinstance(x, str):
            x = Comp.rv(x)
        r = CompArray.empty()
        r.append(x)
        rename_char = PsiOpts.settings["rename_char"]
        if sufp is not None:
            r.append(Comp.rv(iutil.set_suffix_num(x.get_name(), sufp, rename_char, replace_mode = "append")))
        if suff is not None:
            r.append(Comp.rv(iutil.set_suffix_num(x.get_name(), suff, rename_char, replace_mode = "append")))
        return r
    
    
    def __and__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return ExprArray([a & b for a, b in zip(self.x, other.x)], shape = self.shape)
        return ExprArray([a & other for a in self.x], shape = self.shape)
    
    def __or__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return ExprArray([a | b for a, b in zip(self.x, other.x)], shape = self.shape)
        return ExprArray([a | other for a in self.x], shape = self.shape)
    
    def __rand__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return ExprArray([b & a for a, b in zip(self.x, other.x)], shape = self.shape)
        return ExprArray([other & a for a in self.x], shape = self.shape)
    
    def __ror__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return ExprArray([b | a for a, b in zip(self.x, other.x)], shape = self.shape)
        return ExprArray([other | a for a in self.x], shape = self.shape)
        
    def mark(self, *args):
        for a in self.x:
            a.mark(*args)
        return self
    
    def set_card(self, m):
        for a in self.x:
            a.set_card(m)
        return self
    
    def get_card(self):
        return self.get_comp().get_card()
    
    def get_shape(self):
        r = []
        for a in self.x:
            t = a.get_card()
            if t is None:
                raise ValueError("Cardinality of " + str(a) + " not set. Use " + str(a) + ".set_card(m) to set cardinality.")
                return
            r.append(t)
        return tuple(r)
        
        
    
class ExprArray(IBaseArray):
    cls_name = "ExprArray"
    entry_cls = Expr
    entry_cls_zero = Expr.zero
    entry_cls_one = Expr.one
    tostring_bracket_needed = False
    
    
    @staticmethod
    def entry_convert(a):
        if isinstance(a, Expr):
            return a.copy()
        elif isinstance(a, Term):
            return a.copy()
        elif isinstance(a, (int, float)):
            return Expr.const(a)
        return None
    
    
    def get_expr(self):
        return sum(self.x, Expr.zero())
    
    
    
    def series(self, vdir):
        """Get past or future sequence.
        Parameters:
            vdir  : Direction, 1: future non-strict, 2: future strict,
                    -1: past non-strict, -2: past strict
        """
        if vdir == 1:
            return ExprArray([sum(self.x[i:], Expr.zero()) for i in range(len(self.x))])
        elif vdir == 2:
            return ExprArray([sum(self.x[i+1:], Expr.zero()) for i in range(len(self.x))])
        elif vdir == -1:
            return ExprArray([sum(self.x[:i+1], Expr.zero()) for i in range(len(self.x))])
        elif vdir == -2:
            return ExprArray([sum(self.x[:i], Expr.zero()) for i in range(len(self.x))])
        return self.copy()
        
        
    def past_ns(self):
        return self.series(-1)
        
    def past(self):
        return self.series(-2)
    
    def future_ns(self):
        return self.series(1)
    
    def future(self):
        return self.series(2)
    
    
    
    def __abs__(self):
        return eabs(self)
    
    def __and__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return ExprArray([a & b for a, b in zip(self.x, other.x)], shape = self.shape)
        return ExprArray([a & other for a in self.x], shape = self.shape)
    
    def __or__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return ExprArray([a | b for a, b in zip(self.x, other.x)], shape = self.shape)
        return ExprArray([a | other for a in self.x], shape = self.shape)
    
    def __rand__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return ExprArray([b & a for a, b in zip(self.x, other.x)], shape = self.shape)
        return ExprArray([other & a for a in self.x], shape = self.shape)
    
    def __ror__(self, other):
        if isinstance(other, CompArray) or isinstance(other, ExprArray):
            return ExprArray([b | a for a, b in zip(self.x, other.x)], shape = self.shape)
        return ExprArray([other | a for a in self.x], shape = self.shape)
    
    def ge_region(self):
        r = Region.universe()
        for a in self.x:
            r.exprs_ge.append(a.copy())
        return r
    
    def eq_region(self):
        r = Region.universe()
        for a in self.x:
            r.exprs_eq.append(a.copy())
        return r
    
    def __ge__(self, other):
        return (self - other).ge_region()
    
    def __le__(self, other):
        return (other - self).ge_region()
    
    def __eq__(self, other):
        return (self - other).eq_region()
    
    
    
    def fcn(fcncall, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return ExprArray([Expr.fcn(
            (lambda txs: (lambda P: fcncall(P, txs[0] if len(txs) == 1 else txs)))(xs)) 
            for xs in itertools.product(*[range(t) for t in shape])],
            shape = shape)
        
    
    # def tostring(self, style = 0, tosort = False):
    #     """Convert to string
    #     Parameters:
    #         style   : Style of string conversion
    #                   STR_STYLE_STANDARD : I(X,Y;Z|W)
    #                   STR_STYLE_PSITIP : I(X+Y&Z|W)
    #     """
    #     style = iutil.convert_str_style(style)
    #     nlstr = "\n"
    #     if style & PsiOpts.STR_STYLE_LATEX:
    #         nlstr = "\\\\\n"
            
    #     r = ""
    #     add_bracket = True
    #     if style & PsiOpts.STR_STYLE_PSITIP:
    #         r += "ExprArray([ "
    #         add_bracket = False
    #     elif style & PsiOpts.STR_STYLE_LATEX:
    #         if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
    #             r += "\\left\\[\\begin{array}{l}\n"
    #             add_bracket = False
    #         else:
    #             r += "\\[ "
    #     else:
    #         r += "[ "
        
    #     for i, a in enumerate(self.x):
    #         if i:
    #             if style & PsiOpts.STR_STYLE_LATEX:
    #                 if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
    #                     r += nlstr
    #                 else:
    #                     r += ", "
    #             else:
    #                 r += ", "
    #         r += a.tostring(style = style, tosort = tosort)
            
    #     if style & PsiOpts.STR_STYLE_PSITIP:
    #         r += " ])"
    #     elif style & PsiOpts.STR_STYLE_LATEX:
    #         if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
    #             r += "\\end{array}\\right\\]"
    #         else:
    #             r += " \\]"
    #     else:
    #         r += " ]"
        
    #     return r
    
    
    # def __str__(self):
    #     return self.tostring(PsiOpts.settings["str_style"], PsiOpts.settings["str_tosort"])
    
    # def __repr__(self):
    #     return self.tostring(PsiOpts.STR_STYLE_PSITIP)
        

class ivenn:
    
    def ellipse_contains(el, p):
        if len(el) < 4:
            el = (el[0], el[1], numpy.linalg.inv(el[1]), True)
        elcen, elm, elmi, elinc = el
        return not elinc ^ (numpy.linalg.norm(elmi.dot(p - elcen)) <= 1.0)
        
    def ellipse_angle(el, p):
        if len(el) < 4:
            el = (el[0], el[1], numpy.linalg.inv(el[1]), True)
        elcen, elm, elmi, elinc = el
        t = elmi.dot(p - elcen)
        return numpy.arctan2(t[1], t[0])
    
    def ellipse_clamp(el, p, border, inc, is_it):
        if len(el) < 4:
            el = (el[0], el[1], numpy.linalg.inv(el[1]), True)
        elcen, elm, elmi, elinc = el
        pme = p - elcen
        
        
        t = elmi.dot(pme)
        tn = numpy.linalg.norm(t)
        if tn <= 1e-9:
            return None
        if is_it:
            ndiv = 600
            md = 1e20
            mg = numpy.array([0.0, 0.0])
            for i in range(ndiv + 1):
                ia = i * numpy.pi * 2 / ndiv
                g = elm.dot(numpy.array([numpy.cos(ia), numpy.sin(ia)]))
                cd = numpy.linalg.norm(g - pme)
                if cd < md:
                    md = cd
                    mg = g
            if md < 1e-9:
                return None
            if (inc ^ (tn > 1.0)) and md >= border:
                return None
            if inc ^ (tn > 1.0):
                return (pme - mg) * (border / md) + mg + elcen
            else:
                return -(pme - mg) * (border / md) + mg + elcen
            
        else:
            border /= numpy.linalg.norm(pme) / tn
            if inc:
                if tn <= 1.0 - border:
                    return None
                t = t * (1.0 - border) / tn
            else:
                if tn >= 1.0 + border:
                    return None
                t = t * (1.0 + border) / tn
            return elm.dot(t) + elcen
    
    def ellipse_intersect_it(els, r, ndiv = 500, maxnp = 10000, cp = None, stop0 = None, stop1 = None, stopp = None):
        elcen, elm, elmi, elinc = els[0]
        
        started = False
        pcontain = True
        pcontainxi = -1
        
        angle0 = 0.0
        anglesn = 1.0 if elinc else -1.0
        if cp is not None:
            angle0 = ivenn.ellipse_angle(els[0], cp)
        
        for i in range(ndiv + 1):
            ia = anglesn * i * numpy.pi * 2 / ndiv + angle0
            p = elm.dot(numpy.array([numpy.cos(ia), numpy.sin(ia)])) + elcen
            ccontains = [e is els[0] or ivenn.ellipse_contains(e, p) for e in els]
            ccontain = all(ccontains)
            ccontainxi = -1
            for ei in range(len(els)):
                if not ccontains[ei]:
                    ccontainxi = ei
            if cp is None:
                if ccontain and not pcontain:
                    ivenn.ellipse_intersect_it(els, r, ndiv, maxnp, p, els[pcontainxi], els[0], p)
                    return
            else:
                if started and not ccontain:
                    for ei in range(len(els)):
                        if not ccontains[ei]:
                            if els[0] is stop0 and els[ei] is stop1 and numpy.linalg.norm(p - stopp) <= 0.02:
                                return
                            cels = [els[ei]] + [e for e in els if e is not els[ei]]
                            ivenn.ellipse_intersect_it(cels, r, ndiv, maxnp, p, stop0, stop1, stopp)
                            return
                    return
                if ccontain:
                    started = True
                if started:
                    r.append(p)
                    if len(r) >= maxnp:
                        print("Venn diagram intersection failure!")
                        return
            pcontain = ccontain
            pcontainxi = ccontainxi
            
        if cp is None:
            if len(els) >= 2:
                cels = els[1:] + [els[0]]
                ivenn.ellipse_intersect_it(cels, r, ndiv, maxnp, cp, stop0, stop1, stopp)
                return
                # print("Venn diagram includes all!")
            for i in range(ndiv):
                ia = anglesn * i * numpy.pi * 2 / ndiv + angle0
                p = elm.dot(numpy.array([numpy.cos(ia), numpy.sin(ia)])) + elcen
                r.append(p)
            return
    
    def ellipse_intersect(els, ndiv = 700):
        cels = [(e[0], e[1], numpy.linalg.inv(e[1]), e[2] if len(e) >= 3 else True) for e in els]
        r = []
        ivenn.ellipse_intersect_it(cels, r, ndiv)
        return r
        
    
    def ellipses(n):
        if n == 1:
            return [(numpy.array([0.0, 0.0]), numpy.eye(2))]
        elif n == 2:
            ratio = 0.68
            return [(numpy.array([ratio - 1.0, 0.0]), numpy.eye(2) * ratio),
                    (numpy.array([-ratio + 1.0, 0.0]), numpy.eye(2) * ratio)]
        elif n == 3:
            ratio = 0.65
            rad = -ratio + 1.0
            return [(numpy.array([-numpy.sin(i * numpy.pi * 2 / 3) * rad, 
                                  numpy.cos(i * numpy.pi * 2 / 3) * rad]), 
                     numpy.eye(2) * ratio) for i in range(3)]
        elif n == 4:
            mtilt0 = numpy.array([[1.0, 0.0], [-0.7, 1.0]])
            mtilt1 = numpy.array([[1.0, 0.0], [-0.8, 1.0]])
            mtilt2 = numpy.array([[1.0, 0.0], [0.8, 1.0]])
            mtilt3 = numpy.array([[1.0, 0.0], [0.7, 1.0]])
            return [(numpy.array([-0.4 * 1.011, -0.17]), mtilt0 * 0.7),
                    (numpy.array([-0.1, 0.15]), mtilt1 * 0.62),
                    (numpy.array([0.1, 0.15]), mtilt2 * 0.62),
                    (numpy.array([0.4 * 1.011, -0.17]), mtilt3 * 0.7)]
        elif n == 5:
            # 5-set Venn diagram by Branko Grunbaum
            ratio = 0.85
            r = []
            for i in range(5):
                a = -i * numpy.pi * 2 / 5 + 0.2
                cosa = numpy.cos(a)
                sina = numpy.sin(a)
                rmat = numpy.array([[cosa, -sina], [sina, cosa]])
                r.append((rmat.dot(numpy.array([0.04 * 2, 0.087 * 2])) * ratio, 
                     rmat.dot(numpy.array([[0.63, 0.0], [0.0, 1.0]])) * ratio))
            return r
    
    def intersect(n, inc):
        els = ivenn.ellipses(n)
        # ndiv = 500
        # if n >= 4:
        #     ndiv = 700
        # return ivenn.ellipse_intersect([e for i, e in enumerate(els) if inc[i]], ndiv = ndiv)
        return ivenn.ellipse_intersect([e for i, e in enumerate(els) if inc[i]])
    
    
    def calc_ellipses(n):
        els = ivenn.ellipses(n)
        elsi = [(el[0], el[1], numpy.linalg.inv(el[1]), True) for el in els]
        r = [[None, None, None] for i in range(1 << n)]
        
        tpolys = [None] * (1 << n)
        
        for mask in range((1 << n) - 1, 0, -1):
            maskbc = iutil.bitcount(mask)
            # r[mask][0] = ivenn.ellipse_intersect([e for i, e in enumerate(els) if mask & (1 << i)])
            r[mask][0] = ivenn.ellipse_intersect(
                [(e[0], e[1], bool(mask & (1 << i))) for i, e in enumerate(els)])
            if len(r[mask][0]) == 0:
                print("Venn diagram intersection empty!")
                
            
            if False:
                cen = sum(r[mask][0], numpy.array([0.0, 0.0])) / len(r[mask][0])
                # print(cen)
                subcen = numpy.array([0.0, 0.0])
                mask_other = ((1 << n) - 1) ^ mask
                for mask2 in igen.subset_mask(mask_other):
                    if mask2 == 0:
                        continue
                    mask3 = mask | mask2
                    subcen += r[mask3][1]
                
                # cborder = 0.55
                # if n == 3:
                #     if maskbc == 2:
                #         cborder = 0.7
                # elif n == 4:
                #     cborder = 0.8
                #cborder = 0.4 * 3.0 / (n + 2.0)
                cborder_max = 0.4
                nit = 500
                cborder_shrink = 0.05
                walk_ratio = 1.0
                # if n == 4:
                #     nit += 400
                #     # nit = 3
                #     cborder_shrink = 0.975
                
                cborder_step_init = 0.02
                cborder_step_shrink = 0.1
                    
                if n == 5:
                    # nit += 400
                    # nit = 3
                    cborder_shrink = 0.005
                
                if mask_other != 0:
                    subcen /= ((1 << iutil.bitcount(mask_other)) - 1)
                    # r[mask][1] = cen * 5 - subcen * 4
                    r[mask][1] = cen
                    
                    
                    # tpoly = ivenn.ellipse_intersect([(e[0], e[1], bool(mask & (1 << i))) for i, e in enumerate(els)], ndiv = 2000)
                    # r[mask][1] = sum(tpoly, numpy.array([0.0, 0.0])) / len(tpoly)
                    # tpolys[mask] = tpoly
                    
                    cborder = 0.0
                    for it in range(nit):
                        # cborder = cborder_max * (cborder_shrink ** (it * 1.0 / nit))
                        cborder_step = cborder_step_init * (cborder_step_shrink ** (it * 1.0 / nit))
                        tgts = []
                        for i, e in enumerate(elsi):
                            t = ivenn.ellipse_clamp(e, r[mask][1], cborder, bool(mask & (1 << i)), n >= 4)
                            if t is not None:
                                tgts.append(t)
                        if len(tgts):
                            r[mask][1] = (sum(tgts, numpy.array([0.0, 0.0])) / len(tgts)) * walk_ratio + r[mask][1] * (1.0 - walk_ratio)
                            cborder = max(cborder - cborder_step, 0.0)
                            # print("  " + str(it) + ": " + str(cborder) + "  " + str(r[mask][1]))
                        else:
                            cborder += cborder_step
                else:
                    r[mask][1] = cen
                # r[mask][1] = cen * (1 << iutil.bitcount(mask_other)) - subcen
                # print(r[mask][1])
                print("r[" + str(mask) + "][1] = numpy." + repr(r[mask][1]))
                
                for i, e in enumerate(elsi):
                    # if ivenn.ellipse_contains(e, r[mask][1]) ^ bool(mask & (1 << i)):
                    #     print("FAIL " + str(i))
                    if ivenn.ellipse_clamp(e, r[mask][1], 0.0, bool(mask & (1 << i)), n >= 4) is not None:
                        print("FAIL " + str(i))
            
            
            
            r[mask][2] = numpy.array([0.0, 0.0])
            
        
        
        textps = []
        if n == 1:
            textps = [(0.0, 1.1)]
        elif n == 2:
            textps = [(-0.45, 0.8), (0.45, 0.8)]
        elif n == 3:
            textps = [(0.0, 1.1), (-1.0, -0.6), (1.0, -0.6)]
        elif n == 4:
            textps = [(-0.95, 0.85), (-0.4, 1.1), (0.4, 1.1), (0.95, 0.85)]
        elif n == 5:
            textps = [(-numpy.sin(-i * numpy.pi * 2 / 5 + 0.12) * 1.1, 
                       numpy.cos(-i * numpy.pi * 2 / 5 + 0.12) * 1.1) for i in range(5)]
            
        
        
        if True:
            if n == 1:
                r[1][1] = numpy.array([0.0, 0.0])
            elif n == 2:
                r[3][1] = numpy.array([-5.19979865e-06, 0.0])
                r[2][1] = numpy.array([6.39999982e-01, 0.0])
                r[1][1] = numpy.array([-6.39999982e-01, 0.0])
            elif n == 3:
                r[7][1] = numpy.array([-0.00011776,  0.00020427])
                r[6][1] = numpy.array([ 8.10570556e-05, -4.61882452e-01])
                r[5][1] = numpy.array([0.40013006, 0.2308303 ])
                r[4][1] = numpy.array([ 0.60932452, -0.35185475])
                r[3][1] = numpy.array([-0.40005325,  0.23087825])
                r[2][1] = numpy.array([-0.6093162 , -0.35186916])
                r[1][1] = numpy.array([1.71157142e-05, 7.03617892e-01])
            elif n == 4:
                r[15][1] = numpy.array([ 0.0, -0.24960125])
                r[14][1] = numpy.array([0.24123917, 0.14045396 - 0.04])
                r[13][1] = numpy.array([-0.18729334, -0.53280546])
                r[12][1] = numpy.array([0.51662578, 0.40616016])
                r[11][1] = numpy.array([ 0.19303297, -0.51624427])
                r[10][1] = numpy.array([ 0.41826399 + 0.01, -0.41951261])
                r[9][1] = numpy.array([ 0.0, -0.78949216])
                r[8][1] = numpy.array([0.85559034,  0.22166595 - 0.02])
                r[7][1] = numpy.array([-0.24252916,  0.14045396 - 0.04])
                r[6][1] = numpy.array([0.0, 0.36160384 + 0.04])
                r[5][1] = numpy.array([-0.41826399 - 0.01, -0.41951261])
                r[4][1] = numpy.array([0.40438341, 0.74829018])
                r[3][1] = numpy.array([-0.51706528,  0.40634087])
                r[2][1] = numpy.array([-0.4085948 ,  0.74881886])
                r[1][1] = numpy.array([-0.85559034,  0.22166595 - 0.02])
            elif n == 5:
                r[31][1] = numpy.array([ 0.00123121, -0.00035098])
                r[30][1] = numpy.array([-0.50522841, -0.15738005])
                r[29][1] = numpy.array([-0.30334848,  0.43405041])
                r[28][1] = numpy.array([-0.5759732 ,  0.08623396])
                r[27][1] = numpy.array([0.31906663, 0.42263051])
                r[26][1] = numpy.array([-0.56718932, -0.29605626])
                r[25][1] = numpy.array([-0.09599438,  0.57443462])
                r[24][1] = numpy.array([-0.66609369, -0.20490312])
                r[23][1] = numpy.array([ 0.49741082, -0.18604116])
                r[22][1] = numpy.array([ 0.63288148, -0.09387176])
                r[21][1] = numpy.array([-0.45683543,  0.44794225])
                r[20][1] = numpy.array([-0.57279059,  0.30624   ])
                r[19][1] = numpy.array([0.51664827, 0.26884256])
                r[18][1] = numpy.array([0.64339464, 0.0889563 ])
                r[17][1] = numpy.array([-0.36041924,  0.57743233])
                r[16][1] = numpy.array([-0.8230906 ,  0.14612619])
                r[15][1] = numpy.array([-0.00456696, -0.52900733])
                r[14][1] = numpy.array([-0.25995857, -0.52117389])
                r[13][1] = numpy.array([ 0.10629509, -0.63091551])
                r[12][1] = numpy.array([-0.01095303, -0.69681429])
                r[11][1] = numpy.array([0.28484849, 0.57289808])
                r[10][1] = numpy.array([-0.46825182, -0.45012615])
                r[9][1] = numpy.array([0.11424952, 0.63938959])
                r[8][1] = numpy.array([-0.39334952, -0.73764132])
                r[7][1] = numpy.array([ 0.41524152, -0.40832889])
                r[6][1] = numpy.array([ 0.63606   , -0.25034334])
                r[5][1] = numpy.array([ 0.28339764, -0.58443058])
                r[4][1] = numpy.array([ 0.5799869 , -0.60204133])
                r[3][1] = numpy.array([0.43456033, 0.52772247])
                r[2][1] = numpy.array([0.75179328, 0.36557041])
                r[1][1] = numpy.array([-0.11537642,  0.82796063])
                
                if False:
                    for mask in range(1, (1 << n) - 1):
                        maski = mask
                        for i in range(1, 5):
                            maski = maski << 1
                            if maski & (1 << n):
                                maski -= 1 << n
                                maski += 1
                            a = i * numpy.pi * 2 / 5
                            cosa = numpy.cos(a)
                            sina = numpy.sin(a)
                            r[maski][1] = numpy.array([[cosa, -sina], [sina, cosa]]).dot(r[mask][1])
                    
                
        for mask in range((1 << n) - 1, 0, -1):
            mdist = 1e10
            for x in r[mask][0]:
                cdist = x[1]
                cdist += numpy.linalg.norm(r[mask][1] - x)
                if cdist < mdist:
                    mdist = cdist
                    r[mask][2] = x
            
        return (r, textps)
        
    
    def patch(n, inc, **kwargs):
        return matplotlib.patches.Polygon(ivenn.intersect(n, inc), True, **kwargs)

class CellTable:
    def __init__(self, x):
        self.x = x
        self.cells = [{} for i in range(1 << len(self.x))]
        self.fontsize = 22
        self.linewidth = 1.5
        self.exprs = []
    
    def set_enabled(self, mask, enabled = True):
        self.cells[mask]["enabled"] = enabled
    
    def get_enabled(self, mask):
        return self.cells[mask].get("enabled", True)
    
    def set_attr(self, mask, key, val):
        self.cells[mask][key] = val
    
    def get_attr(self, mask, key, default_val = None):
        return self.cells[mask].get(key, default_val)
    
    def add_expr(self, expr, cval):
        self.exprs.append({"expr": expr, "cval": cval})
    
    def set_expr_val(self, mask, val):
        self.cells[mask]["val_" + str(len(self.exprs) - 1)] = val
    
    def get_pos(self, mask):
        n = len(self.x)
        nv = n // 2
        maskv = mask & ((1 << nv) - 1)
        maskh = mask >> nv
        return (iutil.gray_to_bin(maskh), iutil.gray_to_bin(iutil.bit_reverse(maskv, nv)))
        
        # nh = (n + 1) // 2
        # maskh = mask & ((1 << nh) - 1)
        # maskv = mask >> nh
        # return (iutil.gray_to_bin(iutil.bit_reverse(maskh, nh)), iutil.gray_to_bin(maskv))
        
    def get_x_poss(self, xi):
        n = len(self.x)
        nv = n // 2
        cn = nv
        ax = 1
        if xi >= nv:
            xi -= nv
            ax = 0
            cn = n - nv
        else:
            xi = nv - 1 - xi
        
        r = []
        for mask in range(1, 1 << cn):
            if iutil.bin_to_gray(mask) & (1 << xi):
                r.append(mask)
        return (ax, r)
        # nh = (n + 1) // 2
        # cn = nh
        # ax = 0
        # if xi >= nh:
        #     xi -= nh
        #     ax = 1
        #     cn = n - nh
        # else:
        #     xi = nh - 1 - xi
        
        # r = []
        # for mask in range(1, 1 << cn):
        #     if iutil.bin_to_gray(mask) & (1 << xi):
        #         r.append(mask)
        # return (ax, r)
    
    def get_expr_color(self, i, color_shift):
        r = None
        if i is not None:
            r = self.exprs[i].get("color", None)
        if r is None:
            # return [(1, 0.5, 0.5), (0.5, 1, 0.5), (0.5, 0.5, 1), (1, 1, 0.2), (1, 0.3, 1), (0.3, 1, 1), 
            #         (1, 0.8, 0.3), (0.6, 0.6, 0.6)][i]
            return [(1, 0.5, 0.5), (0.5, 0.5, 1), (0.5, 1, 0.5), (0.6, 0.6, 0.6), (1, 0.3, 1), (1, 1, 0.2), (0.3, 1, 1), 
                    (1, 0.8, 0.3)][(0 if i is None else i) + color_shift]
        return r
    
    def plot(self, style = "hsplit", legend = True, use_latex = True):
        label_interval = 0.32
        label_width = 0.29
        fontsize = self.fontsize
        fontsize_in_mul = 1.0
        linewidth = self.linewidth
        
        rcParams_orig = None
        if use_latex:
            rcParams_orig = plt.rcParams.copy()
            plt.rcParams.update({"text.usetex": True,
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Helvetica"]})
        
        fig, ax = plt.subplots(figsize = [10, 8])
        # patches = []
        xlim = [0, 0]
        ylim = [0, 0]
        
        expr_hatch = [None] * len(self.exprs)
        mask_nexpr = [0] * (1 << len(self.x))
        hatches = ['//', '\\\\', '||', '--', '+', 'x', 'o', 'O', '.', '*']
        
        cval_present = False
        cval_min = 0.0 # 1e20
        cval_max = 0.0 # -1e20
        for mask in range(1, 1 << len(self.x)):
            cval = self.get_attr(mask, "cval")
            if cval is not None:
                cval = float(cval)
                cval_present = True
                cval_min = min(cval_min, cval)
                cval_max = max(cval_max, cval)
        
        rect_draw = True
        pm = False
        text_draw = True
        val_outline = None
        neg_hatch = None
        is_blend = None
        is_venn = False
        is_venn_overlap = False
        text_sub_draw = True
        neg_hatch_style = "//"
        color_shift = 0
        numdp = 4
        cval_color_enabled = True
        
        cval_ignore = False
        
        style_split = style.split(",")
        style = None
        
        for cstyle2 in style_split:
            cstyle = cstyle2.strip()
            if len(cstyle) == 0:
                continue
            if cstyle == "nofill":
                rect_draw = False
            elif cstyle == "pm":
                pm = True
            elif cstyle == "num":
                pm = False
            elif cstyle == "text":
                text_draw = True
            elif cstyle == "notext":
                text_draw = False
            elif cstyle == "sign":
                text_sub_draw = False
            elif cstyle == "nosign":
                text_sub_draw = False
            elif cstyle == "signhatch":
                neg_hatch = True
            elif cstyle == "nosignhatch":
                neg_hatch = False
            elif cstyle == "legend":
                legend = True
            elif cstyle == "nolegend":
                legend = False
            elif cstyle == "val_outline":
                val_outline = True
            elif cstyle == "val_nooutline":
                val_outline = False
            elif cstyle == "nocval":
                cval_ignore = True
            elif cstyle == "nocval_color":
                cval_color_enabled = False
            elif cstyle == "dp0":
                numdp = 0
            elif cstyle == "dp1":
                numdp = 1
            elif cstyle == "dp2":
                numdp = 2
            elif cstyle == "dp3":
                numdp = 3
            elif cstyle == "dp4":
                numdp = 4
            elif cstyle == "dp5":
                numdp = 5
            elif cstyle == "dp6":
                numdp = 6
            elif cstyle == "dp7":
                numdp = 7
            elif cstyle == "dp8":
                numdp = 8
            elif cstyle == "dp9":
                numdp = 9
            elif cstyle == "venn":
                is_venn = True
            else:
                style = cstyle
        
        if style is None:
            style = "hsplit"
        
        if cval_ignore:
            cval_present = False
        
        cval_color = None
        if cval_min >= cval_max:
            cval_color_enabled = False
        if cval_present:
            if cval_color_enabled:
                cval_color = self.get_expr_color(None, color_shift)
                color_shift += 1
        else:
            cval_color_enabled = False
        
        if cval_color_enabled:
            style = "hatch"
        
        if len(self.x) > 5:
            is_venn = False
        if is_venn:
            if len(self.x) == 4:
                fontsize_in_mul = 0.9
            if len(self.x) == 5:
                fontsize_in_mul = 0.8
            
        if val_outline is None:
            val_outline = style == "hatch" or style == "text" or style == "blend"
        if neg_hatch is None:
            neg_hatch = style == "hsplit_fixed" or style == "hsplit" or style == "blend"
        if is_blend is None:
            is_blend = style == "blend"
        
        for mask in range(1, 1 << len(self.x)):
            expr_pres = [ei for ei in range(len(self.exprs)) 
                         if self.get_attr(mask, "val_" + str(ei)) is not None]
            mask_nexpr[mask] = len(expr_pres)
            if style == "hatch":
                if len(expr_pres) >= 2:
                    for ei in expr_pres:
                        expr_hatch[ei] = True
                    
        if style == "hatch":
            nover = 0
            for ei in range(len(self.exprs)):
                if expr_hatch[ei]:
                    expr_hatch[ei] = hatches[nover]
                    nover += 1
                
        
        n = len(self.x)
        axlen = [0, 0]
        axslen = [0, 0]
        
        xstr = []
        for xi in range(n):
            if use_latex:
                xstr.append("$" + self.x[xi].latex() + "$")
            else:
                xstr.append(str(self.x[xi]))
            
        
        polys = [None] * (1 << n)
        poly_els = [None] * n
        textps = [None] * n
        if is_venn:
            polys, textps = ivenn.calc_ellipses(n)
            poly_els = [ivenn.intersect(n, [i == j for j in range(n)]) for i in range(n)]
            for xi in range(n):
                ax.text(textps[xi][0], textps[xi][1], xstr[xi], 
                        horizontalalignment="center", verticalalignment="center", fontsize = fontsize)
                
            
        if not is_venn:
            for xi in range(n):
                cax, clist = self.get_x_poss(xi)
                axslen[cax] += 1
            for xi in range(n):
                cax, clist = self.get_x_poss(xi)
                avgpos = [0.0, 0.0]
                caxlen = axlen[cax]
                if cax:
                    caxlen = axslen[cax] - 1 - caxlen
                for v in clist:
                    pos = (v, -(caxlen + 1) * label_interval)
                    size = (1, label_width)
                    if cax:
                        pos = (pos[1], pos[0])
                        size = (size[1], size[0])
                    
                    pos = (pos[0], -pos[1] - size[1])
                    avgpos[0] += pos[0] + size[0] * 0.5
                    avgpos[1] += pos[1] + size[1] * 0.5
                    
                    xlim[0] = min(xlim[0], pos[0])
                    xlim[1] = max(xlim[1], pos[0] + size[0])
                    ylim[0] = min(ylim[0], pos[1])
                    ylim[1] = max(ylim[1], pos[1] + size[1])
                    
                    ax.add_patch(matplotlib.patches.Rectangle(
                        pos, size[0], size[1], facecolor = "lightgray"))
                
                avgpos[0] /= len(clist)
                avgpos[1] /= len(clist)
                ax.text(avgpos[0], avgpos[1], xstr[xi], 
                        horizontalalignment="center", verticalalignment="center", fontsize = fontsize)
                
                axlen[cax] += 1
                
        # passes = [0, 4]
        # if is_venn:
        passes = [0, 2, 4]
        
        for cpass in passes:
            for mask in range(1, 1 << len(self.x)):
                if cpass == 2:
                    if is_venn and iutil.bitcount(mask) != 1:
                        continue
                
                pos = None
                size = None
                if is_venn:
                    pos = polys[mask][1]
                    size = 1.2 / n
                    if n == 4:
                        size *= 0.95
                    elif n == 5:
                        nbit = iutil.bitcount(mask)
                        if nbit >= 2 and nbit <= 4:
                            size *= 0.55
                        elif nbit == 5:
                            size *= 1.5
                        else:
                            size *= 1.2
                            
                    size = (size, size)
                    pos = (pos[0] - size[0] * 0.5, pos[1] - size[1] * 0.5)
                else:
                    pos = self.get_pos(mask)
                    pos = (pos[0], -pos[1] - 1)
                    size = (1, 1)
                xlim[0] = min(xlim[0], pos[0])
                xlim[1] = max(xlim[1], pos[0] + size[0])
                ylim[0] = min(ylim[0], pos[1])
                ylim[1] = max(ylim[1], pos[1] + size[1])
                
                isenabled = self.get_enabled(mask)
                
                cval = None
                if cval_present:
                    cval = self.get_attr(mask, "cval")
                    if cval is not None:
                        cval = float(cval)
                
                color = "none"
                if not isenabled:
                    color = "k"
                elif (cval is not None) and cval_color_enabled:
                    cval_scaled = (cval - cval_min) / (cval_max - cval_min)
                    color = tuple(cval_color[i] * cval_scaled + 1.0 - cval_scaled for i in range(3))
                
                
                if cpass == 0 and color != "none":
                    params = {
                        "facecolor": color
                        }
                    if is_venn:
                        ax.add_patch(matplotlib.patches.Polygon(
                            polys[mask][0], True, **params))
                    else:
                        ax.add_patch(matplotlib.patches.Rectangle(
                            pos, size[0], size[1], **params))
                
                if cpass == 0:
                    params = {
                        "facecolor": "white", 
                        "edgecolor": "none"
                        }
                    if is_venn and is_venn_overlap:
                        ax.add_patch(matplotlib.patches.Polygon(
                            polys[mask][0], True, **params))
                
                if cpass == 2:
                    params = {
                        "facecolor": "none", 
                        "linestyle": "-", 
                        "linewidth": linewidth, 
                        "edgecolor": "k"
                        }
                    if is_venn:
                        # ax.add_patch(matplotlib.patches.Polygon(
                        #     polys[mask][0], True, **params))
                        for i in range(n):
                            if mask & (1 << i):
                                ax.add_patch(matplotlib.patches.Polygon(
                                    poly_els[i], True, **params))
                                break
                    else:
                        ax.add_patch(matplotlib.patches.Rectangle(
                            pos, size[0], size[1], **params))
                    continue
                
                if isenabled:
                    cnexpr = 0
                    snexpr = mask_nexpr[mask]
                    colsum = [0.0, 0.0, 0.0]
                    colsumsol = [0.0, 0.0, 0.0]
                    nsol = 0
                    
                    if cval is not None:
                        if cpass == 4:
                            if text_draw:
                                text_y = 0.7
                                if len(self.exprs) == 0:
                                    text_y = 0.5
                                ctext = ("{:." + str(numdp) + "f}").format(cval)
                                ax.text(pos[0] + 0.5 * size[0], pos[1] + size[1] * text_y, ctext, 
                                    horizontalalignment="center", verticalalignment="center", fontsize = fontsize * fontsize_in_mul,
                                    color = "k")
                        
                    
                    for ei in range(len(self.exprs)):
                        v = self.get_attr(mask, "val_" + str(ei))
                        if v is None:
                            continue
                        
                        if pm:
                            ctext = iutil.float_tostr(abs(v), bracket = False)
                            if ctext == "1":
                                ctext = ""
                            if v >= 0:
                                ctext = "+" + ctext
                            else:
                                ctext = "-" + ctext
                        else:
                            ctext = iutil.float_tostr(v, bracket = False)
                            
                        ccolor = self.get_expr_color(ei, color_shift)
                        chatch = expr_hatch[ei]
                        hatch_invert = False
                        if chatch is not None:
                            hatch_invert = True
                        if neg_hatch and v < 0:
                            chatch = neg_hatch_style
                        
                        rect_x = 0.0
                        rect_w = 1.0
                        text_x = (cnexpr + 0.5) * 1.0 / snexpr
                        text_y = 0.5
                        if cval is not None:
                            text_y = 0.3
                        
                        if style == "hsplit_fixed":
                            rect_x = ei * 1.0 / len(self.exprs)
                            rect_w = 1.0 / len(self.exprs)
                            text_x = (ei + 0.5) * 1.0 / len(self.exprs)
                        elif style == "hsplit":
                            rect_x = cnexpr * 1.0 / snexpr
                            rect_w = 1.0 / snexpr
                        
                        if isinstance(ccolor, tuple):
                            colsum[0] += ccolor[0]
                            colsum[1] += ccolor[1]
                            colsum[2] += ccolor[2]
                            if not (neg_hatch and v < 0):
                                colsumsol[0] += ccolor[0]
                                colsumsol[1] += ccolor[1]
                                colsumsol[2] += ccolor[2]
                                nsol += 1
                                
                        
                        
                        if cpass == 0:
                            if rect_draw and (not is_blend or cnexpr == snexpr - 1):
                                hatch_col = (1.0, 1.0, 1.0)
                                if is_blend and cnexpr == snexpr - 1:
                                    ccolor = (colsum[0] / snexpr, colsum[1] / snexpr, colsum[2] / snexpr)
                                    if nsol > 0:
                                        hatch_col = (colsumsol[0] / nsol, colsumsol[1] / nsol, colsumsol[2] / nsol)
                                    if abs(ccolor[0] - hatch_col[0]) + abs(ccolor[1] - hatch_col[1]) + abs(ccolor[2] - hatch_col[2]) <= 0.001:
                                        chatch = None
                                    else:
                                        chatch = neg_hatch_style
                                
                                params = {
                                    "facecolor": "none" if hatch_invert else ccolor, 
                                    "hatch": chatch,
                                    "edgecolor": ccolor if hatch_invert else hatch_col if chatch else "none", 
                                    "linewidth": 0
                                    }
                                
                                if is_venn:
                                    # print(mask)
                                    # print(polys[mask])
                                    ax.add_patch(matplotlib.patches.Polygon(
                                        polys[mask][0], True, **params))
                                else:
                                    ax.add_patch(matplotlib.patches.Rectangle(
                                        (pos[0] + rect_x * size[0], pos[1]), 
                                        size[0] * rect_w, size[1], **params))
                                    
                                # ax.add_patch(matplotlib.patches.Rectangle(
                                #     (pos[0] + rect_x * size[0], pos[1]), 
                                #     size[0] * rect_w, size[1], 
                                #     facecolor = "none" if hatch_invert else ccolor, hatch = chatch,
                                #     edgecolor = ccolor if hatch_invert else "white" if chatch else "none", 
                                #     linewidth = 0))
                        elif cpass == 4:
                            if text_draw:
                                ax.text(pos[0] + text_x * size[0], pos[1] + size[1] * text_y, ctext, 
                                    horizontalalignment="center", verticalalignment="center", fontsize = fontsize * fontsize_in_mul,
                                    color = ccolor if val_outline else "k",
                                    path_effects = [matplotlib.patheffects.withStroke(linewidth = 3.5, foreground = "k")] if val_outline else None)
                        
                        cnexpr += 1
                        
                if cpass == 0:
                    
                    remtext = ""
                    if self.get_attr(mask, "ispos"):
                        remtext += "+"
                    if self.get_attr(mask, "isneg"):
                        remtext += "-"
                    if text_sub_draw and len(remtext):
                        rempos = None
                        if is_venn:
                            # vpos = -0.08
                            # if n == 5:
                            #     vpos = 0.05
                            # rempos = (pos[0] + size[0] * 0.5, pos[1] + size[1] * vpos)
                            rempos = polys[mask][2]
                            rempos = (rempos[0], rempos[1] + size[1] * 0.033)
                        else:
                            rempos = (pos[0] + size[0] * 0.015, pos[1] + size[1] * 0.02)
                        ax.text(rempos[0], rempos[1], remtext, 
                            horizontalalignment="center" if is_venn else "left", verticalalignment="bottom", 
                            fontsize = fontsize * fontsize_in_mul * 0.85)
        
        if is_venn:
            xlim = [-1.1, 1.1]
            ylim = [-1.1, 1.1]
        
        if legend and len(self.exprs):
            legends = []
            for ei in range(len(self.exprs)):
                ccolor = self.get_expr_color(ei, color_shift)
                clabel = ""
                cexpr = self.exprs[ei].get("expr")
                if cexpr is not None:
                    with PsiOpts(str_lhsreal = False):
                        if use_latex:
                            clabel = "$" + cexpr.latex(skip_simplify = True) + "$"
                        else:
                            clabel = str(cexpr)
                    # if isinstance(cexpr, Region):
                    #     clabel = cexpr.tostring(lhsvar = None)
                    # else:
                    #     clabel = str(cexpr)
                    
                cval = self.exprs[ei].get("cval")
                if cval_present and (cval is not None):
                    if not isinstance(cval, bool):
                        cval = float(cval)
                    if isinstance(cval, float):
                        clabel += " = " + ("{:." + str(numdp) + "f}").format(cval)
                    else:
                        clabel += " = " + str(cval)
                        
                if not len(clabel):
                    continue
                # legends.append(matplotlib.lines.Line2D([0], [0], color=ccolor, lw=4, label=clabel))
                legends.append(matplotlib.patches.Patch(facecolor=ccolor, label=clabel))
                
            ax.legend(handles = legends, fontsize = fontsize, bbox_to_anchor=(0.5, -0.03), 
                      loc="upper center", frameon=False)
                
        if is_venn:
            ax.set_aspect("equal")
            
        # ax.add_collection(PatchCollection(patches))
        plt.axis("off")
        plt.xlim([xlim[0] - 0.011, xlim[1] + 0.011])
        plt.ylim([ylim[0] - 0.011, ylim[1] + 0.011])
        plt.show()
        plt.tight_layout()
        
        if use_latex:
            plt.rcParams = rcParams_orig.copy()

class ProofObj(IBaseObj):
    def __init__(self, claim, desc = None, steps = None, parent = None):
        self.claim = claim
        self.desc = desc
        if steps is None:
            steps = []
        self.steps = steps
        self.parent = parent
    
    def empty():
        return ProofObj(None, None, [])
    
    def copy(self):
        # return ProofObj([(x.copy(), list(d), c) for x, d, c in self.steps])
        return ProofObj(iutil.copy(self.claim), iutil.copy(self.desc), 
                        iutil.copy(self.steps), self.parent)
    
    def from_region(x, c = ""):
        # return ProofObj([(x.copy(), [], c)])
        return ProofObj(iutil.copy(x), c)
    
    def __iadd__(self, other):
        # n = len(self.steps)
        # self.steps += [(x.copy(), [di + n for di in d], c) for x, d, c in other.steps]
        other = other.copy()
        other.parent = self
        self.steps.append(other)
        return self
    
    def __add__(self, other):
        r = self.copy()
        r += other
        return r
    
    def step_in(self, other):
        other = other.copy()
        other.parent = self
        self.steps.append(other)
        return other
    
    def step_out(self):
        return self.parent
    
    def clear(self):
        self.steps = []
          
    # def tostring(self, style = 0, prev = None):
    #     """Convert to string. 
    #     Parameters:
    #         style   : Style of string conversion
    #                   STR_STYLE_STANDARD : I(X,Y;Z|W)
    #                   STR_STYLE_PSITIP : I(X+Y&Z|W)
    #     """
    #     style = iutil.convert_str_style(style)
    #     r = ""
    #     start_n = 0
    #     if prev is not None:
    #         start_n = len(prev.steps)
        
    #     for i, (x, d, c) in enumerate(self.steps):
    #         if i > 0:
    #             r += "\n"
    #         r += "STEP #" + str(start_n + i)
    #         if c != "":
    #             r += " " + c
                
    #         if x is None or x.isuniverse():
    #             r += "\n"
    #         else:
    #             r += ":\n"
    #             r += x.tostring(style = style)
            
    #         r += "\n"
    #     return r
    
    def tostring(self, style = 0, prefix = ""):
        style = iutil.convert_str_style(style)
        
        nlstr = "\n"
        if style & PsiOpts.STR_STYLE_LATEX:
            nlstr = "\\\\\n"
            
        spacestr = " "
        if style & PsiOpts.STR_STYLE_LATEX:
            spacestr = "\\;"
            
        r = ""
        cinden = 4
        cprefix = prefix
        
        if self.desc is None:
            cinden = 0
        else:
            if prefix:
                r += prefix + ": "
                cprefix += "."
            r += iutil.tostring_join(self.desc, style)
            if self.claim is not None:
                r += nlstr
                if isinstance(self.claim, tuple):
                    if self.claim[0] == "equiv":
                        r += self.claim[1].tostring(style = style)
                        if style & PsiOpts.STR_STYLE_PSITIP:
                            r += nlstr + "==" + nlstr
                        elif style & PsiOpts.STR_STYLE_LATEX:
                            r += nlstr + PsiOpts.settings["latex_equiv"] + nlstr
                        else:
                            r += nlstr + "<=>" + nlstr
                        r += self.claim[2].tostring(style = style)
                        
                    elif self.claim[0] == "implies":
                        r += self.claim[1].tostring(style = style)
                        if style & PsiOpts.STR_STYLE_PSITIP:
                            r += nlstr + ">>" + nlstr
                        elif style & PsiOpts.STR_STYLE_LATEX:
                            r += nlstr + PsiOpts.settings["latex_implies"] + nlstr
                        else:
                            r += nlstr + "=>" + nlstr
                        r += self.claim[2].tostring(style = style)
                        
                else:
                    r += self.claim.tostring(style = style)
                    
            r += nlstr
        for i, x in enumerate(self.steps):
            r += nlstr
            r += iutil.str_inden(x.tostring(style, cprefix + str(i + 1)), cinden, spacestr = spacestr)
        return r
        
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"])
    
    
    def _latex_(self):
        return self.tostring(iutil.convert_str_style("latex"))
    

class CodingNode(IBaseObj):
    
    def __init__(self, rv_out, aux_out = None, aux_dec = None, aux_ndec = None, rv_in_causal = None, rv_in_scausal = None, ndec_mode = None, label = None, rv_ndec_force = None):
        
        self.rv_out = rv_out
        
        if isinstance(aux_out, tuple):
            aux_out = [aux_out]
        if isinstance(aux_out, list):
            self.aux_out_sublist = aux_out
            self.aux_out = Comp.empty()
            for v0, v1 in self.aux_out_sublist:
                self.aux_out += v0
        else:
            self.aux_out_sublist = []
            self.aux_out = aux_out
            
        self.aux_dec = aux_dec
        self.aux_ndec = aux_ndec
        
        if rv_in_causal is None:
            self.rv_in_causal = Comp.empty()
        else:
            self.rv_in_causal = rv_in_causal.copy()
        
        if rv_in_scausal is None:
            self.rv_in_scausal = Comp.empty()
        else:
            self.rv_in_scausal = rv_in_scausal.copy()
        
        self.aux_ndec_try = Comp.empty()
        self.aux_ndec_force = Comp.empty()
        if rv_ndec_force is None:
            self.rv_ndec_force = Comp.empty()
        else:
            self.rv_ndec_force = rv_ndec_force.copy()
        self.ndec_mode = ndec_mode

        self.label = label
            
    def copy(self):
        r = CodingNode(self.rv_out.copy())
        r.aux_out = iutil.copy(self.aux_out)
        r.aux_out_sublist = [(a.copy(), b.copy()) for a, b in self.aux_out_sublist]
        r.aux_dec = iutil.copy(self.aux_dec)
        r.aux_ndec = iutil.copy(self.aux_ndec)
        r.rv_in_causal = iutil.copy(self.rv_in_causal)
        r.aux_ndec_try = iutil.copy(self.aux_ndec_try)
        r.aux_ndec_force = iutil.copy(self.aux_ndec_force)
        r.rv_ndec_force = iutil.copy(self.rv_ndec_force)
        r.ndec_mode = self.ndec_mode
        return r
    
    def clear(self):
        self.aux_out = None
        self.aux_out_sublist = []
        self.aux_dec = None
        self.aux_ndec = None
        self.aux_ndec_try = Comp.empty()
        self.aux_ndec_force = Comp.empty()
        # self.rv_ndec_force = Comp.empty()
    
    def record_to(self, index, skip_msg = False):
        self.rv_out.record_to(index)
        if self.aux_out is not None:
            self.aux_out.record_to(index)
        if self.aux_dec is not None:
            self.aux_dec.record_to(index)
        if self.aux_ndec is not None:
            self.aux_ndec.record_to(index)
    

class CodingModel(IBaseObj):
    
    def __init__(self, bnet = None, sublist = None, nodes = None, reg = None):
        if bnet is None:
            self.bnet = BayesNet()
        else:
            if isinstance(bnet, BayesNet):
                self.bnet = bnet
            else:
                self.bnet = BayesNet(bnet)
            
        if reg is None:
            self.reg = Region.universe()
        else:
            self.reg = reg
            
        if sublist is None:
            self.sublist = []
        else:
            self.sublist = sublist
            
        self.sublist_rate = []
        self.sublist_zero = []

        self.nodes = []
        if nodes is None:
            pass
        else:
            for node in nodes:
                if isinstance(node, CodingNode):
                    self.nodes.append(node)
                else:
                    self.nodes.append(CodingNode(node))
        
        self.inner_mode = "plain"
        self.bnet_out = None
        self.bnet_out_arrays = None
    
    def copy(self):
        r = CodingModel()
        r.bnet = iutil.copy(self.bnet)
        r.reg = iutil.copy(self.reg)
        r.sublist = [(a.copy(), b.copy()) for a, b in self.sublist]
        r.sublist_rate = [(a.copy(), b.copy()) for a, b in self.sublist_rate]
        r.sublist_zero = [a.copy() for a in self.sublist_zero]
        r.nodes = [a.copy() for a in self.nodes]
        r.inner_mode = self.inner_mode
        r.bnet_out = iutil.copy(self.bnet_out)
        return r
    
    def __iadd__(self, other):
        if isinstance(other, CodingNode):
            self.nodes.append(other)
        else:
            self.bnet += other
        return self
        
    def __iand__(self, other):
        if isinstance(other, Region):
            self.reg = self.reg & other
        return self
        

    def add_edge(self, a, b, coded = False, children_edge = None, is_fcn = False, 
                 rv_in_causal = None, rv_in_scausal = None, **kwargs):
        
        if rv_in_causal is None:
            rv_in_causal = Comp.empty()
        if rv_in_scausal is None:
            rv_in_scausal = Comp.empty()
            
        ex = Comp.empty()
        b2 = Comp.empty()
        for c in b:
            if self.bnet.index.get_index(c) >= 0:
                ex += c
            else:
                b2 += c
                
        ac = a.copy() + rv_in_causal - rv_in_scausal
        acb = Comp.empty()

        if not ex.isempty():
            cname = str(ex) + "?@@100@@#EX_"
            for c in a:
                cname += str(c)
            cname += "_"
            for c in ex:
                cname += str(c)
            exc = Comp.rv(cname)
            self.bnet.add_edge(ac, exc)
            if is_fcn:
                self.bnet.set_fcn(exc)
            # if children_edge:
            #     ac += ex
            for c in ex:
                if children_edge is True or (children_edge is None and not self.is_rate(c)):
                    acb += c
            if coded:
                self.nodes.append(CodingNode(exc, rv_in_causal = rv_in_causal, 
                                             rv_in_scausal = rv_in_scausal, **kwargs))
            self.sublist.append((exc, ex))
        
        for c in b2:
            if children_edge is True or (children_edge is None and not self.is_rate(c)):
                self.bnet.add_edge(ac + acb, c)
                acb += c
            else:
                self.bnet.add_edge(ac, c)

            if is_fcn:
                self.bnet.set_fcn(c)

            if coded:
                self.nodes.append(CodingNode(c, rv_in_causal = rv_in_causal, 
                                             rv_in_scausal = rv_in_scausal, **kwargs))
        
    def add_node(self, a, b, **kwargs):
        self.add_edge(a, b, coded = True, **kwargs)
            
    def set_rate(self, a, rate):
        if isinstance(rate, (int, float)) and rate == 0:
            rate = Expr.real("#TMPRATE" + str(a))
            self.sublist_zero.append(rate)
        self.sublist_rate.append((a, rate))
        

    def get_aux(self):
        r = Comp.empty()
        for node in self.nodes:
            r += node.aux_out
        return r
        
    
    def get_rv_rates(self, v):
        v = v.copy()
        r = Expr.zero()
        for v0, v1 in self.sublist + self.sublist_rate:
            if v0.ispresent(v):
                if isinstance(v1, Expr):
                    r += v1
                else:
                    v.substitute(v0, v1)
        return r
    
    def is_rate(self, v):
        return not self.get_rv_rates(v).iszero()
    
    def is_src_rate(self, v):
        if not self.is_rate(v):
            return False
        rate = self.get_rv_rates(v)
        if not rate.isrealvar():
            return False
        count = 0
        for v0, v1 in self.sublist + self.sublist_rate:
            if v1.ispresent(v):
                return False
            if v1.ispresent(rate):
                count += 1
                if count >= 2:
                    return False
        return True

    def get_refs(self, v):
        r = v.copy()
        for a in self.bnet.index.comprv:
            if self.get_rv_ratervs(a).ispresent(v):
                r += a
        return r
    
    def is_ch_rate(self, v):
        if not self.is_rate(v):
            return False
        rate = self.get_rv_rates(v)
        if not rate.isrealvar():
            return False
        if not self.bnet.get_parents(v).isempty():
            return False
        g = self.get_refs(v) - v
        for a in g:
            if not self.bnet.get_children(a).isempty():
                return False
        return True

    def get_rv_sub(self, v):
        v = v.copy()
        for v0, v1 in self.sublist + self.sublist_rate:
            if v0.ispresent(v):
                if isinstance(v1, Expr):
                    v.substitute(v0, v0[0])
                    # v = v0[0].copy()
                else:
                    v.substitute(v0, v1)
        return v
    
    def get_rv_ratervs(self, v):
        v = v.copy()
        r = Comp.empty()
        for v0, v1 in self.sublist + self.sublist_rate:
            if v0.ispresent(v):
                if isinstance(v1, Expr):
                    r += v0[0]
                else:
                    v.substitute(v0, v1)
        return r
    
    def get_node_immediate_rates(self, node):
        x = node.rv_out + self.bnet.get_parents(node.rv_out)
        return self.get_rv_rates(x)
    
    def get_node_immediate_ratervs(self, node):
        x = node.rv_out + self.bnet.get_parents(node.rv_out)
        return self.get_rv_ratervs(x)
    
    
    def get_node_descendants(self, node):
        x = self.bnet.get_descendants(node.rv_out) - node.rv_out
        r = []
        for node2 in self.nodes:
            if x.ispresent(node2.rv_out):
                r.append(node2)
        return r
        
    def get_nodes_rv_out(self):
        r = Comp.empty()
        for node in self.nodes:
            r += node.rv_out
        return r
    
        
    def find_node_rv_out(self, x):
        for node in self.nodes:
            if x.ispresent(node.rv_out):
                return node
        return None
    
    def find_nodes_immediate_ratervs(self, a):
        r = []
        for node in self.nodes:
            if not self.get_node_descendants(node):
                continue
            ratervs = self.get_node_immediate_ratervs(node)
            if ratervs.ispresent(a):
                r.append(node)
        return r

    def rv_single_node(self, a):
        nodes = self.find_nodes_immediate_ratervs(a)
        node2 = self.find_node_rv_out(a)
        c = len(nodes)
        if node2 is not None and node2 not in nodes:
            c += 1
        return c <= 1

    def calc_node(self, node):
        if node.aux_out is None:
            node.aux_out = Comp.empty()
            
            ratervs = self.get_node_immediate_ratervs(node)
            des = self.get_node_descendants(node)
            crvs = list(ratervs)
            if len(crvs) == 0:
                crvs.append(Comp.empty())
            
            if self.inner_mode == "combinations":
                for c in crvs:
                    for mask in range(1, 1 << len(des)):
                        cname = ("A_" + node.rv_out.tostring(style = PsiOpts.STR_STYLE_STANDARD)
                                 + ("" if c.isempty() else "_" + c.tostring(style = PsiOpts.STR_STYLE_STANDARD)))
                        # cname_l = ("A_{" + node.rv_out.tostring(style = PsiOpts.STR_STYLE_LATEX)
                        #          + ("" if c.isempty() else "," + c.tostring(style = PsiOpts.STR_STYLE_LATEX)) + "}")
                        cname_l = ("A_{" + node.rv_out.tostring(style = PsiOpts.STR_STYLE_LATEX) + "}"
                                 + ("" if c.isempty() else "^{" + c.tostring(style = PsiOpts.STR_STYLE_LATEX) + "}"))
                        
                        for i, d in enumerate(des):
                            if mask & (1 << i):
                                cname += "_" + d.rv_out.tostring(style = PsiOpts.STR_STYLE_STANDARD)
                                cname_l += "," + d.rv_out.tostring(style = PsiOpts.STR_STYLE_LATEX)
                        cname_l += "}"
                        cname += "@@" + str(PsiOpts.STR_STYLE_LATEX) + "@@" + cname_l
                        a = Comp.rv(cname)
                        node.aux_out += a
                        node.aux_out_sublist.append((a, a + c))
                        for i, d in enumerate(des):
                            if mask & (1 << i):
                                if d.aux_dec is None:
                                    d.aux_dec = Comp.empty()
                                d.aux_dec += a
                            else:
                                d.aux_ndec_try += a
                                if d.rv_ndec_force.ispresent(c):
                                    d.aux_ndec_force += a
            
            elif self.inner_mode == "plain":
                if len(des):
                    for c in crvs:
                        single_node = self.rv_single_node(c)

                        if single_node:
                            cname = ("A_" + c.tostring(style = PsiOpts.STR_STYLE_STANDARD))
                        else:
                            cname = ("A_" + node.rv_out.tostring(style = PsiOpts.STR_STYLE_STANDARD) 
                                    + ("" if c.isempty() or len(crvs) == 1 else "_" + c.tostring(style = PsiOpts.STR_STYLE_STANDARD)))

                        cname += "@@" + str(PsiOpts.STR_STYLE_LATEX) + "@@"

                        # print(str(c) + "  " + str(len(self.find_nodes_immediate_ratervs(c))))
                        if single_node:
                            cname += ("A_{" + c.tostring(style = PsiOpts.STR_STYLE_LATEX) + "}")
                        else:
                            cname += ("A_{" + node.rv_out.tostring(style = PsiOpts.STR_STYLE_LATEX) + "}"
                                    + ("" if c.isempty() or len(crvs) == 1 else "^{" + c.tostring(style = PsiOpts.STR_STYLE_LATEX) + "}"))
                        
                        a = Comp.rv(cname)
                        node.aux_out += a
                        node.aux_out_sublist.append((a, a + c))
                        
                        for d in des:
                            if self.get_node_immediate_ratervs(d).ispresent(c):
                                if d.aux_dec is None:
                                    d.aux_dec = Comp.empty()
                                d.aux_dec += a
                            else:
                                d.aux_ndec_try += a
                                if d.rv_ndec_force.ispresent(c):
                                    d.aux_ndec_force += a
                            
                            
        
    
    def calc_nodes(self):
        for node in self.nodes:
            self.calc_node(node)

    def clear_cache(self):
        for node in self.nodes:
            node.clear()
        
    def presolve(self):
        self.sublist += self.sublist_rate
        self.sublist_rate = []
    

    def get_region(self):
        r = self.bnet.copy()
            
        tosublist = []
        for node in self.nodes:
            tosublist += node.aux_out_sublist
        tosublist += self.sublist + self.sublist_rate
        
        for v0, v1 in tosublist:
            if isinstance(v1, Expr) and not isinstance(v0, Expr):
                r = r.contracted_node(v0)
            else:
                r = r.contracted_node(v0)
        
        return r.get_region()


    def ch_combine(self):
        groups = []
        for a in self.bnet.index.comprv:
            if self.is_ch_rate(a):
                p = self.bnet.get_children(a)
                for g in groups:
                    if g[0] == p:
                        g[1].append(a)
                        break
                else:
                    groups.append((p, [a]))
        
        for p, g in groups:
            decs = []
            for i, a in enumerate(g):
                for b in self.get_refs(a) - a:
                    bp = self.bnet.get_parents(b)
                    if bp not in decs:
                        decs.append(bp)
            
            # Whether y is degraded compared to x
            deg = [[self.bnet.check_ic(Expr.Ic(p, x, y)) for y in decs] for x in decs]

            idecs = [0 for i in range(len(g))]
            for i, a in enumerate(g):
                for b in self.get_refs(a) - a:
                    bp = self.bnet.get_parents(b)
                    bpi = decs.index(bp)
                    for j in range(len(decs)):
                        if deg[bpi][j]:
                            idecs[i] |= 1 << j

            # print(idecs)
            dec_vis = list(idecs)

            for mask in range(1, 1 << len(g)):
                cdec = 0
                for i, a in enumerate(g):
                    if mask & (1 << i):
                        cdec |= idecs[i]
                
                # print(str(mask) + "  " + str(cdec))
                if cdec in dec_vis:
                    continue

                crv = Comp.rv(str(sum(g)) + "@@" + str(PsiOpts.STR_STYLE_LATEX) + "@@" + sum(g).tostring(style = "latex"))
                # crate = Expr.real("MRATE_" + str(p) + "_" + str(mask))
                # self.set_rate(crv, crate)
                self.set_rate(crv, 0)
                for pb in p:
                    self.bnet.add_edge(crv, pb)
                # self.add_edge(crv, p, children_edge = False)
                # for j in range(len(decs)):
                #     if cdec & (1 << j):
                #         self.add_node(decs[j], crv)
                for i, a in enumerate(g):
                    if idecs[i] | cdec == cdec:
                        # print("TRY  " + str(a) + "  " + str(crv))
                        for b in self.get_refs(a) - a:
                            node = self.find_node_rv_out(b)
                            if node is not None:
                                node.rv_ndec_force += crv
                                # print("ADD  " + str(node.rv_out) + "  " + str(crv))

                # self.sublist_zero.append(crate)

                dec_vis.append(cdec)
        
        return self

    def get_src_splice(self):
        r = []
        groups = []
        for a in self.bnet.index.comprv:
            if self.is_src_rate(a):
                p = self.bnet.get_parents(a)
                c = self.bnet.get_children(a)
                for g in groups:
                    if g[0] == p:
                        g[1].append((a, c))
                        break
                else:
                    groups.append((p, [(a, c)]))
        
        def covered(g, i, mask):
            ci = g[i][1]
            cj = Comp.empty()
            for j in range(len(g)):
                if mask & (1 << j):
                    cj += g[j][1]
            return cj.super_of(ci)


        for p, g in groups:
            for i in range(len(g)):
                ci = g[i][1]
                for mask in igen.subset_mask((1 << len(g)) - 1 - (1 << i)):
                    if not covered(g, i, mask):
                        continue
                    bad = False
                    for j in range(len(g)):
                        if mask & (1 << j) and covered(g, i, mask - (1 << j)):
                            bad = True
                            break
                    if bad:
                        continue
                    r.append((self.get_rv_rates(g[i][0]), 
                        [self.get_rv_rates(g[j][0]) for j in range(len(g)) if mask & (1 << j)]))

        return r



    def get_inner(self, convexify = None, ndec_mode = "all", skip_simplify = False, skip_aux = False, splice = True, combine = True):
        """Get an inner bound of the capacity region via [Lee-Chung 2015] together with simplification procedures.
        Si-Hyeon Lee, and Sae-Young Chung. "A unified approach for network information theory." 
        2015 IEEE International Symposium on Information Theory (ISIT). IEEE, 2015.
        """

        if combine:
            cs = self.copy()
            cs.ch_combine()
            return cs.get_inner(convexify, ndec_mode, skip_simplify, skip_aux, splice, False)

        verbose = PsiOpts.settings.get("verbose_codingmodel", False)

        self.presolve()
        
        if self.bnet.tsorted() is None:
            raise ValueError("Non-strictly-causal cycle detected.")
            return None
        
        if convexify is None:
            convexify = self.convexify_needed()
            
        if convexify is not False:
            if convexify is True:
                convexify = Comp.rv("Q_T")
        
        self.calc_nodes()
        for node in self.nodes:
            if node.aux_ndec is None:
                r = RegionOp.union([])
                
                cndec_mode = node.ndec_mode
                if cndec_mode is None:
                    cndec_mode = ndec_mode
                    
                if cndec_mode == "all":
                    # print("NODE  " + str(node.rv_out) + "  " + str(node.aux_ndec_force))

                    for cndec0 in igen.subset(node.aux_ndec_try - node.aux_ndec_force):
                        cndec = node.aux_ndec_force + cndec0
                        # print(str(node.rv_out) + "  " + str(cndec))
                        node.aux_ndec = cndec if isinstance(cndec, Comp) else Comp.empty()
                        r |= self.get_inner(convexify, ndec_mode = ndec_mode, skip_simplify = True, skip_aux = True, splice = splice, combine = combine)
                        
                elif cndec_mode == "min" or cndec_mode == "none":
                    # node.aux_ndec = Comp.empty()
                    node.aux_ndec = node.aux_ndec_force.copy()
                    r = self.get_inner(convexify, ndec_mode = ndec_mode, skip_simplify = True, skip_aux = True, splice = splice, combine = combine)
                        
                elif cndec_mode == "max":
                    node.aux_ndec = node.aux_ndec_try.copy()
                    r = self.get_inner(convexify, ndec_mode = ndec_mode, skip_simplify = True, skip_aux = True, splice = splice, combine = combine)
                
                node.aux_ndec = None
                
                if not skip_aux:
                    aux = self.get_aux()
                    caux = aux.copy()
                    if convexify is not False:
                        caux += convexify
                    r = r.eliminate(caux.inter(r.allcomprv()))
                    
                if skip_simplify:
                    return r
                else:
                    r.simplify()
                    r.simplify_union()
                    # print(r)
                    r.remove_missing_aux()
                    return r.simplified_quick()
        
        
        aux = self.get_aux()
        aux_rates = [Expr.real("#TR" + str(i)) for i in range(len(aux))]
        cbnet = self.bnet.copy()
        
        r = Region.universe()
        for rate in aux_rates:
            r &= rate >= 0
            
        for node in self.nodes:
            y = self.bnet.get_parents(node.rv_out) - node.rv_in_causal
            aux_dec = Comp.empty()
            if node.aux_dec is not None:
                aux_dec = node.aux_dec
                
            cbnet.add_edge(aux_dec + y, node.aux_out + node.rv_out)
            
            for cdec in igen.subset(aux_dec, minsize = 1):
                for cndec in igen.subset(node.aux_ndec):
                    cd = cdec + cndec
                    expr = Expr.zero()
                    cc = Comp.empty()
                    for c in cd:
                        ci = aux.index_of(c)
                        expr += aux_rates[ci]
                        expr -= I(c & cc + (aux_dec + node.aux_ndec - cd) + y)
                        cc += c
                    r &= expr <= 0
            
            for ce in igen.subset(node.aux_out, minsize = 1):
                expr = Expr.zero()
                cc = Comp.empty()
                for c in ce:
                    ci = aux.index_of(c)
                    expr += aux_rates[ci]
                    expr -= I(c & cc + aux_dec + y)
                    cc += c
                r &= expr >= 0
        
        
        if convexify is not False:
            r.condition(convexify)
            for node in self.nodes:
                cbnet.add_edge(convexify, node.aux_out + node.rv_out)
            
        r &= cbnet.get_region()
            
        tosublist = []
        for node in self.nodes:
            tosublist += node.aux_out_sublist
        tosublist += self.sublist
        
        for v0, v1 in tosublist:
            r.substitute(v0, v1)
            if isinstance(v1, Expr) and not isinstance(v0, Expr):
                r &= v1 >= 0
        
        
        if verbose:
            print("============== Inner bound ==============")
            print(r)
        
        r = r.eliminate(sum(aux_rates))
        
        
        # if convexify is not False:
        #     r.condition(convexify)
        #     rallcomprv = r.allcomprv()
        #     encout = self.get_nodes_rv_out().inter(rallcomprv)
        #     r &= markov(rallcomprv - encout - convexify, encout, convexify)
            
        caux = aux.copy()
        if convexify is not False:
            caux += convexify
        
        # r_prior = self.bnet.get_region().copy()

        # assume_reg = Region.universe()
        if not self.reg.isuniverse():
            # r = r & self.reg
            # r = r.and_cause_consequence(self.reg, avoid = caux)
            # r = r.and_cause_consequence(self.reg, added_reg = r_prior)
            r = r.and_cause_consequence(self.reg)
            
        if not skip_aux:
            r = r.eliminate(caux.inter(r.allcomprv()))
        
        
        if verbose:
            print("============== Elim aux rate ==============")
            print(r)
            
        if splice:
            splice_list = self.get_src_splice()
            for w0, w1 in splice_list:
                r.splice_rate(w0, w1)
            if verbose:
                print("============== After splice ==============")
                print(r)

        for tozero in self.sublist_zero:
            r.substitute(tozero, Expr.zero())
            
        # print(r)
        if skip_simplify:
            return r
        else:
            # r = r.simplified(reg = r_prior)
            r = r.simplified()
            # r = r.simplified(self.reg)
            r.remove_missing_aux()
        
            if verbose:
                print("============== Simplified ==============")
                print(r)
                
            return r
            
        
    def nfold(self, n = 2, natural_eqprob = True, all_eqprob = True):
        """
        Return the n-letter setting.

        Parameters
        ----------
        n : int
            The blocklength.

        Returns
        -------
        CodingModel.

        """
        
        r = self.copy()
        
        bnet_out = BayesNet()
        reg_out_map = {}
        clen = n
        
        toeqprob = Comp.empty()
            
        for x in self.bnet.allcomp() + self.reg.allcomprv():
            
            t = None
            if self.is_rate(x):
                t = x.copy()
            else:
                t = rv_array(x.get_name(), n)
                t[0] = x.copy()
                if natural_eqprob or all_eqprob:
                    toeqprob += x
                    
            reg_out_map[x.get_name()] = t
        
        for (a, b) in self.bnet.edges():
            if not all_eqprob:
                toeqprob -= b
            am = reg_out_map[a.get_name()]
            bm = reg_out_map[b.get_name()]
            node = self.find_node_rv_out(b)
            if node is None:
                bnet_out += (am, bm)
            else:
                if node.rv_in_causal.ispresent(a) and isinstance(am, CompArray) and isinstance(bm, CompArray):
                    for t1 in range(clen):
                        for t2 in range(t1 + 1, clen):
                            bnet_out += (am[t1]+bm[t1], bm[t2])
                else:
                    bnet_out += (am.allcomp(), bm.allcomp())
                    
        for x in self.bnet.allcomp():
            if self.bnet.is_fcn(x):
                xm = reg_out_map[x.get_name()]
                bnet_out.set_fcn(xm)
                
        r.bnet = bnet_out
        
        r.reg = Region.universe()
        for i in range(n):
            tr = self.reg.copy()
            for x in self.reg.allcomprv():
                v = reg_out_map[x.get_name()]
                if isinstance(v, CompArray):
                    tr.substitute(x, v[i])
            r.reg = r.reg & tr
        
        if not toeqprob.isempty():
            for i in range(1, n):
                v0 = CompArray.empty()
                vi = CompArray.empty()
                for x in toeqprob:
                    v0.append(reg_out_map[x.get_name()][0])
                    vi.append(reg_out_map[x.get_name()][i])
                r.reg = r.reg & (ent_vector(*v0) == ent_vector(*vi))
        
        def map_getlist_id(a, i):
            if isinstance(a, Expr):
                return a.copy()
            t = None
            for c in a:
                cm = reg_out_map[c.get_name()]
                if isinstance(cm, CompArray):
                    cm = cm[i]
                if t is None:
                    t = cm.copy()
                else:
                    t += cm
            return t
        
        
        r.sublist = []
        for i in range(n):
            for v0, v1 in self.sublist:
                r.sublist.append((map_getlist_id(v0, i), map_getlist_id(v1, i)))
                
        r.nodes = []
        for i in range(n):
            for node in self.nodes:
                v = reg_out_map[node.rv_out.get_name()]
                if isinstance(v, CompArray):
                    v = v[i]
                else:
                    if i != 0:
                        continue
                
                cnode = node.copy()
                cnode.clear()
                cnode.rv_out = v.copy()
                r.nodes.append(cnode)
                
        return r
        
    def convexify_needed(self):
        return self.get_outer(convexify = False, convexify_test = True)
        
    def get_outer(self, oneshot = False, future = True, convexify = None, 
                  add_csiszar_sum = True, leaf_remove = True, node_fcn = True,
                  node_fcn_force = False, skip_simplify = True, convexify_test = False):
        """Get an outer bound of the capacity region.
        """

        self.presolve()

        if self.bnet.tsorted() is None:
            raise ValueError("Non-strictly-causal cycle detected.")
            return None
        
        if convexify is None:
            convexify = self.convexify_needed()
            
        if convexify is not False:
            if convexify is True:
                convexify = Comp.rv("Q_T")
                
        # auxs = self.get_aux()
        # reg_out_aux = auxs.copy()
        reg_out_aux = Comp.empty()
        
        bnet_out = BayesNet()
        reg_out_map = {}
        clen = 1 if oneshot else 3 if future else 2
        ttr = [1, 0, 2]
        
        # for v0, v1 in self.sublist:
        #     print(str(v0) + "  " + str(v1))
        
        rv_wfuture = Comp.empty()
        to_csiszar = []
        rv_series = Comp.empty()
        rv_timedep = Comp.empty()
            
        for x in self.bnet.allcomp():
            removable = False
            removed = False
            if leaf_remove and self.bnet.get_children(x).isempty() and x.ispresent(self.get_rv_sub(x)):
                removable = True
            
            # print(str(x) + "   " + str(self.is_rate(x)) + "  " + str(self.get_rv_rates(x)))
            t = None
            if self.is_rate(x):
                # t = CompArray([x.copy()])
                t = x.copy()
            elif oneshot:
                t = CompArray([x.copy()])
            elif future:
                t = CompArray.series_sym(x)
                if removable:
                    t[1] = Comp.empty()
                    t[2] = Comp.empty()
                    removed = True
                else:
                    rv_wfuture += x
            else:
                t = CompArray.series_sym(x, suff = None)
            reg_out_map[x.get_name()] = t
            if isinstance(t, CompArray):
                for tt in t:
                    rv_timedep += tt
                for tt in t[1:]:
                    rv_series += tt
            if not removed:
                to_csiszar.append(t)
        
        for (a, b) in self.bnet.edges():
            # if not self.get_rv_sub(b).ispresent(b):
            #     continue
            
            am = reg_out_map[a.get_name()]
            bm = reg_out_map[b.get_name()]
            node = self.find_node_rv_out(b)
            if node is None:
                bnet_out += (am.swapped_id(0, 1), bm.swapped_id(0, 1))
            else:
                if convexify is False and self.is_rate(b) and am.allcomp().ispresent(rv_series):
                    rv_series += bm.allcomp()
                    
                if node.rv_in_causal.ispresent(a) and isinstance(am, CompArray) and isinstance(bm, CompArray):
                    for t1 in range(clen):
                        for t2 in range(t1, clen):
                            if convexify is False:
                                bnet_out += (am[ttr[t1]] + bm[ttr[t1]], bm[ttr[t2]])
                            else:
                                bnet_out += (am[ttr[t1]], bm[ttr[t2]])
                else:
                    if convexify is False:
                        bnet_out += (am.swapped_id(0, 1).allcomp(), bm.swapped_id(0, 1).allcomp())
                    else:
                        for tbm in bm.swapped_id(0, 1).allcomp():
                            bnet_out += (am.swapped_id(0, 1).allcomp(), tbm)
                    
        for b in self.bnet.allcomp():
            bm = reg_out_map[b.get_name()]
            if not isinstance(bm, CompArray):
                continue
            node = self.find_node_rv_out(b)
            if node is None:
                continue
            for a in node.rv_in_scausal:
                am = reg_out_map[a.get_name()]
                if not isinstance(am, CompArray):
                    continue
                for t1 in range(clen):
                    for t2 in range(t1, clen):
                        if t1 == 1 and t2 == 1:
                            continue
                        if convexify is False:
                            bnet_out += (am[ttr[t1]] + bm[ttr[t1]], bm[ttr[t2]])
                        else:
                            bnet_out += (am[ttr[t1]], bm[ttr[t2]])
            
            
        if convexify is False:
            for x in self.bnet.allcomp():
                parent_empty = self.bnet.get_parents(x).isempty()
                if parent_empty:
                    xm = reg_out_map[x.get_name()]
                    if isinstance(xm, CompArray) and len(xm) >= 3:
                        bnet_out += (xm[1], xm[2])
            
        rvnats = Comp.empty()
        
        for x in self.bnet.allcomp():
            parent_empty = self.bnet.get_parents(x).isempty()
            if parent_empty or self.find_node_rv_out(x) is not None:
                xm = reg_out_map[x.get_name()]
                if isinstance(xm, CompArray):
                    for txm in (xm[1:] if parent_empty else xm):
                        if txm.isempty():
                            continue
                        series_parent = bnet_out.get_parents(txm).ispresent(rv_series)
                        if series_parent:
                            continue
                        rvnats += txm
                        if convexify is not False:
                            bnet_out += (convexify, txm)
                        else:
                            if not txm.ispresent(rv_series[0]):
                                if convexify_test:
                                    # print((rv_series[0], txm))
                                    return True
                                bnet_out += (rv_series[0], txm)
        
        
        # print(rv_series)
        
        for x in self.bnet.allcomp():
            if self.bnet.is_fcn(x):
                xm = reg_out_map[x.get_name()]
                bnet_out.set_fcn(xm)
            elif node_fcn and self.find_node_rv_out(x) is not None:
                xm = reg_out_map[x.get_name()]
                if isinstance(xm, CompArray):
                    for txm in xm:
                        if txm.isempty():
                            continue
                        series_parent = bnet_out.get_parents(txm).ispresent(rv_series)
                        if node_fcn_force or convexify is not False or series_parent:
                            bnet_out.set_fcn(txm)
                            # print(txm)
                else:
                    bnet_out.set_fcn(xm)
        
        bnet_out = bnet_out.scc()
        
        if convexify_test:
            return False
        
        # print(bnet_out)
        self.bnet_out = bnet_out
        self.bnet_out_arrays = []
        for x in self.bnet.allcomp():
            xm = reg_out_map[x.get_name()]
            if isinstance(xm, CompArray):
                t = xm.swapped_id(0, 1).allcomp()
                if len(t) > 1:
                    self.bnet_out_arrays.append(t)
        
        r = bnet_out.get_region()
        
        if not self.reg.isuniverse():
            # r = r & self.reg
            r = r.and_cause_consequence(self.reg)
            
        if not oneshot and future and add_csiszar_sum:
            # r &= csiszar_sum(*(list(x for _, x in reg_out_map.items())))
            r &= csiszar_sum(*to_csiszar)
        
        if convexify is not False:
            reg_out_aux += convexify
            for x in rv_series:
                r &= Expr.Hc(convexify, x) == 0
        
        reg_out_aux += bnet_out.allcomp() - self.bnet.allcomp()

        # for b in self.bnet.allcomp():
        #     b2 = self.get_rv_sub(b)
        #     if b2.ispresent(b):
        #         continue
            # if self.is_rate(b):
            #     rate = self.get_rv_rate(b)
            #     r &= rate >= 0
        
        def map_getlist(a):
            t = None
            for c in a:
                cm = reg_out_map[c.get_name()]
                if t is None:
                    t = cm.copy()
                else:
                    t += cm
            return t
        
        def map_substitute(r, a, b):
            am = map_getlist(a)
            bm = map_getlist(b)
            if isinstance(am, Comp) or isinstance(bm, Comp):
                r.substitute(am.allcomp(), bm.allcomp())
            else:
                for a2, b2 in zip(am, bm):
                    r.substitute(a2, b2)
                
        msgs = Comp.empty()
        msgnats = Comp.empty()
        for v0, v1 in self.sublist:
            if isinstance(v1, Expr) and not isinstance(v0, Expr):
                msgs += reg_out_map[v0[0].get_name()]
                
                vout = self.bnet.get_parents(v0[0])
                
                if vout.isempty():
                    msgnats += reg_out_map[v0[0].get_name()]
        
        
        for v0, v1 in self.sublist:
            if isinstance(v0, Comp):
                rv_wfuture -= v0
            if isinstance(v1, Expr) and not isinstance(v0, Expr):
                map_substitute(r, v0, v0[0])
                
                isnat = False
                vout = self.bnet.get_parents(v0[0])
                
                if vout.isempty():
                    vout = self.bnet.get_children(v0[0])
                    isnat = True
                    
                voutm = map_getlist(vout)
                voutm0 = voutm[0]
                voutm1 = Comp.empty()
                if len(voutm) >= 2:
                    voutm1 = voutm[1]
                
                am = reg_out_map[v0[0].get_name()]
                
                if isnat:
                    if oneshot:
                        r &= Expr.H(am) >= v1
                        r &= v1 >= 0
                    else:
                        for cmsg in igen.subset(msgnats - am, minsize = 1):
                            r &= Expr.Ic(am, voutm0, voutm1 + cmsg) >= Expr.Ic(am, voutm0, voutm1)
                        r &= Expr.Ic(am, voutm0, voutm1) >= v1
                        r &= v1 >= 0
                        # r &= Expr.Ic(am, voutm0, voutm1) == v1
                else:
                    if oneshot:
                        r &= Expr.H(am) <= v1
                    else:
                        for ctv in igen.subset(rv_wfuture - vout):
                            tv = map_getlist(ctv)
                            if tv is None:
                                tv = Comp.empty()
                            else:
                                tv = tv[0] + tv[1] + tv[2]
                            for cmsg in igen.subset(msgs - am):
                                r &= Expr.Ic(am, voutm0, voutm1 + tv + cmsg) <= v1
                    
                    # r &= Expr.Ic(am, voutm0, voutm1) == v1
                    
                # print(v0)
                # print(v1)
                reg_out_aux += am
            else:
                map_substitute(r, v0, v1)
                
        reg_out_aux = reg_out_aux.inter(r.allcomprv())
        r = r.exists(reg_out_aux)
        
        # print(r)

        for tozero in self.sublist_zero:
            r.substitute(tozero, Expr.zero())
        
        if not skip_simplify:
            return r.simplified()
        else:
            return r
        
    def get_outer_nfold(self, n = 2, **kwargs):
        return self.nfold(n).get_outer(**kwargs) / n
        

    def optimum(self, v, b, sn, name = None, inner = True, outer = True, inner_kwargs = None, outer_kwargs = None, tighten = False, **kwargs):
        """Return the variable obtained from maximizing (sn=1)
        or minimizing (sn=-1) the expression v over variables b (Comp, Expr or list)
        """
        if inner_kwargs is None:
            inner_kwargs = dict()
        if outer_kwargs is None:
            outer_kwargs = dict()
        
        if inner:
            inner = self.get_inner(**inner_kwargs)
        else:
            inner = None
        if outer:
            outer = self.get_outer(**outer_kwargs)
        else:
            outer = None
        
        if name is not None:
            name += PsiOpts.settings["fcn_suffix"]

        if inner is not None:
            return inner.optimum(v, b, sn, name = name, reg_outer = outer, tighten = tighten, quick = None, quick_outer = True, **kwargs)
        if outer is not None:
            return outer.optimum(v, b, sn, name = name, tighten = tighten, quick = True, **kwargs)
        return None
    
    def maximum(self, expr, vs = None, **kwargs):
        """Return the variable obtained from maximizing the expression expr
        over variables vs (Comp, Expr or list)
        """
        return self.optimum(expr, vs, 1, **kwargs)
    
    def minimum(self, expr, vs = None, **kwargs):
        """Return the variable obtained from minimizing the expression expr
        over variables vs (Comp, Expr or list)
        """
        return self.optimum(expr, vs, -1, **kwargs)

    def node_groups(self):
        r = []
        for node in self.nodes:
            a = self.bnet.get_parents(node.rv_out)
            b = node.rv_out.copy()
            for t in r:
                if a.super_of(t[1]) and (t[1] + t[0]).super_of(a):
                    t[0] += b
                    break
            else:
                r.append([b, a, node])
        return r
    
    def node_group_info(self):
        groups = self.node_groups()
        return groups
        
    def graph(self, lr = True, enc_node = True, ortho = False):
        """Return the graphviz digraph of the network that can be displayed in the console.
        """
        
        r = graphviz.Digraph()
        if lr:
            r.graph_attr["rankdir"] = "LR"
        
        if ortho:
            r.graph_attr["splines"] = "ortho"

        groups = self.node_groups()
        
        rvs = self.bnet.allcomp()
        
        for a in self.bnet.allcomp():
            shape = "plaintext" #"oval"
            node = self.find_node_rv_out(a)
            label = str(self.get_rv_sub(a))
            if node is not None and not enc_node:
                shape = "rect"
            
            if shape == "plaintext":
                r.node(a.get_name(), label, shape = shape, margin = "0")
            else:
                r.node(a.get_name(), label, shape = shape)
        
        if enc_node:
            for i, (b, a, node) in enumerate(groups):
                cname = "enc_" + str(a) + "_" + str(b)
                label = str(i + 1)
                if node.label is not None:
                    label = node.label
                r.node(cname, label, shape = "rect")
                for ai in a:
                    r.edge(ai.get_name(), cname)
                for bi in b:
                    r.edge(cname, bi.get_name())
                
                for ai in node.rv_in_scausal:
                    r.edge(ai.get_name(), cname, style = "dashed")

                rvs -= b
                
        for (a, b) in self.bnet.edges():
            if b in rvs:
                r.edge(a.get_name(), b.get_name())
            
        return r
        
    def graph_outer(self, **kwargs):
        return self.bnet_out.graph(groups = self.bnet_out_arrays, **kwargs)
    

class CommEnc(IBaseObj):
    
    def __init__(self, rv_out, msgs, rv_in_causal = None):
        # self.rv_in = Comp.empty()
        # self.msgs = []
        
        # for x in list_in:
        #     if isinstance(x, Comp):
        #         self.rv_in += x
        #     else:
        #         self.msgs.append(x)
        self.msgs = msgs
        
        self.rv_out = rv_out
        
        if rv_in_causal is None:
            self.rv_in_causal = Comp.empty()
        else:
            self.rv_in_causal = rv_in_causal
            
    
    def record_to(self, index, skip_msg = False):
        # self.rv_in.record_to(index)
        self.rv_out.record_to(index)
        self.rv_in_causal.record_to(index)
        if not skip_msg:
            for x in self.msgs:
                for y in x:
                    y.record_to(index)
        
class CommDec(IBaseObj):
    
    def __init__(self, rv_in, msgs, msgints = None):
        self.rv_in = rv_in
        self.msgs = msgs
        self.msgints = msgints
        
    
    def record_to(self, index):
        self.rv_in.record_to(index)
        for x in self.msgs:
            for y in x:
                y.record_to(index)
        
        if self.msgints is not None:
            for x in self.msgints:
                for y in x:
                    y.record_to(index)

class CommModel(IBaseObj):
    
    def __init__(self, bnet = None, reg = None, nature = None):
        if bnet is None:
            self.bnet = BayesNet()
        else:
            self.bnet = bnet
        if reg is None:
            self.reg = Region.universe()
        else:
            self.reg = reg
        
        self.bnet_in = None
        self.reg_in = None
        
        self.bnet_out = None
        self.reg_out = None
        self.reg_out_aux = None
        self.reg_out_tsrv = None
        self.reg_out_vs = None
        self.reg_out_map = None
        
        self.msgs = None
        self.maux_rt = None
        self.maux_rtsub = None
        self.decs = None
        if nature is None:
            self.nature = Comp.empty()
        else:
            self.nature = nature
        self.enclist = []
        self.declist = []
        
    def __iadd__(self, other):
        if isinstance(other, CommEnc):
            self.enclist.append(other)
        elif isinstance(other, CommDec):
            self.declist.append(other)
        else:
            self.bnet += other
        return self
        
                
    def create_reg_in(self, name_prefix = "A_"):
        self.bnet_in = self.bnet.copy()
        self.msgs = []
        self.decs = []
        
        index = IVarIndex()
        self.reg.record_to(index)
        for x in self.enclist:
            x.record_to(index)
        for x in self.declist:
            x.record_to(index)
            
        for x in self.enclist:
            cauxall = Comp.empty()
            rv_in = self.bnet.get_parents(x.rv_out)
            rv_in = rv_in - x.rv_in_causal
            for m in x.msgs:
                caux = None
                if isinstance(m, Expr):
                    tname = name_prefix + str(m)
                    tname = index.name_avoid(tname)
                    caux = Comp.rv(tname)
                    caux.record_to(index)
                else:
                    caux = m[1].copy()
                self.msgs.append([rv_in.copy(), m[0], caux, x.rv_out.copy()])
                cauxall += caux
            self.bnet_in += (rv_in, cauxall)
            self.bnet_in += (cauxall, x.rv_out)
            # self.reg_in &= (Expr.Ic(cauxall, index.comprv - cauxall - (x.rv_in + x.rv_out), 
            #                          x.rv_in + x.rv_out) == 0)
                
        # for x in self.declist:
        #     self.decs.append([x.rv_in.copy(), x.msgs.copy(), iutil.copy(x.msgints)])
        
        self.reg_in = self.bnet_in.get_region() & self.reg
             
    def get_encout(self):
        r = Comp.empty()
        for x in self.enclist:
            r += x.rv_out
        return r
    
    def enc_id_get_aux(self, i):
        r = Comp.empty()
        for rv_in, rtbs, b, rv_out in self.msgs:
            if self.enclist[i].rv_out.ispresent(rv_out):
                r += b
        return r
        
    
    def create_reg_out(self, future = True):
        self.create_reg_in("M_")
        auxs = sum(x[2] for x in self.msgs)
        self.reg_out_aux = auxs.copy()
        #self.reg_out_tsrv = Comp.rv("#TS")
        self.bnet_out = BayesNet()
        self.reg_out_map = {}
        clen = 3 if future else 2
        ttr = [1, 0, 2]
        
        for x in self.bnet.allcomp():
            t = None
            if future:
                t = CompArray.series_sym(x)
            else:
                t = CompArray.series_sym(x, suff = None)
            self.reg_out_map[x.get_name()] = t
        
        for (a, b) in self.bnet.edges():
            self.bnet_out += (self.reg_out_map[a.get_name()], self.reg_out_map[b.get_name()])
        
        for i in range(len(self.enclist)):
            auxs = self.enc_id_get_aux(i)
            rv_out = self.enclist[i].rv_out
            rv_in_causal = self.enclist[i].rv_in_causal
            rv_out_map = self.reg_out_map[rv_out.get_name()]
            
            for x in rv_out_map:
                self.bnet_out += (auxs, x)
            
            for pa in self.bnet.get_parents(rv_out):
                pa_map = self.reg_out_map[pa.get_name()]
                if rv_in_causal.ispresent(pa):
                    for t1 in range(clen):
                        for t2 in range(t1 + 1, clen):
                            self.bnet_out += (pa_map[ttr[t1]], rv_out_map[ttr[t2]])
                else:
                    for t1 in range(clen):
                        for t2 in range(clen):
                            if t1 != t2:
                                self.bnet_out += (pa_map[t1], rv_out_map[t2])
            
        
            
        self.reg_out = self.bnet_out.get_region() & self.reg
        if future:
            self.reg_out &= csiszar_sum(*(list(x for _, x in self.reg_out_map.items()) + list(auxs)))
            
        self.reg_out_aux += self.bnet_out.allcomp() - self.bnet.allcomp()

        
    def get_outer(self, future = True, convexify = False, skip_simplify = False):
        self.create_reg_out(future = future)
        r = self.reg_out.copy()
        
        rts = self.get_rates()
        for rt in rts:
            r &= rt >= 0
        for dec in self.declist:
            a = dec.rv_in
            rts = dec.msgs
            ap = self.reg_out_map[a.get_name()][1]
            for rt in rts:
                for i, (_, rtbs, b, _) in enumerate(self.msgs):
                    if rtbs.ispresent(rt):
                        r &= rt <= Expr.Ic(b, a, ap)
                        
        aux = self.reg_out_aux
        
        # if convexify is not False:
        #     if convexify is True:
        #         convexify = Comp.rv("Q_T")
        #     r.condition(convexify)
        #     encout = self.get_encout()
        #     encout_map
        #     r &= markov(r.allcomprv() - encout - convexify, encout, convexify)
        #     aux += convexify
        
        r = r.exists(aux)
        
        if not skip_simplify:
            return r.simplified()
        else:
            return r
        
                
    
    def get_decreq(self):
        r = []
        for dec in self.declist:
            a = dec.rv_in
            rts = dec.msgs
            reqs = []
            reqints = None
            if dec.msgints is not None:
                reqints = []
            for i, (_, rtbs, b, _) in enumerate(self.msgs):
                if any(rtbs.ispresent(rt) for rt in rts):
                    reqs.append(i)
                if dec.msgints is not None and any(rtbs.ispresent(rt) for rt in dec.msgints):
                    reqints.append(i)
            # cknown = a.copy()
            
            for j in range(len(reqs) - 1, -1, -1):
                
                # cdec = a.copy()
                # for cc in cknown:
                #     ccan = self.bnet_in.get_ancestors(cc)
                #     if cknown.super_of(ccan):
                #         cdec += cc
                # r.append((reqs[j], cdec))
                
                # TODO ?????????????
                r.append((reqs[j], a, reqs[j+1:], None if reqints is None else [a for a in reqints if a < reqs[j]]))
                # r.append((reqs[j], a+cknown))
                
                # cknown += self.msgs[reqs[j]][2]
        return r
    
    def get_ratereq(self, known, reqprevs, i, tlist):
        
        verbose = PsiOpts.settings.get("verbose_commmodel", False)
        
        r = Region.universe()
        lmask = (1 << i) + sum(1 << t for t in tlist)
        submask = sum(1 << j for j in range(i) if self.maux_rtsub[j] is not None)
        
        for tmask in range(1 << len(tlist)):
            mask = (1 << i) + sum(1 << tlist[j] for j in range(len(tlist)) if tmask & (1 << j))
            
            expr = Expr.zero()
            
            for j in range(i + 1):
                if mask & (1 << j):
                    
                    expr += self.maux_rt[j]
                    
                    if self.maux_rtsub[j] is not None:
                        expr += self.maux_rtsub[j]
                        
                    expr -= mi_rect_max(self.msgs[j][2], [known] +
                                        [(self.msgs[j2][2], self.maux_rtsub[j2]) if self.maux_rtsub[j2] is not None
                                         else self.msgs[j2][2] for j2 in reqprevs] +
                                        [(self.msgs[j2][2], self.maux_rtsub[j2]) if self.maux_rtsub[j2] is not None
                                         else self.msgs[j2][2] for j2 in range(j) if lmask & (1 << j2)])
                    
                    expr += mi_rect_max([self.msgs[j][0]] + 
                                        [(self.msgs[j2][2], self.maux_rtsub[j2]) if self.maux_rtsub[j2] is not None
                                         else self.msgs[j2][2] for j2 in range(j)],
                                        self.msgs[j][2])
                
            expr.simplify()
            r &= (expr <= 0)
        
        r_st = None
        if verbose:
            print("========== inner bound ==========")
            print("known:" + str(known) + "  decoded:" + str(sum(self.msgs[t][2] for t in reqprevs))
                  + "  nonunique:" + str(sum(self.msgs[t][2] for t in tlist)) + "  target:" + str(self.msgs[i][2]))
            r_st = str(r)
            print(r_st)
            
        r = r.flattened(minmax_elim = True)
        
        if verbose:
            r_st2 = str(r)
            if r_st2 != r_st:
                print("expanded:")
                print(r_st2)
            print("")
            
        return r
    
    def get_ratereq_old(self, known, i, tlist):
        r = Region.universe()
        lmask = (1 << i) + sum(1 << t for t in tlist)
        submask = sum(1 << j for j in range(i) if self.maux_rtsub[j] is not None)
        
        for tmask in range(1 << len(tlist)):
            mask = (1 << i) + sum(1 << tlist[j] for j in range(len(tlist)) if tmask & (1 << j))
            
            for cp in itertools.product(*[igen.subset_mask(submask & ((1 << j) - 1))
                                        for j in range(i + 1) if mask & (1 << j)]):
                expr = Expr.zero()
                caux = Comp.empty()
                #cauxmiss = Comp.empty()
                ji = 0
                for j in range(i + 1):
                    if mask & (1 << j):
                        
                        # TODO ??????????????????????????????????????
                        expr += self.maux_rt[j] - Expr.I(self.msgs[j][2], known + caux)
                        if self.maux_rtsub[j] is not None:
                            expr += self.maux_rtsub[j]
                            
                        cauxmiss = self.msgs[j][0].copy()
                        for j2 in range(j):
                            if submask & (1 << j2):
                                if cp[ji] & (1 << j2):
                                    expr -= self.maux_rtsub[j2]
                                    cauxmiss += self.msgs[j2][2]
                            else:
                                cauxmiss += self.msgs[j2][2]
                        if not cauxmiss.isempty():
                            expr += Expr.I(self.msgs[j][2], cauxmiss)
                            
                        ji += 1
                    if lmask & (1 << j):
                        caux += self.msgs[j][2]
                    #cauxmiss += self.msgs[j][2]
                expr.simplify()
                r &= (expr <= 0)
        # print("==========")
        # print(str(known) + "  " + str(self.msgs[i][2]) + "  " + str(sum(self.msgs[t][2] for t in tlist)))
        # print(r)
        return r
    
    def get_rates(self):
        r = Expr.zero()
        for _, rts, a, _ in self.msgs:
            for rt in rts:
                if not r.ispresent(rt):
                    r += rt
        return r
    
    
    def get_aux(self):
        r = Comp.empty()
        for _, rts, a, _ in self.msgs:
            r += a
        return r
                    
        
    
    def get_inner_iter(self, subcodebook = True, skip_simplify = False):
        self.maux_rt = [rt[0] for _, rt, a, _ in self.msgs]
        self.maux_rtsub = [None] * len(self.msgs)
        if subcodebook is True:
            self.maux_rtsub = [Expr.real("#RS_" + x.get_name()) for _, rt, x, _ in self.msgs]
            self.maux_rtsub[-1] = None
        elif isinstance(subcodebook, Comp):
            self.maux_rtsub = [Expr.real("#RS_" + x.get_name()) 
                               if subcodebook.ispresent(x) else None for _, rt, x, _ in self.msgs]
            self.maux_rtsub[-1] = None
            subcodebook = True
            
        reqs = self.get_decreq()
        #r = Region.universe()
        r = self.reg_in.copy()
        rts = self.get_rates()
        for rt in rts:
            r &= rt >= 0
        if subcodebook:
            for rt in self.maux_rtsub:
                if rt is not None:
                    r &= rt >= 0
                    
        for reqi, known, reqprevs, reqints in reqs:
            r2 = Region.empty()
            if reqints is None:
                for mask in range(1 << reqi):
                    tlist = []
                    for i in range(reqi):
                        if mask & (1 << i):
                            tlist.append(i)
                    r2 |= self.get_ratereq(known, reqprevs, reqi, tlist)
            else:
                r2 = self.get_ratereq(known, reqprevs, reqi, reqints)
            if not skip_simplify:
                r2 = r2.simplified()
            
            r &= r2
            
        if subcodebook and any(x is not None for x in self.maux_rtsub):
            r.eliminate(sum(x for x in self.maux_rtsub if x is not None))
        return r
    
    def rate_splitting(self, r):
        if self.msgs is None:
            self.create_reg_in()
        
        rttmp = [Expr.real("#RT_" + x.get_name()) for _, rt, x, _ in self.msgs]
        
        for rt1, rt2 in zip((rt for _, rt, x, _ in self.msgs), rttmp):
            r.substitute(rt1, rt2)
            
        decflag = [0] * len(self.msgs)
        for i, (_, rtbs, b, _) in enumerate(self.msgs):
            for j, dec in enumerate(self.declist):
                a = dec.rv_in
                rts = dec.msgs
                if any(rtbs.ispresent(rt) for rt in rts):
                    decflag[i] |= 1 << j
        
        rtauxs = []
        n = len(self.msgs)
        rn = Region.universe()
        
        bexps = [Expr.zero() for i in range(n)]
        
        for i, (_, rtbs, b, _) in enumerate(self.msgs):
            vis = []
            cexp = Expr.zero()
            for mask in range(1, 1 << n):
                if any(mask | v == mask for v in vis):
                    continue
                
                cdecflag = 0
                for i2 in range(n):
                    if mask & (1 << i2):
                        cdecflag |= decflag[i2]
                if cdecflag | decflag[i] != cdecflag:
                    continue
                
                vis.append(mask)
                crtaux = Expr.real("#RTA_" + str(i) + "_" + str(mask))
                rn &= crtaux >= 0
                
                for i2 in range(n):
                    if mask & (1 << i2):
                        bexps[i2] += crtaux
                cexp += crtaux
                rtauxs.append(crtaux)
                
            rn &= rtbs <= cexp
            
        
        for i in range(n):
            rn &= bexps[i] <= rttmp[i]
        
        r &= rn
            
        rts = self.get_rates()
        for rt in rts:
            r &= rt >= 0
        
        # print(r)
        r.eliminate(sum(rttmp + rtauxs, Expr.zero()))
        # print(r)
        
        return r
        
    
    def get_inner(self, subcodebook = True, rate_split = False, shuffle = False, 
                  convexify = False, convexify_diag = False,
                  skip_simplify = False, skip_simplify_iter = False):
        r = None
        self.create_reg_in()
        aux = self.get_aux()
        
        if shuffle:
            r = Region.empty()
            tmaux = self.msgs
            for cmaux in itertools.permutations(tmaux):
                tbnet = self.bnet.copy()
                for i in range(len(cmaux) - 1):
                    tbnet += (cmaux[i][3], cmaux[i+1][3])
                if tbnet.iscyclic():
                    # print("CYCLIC " + str(cmaux))
                    continue
                
                self.msgs = list(cmaux)
                r |= self.get_inner_iter(subcodebook = subcodebook, skip_simplify = skip_simplify_iter)
            self.msgs = tmaux
            
        else:
            r = self.get_inner_iter(subcodebook = subcodebook, skip_simplify = skip_simplify_iter)
            
        if convexify_diag:
            r = r.convexified_diag(self.get_rates(), skip_simplify = skip_simplify)
            
        if convexify is not False:
            if convexify is True:
                convexify = Comp.rv("Q_T")
            r.condition(convexify)
            encout = self.get_encout()
            r &= markov(r.allcomprv() - encout - convexify, encout, convexify)
            aux += convexify
        
        if rate_split:
            r = self.rate_splitting(r)
        
        if not skip_simplify:
            #return r.distribute().simplified()
            r = r.exists(aux)
            # print(r)
            r.simplify_union()
            # print(r)
            r.remove_missing_aux()
            return r.simplified_quick()
        
        else:
            return r.exists(aux)
            

# Generators

class igen:
    
    def subset_mask(mask):
        x = 0
        while True:
            yield x
            if x >= mask:
                break
            x = ((x | ~mask) + 1) & mask
    
    def partition_mask(mask, n, max_mask = None):
        if n == 1:
            if mask != 0 and not (max_mask is not None and mask > max_mask):
                yield (mask,)
            return
        
        for xmask in igen.subset_mask(mask):
            if xmask == 0:
                continue
            if max_mask is not None and xmask > max_mask:
                break
            for t in igen.partition_mask(mask & ~xmask, n - 1, xmask):
                yield t + (xmask,)
            
    def partition(x, n):
        m = len(x)
        for mask in igen.partition_mask((1 << m) - 1, n):
            yield tuple(sum(t for i, t in enumerate(x) if xmask & (1 << i)) for xmask in mask)
        
    def subset(x, minsize = 0, maxsize = 100000, size = None, reverse = False, coeffmode = 0, replacement = False):
        """Iterate sum of subset.
        Parameters:
            coeffmode : Set to 0 to only allow positive terms.
                        Set to 1 to allow positive/negative terms, but not all negative.
                        Set to 2 to allow positive/negative terms.
                        Set to -1 to allow positive/negative terms, but not all positive/negative.
        """
        
        #if isinstance(x, types.GeneratorType):
        if not hasattr(x, '__len__'):
            x = list(x)
        
        if size is not None:
            minsize = size
            maxsize = size
        
        iterfcn = None
        if replacement:
            iterfcn = itertools.combinations_with_replacement
        else:
            iterfcn = itertools.combinations
            
        n = len(x)
        cr = None
        if reverse:
            maxsize = min(maxsize, n)
            cr = range(maxsize, minsize - 1, -1)
        else:
            cr = range(minsize, maxsize + 1)
            
        for s in cr:
            if s > n:
                return
            if s == 0:
                if len(x) > 0:
                    if isinstance(x[0], Comp):
                        yield Comp.empty()
                    elif isinstance(x[0], Expr):
                        yield Expr.zero()
                    else:
                        yield 0
                else:
                    if isinstance(x, Comp):
                        yield Comp.empty()
                    elif isinstance(x, Expr):
                        yield Expr.zero()
                    else:
                        yield 0
                    
            elif s == 1:
                if coeffmode == 2:
                    for y in x:
                        yield y
                        yield -y
                else:
                    for y in x:
                        yield y
            else:
                if coeffmode == 0:
                    for comb in iterfcn(x, s):
                        yield sum(comb)
                else:
                    for comb in iterfcn(x, s):
                        for sflag in range(int(coeffmode == -1),
                        (1 << s) - int(coeffmode == 1 or coeffmode == -1)):
                            yield sum(comb[i] * (-1 if sflag & (1 << i) else 1)
                                      for i in range(s))
    
    def pm(x):
        """Plus or minus.
        """
        for a in x:
            yield a
            yield -a
    
    def sI(x, maxsize = 100000, cond = True, ent = True, pm = False):
        """Iterate mutual information of variables in x.
        Parameters:
            cond : Whether to include conditional mutual information.
            ent  : Whether to include entropy.
            pm   : Whether to include both positive and negative.
        """
        
        n = len(x)
        
        def xmask(mask):
            return sum((x[i] for i in range(n) if mask & (1 << i)), Comp.empty())
        
        
        for mask in igen.subset([1 << i for i in range(n)], minsize = 1, maxsize = maxsize):
            bmask = mask
            
            if not cond:
                if ent:
                    yield Expr.H(xmask(bmask))
                    if pm:
                        yield -Expr.H(xmask(bmask))
                    
            while True:
                if not cond:
                    if bmask <= 0:
                        break
                    amask = mask - bmask
                    if amask > bmask:
                        break
                    if amask > 0:
                        yield Expr.I(xmask(amask), xmask(bmask))
                        if pm:
                            yield -Expr.I(xmask(amask), xmask(bmask))
                else:
                    a0mask = mask - bmask
                    if ent:
                        if a0mask > 0:
                            yield Expr.Hc(xmask(a0mask), xmask(bmask))
                            if pm:
                                yield -Expr.Hc(xmask(a0mask), xmask(bmask))
                            
                    while a0mask > 0:
                        a1mask = mask - bmask - a0mask
                        if a1mask > a0mask:
                            break
                        if a1mask > 0:
                            yield Expr.Ic(xmask(a1mask), xmask(a0mask), xmask(bmask))
                            if pm:
                                yield -Expr.Ic(xmask(a1mask), xmask(a0mask), xmask(bmask))
                        a0mask = (a0mask - 1) & (mask - bmask)
                    if bmask <= 0:
                        break
                bmask = (bmask - 1) & mask
    
    def test(x, fcn, sgn = 0, yield_set = False):
        """Test fcn for values in generator x.
        Set sgn = 1 if fcn is increasing.
        Set sgn = -1 if fcn is decreasing.
        """
        ub = MonotoneSet(sgn = 1)
        lb = MonotoneSet(sgn = -1)
        
        for a in x:
            if sgn != 0:
                if a in ub:
                    continue
                if a in lb:
                    continue
                    
            if fcn(a):
                if not yield_set:
                    yield a
                if sgn > 0:
                    ub.add(a)
                    if yield_set:
                        yield (a, ub)
                elif sgn < 0:
                    lb.add(a)
                    if yield_set:
                        yield (a, lb)
            else:
                if sgn > 0:
                    lb.add(a)
                elif sgn < 0:
                    ub.add(a)
                


# Shortcuts

def alland(a):
    """Intersection of elements in list a (using operator &)."""
    r = None
    for x in a:
        if r is None:
            r = x
        else:
            r &= x
    return r

def anyor(a):
    """Union of elements in list a (using operator |)."""
    r = None
    for x in a:
        if r is None:
            r = x
        else:
            r |= x
    return r
    
def rv(*args, **kwargs):
    """Random variable"""
    
    args = [x for b in args for x in iutil.split_comma(b)]
    r = Comp.empty()
    for a in args:
        r += Comp.rv(a)
    
    for key, value in kwargs.items():
        r.add_markers([(key, value)])
        
    return r
    
def rv_seq(name, st, en = None):
    """Sequence of random variables"""
    return Comp.array(name, st, en)
    
def rv_array(name, st, en = None):
    """Array of random variables"""
    return CompArray.make(*(Comp.array(name, st, en)))
    
def real(*args):
    """Real variable"""
    args = [x for b in args for x in iutil.split_comma(b)]
    if len(args) == 1:
        return Expr.real(args[0])
    return ExprArray([Expr.real(a) for a in args])
    
def real_array(name, st, en = None):
    """Array of random variables"""
    t = rv_seq(name, st, en)
    return ExprArray([Expr.real(a.name) for a in t.varlist])
    
def real_seq(name, st, en = None):
    """Array of random variables"""
    t = rv_seq(name, st, en)
    return ExprArray([Expr.real(a.name) for a in t.varlist])


def rv_empty():
    """Empty random variable"""
    return Comp.empty()
        
def zero():
    """Zero expression"""
    return Expr.zero()
        
def universe():
    """Universal set. 
    Returns a region with no constraints.
    """
    return Region.universe()
        
def empty():
    """Empty set. 
    Returns a region that is empty.
    """
    return RegionOp.empty()
        
def emptyset():
    """Empty set. 
    Returns a region that is empty.
    """
    return RegionOp.empty()
    
@fcn_list_to_list
def H(x):
    """Entropy. 
    Returns a symbolic expression for the entropy 
    e.g. use H(X + Y | Z + W) for H(X,Y|Z,W)
    """
    if isinstance(x, Comp):
        if x.isempty():
            return Expr.zero()
        return Expr.H(x)
    if isinstance(x, ConcDist):
        return x.entropy()
    if x.isempty():
        return Expr.zero()
    return Expr([(x.copy(), 1.0)])
    
def I(x):
    """Mutual information. 
    Returns a symbolic expression for the mutual information 
    e.g. use I(X + Y & Z | W) for I(X,Y;Z|W)
    """
    return H(x)
    
def I0(x):
    """Shorthand for I(...)==0. 
    e.g. use I0(X + Y & Z | W) for I(X,Y;Z|W)==0
    """
    return H(x) == 0
    
@fcn_list_to_list
def Hc(x, z):
    """Conditional entropy. 
    Hc(X, Z) is the same as H(X | Z)
    """
    return Expr.Hc(x, z)

@fcn_list_to_list
def Ic(x, y, z):
    """Conditional mutual information. 
    Ic(X, Y, Z) is the same as I(X & Y | Z)
    """
    return Expr.Ic(x, y, z)

@fcn_list_to_list
def indep(*args):
    """Return Region where the arguments are independent."""
    r = Region.universe()
    for i in range(1, len(args)):
        r &= Expr.I(iutil.sumlist(args[:i]), iutil.sumlist(args[i])) == 0
    return r

def indep_across(*args):
    """Take several arrays, return Region where entries are independent across dimension."""
    n = max([len(a) for a in args])
    vec = [iutil.sumlist([a[i] for a in args if i < len(a)]) for i in range(n)]
    return indep(*vec)

@fcn_list_to_list
def equiv(*args):
    """Return Region where the arguments contain the same information."""
    if len(args) <= 1:
        return Region.universe()
    r = Region.universe()
    for i in range(1, len(args)):
        if isinstance(args[0], Comp):
            r &= (Expr.Hc(args[i], args[0]) == 0) & (Expr.Hc(args[0], args[i]) == 0)
        elif isinstance(args[0], Expr):
            r &= args[0] == args[i]
        elif isinstance(args[0], Region):
            r &= args[0] == args[i]
    return r

@fcn_list_to_list
def markov(*args):
    """Return Region where the arguments form a Markov chain."""
    r = Region.universe()
    for i in range(2, len(args)):
        r &= Expr.Ic(iutil.sumlist(args[:i-1]), iutil.sumlist(args[i]), iutil.sumlist(args[i-1])) == 0
    return r

def eqdist(*args):
    """Return Region where the argument lists have the same distribution.
    Only equalities of entropies are enforced.
    e.g. eqdist([X, Y], [Z, W])
    """
    m = min(len(a) for a in args)
    r = Region.universe()
    for i in range(1, len(args)):
        for mask in range(1, 1 << m):
            x = Comp.empty()
            y = Comp.empty()
            for j in range(m):
                if mask & (1 << j) != 0:
                    x += args[0][j]
                    y += args[i][j]
            r &= Expr.H(x) == Expr.H(y)
    return r


def eqdist_across(*args):
    """Take several arrays, return Region where entries have the same distribution 
    across dimension. Only equalities of entropies are enforced.
    """
    n = min([len(a) for a in args])
    vec = [[a[i] for a in args] for i in range(n)]
    return eqdist(*vec)


def exchangeable(*args):
    """Return Region where the arguments are exchangeable random variables.
    Only equalities of entropies are enforced.
    e.g. exchangeable(X, Y, Z)
    """
    r = Region.universe()
    for tsize in range(1, len(args)):
        cvar = sum(args[:tsize])
        for comb in itertools.combinations(args, tsize):
            tvar = sum(comb)
            if tvar != cvar:
                r &= Expr.H(cvar) == Expr.H(tvar)
    return r


def iidseq(*args):
    """Return Region where the arguments form an i.i.d. sequence.
    Only equalities of entropies are enforced.
    e.g. iidseq(X, Y, Z), iidseq([X1,X2], [Y1,Y2], [Z1,Z2])
    """
    return indep(*args) & eqdist(*args)


def iidseq_across(*args):
    """Take several arrays, return Region where entries are i.i.d. 
    across dimension. Only equalities of entropies are enforced.
    """
    n = min([len(a) for a in args])
    vec = [[a[i] for a in args] for i in range(n)]
    return indep(*vec) & eqdist(*vec)


@fcn_list_to_list
def sfrl_cons(x, y, u, k = None, gap = None):
    """Strong functional representation lemma. 
    Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
    applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978.
    """
    if k is None:
        r = (Expr.Hc(y, x + u) == 0) & (Expr.I(x, u) == 0)
        if gap is not None:
            if not isinstance(gap, Expr):
                gap = Expr.const(gap)
            r &= Expr.Ic(x, u, y) <= gap
        return r
    else:
        r = (Expr.Hc(k, x + u) == 0) & (Expr.Hc(y, u + k) == 0) & (Expr.I(x, u) == 0)
        if gap is not None:
            if not isinstance(gap, Expr):
                gap = Expr.const(gap)
            r &= Expr.H(k) <= Expr.I(x, y) + gap
        return r



@fcn_list_to_list
def sunflower(*args):
    """Sunflower dependency.
    """
    r = Region.universe()
    for i in range(len(args)):
        r &= indep(*(args[:i] + args[i+1:])).conditioned(args[i])
    
    return r

@fcn_list_to_list
def stardep(*args):
    """Star dependency.
    """
    SU = Comp.rv("SU")
    r = Region.universe()
    for a in args:
        r &= H(SU | a) == 0
    return (r & indep(*args).conditioned(SU)).exists(SU)

@fcn_list_to_list
def cardbd(x, n):
    """Return Region where the cardinality of x is upper bounded by n."""
    if n <= 1:
        return H(x) == 0
    
    loge = PsiOpts.settings["ent_coeff"]
    
    V = rv_seq("V", 0, n-1)
    r = Expr.H(V[n - 2]) == 0
    r2 = Region.universe()
    for i in range(0, n - 1):
        r2 &= Expr.Hc(V[i], V[i - 1] if i > 0 else x) == 0
        r |= Expr.Hc(V[i - 1] if i > 0 else x, V[i]) == 0
    r = r.implicated(r2, skip_simplify = True).forall(V)
    return r & (H(x) <= numpy.log(n) * loge)


@fcn_list_to_list
def isbin(x):
    """Return Region where x is a binary random variable."""
    return cardbd(x, 2)


def sfrl(gap = None):
    """Strong functional representation lemma. 
    Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
    applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978.
    """
    disjoint_id = iutil.get_count()
    SX, SY, SU = rv("SX", "SY", "SU")
    SX.add_markers([("disjoint", disjoint_id), ("nonempty", 1)])
    SY.add_markers([("disjoint", disjoint_id), ("nonempty", 1)])
    #SU.add_markers([("mustuse", 1)])
    
    r = (Expr.Hc(SY, SX + SU) == 0) & (Expr.I(SX, SU) == 0)
    if gap is not None:
        if not isinstance(gap, Expr):
            gap = Expr.const(gap)
        r &= Expr.Ic(SX, SU, SY) <= gap
    return r.exists(SU).forall(SX + SY)


def copylem(n = 2, m = 1):
    """Copy lemma: for any X, Y, there exists Z such that (X, Y) has the same
    distribution as (X, Z), and Y-X-Z forms a Markov chain.
    n, m are the dimensions of X, Y respectively.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    Randall Dougherty, Chris Freiling, and Kenneth Zeger. "Non-Shannon information 
    inequalities in four random variables." arXiv preprint arXiv:1104.3602 (2011).
    """
    disjoint_id = iutil.get_count()
    symm_id_x = iutil.get_count()
    symm_id_y = iutil.get_count()
    
    X = rv_seq("CX", 0, n)
    for i in range(n):
        X[i].add_markers([("disjoint", disjoint_id), ("symm", symm_id_x), ("symm_nonempty", 1)])
    Y = rv_seq("CY", 0, m)
    Z = rv_seq("CZ", 0, m)
    for i in range(m):
        Y[i].add_markers([("disjoint", disjoint_id), ("symm", symm_id_y), ("symm_nonempty", 1)])
    return (eqdist(X + Y, X + Z) & markov(Y, X, Z)).exists(Z).forall(X + Y)


def dblmarkov():
    """Double Markov property: If X-Y-Z and Y-X-Z are Markov chains, then there
    exists W that is a function of X, a function of Y, and (X,Y)-W-Z is Markov chain.
    Imre Csiszar and Janos Korner. Information theory: coding theorems for 
    discrete memoryless systems. Cambridge University Press, 2011.
    """
    symm_id_x = iutil.get_count()
    nonsubset_id_x = iutil.get_count()
    X = Comp.rv("DX")
    Y = Comp.rv("DY")
    Z = Comp.rv("DZ")
    W = Comp.rv("DW")
    X.add_markers([("symm", symm_id_x), ("nonsubset", nonsubset_id_x), ("nonempty", 1)])
    Y.add_markers([("symm", symm_id_x), ("nonsubset", nonsubset_id_x), ("nonempty", 1)])
    Z.add_markers([("nonempty", 1)])
    return ((markov(X, Y, Z) & markov(Y, X, Z))
        >> ((H(W|X) == 0) & (H(W|Y) == 0) & markov(X+Y, W, Z)).exists(W)).forall(X+Y+Z)
    

def mmrv_thm(n = 2):
    """The non-Shannon inequality in the paper:
    Makarychev, K., Makarychev, Y., Romashchenko, A., & Vereshchagin, N. (2002). A new class of 
    non-Shannon-type inequalities for entropies. Communications in Information and Systems, 2(2), 147-166.
    """
    disjoint_id = iutil.get_count()
    symm_id_x = iutil.get_count()
    symm_id_u = iutil.get_count()
    
    X = rv_seq("CX", 0, n)
    for i in range(n):
        X[i].add_markers([("disjoint", disjoint_id), ("symm", symm_id_x), ("symm_nonempty", 2)])
    U = Comp.rv("CU")
    V = Comp.rv("CV")
    Z = Comp.rv("CZ")
    U.add_markers([("nonempty", 1), ("symm", symm_id_u)])
    V.add_markers([("nonempty", 1), ("symm", symm_id_u)])
    Z.add_markers([("nonempty", 1)])
    
    expr = H(X) + n * I(U & V & Z)
    expr -= sum(I(U & V | Xi) for Xi in X)
    expr -= sum(H(Xi) for Xi in X)
    expr -= I(U+V & Z)
    
    return (expr <= 0).forall(X + U + V + Z)

    

def zydfz_thm():
    """The non-Shannon inequalities in the paper:
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    Randall Dougherty, Christopher Freiling, and Kenneth Zeger. "Six new non-Shannon 
    information inequalities." 2006 IEEE International Symposium on Information Theory. IEEE, 2006.
    """
    disjoint_id = iutil.get_count()
    A = Comp.rv("CA")
    B = Comp.rv("CB")
    C = Comp.rv("CC")
    D = Comp.rv("CD")
    A.add_markers([("disjoint", disjoint_id), ("nonempty", 1)])
    B.add_markers([("disjoint", disjoint_id), ("nonempty", 1)])
    C.add_markers([("disjoint", disjoint_id), ("nonempty", 1)])
    D.add_markers([("disjoint", disjoint_id), ("nonempty", 1)])
    
    r = Region.universe()
    r &= 2*I(C&D) <= I(A&B) + I(A&C+D) + 3*I(C&D|A) + I(C&D|B)                                           # ZY
    r &= 2*I(A&B) <= 3*I(A&B|C) + 3*I(A&C|B) + 3*I(B&C|A) + 2*I(A&D) + 2*I(B&C|D)                        # DFZ1
    r &= 2*I(A&B) <= 4*I(A&B|C) +   I(A&C|B) + 2*I(B&C|A) + 3*I(A&B|D)            + I(B&D|A) + 2*I(C&D)  # DFZ2
    r &= 2*I(A&B) <= 3*I(A&B|C) + 2*I(A&C|B) + 4*I(B&C|A) + 2*I(A&C|D) + I(A&D|C) + 2*I(B&D) + I(C&D|A)  # DFZ3
    r &= 2*I(A&B) <= 5*I(A&B|C) + 3*I(A&C|B) +   I(B&C|A) + 2*I(A&D) + 2*I(B&C|D)                        # DFZ4
    r &= 2*I(A&B) <= 4*I(A&B|C) + 4*I(A&C|B) +   I(B&C|A) + 2*I(A&D) + 3*I(B&C|D) + I(C&D|B)             # DFZ5
    r &= 2*I(A&B) <= 3*I(A&B|C) + 2*I(A&C|B) + 2*I(B&C|A) + 2*I(A&B|D) + I(A&D|B) + I(B&D|A) + 2*I(C&D)  # DFZ6
    
    return r.forall(A+B+C+D)
    


def ainfdiv(n = 2, coeff = None, cadd = None):
    """The approximate infinite divisibility of information.
    C. T. Li, "Infinite Divisibility of Information," 
    arXiv preprint arXiv:2008.06092 (2020).
    """
    X = Comp.rv("X")
    Z = rv_seq("Z", 0, n)
    
    if coeff is None:
        coeff = 1.0 / (1.0 - (1.0 - 1.0 / n) ** n) / n
    else:
        coeff = coeff / n
    
    if cadd is None:
        cadd = 2.43
        
    r = iidseq(*Z) & (H(X | Z) == 0)
    r &= H(Z[0]) <= H(X) * coeff + cadd
    return r.exists(Z).forall(X)
    
def exists_xor():
    """For any X, there exists Z0 uniformly distributed and Z1 = X XOR Z0.
    """
    X = Comp.rv("X")
    Z = rv_seq("Z", 0, 2)
    r = indep(X, Z[0]) & indep(X, Z[1]) & (H(X | Z) == 0)
    return r.exists(Z).forall(X)
    
    
    
def existence(f, numarg = 2, nonempty = False):
    """A region for the existence of the random variable f(X, Y).
    e.g. existence(meet), existence(mss)
    """
    T = rv_seq("T", 0, numarg)
    X = f(*T)
    r = X.varlist[0].reg
    
    if X.get_marker_key("symm_args") is not None: # r.issymmetric(T):
        symm_id_x = iutil.get_count()
        
        for i in range(numarg):
            T[i].add_markers([("symm", symm_id_x)])
    
    if X.get_marker_key("nonsubset_args") is not None:
        nonsubset_id_x = iutil.get_count()
        
        for i in range(numarg):
            T[i].add_markers([("nonsubset", nonsubset_id_x)])
        
    X = f(*T)
    r = X.varlist[0].reg
    
    X = X.copy_noreg()
    if nonempty:
        X.add_markers([("nonempty", 1)])
    r.substitute(X, X)
    
    return r.exists(X).forall(T)
    

def rv_bit():
    """A random variable with entropy 1."""
    U = Comp.rv("BIT")
    return Comp.rv_reg(U, H(U) == 1)
    

def exists_bit(n = 1):
    """There exists a random variable with entropy 1."""
    U = rv_seq("BIT", 0, n)
    return (alland([H(x) == 1 for x in U]) & indep(*U)).exists(U)


@fcn_list_to_list
def emin(*args):
    """Return the minimum of the expressions."""
    R = real(iutil.fcn_name_maker("min", args, pname = "emin", lname = "\\min", fcn_suffix = False))
    R = real(str(R))
    r = universe()
    for x in args:
        r &= R <= x
    return r.maximum(R, allow_reuse = True)

@fcn_list_to_list
def emax(*args):
    """Return the maximum of the expressions."""
    R = real(iutil.fcn_name_maker("max", args, pname = "emax", lname = "\\max", fcn_suffix = False))
    R = real(str(R))
    r = universe()
    for x in args:
        r &= R >= x
    return r.minimum(R, allow_reuse = True)


@fcn_list_to_list
def eabs(x):
    """Absolute value of expression."""
    R = real(iutil.fcn_name_maker("abs", x, fcn_suffix = False))
    return ((R >= x) & (R >= -x)).minimum(R, allow_reuse = True)

@fcn_list_to_list
def meet(*args):
    """Gacs-Korner common part. 
    Peter Gacs and Janos Korner. Common information is far less than mutual information.
    Problems of Control and Information Theory, 2(2):149-162, 1973.
    """
    U = Comp.rv(iutil.fcn_name_maker("meet", args))
    V = Comp.rv("V")
    U.add_markers([("mustuse", 1), ("symm_args", 1), ("nonsubset_args", 1)])
    V.add_markers([("nonempty", 1)])
    r = Region.universe()
    r2 = Region.universe()
    for a in args:
        r &= Expr.Hc(U, a) == 0
        r2 &= (Expr.Hc(V, a) == 0)
    r = r & (Expr.Hc(V, U) == 0).implicated(r2, skip_simplify = True).forall(V)
    ret = Comp.rv_reg(U, r, reg_det = True)
    #ret.add_markers([("symm_args", 1), ("nonsubset_args", 1)])
    return ret


@fcn_list_to_list
def mss(x, y):
    """Minimal sufficient statistic of x about y."""
    
    U = Comp.rv(iutil.fcn_name_maker("mss", [x, y]))
    V = Comp.rv("V")
    U.add_markers([("mustuse", 1), ("nonsubset_args", 1)])
    r = (Expr.Hc(U, x) == 0) & (Expr.Ic(x, y, U) == 0)
    r2 = (Expr.Hc(V, x) == 0) & (Expr.Ic(x, y, V) == 0)
    r = r & (Expr.Hc(U, V) == 0).implicated(r2, skip_simplify = True).forall(V)
    ret = Comp.rv_reg(U, r, reg_det = True)
    #ret.add_markers([("nonsubset_args", 1)])
    return ret


@fcn_list_to_list
def sfrl_rv(x, y, gap = None):
    """Strong functional representation lemma. 
    Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
    applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978.
    """
    U = Comp.rv(iutil.fcn_name_maker("sfrl", [x, y], pname = "sfrl_rv"))
    #U = Comp.rv(y.tostring(add_bracket = True) + "%" + x.tostring(add_bracket = True))
    r = (Expr.Hc(y, x + U) == 0) & (Expr.I(x, U) == 0)
    if gap is not None:
        if not isinstance(gap, Expr):
            gap = Expr.const(gap)
        r &= Expr.Ic(x, U, y) <= gap
    return Comp.rv_reg(U, r, reg_det = False)


def esfrl_rv(x, y, gap = None):
    """Strong functional representation lemma, extended form. 
    Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
    applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978.
    """
    U = Comp.rv(iutil.fcn_name_maker("esfrl", [x, y], pname = "esfrl_rv"))
    K = Comp.rv(iutil.fcn_name_maker("esfrl_K", [x, y], pname = "esfrl_rv_K"))
    r = (Expr.Hc(K, x + U) == 0) & (Expr.Hc(Y, U + K) == 0) & (Expr.I(x, U) == 0)
    if gap is not None:
        if not isinstance(gap, Expr):
            gap = Expr.const(gap)
        r &= Expr.H(K) <= Expr.I(x, y) + gap
    return (Comp.rv_reg(U, r, reg_det = False), Comp.rv_reg(K, r, reg_det = False))


def copylem_rv(x, y):
    """Copy lemma: for any X, Y, there exists Z such that (X, Y) has the same
    distribution as (X, Z), and Y-X-Z forms a Markov chain.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    Randall Dougherty, Chris Freiling, and Kenneth Zeger. "Non-Shannon information 
    inequalities in four random variables." arXiv preprint arXiv:1104.3602 (2011).
    """
    U = Comp.rv(iutil.fcn_name_maker("copy", [x, y], pname = "copylem_rv"))
    r = eqdist(list(x) + [y], list(x) + [U]) & markov(y, sum(x), U)
    return Comp.rv_reg(U, r, reg_det = False)


@fcn_list_to_list
def total_corr(x):
    """Total correlation. 
    Watanabe S (1960). Information theoretical analysis of multivariate correlation, 
    IBM Journal of Research and Development 4, 66-82. 
    e.g. total_corr(X & Y & Z | W)
    """
    if isinstance(x, Comp):
        return Expr.H(x)
    return sum([Expr.Hc(a, x.z) for a in x.x]) - Expr.Hc(sum(x.x), x.z)


@fcn_list_to_list
def dual_total_corr(x):
    """Dual total correlation. 
    Han T. S. (1978). Nonnegative entropy measures of multivariate symmetric 
    correlations, Information and Control 36, 133-156. 
    e.g. dual_total_corr(X & Y & Z | W)
    """
    if isinstance(x, Comp):
        return Expr.H(x)
    r = Expr.Hc(sum(x.x), x.z)
    for i in range(len(x.x)):
        r -= Expr.Hc(x.x[i], sum([x.x[j] for j in range(len(x.x)) if j != i]) + x.z)
    return r



@fcn_list_to_list
def gacs_korner(x):
    """Gacs-Korner common information. 
    Peter Gacs and Janos Korner. Common information is far less than mutual information.
    Problems of Control and Information Theory, 2(2):149-162, 1973. 
    e.g. gacs_korner(X & Y & Z | W)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("K", x, pname = "gacs_korner", cropi = True))
    r = universe()
    for a in x.x:
        r &= Expr.Hc(U, a+x.z) == 0
    r &= R <= Expr.Hc(U, x.z)
    return r.exists(U).maximum(R, allow_reuse = True)


@fcn_list_to_list
def wyner_ci(x):
    """Wyner's common information. 
    A. D. Wyner. The common information of two dependent random variables.
    IEEE Trans. Info. Theory, 21(2):163-179, 1975. 
    e.g. wyner_ci(X & Y & Z | W)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("J", x, pname = "wyner_ci", cropi = True))
    r = indep(*(x.x)).conditioned(U + x.z)
    r &= R >= Expr.Ic(U, sum(x.x), x.z)
    return r.exists(U).minimum(R, allow_reuse = True)


@fcn_list_to_list
def exact_ci(x):
    """Common entropy (one-shot exact common information). 
    G. R. Kumar, C. T. Li, and A. El Gamal. Exact common information. In Information
    Theory (ISIT), 2014 IEEE International Symposium on, 161-165. IEEE, 2014. 
    e.g. exact_ci(X & Y & Z | W)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("G", x, pname = "exact_ci", cropi = True))
    r = indep(*(x.x)).conditioned(U + x.z)
    r &= R >= Expr.Hc(U, x.z)
    return r.exists(U).minimum(R, allow_reuse = True)


@fcn_list_to_list
def H_nec(x):
    """Necessary conditional entropy. 
    Cuff, P. W., Permuter, H. H., & Cover, T. M. (2010). Coordination capacity.
    IEEE Transactions on Information Theory, 56(9), 4181-4206. 
    e.g. H_nec(X + Y | W)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("Hnec", x, pname = "H_nec", lname = "H^\\dagger", cropi = True))
    r = markov(x.z, U, x.x[0]) & (Expr.Hc(U, x.x[0]) == 0)
    r &= R >= Expr.Hc(U, x.z)
    return r.exists(U).minimum(R, allow_reuse = True)


@fcn_list_to_list
def excess_fi(x, y):
    """Excess functional information. 
    Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
    applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978. 
    e.g. excess_fi(X, Y)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("excess_fi", [x, y], pname = "excess_fi", lname = "\\Psi"))
    r = indep(U, x)
    r &= R >= Expr.Hc(y, U) - Expr.I(x, y)
    return r.exists(U).minimum(R, allow_reuse = True)


@fcn_list_to_list
def korner_graph_ent(x, y):
    """Korner graph entropy. 
    J. Korner, "Coding of an information source having ambiguous alphabet and the 
    entropy of graphs," in 6th Prague conference on information theory, 1973, pp. 411-425.
    C. T. Li and A. El Gamal, "Extended Gray-Wyner system with complementary 
    causal side information," IEEE Transactions on Information Theory 64.8 (2017): 5862-5878.
    e.g. korner_graph_ent(X, Y)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("korner_graph_ent", [x, y], lname = "H_K"))
    r = markov(U, x, y) & (Expr.Hc(x, y+U) == 0)
    r &= R >= Expr.I(x, U)
    return r.exists(U).minimum(R, allow_reuse = True)


@fcn_list_to_list
def perfect_privacy(x, y):
    """Perfect privacy rate. 
    A. Makhdoumi, S. Salamatian, N. Fawaz, and M. Medard, "From the information bottleneck 
    to the privacy funnel," in Information Theory Workshop (ITW), 2014 IEEE, Nov 2014, pp. 501-505.
    S. Asoodeh, F. Alajaji, and T. Linder, "Notes on information-theoretic privacy," in Communication, 
    Control, and Computing (Allerton), 2014 52nd Annual Allerton Conference on, Sept 2014, pp. 1272-1278.
    e.g. perfect_privacy(X, Y)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("perfect_privacy", [x, y], lname = "g_0"))
    r = markov(x, y, U) & (Expr.I(x, U) == 0)
    r &= R <= Expr.I(y, U)
    return r.exists(U).maximum(R, allow_reuse = True)


@fcn_list_to_list
def max_interaction_info(x, y):
    """Maximal interaction information.
    C. T. Li and A. El Gamal, "Extended Gray-Wyner system with complementary 
    causal side information," IEEE Transactions on Information Theory 64.8 (2017): 5862-5878.
    e.g. max_interaction_info(X, Y)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("max_interaction_info", [x, y], lname = "G_{NNI}"))
    r = Region.universe()
    r &= R <= Expr.Ic(x, y, U) - Expr.I(x, y)
    return r.exists(U).maximum(R, allow_reuse = True)


@fcn_list_to_list
def asymm_interaction_info(x, y):
    """Asymmetric private interaction information.
    C. T. Li and A. El Gamal, "Extended Gray-Wyner system with complementary 
    causal side information," IEEE Transactions on Information Theory 64.8 (2017): 5862-5878.
    e.g. max_interaction_info(X, Y)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("asymm_interaction_info", [x, y], lname = "G_{PNI}"))
    r = indep(x, U)
    r &= R <= Expr.Ic(x, y, U) - Expr.I(x, y)
    return r.exists(U).maximum(R, allow_reuse = True)


@fcn_list_to_list
def symm_interaction_info(x, y):
    """Symmetric private interaction information.
    C. T. Li and A. El Gamal, "Extended Gray-Wyner system with complementary 
    causal side information," IEEE Transactions on Information Theory 64.8 (2017): 5862-5878.
    e.g. max_interaction_info(X, Y)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("symm_interaction_info", [x, y], lname = "G_{PPI}"))
    r = indep(x, U) & indep(y, U)
    r &= R <= Expr.Ic(x, y, U) - Expr.I(x, y)
    return r.exists(U).maximum(R, allow_reuse = True)


@fcn_list_to_list
def minent_coupling(x, y):
    """Minimum entropy coupling of the distributions p_{Y|X=x}. 
    M. Vidyasagar, "A metric between probability distributions on finite sets of 
    different cardinalities and applications to order reduction," IEEE Transactions 
    on Automatic Control, vol. 57, no. 10, pp. 2464-2477, 2012.
    A. Painsky, S. Rosset, and M. Feder, "Memoryless representation of Markov processes," 
    in 2013 IEEE International Symposium on Information Theory. IEEE, 2013, pp. 2294-298.
    M. Kovacevic, I. Stanojevic, and V. Senk, "On the entropy of couplings," 
    Information and Computation, vol. 242, pp. 369-382, 2015.
    M. Kocaoglu, A. G. Dimakis, S. Vishwanath, and B. Hassibi, "Entropic causal inference," 
    in Thirty-First AAAI Conference on Artificial Intelligence, 2017.
    F. Cicalese, L. Gargano, and U. Vaccaro, "Minimum-entropy couplings and their 
    applications," IEEE Transactions on Information Theory, vol. 65, no. 6, pp. 3436-3451, 2019.
    Cheuk Ting Li, "Efficient Approximate Minimum Entropy Coupling of Multiple 
    Probability Distributions," https://arxiv.org/abs/2006.07955 , 2020.
    e.g. minent_coupling(X, Y)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("MEC", [x, y], pname = "minent_coupling", lname = "H_{couple}"))
    r = indep(U, x) & (Expr.Hc(y, x + U) == 0)
    r &= R >= Expr.H(U)
    return r.exists(U).minimum(R, allow_reuse = True)


@fcn_list_to_list
def mutual_dep(x):
    """Mutual dependence. 
    Csiszar, Imre, and Prakash Narayan. "Secrecy capacities for multiple terminals." 
    IEEE Transactions on Information Theory 50, no. 12 (2004): 3047-3061.
    """
    n = len(x.x)
    if n <= 2:
        return I(x)
    R = real(iutil.fcn_name_maker("MD", x, pname = "mutual_dep", lname = "C_{MD}", cropi = True))
    Hall = Expr.Hc(sum(x.x), x.z)
    r = universe()
    for part in iutil.enum_partition(n):
        if len(part) <= 1:
            continue
        expr = Expr.zero()
        for cell in part:
            xb = Comp.empty()
            for i in range(n):
                if cell & (1 << i) != 0:
                    xb += x.x[i]
            expr += Expr.Hc(xb, x.z)
        r &= R <= (expr - Hall) / (len(part) - 1)
    return r.maximum(R, allow_reuse = True)


@fcn_list_to_list
def intrinsic_mi(x):
    """Intrinsic mutual information. 
    U. Maurer and S. Wolf. "Unconditionally secure key agreement and the intrinsic 
    conditional information." IEEE Transactions on Information Theory 45.2 (1999): 499-514.
    e.g. intrinsic_mi(X & Y | Z)
    """
    U = Comp.rv("U")
    R = real(iutil.fcn_name_maker("IMI", x, pname = "intrinsic_mi", lname = "I_{intrinsic}", cropi = True))
    r = markov(sum(x.x), x.z, U) & (R >= mutual_dep(Term(x.x, U)))
    return r.exists(U).minimum(R, allow_reuse = True)


def mi_rect(xs, ys, z = None, sgn = 1):
    if z is None:
        z = Comp.empty()
    if not isinstance(xs, list):
        xs = [xs]
    if not isinstance(ys, list):
        ys = [ys]
    xs = [x if isinstance(x, tuple) else (x,) for x in xs]
    ys = [y if isinstance(y, tuple) else (y,) for y in ys]
    
    exprs = []
    for px in itertools.product(*[range(len(x)) for x in xs]):
        for py in itertools.product(*[range(len(y)) for y in ys]):
            cx = sum((x[0] for x, p in zip(xs, px) if p or len(x) == 1), Comp.empty())
            cy = sum((y[0] for y, p in zip(ys, py) if p or len(y) == 1), Comp.empty())
            cxm = sum((x[1] for x, p in zip(xs, px) if p), Expr.zero())
            cym = sum((y[1] for y, p in zip(ys, py) if p), Expr.zero())
            if sgn > 0:
                if cx.isempty() or cy.isempty():
                    if cxm.iszero() and cym.iszero():
                        exprs.append(Expr.zero())
                else:
                    exprs.append(Expr.Ic(cx, cy, z) - cxm - cym)
            else:
                if cx.isempty() or cy.isempty():
                    exprs.append(-cxm - cym)
                else:
                    exprs.append(Expr.Ic(cx, cy, z) - cxm - cym)
    
    if len(exprs) == 1:
        return exprs[0]
    if sgn > 0:
        return emax(*exprs)
    else:
        return emin(*exprs)

def mi_rect_max(xs, ys, z = None):
    return mi_rect(xs, ys, z, 1)

def mi_rect_min(xs, ys, z = None):
    return mi_rect(xs, ys, z, -1)
    

def directed_info(x, y, z = None):
    """Directed information. 
    Massey, James. "Causality, feedback and directed information." Proc. Int. 
    Symp. Inf. Theory Applic.(ISITA-90). 1990.
    Parameters can be either Comp or CompArray.
    """
    x = CompArray.arg_convert(x)
    y = CompArray.arg_convert(y)
    if z is None:
        return sum(I(x.past_ns() & y | y.past()))
    else:
        z = CompArray.arg_convert(z)
        return sum(I(x.past_ns() & y | y.past() + z.past_ns()))


def comp_vector(*args):
    return CompArray.make(igen.subset(args, minsize = 1))


def ent_vector(*args):
    """Entropy vector.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    """
    if len(args) == 0:
        return ExprArray.empty()
    return H(comp_vector(*args))


def mi_vector(*args, minsize = 1):
    """Mutual information vector.
    """
    r = ExprArray.empty()
    for xs in igen.subset([1 << x for x in range(len(args))], minsize = minsize):
        r.append(I(alland(args[x] for x in range(len(args)) if xs & (1 << x))))
    return r

def ent_cells(*args, minsize = 1):
    """Cells of the I-measure.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    """
    allrv = sum(args)
    r = ExprArray.empty()
    for xs in igen.subset([1 << x for x in range(len(args))], minsize = minsize):
        r.append(I(alland(args[x] for x in range(len(args)) if xs & (1 << x)) 
                 | sum(args[x] for x in range(len(args)) if not (xs & (1 << x)))))
    return r


def mi_cells(*args, minsize = 2):
    """Cells of the I-measure, excluding conditional entropies.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    """
    return ent_cells(*args, minsize = minsize)


def ent_region(n, real_name = "R", var_name = "X"):
    """Entropy region.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    """
    xs = rv_seq(var_name, 0, n)
    cv = comp_vector(*xs)
    re = real_array(real_name, 1, 1 << n)
    return (re == H(cv)).exists(xs)


def csiszar_sum(*args):
    """Csiszar sum identity.
    """
    r = Region.universe()
    vecs = []
    ones = []
    for a in args:
        if isinstance(a, CompArray) or isinstance(a, list):
            vecs.append(a)
            # ones.append(sum(a))
        else:
            ones.append(a)
    
    for xmask in range(1, 1 << len(vecs)):
        x = sum(vecs[i] for i in range(len(vecs)) if xmask & (1 << i))
        #for ymask in igen.subset_mask((1 << len(vecs)) - 1 - xmask):
        for ymask in range(1, 1 << len(vecs)):
            if ymask == 0:
                continue
            y = sum(vecs[i] for i in range(len(vecs)) if ymask & (1 << i))
            for umask in range(1 << len(ones)):
                u = sum((ones[i] for i in range(len(ones)) if umask & (1 << i)), Comp.empty())
                r &= Expr.Ic(x[2], y[0], y[1]+u) == Expr.Ic(y[1], x[0], x[2]+u)
                
    return r
        


# Numerical functions

@fcn_list_to_list
def elog(x, base = None):
    """Logarithm."""
    loge = PsiOpts.settings["ent_coeff"]
    
    fcnargs = [x]
    if base is not None:
        fcnargs.append(base)
        
    R = Expr.real(iutil.fcn_name_maker("log", fcnargs, pname = "elog", lname = "\\log"))
        
    reg = Region.universe()
    
    def fcncall(x, cbase = None):
        cloge = loge
        if cbase is not None:
            if iutil.istorch(cbase):
                cloge = 1.0 / torch.log(cbase)
            else:
                cloge = 1.0 / numpy.log(cbase)
        
        if iutil.istorch(x) or iutil.istorch(cloge):
            return torch.log(x) * cloge
        else:
            return numpy.log(x) * cloge
    
    return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), reg, 0, fcncall, fcnargs))

@fcn_list_to_list
def renyi(x, order = 2):
    """Renyi entropy.
    Renyi, Alfred (1961). "On measures of information and entropy". Proceedings of the 
    fourth Berkeley Symposium on Mathematics, Statistics and Probability 1960. pp. 547-561.
    """
    ceps = PsiOpts.settings["eps"]
    loge = PsiOpts.settings["ent_coeff"]
    
    if isinstance(order, int):
        order = float(order)
    
    def fcncall(xdist, corder):
        if isinstance(corder, ConcReal):
            corder = corder.x
        if isinstance(corder, int):
            corder = float(corder)
            
        if isinstance(corder, float):
            if abs(order - 1) < ceps:
                return xdist.entropy()
            if abs(order - 0) < ceps:
                r = 0.0
                for a in xdist.items():
                    if a > ceps:
                        r += 1
                return numpy.log(r) * loge
            if numpy.isinf(order):
                if xdist.istorch():
                    return -torch.log(torch.max(torch.flatten(xdist.p))) * loge
                else:
                    return -numpy.log(max(xdist.items())) * loge
        
        if xdist.istorch():
            return (torch.log(torch.sum(torch.pow(torch.flatten(xdist.p), corder)))
                    / (1.0 - corder) * loge)
        else:
            return (numpy.log(sum(numpy.power(a, corder) for a in xdist.items()))
                    / (1.0 - corder) * loge)
            
        
           
    if isinstance(x, ConcDist):
        return fcncall(x, order)
        
    R = Expr.real(iutil.fcn_name_maker("renyi", [x, order], pname = "renyi"))
    reg = Region.universe()
    
    if isinstance(order, float):
        if abs(order - 1) < ceps:
            return H(x)
        if order > 1:
            reg = (R <= H(x)) & (R >= 0)
        else:
            reg = R >= H(x)
    
    return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), reg, 0, fcncall, [x, order]))


@fcn_list_to_list
def maxcorr(x):
    """Maximal correlation.
    H. O. Hirschfeld, "A connection between correlation and contingency," in Mathematical 
    Proceedings of the Cambridge Philosophical Society, vol. 31, no. 04. Cambridge Univ Press, 1935, pp. 520-524.
    H. Gebelein, "Das statistische problem der korrelation als variations-und eigenwertproblem 
    und sein zusammenhang mit der ausgleichsrechnung," ZAMM-Journal of Applied Mathematics and 
    Mechanics/Zeitschrift fur Angewandte Mathematik und Mechanik, vol. 21, no. 6, pp. 364-379, 1941.
    A. Renyi, "On measures of dependence," Acta mathematica hungarica, vol. 10, no. 3, 
    pp. 441-451, 1959.
    """
    ceps = PsiOpts.settings["eps"]
    ceps_d = PsiOpts.settings["opt_eps_denom"]
    
    def fcncall(xdist):
        xdist = xdist.flattened_sublen()
        if len(xdist.p.shape) != 2:
            raise ValueError("Only maximal correlation between two random variables is supported.")
            return None
        
        if xdist.istorch():
            px = torch.sum(xdist.p, 1)
            # return torch.sum(px)
            py = torch.sum(xdist.p, 0)
            tmat = torch.zeros(xdist.p.shape, dtype=torch.float64)
            for x in range(xdist.p.shape[0]):
                for y in range(xdist.p.shape[1]):
                    rxy = torch.sqrt(px[x] * py[y])
                    tmat[x, y] = xdist.p[x, y] / (rxy + ceps_d) - rxy
            return torch.linalg.norm(tmat, 2)
        else:
            px = numpy.sum(xdist.p, 1)
            py = numpy.sum(xdist.p, 0)
            tmat = numpy.zeros(xdist.p.shape)
            for x in range(xdist.p.shape[0]):
                for y in range(xdist.p.shape[1]):
                    rxy = numpy.sqrt(px[x] * py[y])
                    if rxy > ceps:
                        tmat[x, y] = xdist.p[x, y] / rxy - rxy
            return numpy.linalg.norm(tmat, 2)
            
           
    if isinstance(x, ConcDist):
        return fcncall(x)

    R = Expr.real(iutil.fcn_name_maker("maxcorr", x, pname = "maxcorr"))
    reg = R >= 0
    
    return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), reg, 0, fcncall, [x]))



@fcn_list_to_list
def divergence(x, y, mode = "kl"):
    """
    Divergence between probability distributions.

    Parameters
    ----------
    x : Comp or ConcDist
        The first distribution (if random variable is given, consider its distribution).
    y : Comp or ConcDist
        The second distribution (if random variable is given, consider its distribution).
    mode : str, optional
        Choices are "kl" (Kullback-Leibler divergence or relative entropy), 
        "tv" (total variation distance), "chi2" (Chi-squared divergence),
        "hellinger" (Hellinger distance) and "js" (Jensen-Shannon divergence). 
        The default is "kl".

    Returns
    -------
    Expr, float or torch.Tensor
        The expression of the divergence. If x,y are ConcDist, gives float or torch.Tensor.

    """
    
    
    mode = mode.lower()
    ceps = PsiOpts.settings["eps"]
    ceps_d = PsiOpts.settings["opt_eps_denom"]
    loge = PsiOpts.settings["ent_coeff"]
    
    def fcncall(xdist, ydist):
        r = 0.0
        if mode == "kl":
            for a, b in zip(xdist.items(), ydist.items()):
                r += iutil.xlogxoy(a, b) * loge
                    
        elif mode == "tv":
            for a, b in zip(xdist.items(), ydist.items()):
                r += abs(a - b) * 0.5
                
        elif mode == "chi2":
            if xdist.istorch() or ydist.istorch():
                for a, b in zip(xdist.items(), ydist.items()):
                    r += (a ** 2) / (b + ceps_d)
            else:
                for a, b in zip(xdist.items(), ydist.items()):
                    r += (a ** 2) / b
            r -= 1.0
                    
        elif mode == "hellinger":
            for a, b in zip(xdist.items(), ydist.items()):
                r += (iutil.sqrt(a) - iutil.sqrt(b)) ** 2
            r = iutil.sqrt(r * 0.5)
                    
        elif mode == "js":
            for a, b in zip(xdist.items(), ydist.items()):
                r += iutil.xlogxoy(a, (a + b) * 0.5) * loge * 0.5
                r += iutil.xlogxoy(b, (a + b) * 0.5) * loge * 0.5
                
        return r
           
    if isinstance(x, ConcDist) and isinstance(y, ConcDist):
        return fcncall(x, y)
        
    R = Expr.real(iutil.fcn_name_maker("divergence", [x, y, mode], pname = "divergence"))
    reg = R >= 0
    
    return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), reg, 0, fcncall, [x, y]))



@fcn_list_to_list
def varent(x):
    """Varentropy (variance of self information) and dispersion (variance of information density).
    Kontoyiannis, Ioannis, and Sergio Verdu. "Optimal lossless compression: Source 
    varentropy and dispersion." 2013 IEEE International Symposium on Information Theory. IEEE, 2013.
    Polyanskiy, Yury, H. Vincent Poor, and Sergio Verdu. "Channel coding rate in the finite 
    blocklength regime." IEEE Transactions on Information Theory 56.5 (2010): 2307-2359.
    """
    ceps = PsiOpts.settings["eps"]
    ceps_d = PsiOpts.settings["opt_eps_denom"]
    
    def fcncall(xdist):
        xdist = xdist.flattened_sublen()
        n = len(xdist.p.shape)
        
        pxs = None
        if xdist.istorch():
            pxs = [torch.sum(xdist.p, tuple(j for j in range(n) if j != i)) for i in range(n)]
        else:
            pxs = [numpy.sum(xdist.p, tuple(j for j in range(n) if j != i)) for i in range(n)]
            
        s1 = 0.0
        s2 = 0.0
        for zs in itertools.product(*[range(z) for z in xdist.p.shape]):
            t = 0.0
            if n == 1:
                s1 += -iutil.xlogxoy(xdist.p[zs], 1.0)
                s2 += iutil.xlogxoy2(xdist.p[zs], 1.0)
            else:
                prod = iutil.product(pxs[i][zs[i]] for i in range(n))
                s1 += iutil.xlogxoy(xdist.p[zs], prod)
                s2 += iutil.xlogxoy2(xdist.p[zs], prod)
        
        return s2 - s1 ** 2
           
    if isinstance(x, ConcDist):
        return fcncall(x)

    R = Expr.real(iutil.fcn_name_maker("varent", x, pname = "varent"))
    reg = R >= 0
            
    return Expr.fromterm(Term(R.terms[0][0].x, Comp.empty(), reg, 0, fcncall, [x]))





