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
Version 1.04
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
    STR_STYLE_LATEX_ARRAY = 8
    STR_STYLE_LATEX_FRAC = 16
    
    SFRL_LEVEL_SINGLE = 1
    SFRL_LEVEL_MULTIPLE = 2
    
    settings = {
        "eps": 1e-10,
        "eps_lp": 1e-5,
        "max_denom": 1000000,
        "max_denom_mul": 10000,
        "str_style": 1,
        "str_lhsreal": True,
        "rename_char": "_",
        
        "truth": None,
        "proof": None,
        
        "solver": "scipy",
        "lptype": LinearProgType.HC1BN,
        "lp_bounded": False,
        "lp_ubound": 1e4,
        "lp_eps": 1e-3,
        "lp_eps_obj": 1e-4,
        "lp_zero_cutoff": -1e-5,
        "fcn_mode": 1,
        "solver_scipy_maxsize": -1,
        
        "imp_noncircular": True,
        "imp_noncircular_allaux": False,
        "imp_simplify": False,
        "tensorize_simplify": False,
        "eliminate_rays": False,
        "ignore_must": False,
        "forall_multiuse": True,
        "forall_multiuse_numsave": 128,
        "auxsearch_local": True,
        "auxsearch_leaveone": False,
        "init_leaveone": True,
        "auxsearch_max_iter": 0,
        "auxsearch_op_casesteplimit": 16,
        "auxsearch_op_caselimit": 512,
        
        "proof_enabled": False,
        "proof_nowrite": False,
        "proof_step_dualsum": True,
        "proof_step_simplify": True,
        
        "repr_simplify": True,
        "repr_check": False,
        
        "discover_hull_frac_enabled": True,
        "discover_hull_frac_denom": 1000,
        
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
        "verbose_eliminate_toreal": False,
        "verbose_semigraphoid": False,
        "verbose_proof": False,
        "verbose_discover": False,
        "verbose_discover_detail": False,
        "verbose_discover_outer": False,
        "verbose_discover_terms": False,
        "verbose_discover_terms_inner": False,
        "verbose_discover_terms_outer": False,
        
        "sfrl_level": 0,
        "sfrl_maxsize": 1,
        "sfrl_gap": ""
    }
    
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
            
        else:
            if key not in d:
                raise KeyError("Option '" + str(key) + "' not found.")
            d[key] = value
    
    def set_setting(**kwargs):
        for key, value in kwargs.items():
            PsiOpts.set_setting_dict(PsiOpts.settings, key, value)
    
    def setting(**kwargs):
        PsiOpts.set_setting(**kwargs)
    
    def get_setting(key, defaultval = None):
        if key in PsiOpts.settings:
            return PsiOpts.settings[key]
        return defaultval
    
    def get_proof():
        return PsiOpts.settings["proof"]
    
    def get_truth():
        return PsiOpts.settings["truth"]
            
    def __init__(self, **kwargs):
        self.cur_settings = PsiOpts.settings.copy()
        for key, value in kwargs.items():
            PsiOpts.set_setting_dict(self.cur_settings, key, value)
    
    def __enter__(self):
        PsiOpts.settings, self.cur_settings = self.cur_settings, PsiOpts.settings
        return PsiOpts.settings
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        PsiOpts.settings, self.cur_settings = self.cur_settings, PsiOpts.settings
    
    
class iutil:
    """Common utilities
    """
    
    
    solver_list = ["pulp.glpk", "pyomo.glpk", "pulp.cbc", "scipy"]
    pulp_solver = None
    pulp_solvers = {}
    
    cur_count = 0
    
    def float_tostr(x, style = 0):
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
                    return "(" + str(frac) + ")"
            else:
                if style & PsiOpts.STR_STYLE_LATEX_FRAC:
                    return "-\\frac{" + str(frac.numerator) + "}{" + str(frac.denominator) + "}"
                else:
                    return "-(" + str(frac) + ")"

    def float_snap(x):
        t = float(fractions.Fraction(x).limit_denominator(PsiOpts.settings["max_denom"]))
        if abs(x - t) <= PsiOpts.settings["eps"]:
            return t
        return x

    def get_count():
        iutil.cur_count += 1
        return iutil.cur_count
        
    def gcd(a, b):
        while b > 0:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return (a // iutil.gcd(a, b)) * b
    
    def convert_str_style(style):
        if style == "standard" or style == "std":
            return PsiOpts.STR_STYLE_STANDARD
        elif style == "psitip" or style == "code":
            return PsiOpts.STR_STYLE_PSITIP
        elif style == "latex":
            return PsiOpts.STR_STYLE_LATEX | PsiOpts.STR_STYLE_LATEX_ARRAY | PsiOpts.STR_STYLE_LATEX_FRAC
        elif style == "latex_noarray":
            return PsiOpts.STR_STYLE_LATEX | PsiOpts.STR_STYLE_LATEX_FRAC
        else:
            return style
        
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
    
    def str_python_multiline(s):
        s = str(s)
        return "(\"" + "\\n\"\n\"".join(s.split("\n")) + "\")"
    
    def hash_short(s):
        s = str(s)
        return hash(s) % 99991
        
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
    
    def pulp_get_solver(solver):
        copt = solver[solver.index(".") + 1 :].upper()
        if copt == "OTHER":
            return iutil.pulp_solver
        
        if copt in iutil.pulp_solvers:
            return iutil.pulp_solvers[copt]
        
        r = None
        if copt == "GLPK":
            #r = pulp.solvers.GLPK(msg = 0)
            r = pulp.GLPK(msg = 0)
        elif copt == "CBC" or copt == "PULP_CBC_CMD":
            #r = pulp.solvers.PULP_CBC_CMD()
            r = pulp.PULP_CBC_CMD()
        
        iutil.pulp_solvers[copt] = r
        return r
    
    def bitcount(x):
        r = 0
        while x != 0:
            x &= x - 1
            r += 1
        return r
    
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
    
    def list_tostr_std(x):
        return iutil.list_tostr(x, tuple_delim = ": ", list_delim = "; ")

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
            
    def iscyclic(x):
        return len(iutil.tsort(x)) < len(x)
        
    def signal_type(x):
        if isinstance(x, tuple) and len(x) > 0 and isinstance(x[0], str):
            return x[0]
        return ""
    
    def mhash(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return hash(tuple(iutil.mhash(y) for y in x))
        return hash(x)
    
    def list_unique(x):
        r = []
        s = set()
        for a in x:
            h = iutil.mhash(a)
            if h not in s:
                s.add(h)
                r.append(a)
        return r
    
    def list_sorted_unique(x):
        x = sorted(x, key = lambda a: iutil.mhash(a))
        return [x[i] for i in range(len(x)) if i == 0 or not x[i] == x[i - 1]]
    
    def sumlist(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return sum(iutil.sumlist(a) for a in x)
        return x
    
    def set_suffix_num(s, k, schar, replace_mode = "set"):
        t = s.split("@@")
        if len(t) >= 2:
            for i in range(0, len(t), 2):
                t[i] = iutil.set_suffix_num(t[i], k, schar, replace_mode)
            return "@@".join(t)
            
        if replace_mode != "suffix":
            i = s.rfind(schar)
            if i >= 0 and s[i + 1 :].isdigit():
                if replace_mode == "add":
                    return s[:i] + schar + str(int(s[i + 1 :]) + k)
                else:
                    return s[:i] + schar + str(k)
        return s + schar + str(k)
    
    def get_name_fromcat(s, style):
        t = s.split("@@")
        for i in range(1, len(t) - 1, 2):
            if int(t[i]) & style:
                return t[i + 1]
        return t[0]
    
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
    
    def fcn_name_maker(name, v, pname = None, lname = None, cropi = False):
        if not isinstance(v, list) and not isinstance(v, tuple):
            v = [v]
        r = ""
        for style in [PsiOpts.STR_STYLE_STANDARD, PsiOpts.STR_STYLE_PSITIP, PsiOpts.STR_STYLE_LATEX]:
            if style != PsiOpts.STR_STYLE_STANDARD:
                r += "@@" + str(style) + "@@"
            if style == PsiOpts.STR_STYLE_STANDARD:
                r += name
            elif style == PsiOpts.STR_STYLE_PSITIP:
                r += pname or name
            elif style == PsiOpts.STR_STYLE_LATEX:
                r += lname or name
                
            if not cropi:
                r += "("
            for i, a in enumerate(v):
                if i:
                    r += ","
                t = ""
                if style == PsiOpts.STR_STYLE_STANDARD and isinstance(a, Comp):
                    t = a.tostring(style = style, add_braket = True)
                else:
                    t = a.tostring(style = style)
                if cropi and len(t) >= 1:
                    r += t[1:]
                else:
                    r += t
            if not cropi:
                r += ")"
        return r
    
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
    
    
    
def fcn_list_to_list(fcn):
    
    @functools.wraps(fcn)
    def wrapper(*args, **kwargs):
        islist = False
        maxlen = 0
        for a in itertools.chain(args, kwargs.values()):
            if isinstance(a, CompList) or isinstance(a, ExprList):
                islist = True
                maxlen = max(maxlen, len(a))
        
        if not islist:
            return fcn(*args, **kwargs)
        
        r = []
        for i in range(maxlen):
            targs = []
            for a in args:
                if isinstance(a, CompList):
                    if i < len(a):
                        targs.append(a[i])
                    else:
                        targs.append(Comp.empty())
                elif isinstance(a, ExprList):
                    if i < len(a):
                        targs.append(a[i])
                    else:
                        targs.append(Expr.zero())
                else:
                    targs.append(a)
                    
            tkwargs = dict()
            for key, a in kwargs.items():
                if isinstance(a, CompList):
                    if i < len(a):
                        tkwargs[key] = a[i]
                    else:
                        tkwargs[key] = Comp.empty()
                elif isinstance(a, ExprList):
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
                return CompList(r)
            elif isinstance(r[0], Expr):
                return ExprList(r)
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
    
class IVar:
    """Random variable or real variable
    Do NOT use this class directly. Use Comp instead
    """
    
    def __init__(self, vartype, name, reg = None, reg_det = False, markers = None):
        self.vartype = vartype
        self.name = name
        self.reg = reg
        self.reg_det = reg_det
        self.markers = markers
        
    def rv(name):
        return IVar(IVarType.RV, name)
        
    def real(name):
        return IVar(IVarType.REAL, name)
        
    def eps():
        return IVar(IVarType.REAL, "EPS")
        
    def one():
        return IVar(IVarType.REAL, "ONE")
        
    def inf():
        return IVar(IVarType.REAL, "INF")
    
    def isrealvar(self):
        return self.vartype == IVarType.REAL and self.name != "ONE" and self.name != "EPS" and self.name != "INF"
    
    def tostring(self, style = 0):
        return iutil.get_name_fromcat(self.name, style)
    
    def __str__(self):
        
        return self.tostring(PsiOpts.settings["str_style"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.STR_STYLE_PSITIP)
        
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

    
class Comp:
    """Compound random variable or real variable
    """
    
    def __init__(self, varlist):
        self.varlist = varlist
        
    def empty():
        """
        The empty random variable.

        Returns
        -------
        Comp

        """
        return Comp([])
        
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
        
    def rv_reg(a, reg, reg_det = False):
        r = a.copy_noreg()
        for i in range(len(r.varlist)):
            r.varlist[i].reg = reg.copy()
            r.varlist[i].reg_det = reg_det
        return r
        #return Comp([IVar(IVarType.RV, str(a), reg.copy(), reg_det)])
        
    def real(name):
        return Comp([IVar(IVarType.REAL, name)])
    
    def array(name, st, en):
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
    
    
    def get_type(self):
        if len(self.varlist) == 0:
            return IVarType.NIL
        return self.varlist[0].vartype
        
    def allcomp(self):
        return self.copy()
    
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
                for v, w in a.markers:
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
        if isinstance(other, CompList) or isinstance(other, ExprList):
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
        return CompList([a.copy() for a in self])
    
    def tostring(self, style = 0, tosort = False, add_braket = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        style = iutil.convert_str_style(style)
        
        namelist = [a.tostring(style) for a in self.varlist]
        if len(namelist) == 0:
            if style == PsiOpts.STR_STYLE_PSITIP:
                return "rv_empty()"
            elif style & PsiOpts.STR_STYLE_LATEX:
                return "\\emptyset"
            return "!"
            
        if tosort:
            namelist.sort()
        r = ""
        if add_braket and len(namelist) > 1:
            r += "("
            
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += "+".join(namelist)
        else:
            r += ",".join(namelist)
            
        if add_braket and len(namelist) > 1:
            r += ")"
        
        return r
    
    
    def __str__(self):
        
        return self.tostring(PsiOpts.settings["str_style"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.STR_STYLE_PSITIP)
        
    def __hash__(self):
        #return hash(self.tostring(tosort = True))
        return hash(frozenset(a.name for a in self.varlist))
    
    
        
    def isregtermpresent(self):
        for b in self.varlist:
            if b.reg is not None:
                return True
        return False
        
    def get_type(self):
        if len(self.varlist) == 0:
            return IVarType.NIL
        return self.varlist[0].vartype
    
    
    def __and__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        return Term.H(self) & Term.H(other)
    
    def __or__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
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
    
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound)"""
        r = []
        for a in self.varlist:
            if v0.ispresent_shallow(a):
                r += [b.copy() for b in v1.varlist]
            else:
                r.append(a)
        self.varlist = r
        for a in self.varlist:
            if a.reg is not None:
                a.reg.substitute(v0, v1)
    
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
        
    
    def record_to(self, index):
        index.record(self)
        for a in self.varlist:
            if a.reg is not None:
                index.record(a.reg.allcomprv_noaux())
        
    def fcn_of(self, b):
        return Expr.Hc(self, b) == 0
    
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
        if x.vartype == IVarType.RV:
            if x.name in self.dictrv:
                return self.dictrv[x.name]
            return -1
        else:
            if x.name in self.dictreal:
                return self.dictreal[x.name]
            return -1
        
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
        

class TermType:
    NIL = 0
    IC = 1
    REAL = 2
    REGION = 3
    
class Term:
    """A term in an expression
    Do NOT use this class directly. Use Expr instead
    """
    
    def __init__(self, x, z, reg = None, sn = 0):
        self.x = x
        self.z = z
        self.reg = reg
        self.sn = sn
        
    def copy(self):
        if self.reg is None:
            return Term([a.copy() for a in self.x], self.z.copy(), None, self.sn)
        else:
            return Term([a.copy() for a in self.x], self.z.copy(), self.reg.copy(), self.sn)
        
    def copy_noreg(self):
        return Term([a.copy_noreg() for a in self.x], self.z.copy_noreg(), None, 0)
        
    def zero():
        return Term([], Comp.empty())
    
    def setzero(self):
        self.x = []
        self.z = Comp.empty()
        self.reg = None
        self.sn = 0
        
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
        
    def size(self):
        r = self.z.size()
        for a in self.x:
            r += a.size()
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
        
    def record_to(self, index):
        if self.get_type() == TermType.REGION:
            index.record(self.reg.allcomprv_noaux())
            
        for a in self.x:
            a.record_to(index)
        self.z.record_to(index)
        
    
    def value(self, method = "", num_iter = 30, prog = None):
        if self.isone():
            return 1.0
        if self.iseps():
            return 0.0
        if self.isinf():
            return float("inf")
        if self.reg is None:
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
        
    
    def tostring(self, style = 0, tosort = False):
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
                r += "H"
            else:
                r += "I"
            r += "("
            
            namelist = [a.tostring(style = style, tosort = tosort) for a in self.x]
            if tosort:
                namelist.sort()
                
            if style == PsiOpts.STR_STYLE_PSITIP:
                r += "&".join(namelist)
            else:
                r += ";".join(namelist)
            if self.z.size() > 0:
                r += "|" + self.z.tostring(style = style, tosort = tosort)
            r += ")"
            return r
        elif termType == TermType.REGION:
            return self.x[0].tostring(style = style, tosort = tosort)
        
        return ""
    
        
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.STR_STYLE_PSITIP)
        
    def __hash__(self):
        #return hash(self.tostring(tosort = True))
        return hash((frozenset(hash(a) for a in self.x), hash(self.z)))
    
    def simplify(self):
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
        return self
    
    def simplified(self):
        r = self.copy()
        r.simplify()
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
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        if isinstance(other, Comp):
            other = Term.H(other)
        return Term([a.copy() for a in self.x] + [a.copy() for a in other.x], self.z + other.z)
        
    
    def __or__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        if isinstance(other, Comp):
            other = Term.H(other)
        if isinstance(other, int):
            return self.copy()
        return Term([a.copy() for a in self.x], self.z + other.allcomp())
        
        
            
    def ispresent(self, x):
        """Return whether any variable in x appears here"""
        
        if isinstance(x, IVar):
            x = Comp([x])
        if not isinstance(x, str) and not isinstance(x, Comp):
            x = x.allcomp()
        
        if self.get_type() == TermType.REGION:
            if self.reg.ispresent(x):
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
            self.reg.rename_var(name0, name1)
        for a in self.x:
            a.rename_var(name0, name1)
        self.z.rename_var(name0, name1)
            
    def rename_map(self, namemap):
        """Rename according to name map
        """
        if self.get_type() == TermType.REGION:
            self.reg.rename_map(namemap)
        for a in self.x:
            a.rename_map(namemap)
        self.z.rename_map(namemap)
        return self
    
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound)"""
        if self.get_type() == TermType.REGION:
            self.reg.substitute(v0, v1)
        for a in self.x:
            a.substitute(v0, v1)
        self.z.substitute(v0, v1)
        
    
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
    
    
    
class Expr:
    """An expression
    """
    
    def __init__(self, terms, mhash = None):
        self.terms = terms
        self.mhash = mhash
    
        
    def copy(self):
        return Expr([(a.copy(), c) for (a, c) in self.terms], self.mhash)
        
        
    def copy_noreg(self):
        return Expr([(a.copy_noreg(), c) for (a, c) in self.terms], None)
    
    
    def fromcomp(x):
        return Expr([(Term.fromcomp(x), 1.0)])
    
    def fromterm(x):
        return Expr([(x.copy(), 1.0)])
    
    def get_const(self):
        r = 0.0
        for (a, c) in self.terms:
            if a.isone():
                r += c
            else:
                return None
        return r
        
    def __len__(self):
        return len(self.terms)
    
    def __getitem__(self, key):
        r = self.terms[key]
        if isinstance(r, list):
            return Expr(r)
        return Expr([r])
        
    def __iadd__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if not isinstance(other, Expr):
            other = Expr.const(other)
        self.terms += [(a.copy(), c) for (a, c) in other.terms]
        self.mhash = None
        return self
        
    def __add__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
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
        if isinstance(other, CompList) or isinstance(other, ExprList):
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
            other = other.get_const()
            if other is None:
                raise ValueError("Multiplication with non-constant expression is not supported.")
        self.terms = [(a, c * other) for (a, c) in self.terms]
        self.mhash = None
        return self
        
    def __mul__(self, other):
        if isinstance(other, Expr):
            other = other.get_const()
            if other is None:
                raise ValueError("Multiplication with non-constant expression is not supported.")
        return Expr([(a.copy(), c * other) for (a, c) in self.terms])
        
    def __rmul__(self, other):
        if isinstance(other, Expr):
            other = other.get_const()
            if other is None:
                raise ValueError("Multiplication with non-constant expression is not supported.")
        return Expr([(a.copy(), c * other) for (a, c) in self.terms])
        
    def __itruediv__(self, other):
        if isinstance(other, Expr):
            other = other.get_const()
            if other is None:
                raise ValueError("Division with non-constant expression is not supported.")
        self.terms = [(a, c / other) for (a, c) in self.terms]
        self.mhash = None
        return self
        
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
    
    def zero():
        """The constant zero expression."""
        return Expr([])
    
    def H(x):
        """Entropy."""
        return Expr([(Term.H(x), 1.0)])
        
    def I(x, y):
        """Mutual information."""
        return Expr([(Term.I(x, y), 1.0)])
        
    def Hc(x, z):
        """Conditional entropy."""
        return Expr([(Term.Hc(x, z), 1.0)])
        
    def Ic(x, y, z):
        """Conditional mutual information."""
        return Expr([(Term.Ic(x, y, z), 1.0)])
        
        
    def real(name):
        """Real variable."""
        return Expr([(Term([Comp.real(name)], Comp.empty()), 1.0)])
    
        
    def eps():
        """Epsilon."""
        return Expr([(Term([Comp([IVar.eps()])], Comp.empty()), 1.0)])
        
    def one():
        """One."""
        return Expr([(Term([Comp([IVar.one()])], Comp.empty()), 1.0)])
        
    def inf():
        """Infinity."""
        return Expr([(Term([Comp([IVar.inf()])], Comp.empty()), 1.0)])
        
    def const(c):
        """Constant."""
        if abs(c) <= PsiOpts.settings["eps"]:
            return Expr.zero()
        return Expr([(Term([Comp([IVar.one()])], Comp.empty()), float(c))])
    
    def tostring(self, style = 0, tosort = False):
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
        
        r = ""
        first = True
        for (a, c) in termlist:
            if abs(c) <= PsiOpts.settings["eps"]:
                continue
            if c > 0.0 and not first:
                r += "+"
            if a.isone():
                r += iutil.float_tostr(c)
            else:
                if abs(c - 1.0) < PsiOpts.settings["eps"]:
                    pass
                elif abs(c + 1.0) < PsiOpts.settings["eps"]:
                    r += "-"
                else:
                    r += iutil.float_tostr(c, style)
                    if style == PsiOpts.STR_STYLE_PSITIP:
                        r += "*"
                r += a.tostring(style = style, tosort = tosort)
            first = False
            
        if r == "":
            return "0"
        return r
        
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"])
    
    def __repr__(self):
        if PsiOpts.settings.get("repr_simplify", False):
            return self.simplified().tostring(PsiOpts.STR_STYLE_PSITIP)
        return self.tostring(PsiOpts.STR_STYLE_PSITIP)
        
    def __hash__(self):
        if self.mhash is None:
            #self.mhash = hash(self.tostring(tosort = True))
            self.mhash = hash(tuple(sorted((hash(a), c) for a, c in self.terms)))
            
        return self.mhash
        
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
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([other - self], [], Comp.empty(), Comp.empty(), Comp.empty())
        
    def __lt__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([other - self - Expr.eps()], [], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __ge__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([self - other], [], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __gt__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([self - other - Expr.eps()], [], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __eq__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        if not isinstance(other, Expr):
            other = Expr.const(other)
        return Region([], [self - other], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __ne__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return NotImplemented
        #return RegionOp.union([self > other, self < other])
        return ~RegionOp.inter([self == other])
        
    
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
                if len([(a, c) for (a, c) in self.terms if abs(c * denom - round(c * denom)) > PsiOpts.settings["eps"]]) == 0:
                    self.terms = [(a, iutil.float_snap(c * denom)) for (a, c) in self.terms]
            
    
    def simplify(self, reg = None):
        """Simplify the expression in place"""
        
        self.mhash = None
        
        for (a, c) in self.terms:
            a.simplify()
            
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
            
            self.terms = [(a, c) for (a, c) in self.terms if abs(c) > PsiOpts.settings["eps"] and not a.iszero()]
            
            for i in range(len(self.terms)):
                if abs(self.terms[i][1]) > PsiOpts.settings["eps"]:
                    for j in range(len(self.terms)):
                        if i != j and abs(self.terms[j][1]) > PsiOpts.settings["eps"]:
                            if abs(self.terms[i][1] - self.terms[j][1]) <= PsiOpts.settings["eps"]:
                                if self.terms[i][0].try_iadd(self.terms[j][0]):
                                    self.terms[j] = (self.terms[j][0], 0.0)
                                    did = True
                            elif abs(self.terms[i][1] + self.terms[j][1]) <= PsiOpts.settings["eps"]:
                                if self.terms[i][0].try_isub(self.terms[j][0]):
                                    self.terms[j] = (self.terms[j][0], 0.0)
                                    did = True
            
            self.terms = [(a, c) for (a, c) in self.terms if abs(c) > PsiOpts.settings["eps"] and not a.iszero()]
        
        #self.terms = [(a, iutil.float_snap(c)) for (a, c) in self.terms 
        #              if abs(c) > PsiOpts.settings["eps"] and not a.iszero()]
            
        return self

    
    def simplified(self, reg = None):
        """Simplify the expression, return simplified expression"""
        r = self.copy()
        r.simplify(reg)
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
                return self.get_ratio(other)
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
        else:
            for (a, c) in self.terms:
                a.substitute(v0, v1)
        return self

    def substituted(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), return result"""
        r = self.copy()
        r.substitute(v0, v1)
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
                lhs.append((a, c))
            else:
                rhs.append((a, c))
        return (Expr(lhs), Expr(rhs))
        
    
    def tostring_eqn(self, eqnstr, style = 0, tosort = False, lhsvar = None):
        style = iutil.convert_str_style(style)
        lhs = self
        rhs = Expr.zero()
        if lhsvar is not None:
            lhs, rhs = self.split_lhs(lhsvar)
            
            if lhs.iszero() or rhs.iszero():
                lhs, rhs = self.split_posneg()
                if lhs.iszero():
                    lhs, rhs = rhs, lhs
                elif lhs.get_const() is not None:
                    lhs, rhs = rhs, lhs
                    rhs *= -1.0
                else:
                    rhs *= -1.0
                
            else:
                rhs *= -1.0
                
            if lhs.coeff_sum() < 0 or (abs(lhs.coeff_sum()) <= PsiOpts.settings["eps"] and rhs.coeff_sum() < 0):
                lhs *= -1.0
                rhs *= -1.0
                eqnstr = iutil.reverse_eqnstr(eqnstr)
        
        return (lhs.tostring(style = style, tosort = tosort) + " "
        + iutil.eqnstr_style(eqnstr, style) + " " + rhs.tostring(style = style, tosort = tosort))
        
        
class BayesNet:
    """Bayesian network"""
    
    def __init__(self):
        self.index = IVarIndex()
        self.parent = []
        self.child = []
        self.fcn = []
        
    def add_edge_id(self, i, j):
        if i < 0 or j < 0 or i == j:
            return
        self.parent[j].append(i)
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
    
    def tsorted(self):
        n = self.index.comprv.size()
        cstack = []
        cnparent = [0] * n
        
        r = BayesNet()
        
        for i in range(n - 1, -1, -1):
            cnparent[i] = len(self.parent[i])
            if cnparent[i] == 0:
                cstack.append(i)
                
        while len(cstack) > 0:
            i = cstack.pop()
            r.record(self.index.comprv[i])
            for j in reversed(self.child[i]):
                if cnparent[j] > 0:
                    cnparent[j] -= 1
                    if cnparent[j] == 0:
                        cstack.append(j)
                        
        for i in range(n):
            for j in self.parent[i]:
                r.add_edge(self.index.comprv[j], self.index.comprv[i])
        
        return r
            
    def check_ic_mask(self, x, y, z):
        if x < 0 or y < 0 or z < 0:
            return False
        
        x &= ~z
        y &= ~z
        if x & y != 0:
            return False
        
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
        for (a, c) in icexpr.terms:
            if not a.isic2():
                return False
            if not self.check_ic_mask(self.index.get_mask(a.x[0]), 
                                      self.index.get_mask(a.x[1]), 
                                      self.index.get_mask(a.z)):
                return False
        return True
    
    def from_ic_inplace(self, icexpr):
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
        
        xk = 0
        zk = 0
        vis = 0
        
        np2 = (1 << n)
        dp = [1000000] * np2
        dpi = [-1] * np2
        dped = [-1] * np2
        dp[np2 - 1] = 0
        for vis in range(np2 - 2, -1, -1):
            nvis = iutil.bitcount(vis)
            for i in range(n):
                if vis & (1 << i) == 0:
                    nedge = nvis + dp[vis | (1 << i)]
                    if nedge < dp[vis]:
                        dp[vis] = nedge
                        dpi[vis] = i
                        dped[vis] = vis
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
                        nedge = iutil.bitcount(zk) + dp[vis | (1 << i)]
                        if nedge < dp[vis]:
                            dp[vis] = nedge
                            dpi[vis] = i
                            dped[vis] = zk
        
        #for vis in range(np2):
        #    print("{0:b}".format(vis) + " " + str(dp[vis]) + " " + str(dpi[vis]) + " " + "{0:b}".format(dped[vis]))
        cvis = 0
        for it in range(n):
            i = dpi[cvis]
            ed = dped[cvis]
            for j in range(n):
                if ed & (1 << j) != 0:
                    self.add_edge_id(j, i)
            cvis |= (1 << i)
            
    
    def from_ic(icexpr):
        r = BayesNet()
        r.from_ic_inplace(icexpr)
        return r
    
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
            return self.tsorted().tostring(tsort = False)
        
        n = self.index.comprv.size()
        r = ""
        for i in range(n):
            first = True
            for j in self.parent[i]:
                if not first:
                    r += ","
                r += self.index.comprv.varlist[j].tostring()
                first = False
            r += " -> " + self.index.comprv.varlist[i].tostring() + "\n"
        return r
        
    def __str__(self):
        return self.tostring()
    
    def __repr__(self):
        return self.tostring()
        
    def __hash__(self):
        return hash(self.tostring())
        
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
    
    def write_pf(self, x):
        r = self.get_dual_region()
        if r is None:
            return
        xstr = str(x) + " >= 0"
        pf = ProofObj.from_region(r, c = "Duals for " + xstr)
        
        if PsiOpts.settings["proof_step_dualsum"]:
            r2 = Region.universe()
            cur = Expr.zero()
            for x in r.exprs_ge:
                cur = (cur + x).simplified()
                r2.exprs_ge.append(cur)
            for x in r.exprs_eq:
                cur = (cur + x).simplified()
                r2.exprs_ge.append(cur)
            pf += ProofObj.from_region(r2, c = "Steps for " + xstr)
            
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
                return (res.fun, [res.x[i] for i in range(self.nxvar)])
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
                res = prob.solve(iutil.pulp_get_solver(self.solver))
                
                if pulp.LpStatus[res] == "Optimal":
                    
                    return (prob.objective.value(), [xvar[str(i)].value() for i in range(self.nxvar)])
                
            return (None, None)
            
        
        elif self.solver.startswith("pyomo."):
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
                res = opt.solve(model)

            if (res.solver.status == pyo.SolverStatus.ok 
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
                self.optval = res.fun + c1
                if optval is not None:
                    optval.append(self.optval)
                    
                if self.val_enabled:
                    self.val_x = [0.0] * self.nxvar
                    for i in range(self.nxvar):
                        self.val_x[i] = res.x[i]
                        
                if self.optval >= zero_cutoff:
                    return True
            
            if res.status == 0 and self.save_res:
                self.saved_var.append(array.array("d", list(res.x)))
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
                res = prob.solve(iutil.pulp_get_solver(self.solver))
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
                res = opt.solve(model)

            if verbose:
                print("  status=" + ("OK" if res.solver.status == pyo.SolverStatus.ok else "NO"))
            
            if self.affine_present and self.lp_bounded and res.solver.termination_condition == pyo.TerminationCondition.infeasible:
                return True
            
            #print("save_res = " + str(self.save_res))
            if (res.solver.status == pyo.SolverStatus.ok 
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
                for sgn in ([1, -1] if i in lset else [1]):
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
    
    
        
        
    
class RegionType:
    NIL = 0
    NORMAL = 1
    UNION = 2
    INTER = 3
    
class Region:
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
        
    def substituted(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), return result"""
        r = self.copy()
        r.substitute(v0, v1)
        return r

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
        
    def substituted_aux(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), and remove auxiliary v0, return result"""
        r = self.copy()
        r.substitute_aux(v0, v1)
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
        
        if isinstance(other, RegionOp) or self.imp_present() or other.imp_present() or self.aux_present() or other.aux_present():
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
        return RegionOp.union([self]) | other
        
    def __ror__(self, other):
        return RegionOp.union([self]) | other
        
    def __ior__(self, other):
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
        return RegionOp.union([~self.implicated(other), ~other.implicated(self)])
        
    
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
            t = real_list("#TMPVAR", 0, len(w))
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
                self.exprs_ge.append(x.copy())
                self.exprs_ge.append(-x)
                x.setzero()
                sn_present[0] = True
                sn_present[1] = True
                
        for x in self.exprs_eqi:
            if x.ispresent(v0):
                self.exprs_gei.append(x.copy())
                self.exprs_gei.append(-x)
                x.setzero()
                sn_present[0] = True
                sn_present[1] = True
        
        for x in self.exprs_ge:
            if x.ispresent(v0):
                c = x.get_coeff(v0term)
                if c > 0.0:
                    x.substitute(v0, v1s[1])
                    sn_present[1] = True
                else:
                    x.substitute(v0, v1s[0])
                    sn_present[0] = True
                
        for x in self.exprs_gei:
            if x.ispresent(v0):
                c = x.get_coeff(v0term)
                if c > 0.0:
                    x.substitute(v0, v1s[0])
                    sn_present[0] = True
                else:
                    x.substitute(v0, v1s[1])
                    sn_present[1] = True
             
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        self.exprs_eqi = [x for x in self.exprs_eqi if not x.iszero()]
        
        return sn_present
    
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
    
    def flattened(self, *args):
        if not self.isregtermpresent() and len(args) == 0:
            return self.copy()
        
        cs = None
        if not isinstance(cs, RegionOp):
            cs = RegionOp.inter([self])
        else:
            cs = self.copy()
        cs.flatten()
        for x in args:
            cs.flatten_term(x)
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
    
    def and_cause_consequence(self, other):
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
            for rr in tcs.check_getaux_gen():
                if iutil.signal_type(rr) == "":
                    #print(rr)
                    tcons = cons.copy()
                    Comp.substitute_list(tcons, rr)
                    cs &= tcons
                    
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
    
    def optimum(self, v, sn):
        """Return the variable obtained from maximizing (sn=1)
        or minimizing (sn=-1) the expression v
        """
        if v.size() == 1 and v.terms[0][0].get_type() == TermType.REAL:
            return Expr.fromterm(Term(v.terms[0][0].copy().x, Comp.empty(), self, sn))
        
        tmpstr = ""
        if sn > 0:
            tmpstr = "max"
        else:
            tmpstr = "min"
        tmpstr += str(iutil.hash_short(self))
        tmpvar = Expr.real(tmpstr + "(" + str(v) + ")")
        cs = self.copy()
        if sn > 0:
            cs.exprs_ge.append(v - tmpvar)
        else:
            cs.exprs_ge.append(tmpvar - v)
        return cs.optimum(tmpvar, sn)
    
    def maximum(self, v):
        """Return the variable obtained from maximizing the expression v
        """
        return self.optimum(v, 1)
    
    def minimum(self, v):
        """Return the variable obtained from minimizing the expression v
        """
        return self.optimum(v, -1)
    
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
            if (not self.implies_ineq_prog(index, progs, x, ">=", save_res = True, saved = True)
            or not self.implies_ineq_prog(index, progs, x, ">=", save_res = True, saved = False)):
                if verbose_subset:
                    print(str(x) + " >= 0 FAIL")
                return False
        for x in other.exprs_eq:
            if (not self.implies_ineq_prog(index, progs, x, "==", save_res = True, saved = True)
            or not self.implies_ineq_prog(index, progs, x, "==", save_res = True, saved = False)):
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
                        if maxcost > 0 and curcost[0] >= maxcost:
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
        
        if n_cond == 0:
            for (x, sg) in eqvs[eqvs_emptyid]:
                if not cs.implies_ineq_prog(index, progs, x, sg, save_res = save_res):
                    if leaveone and sg == ">=":
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
        
        def clear_cache(mini, maxi):
            if verbose_cache:
                print("========= cache clear: " + str(mini) + " - " + str(maxi) + " =========")
            progs[:] = []
            auxcache[mini:maxi] = [{} for i in range(mini, maxi)]
            for a in flagcache:
                a.clear()
            for i in range(mini, maxi):
                auxavoidflag[i] = 0
                
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
                    if len(clist) > forall_multiuse_numsave:
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
                            if len(condflagadded_true) > forall_multiuse_numsave:
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
                                if maxcost > 0 and curcost[0] >= maxcost:
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
                            if maxcost > 0 and curcost[0] >= maxcost:
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
                                    if maxcost > 0 and curcost[0] >= maxcost:
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
                            if maxcost > 0 and curcost[0] >= maxcost:
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
                            if maxcost > 0 and curcost[0] >= maxcost:
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
                    if maxcost > 0 and curcost[0] >= maxcost:
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
                    if maxcost > 0 and curcost[0] >= maxcost:
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
                 
                    if maxcost > 0 and curcost[0] >= maxcost:
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
            
        if maxcost > 0 and curcost[0] >= maxcost:
            yield ("max_iter_reached", )
        return None
    
        
    def add_sfrl_imp(self, x, y, gap = None, noaux = True):
        ccomp = self.allcomprv() - self.aux
        
        newvar = Comp.rv(self.name_avoid(y.tostring(add_braket = True) + "%" + x.tostring(add_braket = True)))
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
    
    def add_sfrl(self, x, y, gap = None, noaux = True):
        self.imp_flip()
        r = self.add_sfrl_imp(x, y, gap, noaux)
        self.imp_flip()
        return r
        
        
    def add_esfrl_imp(self, x, y, gap = None, noaux = True):
        if x.super_of(y):
            return Comp.empty(), y.copy()
        if x.isempty():
            return y.copy(), Comp.empty()
        
        ccomp = self.allcomprv() - self.aux
        
        newvar = Comp.rv(self.name_avoid(y.tostring(add_braket = True) + "%" + x.tostring(add_braket = True)))
        newvark = Comp.rv(self.name_avoid(y.tostring(add_braket = True) + "%" + x.tostring(add_braket = True) + "_K"))
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
                        pf = ProofObj.from_region(self, c = "To prove")
                        PsiOpts.set_setting(proof_add = pf)
                        
                        pf = ProofObj.from_region(Region.universe(), c = "Substitute:\n" + iutil.list_tostr_std(rr))
                        PsiOpts.set_setting(proof_add = pf)
                        
                        cs = self.copy()
                        Comp.substitute_list(cs, rr, isaux = True)
                        if cs.getaux().isempty():
                            with PsiOpts(proof_enabled = True):
                                cs.check_plain()
                        
                return rr
        return None
    
    
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
        
        ceps = PsiOpts.settings["eps"]
        for x in self.exprs_gei:
            if not f(x) >= -ceps:
                return True
        for x in self.exprs_eqi:
            if not abs(f(x)) <= ceps:
                return True
        for x in self.exprs_ge:
            if not f(x) >= -ceps:
                return False
        for x in self.exprs_eq:
            if not abs(f(x)) <= ceps:
                return False
        return True
        
    def implies(self, other):
        """Whether self implies other"""
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
        
    def allcomprv(self):
        index = IVarIndex()
        self.record_to(index)
        return index.comprv
        
    def allcompreal(self):
        index = IVarIndex()
        self.record_to(index)
        return index.compreal
        
    def allcomprealvar(self):
        index = IVarIndex()
        self.record_to(index)
        return index.compreal - Comp([IVar.eps(), IVar.one()])
    
    def allcomprv_noaux(self):
        return self.allcomprv() - self.getauxall()
    
    def aux_remove(self):
        self.aux = Comp.empty()
        self.auxi = Comp.empty()
        return self
        
    def completed_semigraphoid(self, max_iter = None):
        """ Use semi-graphoid axioms to deduce more conditional independence.
        Judea Pearl and Azaria Paz, "Graphoids: a graph-based logic for reasoning 
        about relevance relations", Advances in Artificial Intelligence (1987), pp. 357--363.
        """
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
        return r <= 0
                
    
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
        
        
        write_pf_enabled = (PsiOpts.settings.get("proof_enabled", False) 
                            and PsiOpts.settings.get("proof_step_simplify", False))
        if write_pf_enabled:
            prevself = self.copy()
        
        
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
                pf = ProofObj.from_region(prevself, c = "Simplify")
                pf += ProofObj.from_region(self, c = "Simplified as")
                PsiOpts.set_setting(proof_add = pf)
            
            
    
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
        
    def symmetrized(self, symm_set, skip_simplify = False):
        if symm_set is None:
            return self
        
        r = Region.universe()
        cs = self.copy()
        n = len(symm_set)
        tmpvar = Comp.array("#TMPVAR", 0, n)
        for i in range(n):
            cs.substitute(symm_set[i], tmpvar[i])
        
        for p in itertools.permutations(range(n)):
            tcs = cs.copy()
            for i in range(n):
                tcs.substitute(tmpvar[i], symm_set[p[i]])
            r &= tcs
        
        if skip_simplify:
            return r
        else:
            return r.simplified()
        
        
    def simplify_noredundant(self, reg = None, proc = None):
        write_pf_enabled = (PsiOpts.settings.get("proof_enabled", False) 
                            and PsiOpts.settings.get("proof_step_simplify", False))
        if write_pf_enabled:
            prevself = self.copy()
            red_reg = Region.universe()
        
                
        if reg is None:
            reg = Region.universe()
        
        #if self.isregtermpresent():
        #    return self
        
        for i in range(len(self.exprs_ge) - 1, -1, -1):
            t = self.exprs_ge[i]
            self.exprs_ge[i] = Expr.zero()
            cs = self.imp_intersection_noaux() & reg
            if proc is not None:
                cs = proc(cs)
            if not (cs <= (t >= 0)).check_plain():
                self.exprs_ge[i] = t
            elif write_pf_enabled:
                red_reg.exprs_ge.append(t)
        
        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        
        for i in range(len(self.exprs_eq) - 1, -1, -1):
            t = self.exprs_eq[i]
            self.exprs_eq[i] = Expr.zero()
            cs = self.imp_intersection_noaux() & reg
            if proc is not None:
                cs = proc(cs)
            if not (cs <= (t == 0)).check_plain():
                self.exprs_eq[i] = t
            elif write_pf_enabled:
                red_reg.exprs_eq.append(t)
        
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        for i in range(len(self.exprs_gei) - 1, -1, -1):
            t = self.exprs_gei[i]
            self.exprs_gei[i] = Expr.zero()
            cs = self.imp_flippedonly_noaux() & reg
            if proc is not None:
                cs = proc(cs)
            if not (cs <= (t >= 0)).check_plain():
                self.exprs_gei[i] = t
            elif write_pf_enabled:
                red_reg.exprs_gei.append(t)
        
        self.exprs_gei = [x for x in self.exprs_gei if not x.iszero()]
        
        for i in range(len(self.exprs_eqi) - 1, -1, -1):
            t = self.exprs_eqi[i]
            self.exprs_eqi[i] = Expr.zero()
            cs = self.imp_flippedonly_noaux() & reg
            if proc is not None:
                cs = proc(cs)
            if not (cs <= (t == 0)).check_plain():
                self.exprs_eqi[i] = t
            elif write_pf_enabled:
                red_reg.exprs_eqi.append(t)
        
        self.exprs_eqi = [x for x in self.exprs_eqi if not x.iszero()]
        
        if False:
            if self.imp_present():
                t = self.imp_flippedonly()
                t.simplify_noredundant(reg)
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
            self.simplify_noredundant(proc = lambda t: t.symmetrized(symm_set, skip_simplify = True))
        
        
    def simplify_imp(self, reg = None):
        if not self.imp_present():
            return
        
    def simplify(self, reg = None, zero_group = 0):
        """Simplify a region in place
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """
        
        if reg is None:
            reg = Region.universe()
            
        self.simplify_quick(reg, zero_group)
        
        self.simplify_noredundant(reg)
        
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
    
    def simplified(self, reg = None, zero_group = 0):
        """Returns the simplified region
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        r = self.copy()
        r.simplify(reg, zero_group)
        return r
    
    def get_ic(self, skip_simplify = False):
        cs = self
        if not skip_simplify:
            cs = self.simplified_quick(zero_group = 2)
        icexpr = Expr.zero()
        for x in cs.exprs_ge:
            if x.isnonpos():
                icexpr += x
        for x in cs.exprs_eq:
            if x.isnonpos():
                icexpr += x
            elif x.isnonneg():
                icexpr -= x
        return icexpr
    
    def get_bayesnet(self, skip_simplify = False):
        """Return a Bayesian network containing the conditional independence
        conditions in this region
        """
        icexpr = self.get_ic(skip_simplify)
        return BayesNet.from_ic(icexpr).tsorted()
    
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
        
    
    def eliminate_term_eq(self, w):
        """Do NOT use this directly. Use eliminate instead
        """
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
        
        
        
    def eliminate_term(self, w, forall = False):
        """Do NOT use this directly. Use eliminate instead
        """
        el = []
        er = []
        ee = []
        
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
        
        #if not forall and elni + erni + eeni > 0:
        #    self.setuniverse()
        #    return self
        
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
        
    def eliminate(self, w, reg = None, toreal = False, forall = False):
        """Fourier-Motzkin elimination, in place. 
        w is the Expr object with the real variables to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        
        if isinstance(w, CompList):
            w = w.get_comp()
        
        if isinstance(w, Comp):
            w = Expr.H(w)
        
        if not toreal and not forall and not self.auxi.isempty() and any(a.get_type() == TermType.IC for a, c in w.terms):
            return RegionOp.inter([self]).eliminate(w, reg, toreal, forall)
        
        self.simplify_quick(reg)
        
        if toreal and PsiOpts.settings["eliminate_rays"]:
            w2 = w
            w = Expr.zero()
            toelim = Comp.empty()
            for (a, c) in w2.terms:
                if a.get_type() == TermType.IC:
                    toelim += a.allcomp()
                else:
                    w.terms.append((a, c))
            if not toelim.isempty():
                self.eliminate_toreal_rays(toelim)
        
        for (a, c) in w.terms:
            if a.get_type() == TermType.REAL:
                self.eliminate_term(a, forall = forall)
                self.simplify(reg)
            elif a.get_type() == TermType.IC:
                if toreal:
                    self.eliminate_toreal(a.allcomp(), forall = forall)
                    self.simplify(reg)
                else:
                    if forall:
                        self.auxi += a.allcomp()
                    else:
                        self.aux += a.allcomp()
                
        return self
        
    def eliminate_quick(self, w, reg = None, toreal = False, forall = False):
        """Fourier-Motzkin elimination, in place. 
        w is the Expr object with the real variables to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        
        if isinstance(w, CompList):
            w = w.get_comp()
        
        if isinstance(w, Comp):
            w = Expr.H(w)
        
        if not toreal and not forall and not self.auxi.isempty() and any(a.get_type() == TermType.IC for a, c in w.terms):
            return RegionOp.inter([self]).eliminate(w, reg, toreal, forall)
        
        self.simplify_quick(reg)
        
        for (a, c) in w.terms:
            if a.get_type() == TermType.REAL:
                self.eliminate_term(a, forall = forall)
                self.simplify_quick(reg)
            elif a.get_type() == TermType.IC:
                if toreal:
                    self.eliminate_toreal(a.allcomp(), forall = forall)
                    self.simplify_quick(reg)
                else:
                    if forall:
                        self.auxi += a.allcomp()
                    else:
                        self.aux += a.allcomp()
                
        return self
        
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
        """Alias of eliminated
        """
        if w is None:
            w = self.allcomprv_noaux()
        r = self.copy()
        r = r.eliminate(w, reg, toreal, forall = False)
        return r
        
    def exists_quick(self, w = None, reg = None, toreal = False):
        """Alias of eliminated_quick
        """
        if w is None:
            w = self.allcomprv_noaux()
        r = self.copy()
        r = r.eliminate_quick(w, reg, toreal, forall = False)
        return r
        
    def forall(self, w = None, reg = None, toreal = False):
        """Region of intersection for all variable w
        """
        if w is None:
            w = self.allcomprv_noaux()
        r = self.copy()
        r = r.eliminate(w, reg, toreal, forall = True)
        return r
        
    def forall_quick(self, w = None, reg = None, toreal = False):
        """Region of intersection for all variable w
        """
        if w is None:
            w = self.allcomprv_noaux()
        r = self.copy()
        r = r.eliminate_quick(w, reg, toreal, forall = True)
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
        return not self.implies(Expr.one() <= 0)
        
    def __bool__(self):
        return self.check()
        
    def __call__(self):
        return self.check()
    
    def assume(self):
        PsiOpts.set_setting(truth_add = self)
    
    def assumed(self):
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
        
        plain = cs.isplain()
        #plain = False
        
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
                    
            if isinstance(a2[1], ExprList) or isinstance(a2[1], CompList):
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
                
            if plain and aself.isregtermpresent():
                plain = False
                
            if isinstance(a2[0], Expr):
                varreal.append(a2[0])
            else:
                varrv.append(a2[0])
                
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
        
        terms = list(itertools.chain(varreal, igen.sI(varrv)))
        lastres = [False]
        
        
        def expr_tr(ex):
            exsum = Expr.zero()
            for i in range(maxlen):
                tex = ex.copy()
                for a in entries:
                    if isinstance(a, tuple):
                        if isinstance(a[1], ExprList) or isinstance(a[1], CompList):
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
    
    
    def tostring(self, style = 0, tosort = False, lhsvar = "real", inden = 0):
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
            
        if self.imp_present():
            r += self.imp_flippedonly().tostring(style = style, tosort = tosort, lhsvar = lhsvar, inden = inden)
            if style == PsiOpts.STR_STYLE_PSITIP:
                r += " >>" + nlstr
            elif style & PsiOpts.STR_STYLE_LATEX:
                r += "\to" + nlstr
            else:
                r += " ->" + nlstr
        
        eqnlist = ([x.tostring_eqn(">=", style = style, tosort = tosort, lhsvar = lhsvar) for x in self.exprs_ge]
        + [x.tostring_eqn("==", style = style, tosort = tosort, lhsvar = lhsvar) for x in self.exprs_eq])
        if tosort:
            eqnlist = sorted(eqnlist, key=lambda a: (len(a), a))
        
        first = True
        isplu = not self.aux.isempty() or len(eqnlist) > 1
        
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += " " * inden + "("
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += " " * inden + "\\left\\{\\begin{array}{l}\n"
            else:
                r += " " * inden + "\\{"
        else:
            r += " " * inden + "{"
        
        for x in eqnlist:
            if style == PsiOpts.STR_STYLE_PSITIP:
                if first:
                    r += " "
                else:
                    r += nlstr + " " * inden + " &"
                if isplu:
                    r += "("
                r += " "
            elif style & PsiOpts.STR_STYLE_LATEX:
                if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                    if first:
                        r += " " * inden + "  "
                    else:
                        r += "," + nlstr + " " * inden + "  "
                else:
                    if first:
                        r += " "
                    else:
                        r += "," + nlstr + " " * inden + "  "
            else:
                if first:
                    r += " "
                else:
                    r += "," + nlstr + " " * inden + "  "
            
            r += x
            
            if style == PsiOpts.STR_STYLE_PSITIP:
                r += " "
                if isplu:
                    r += ")"
                
            first = False
            
        if len(eqnlist) == 0:
            if style == PsiOpts.STR_STYLE_PSITIP:
                r += " universe()"
        
        if not self.aux.isempty():
            if style == PsiOpts.STR_STYLE_PSITIP:
                r += " ^ "
            elif style & PsiOpts.STR_STYLE_LATEX:
                r += " , \\exists "
            else:
                r += " | "
            r += self.aux.tostring(style = style, tosort = tosort)
            
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += " )"
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += nlstr + " " * inden + "\\end{array}\\right\\}"
            else:
                r += " \\}"
        else:
            r += " }"
        return r
    
        
    def __str__(self):
        lhsvar = None
        if PsiOpts.settings.get("str_lhsreal", False):
            lhsvar = "real"
        return self.tostring(PsiOpts.settings["str_style"], lhsvar = lhsvar)
    
    def __repr__(self):
        if PsiOpts.settings.get("repr_check", False):
            #return str(self.check())
            if self.check():
                return str(True)
        
        lhsvar = None
        if PsiOpts.settings.get("str_lhsreal", False):
            lhsvar = "real"
            
        if PsiOpts.settings.get("repr_simplify", False):
            return self.simplified_quick().tostring(PsiOpts.STR_STYLE_PSITIP, lhsvar = lhsvar)
        
        return self.tostring(PsiOpts.STR_STYLE_PSITIP, lhsvar = lhsvar)
    
        
    def __hash__(self):
        #return hash(self.tostring(tosort = True))
        
        return hash((
            hash(frozenset(hash(x) for x in self.exprs_ge)),
            hash(frozenset(hash(x) for x in self.exprs_eq)),
            hash(frozenset(hash(x) for x in self.exprs_gei)),
            hash(frozenset(hash(x) for x in self.exprs_eqi)),
            hash(self.aux), hash(self.inp), hash(self.oup), hash(self.auxi)
            ))
        
        

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
    
    def union(xs):
        return RegionOp(RegionType.UNION, [(x.copy(), True) for x in xs], [])
    
    def inter(xs):
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
    
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), in place."""
        for x, c in self.regs:
            x.substitute(v0, v1)
        if not isinstance(v0, Expr):
            for x, c in self.auxs:
                x.substitute(v0, v1)
        return self
    
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
        warnings.warn("Minkowski sum of union or intersection regions is unsupported.", RuntimeWarning)
        
        auxs = [(x.copy(), c) for x, c in self.getauxs() + other.getauxs()]
        if self.get_type() == RegionType.UNION:
            return RegionOp(RegionType.UNION, [(x.sum_minkowski(other), c) for x, c in self.regs], auxs)
        if other.get_type() == RegionType.UNION:
            return RegionOp(RegionType.UNION, [(self.sum_minkowski(x), c) for x, c in other.regs], auxs)
        
        # The following are technically wrong
        if self.get_type() == RegionType.INTER:
            return RegionOp(RegionType.INTER, [(x.sum_minkowski(other), c) for x, c in self.regs], auxs)
        if other.get_type() == RegionType.INTER:
            return RegionOp(RegionType.INTER, [(self.sum_minkowski(x), c) for x, c in other.regs], auxs)
        return self.copy()
    
    
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
                t = x.substitute_sign(v0, list(reversed(v1s)))
                t.reverse()
            r[0] |= t[0]
            r[1] |= t[1]
        return r
    
    def flatten_regterm(self, term, isimp = True):
        if term.reg is None:
            return
        
        self.simplify_quick()
        sn = term.sn
        
        v1s = [Expr.real(self.name_avoid(str(term) + "_L")), 
               Expr.real(self.name_avoid(str(term) + "_U"))]
        sn_present = self.substitute_sign(Expr.fromterm(term), v1s)
        if sn < 0:
            sn_present.reverse()
            v1s.reverse()
        
        if sn_present[1]:
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
        
    def flatten(self):
        
        verbose = PsiOpts.settings.get("verbose_flatten", False)
        
        write_pf_enabled = PsiOpts.settings.get("proof_enabled", False)
        
        if write_pf_enabled:
            prevself = self.copy()
        
        did = True
        didall = False
        
        while did:
            did = False
            regterms = {}
            self.regtermmap(regterms, True)
            for (name, term) in regterms.items():
                regterms_in = {}
                term.reg.regtermmap(regterms_in, False)
                if not regterms_in:
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
                        self.flatten_regterm(term)
                    did = True
                    didall = True
                    
                    if verbose:
                        print("=========     to     ========")
                        print(self)
                        
                    break
        
        if write_pf_enabled:
            if didall:
                pf = ProofObj.from_region(prevself, c = "Expand definitions")
                pf += ProofObj.from_region(self, c = "Expanded definitions to")
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
                if t is None:
                    return None
                r.iand_norename(t)
            
        for x, c in self.auxs:
            if not c:
                return None
            r.eliminate(x)
            
        return r
    
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
        self.flatten()
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
        cs = self.copy()
        cs.presolve()
        r = []
        #allauxs = cs.auxs
        auxi = cs.getauxi()
        
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
                    req &= x
                else:
                    cons.append(x)
            
            ccons = None
            if len(cons) == 1:
                ccons = cons[0]
            else:
                ccons = RegionOp.union(cons)
            
            r.append((req, ccons, auxi.inter(x.getauxi())))
            
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
            
            cons2 = cons
            cons = []
            for x in cons2:
                y = x.copy()
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
        
        oneuse_set = MHashSet()
        oneuse_set.add(([None] * len(csaux), []))
        
        max_iter_pow = 4
        cur_max_iter = 800
        
        max_yield_pow = 3
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
        
        while did and (max_iter <= 0 or cur_max_iter < max_iter):
            cnstep += 1
            prev_did = did
            did = False
            #rcases3 = rcases
            nonsimple_did = False
            
            for i in range(n):
                if len(rcases) == 0:
                    break
                
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
                
                cur_consonly = req.isuniverse() and aux.isempty()
                if cnstep > 1 and cur_consonly:
                    continue
                
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
                
            if not nonsimple_did:
                break
            
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
                    
                    pf = ProofObj.from_region(self, c = "To prove")
                    PsiOpts.set_setting(proof_add = pf)
                    
                    pf = ProofObj.from_region(Region.universe(), c = "Substitute:\n" + iutil.list_tostr_std(resrr))
                    PsiOpts.set_setting(proof_add = pf)
                    
                    cs = self.copy()
                    Comp.substitute_list(cs, resrr, isaux = True)
                    if cs.getaux().isempty():
                        with PsiOpts(proof_enabled = True):
                            cs.check()
                    
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
        
        ceps = PsiOpts.settings["eps"]
        
        isunion = (self.get_type() == RegionType.UNION)
        for x, c in self.regs:
            if isunion ^ c ^ x.evalcheck(f):
                return isunion
        return not isunion
    
    
    def implies_getaux(self, other, hint_pair = None, hint_aux = None):
        """Whether self implies other, with auxiliary search result."""
        return (self <= other).check_getaux(hint_pair, hint_aux)
        
        
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
        #self.distribute()
            
        for x, c in self.regs:
            x.simplify_quick(reg, zero_group)
        
        self.simplify_op()
        return self
    
        
    def simplify(self, reg = None, zero_group = 0):
        """Simplify a region in place. 
        Optional argument reg with constraints assumed to be true. 
        zero_group = 2: group all nonnegative terms as a single inequality.
        """
        #self.distribute()
        for x, c in self.regs:
            x.simplify(reg, zero_group)
        
        self.simplify_op()
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
        if not r.aux_present():
            t = r.tosimple()
            if t is not None:
                return t
        return r
    
    def simplified(self, reg = None, zero_group = 0):
        """Returns the simplified region
        Optional argument reg with constraints assumed to be true
        zero_group = 2: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        r = self.copy()
        r.simplify(reg, zero_group)
        if not r.aux_present():
            t = r.tosimple()
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
        
    def eliminate(self, w, reg = None, toreal = False, forall = False):
        
        if isinstance(w, CompList):
            w = w.get_comp()
            
        for v in w.allcomp():
            if toreal or v.get_type() == IVarType.REAL:
                for x, c in self.regs:
                    x.eliminate(w, reg, toreal, forall ^ (not c))
            elif v.get_type() == IVarType.RV:
                self.add_aux(v, not forall)
        return self
        
    def eliminate_quick(self, w, reg = None, toreal = False, forall = False):
        
        if isinstance(w, CompList):
            w = w.get_comp()
            
        for v in w.allcomp():
            if toreal or v.get_type() == IVarType.REAL:
                for x, c in self.regs:
                    x.eliminate_quick(w, reg, toreal, forall ^ (not c))
            elif v.get_type() == IVarType.RV:
                self.add_aux(v, not forall)
        return self
        
    def marginal_eliminate(self, w):
        for x in self.regs:
            x.marginal_eliminate(w)
        
    def kernel_eliminate(self, w):
        for x in self.regs:
            x.kernel_eliminate(w)
          
    def tostring(self, style = 0, tosort = False, lhsvar = None, inden = 0):
        """Convert to string. 
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        
        style = iutil.convert_str_style(style)
        r = ""
        interstr = ""
        nlstr = "\n"
        notstr = "NOT"
        if style == PsiOpts.STR_STYLE_PSITIP:
            notstr = "~"
        elif style & PsiOpts.STR_STYLE_LATEX:
            notstr = "\\lnot"
            nlstr = "\\\\\n"
            
        if self.get_type() == RegionType.UNION:
            if style == PsiOpts.STR_STYLE_PSITIP:
                interstr = "|"
            elif style & PsiOpts.STR_STYLE_LATEX:
                interstr = "\\vee"
            else:
                interstr = "OR"
        if self.get_type() == RegionType.INTER:
            if style == PsiOpts.STR_STYLE_PSITIP:
                interstr = "&"
            elif style & PsiOpts.STR_STYLE_LATEX:
                interstr = "\\wedge"
            else:
                interstr = "AND"
        
        if self.isuniverse(sgn = True, canon = True):
            if style == PsiOpts.STR_STYLE_PSITIP:
                return " " * inden + "RegionOp.universe()"
            elif style & PsiOpts.STR_STYLE_LATEX:
                return " " * inden + "Universe"
            else:
                return " " * inden + "Universe"
        
        if self.isuniverse(sgn = False, canon = True):
            if style == PsiOpts.STR_STYLE_PSITIP:
                return " " * inden + "RegionOp.empty()"
            elif style & PsiOpts.STR_STYLE_LATEX:
                return " " * inden + "\\emptyset"
            else:
                return " " * inden + "{}"
        
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += " " * inden + "(" + nlstr
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += " " * inden + "\\left\\{\\begin{array}{l}\n"
            else:
                r += " " * inden + "\\{" + nlstr
        else:
            r += " " * inden + "{" + nlstr
        
        rlist = [" " * inden + ("" if c else " " + notstr) + 
                x.tostring(style = style, tosort = tosort, lhsvar = lhsvar, inden = inden + 2)[inden:] 
                for x, c in self.regs]
        if tosort:
            rlist = sorted(rlist, key=lambda a: (len(a), a))
            
        r += (nlstr + " " * inden + " " + interstr + nlstr).join(rlist)
        
        r += nlstr + " " * inden
        for x, c in self.auxs:
            if c:
                if style & PsiOpts.STR_STYLE_LATEX:
                    r += " , \\exists "
                else:
                    r += " | "
            else:
                if style & PsiOpts.STR_STYLE_LATEX:
                    r += " \\forall "
                else:
                    r += " & "
            r += x.tostring(style = style, tosort = tosort)
                
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += ")"
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += nlstr + " " * inden + "\\end{array}\\right\\}"
            else:
                r += "\\}"
        else:
            r += "}"
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
        
    
class CompList:
    def __init__(self, x):
        self.x = list(x)
    
    def empty(n = 0):
        return CompList([Comp.empty() for i in range(n)])
    
    def make(*args):
        r = CompList.empty()
        for a in args:
            if isinstance(a, Comp):
                r.x.append(a.copy())
            else:
                for b in a:
                    r.x.append(b.copy())
        return r
    
    def arg_convert(b):
        if isinstance(b, list) or isinstance(b, Comp):
            return CompList.make(*b)
        return b
    
    def copy(self):
        return CompList([a.copy() for a in self.x])
    
    def get_comp(self):
        return sum(self.x, Comp.empty())
    
    def append(self, a):
        self.x.append(a)
    
    def series(self, vdir):
        """Get past or future sequence.
        Parameters:
            vdir  : Direction, 1: future non-strict, 2: future strict,
                    -1: past non-strict, -2: past strict
        """
        if vdir == 1:
            return CompList([sum(self.x[i:], Comp.empty()) for i in range(len(self.x))])
        elif vdir == 2:
            return CompList([sum(self.x[i+1:], Comp.empty()) for i in range(len(self.x))])
        elif vdir == -1:
            return CompList([sum(self.x[:i+1], Comp.empty()) for i in range(len(self.x))])
        elif vdir == -2:
            return CompList([sum(self.x[:i], Comp.empty()) for i in range(len(self.x))])
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
    
    
    def set_len(self, n):
        if n < len(self.x):
            self.x = self.x[:n]
            return
        while n > len(self.x):
            self.x.append(Comp.empty())
    
    def __iadd__(self, other):
        if isinstance(other, CompList):
            r = []
            for i in range(len(other.x)):
                if i < len(self.x):
                    self.x[i] += other.x[i]
                else:
                    self.x.append(other.x[i].copy())
            return self
        
        for i in range(len(self.x)):
            self.x[i] += other
        return self
    
    def __add__(self, other):
        r = self.copy()
        r += other
        return r
        
    def __radd__(self, other):
        if isinstance(other, Comp) or isinstance(other, CompList):
            return self + other
        return self.copy()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, key):
        r = self.x[key]
        if isinstance(r, list):
            return CompList(r)
        return r
    
    def __setitem__(self, key, item):
        self.x[key] = item
    
    def __and__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return ExprList([a & b for a, b in zip(self.x, other.x)])
        return ExprList([a & other for a in self.x])
    
    def __or__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return ExprList([a | b for a, b in zip(self.x, other.x)])
        return ExprList([a | other for a in self.x])
    
    def __rand__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return ExprList([b & a for a, b in zip(self.x, other.x)])
        return ExprList([other & a for a in self.x])
    
    def __ror__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return ExprList([b | a for a, b in zip(self.x, other.x)])
        return ExprList([other | a for a in self.x])
        
    
    def record_to(self, index):
        for a in self.x:
            a.record_to(index)
            
    def isregtermpresent(self):
        for a in self.x:
            if a.isregtermpresent():
                return True
        return False
    
    def tostring(self, style = 0, tosort = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        style = iutil.convert_str_style(style)
        nlstr = "\n"
        if style & PsiOpts.STR_STYLE_LATEX:
            nlstr = "\\\\\n"
            
        r = ""
        add_braket = True
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += "CompList([ "
            add_braket = False
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += "\\left\\[\\begin{array}{l}\n"
                add_braket = False
            else:
                r += "\\[ "
        else:
            r += "[ "
        
        for i, a in enumerate(self.x):
            if i:
                if style & PsiOpts.STR_STYLE_LATEX:
                    if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                        r += nlstr
                    else:
                        r += ", "
                else:
                    r += ", "
            r += a.tostring(style = style, tosort = tosort, add_braket = add_braket)
            
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += " ])"
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += "\\end{array}\\right\\]"
            else:
                r += " \\]"
        else:
            r += " ]"
        
        return r
    
    
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.STR_STYLE_PSITIP)
        
    
class ExprList:
    def __init__(self, x):
        self.x = list(x)
    
    def empty():
        return ExprList([])
    
    def zeros(n):
        return ExprList([Expr.zero() for i in range(n)])
    
    def ones(n):
        return ExprList([Expr.one() for i in range(n)])
    
    def make(*args):
        r = ExprList.empty()
        for a in args:
            if isinstance(a, Expr):
                r.x.append(a.copy())
            else:
                for b in a:
                    r.x.append(b.copy())
        return r
    
    def copy(self):
        return ExprList([a.copy() for a in self.x])
    
    def get_expr():
        return sum(self.x, Expr.empty())
    
    
    def append(self, a):
        self.x.append(a)
        
    def series(self, vdir):
        """Get past or future sequence.
        Parameters:
            vdir  : Direction, 1: future non-strict, 2: future strict,
                    -1: past non-strict, -2: past strict
        """
        if vdir == 1:
            return ExprList([sum(self.x[i:], Expr.zero()) for i in range(len(self.x))])
        elif vdir == 2:
            return ExprList([sum(self.x[i+1:], Expr.zero()) for i in range(len(self.x))])
        elif vdir == -1:
            return ExprList([sum(self.x[:i+1], Expr.zero()) for i in range(len(self.x))])
        elif vdir == -2:
            return ExprList([sum(self.x[:i], Expr.zero()) for i in range(len(self.x))])
        return self.copy()
        
        
    def past_ns(self):
        return self.series(-1)
        
    def past(self):
        return self.series(-2)
    
    def future_ns(self):
        return self.series(1)
    
    def future(self):
        return self.series(2)
    
    
    def set_len(self, n):
        if n < len(self.x):
            self.x = self.x[:n]
            return
        while n > len(self.x):
            self.x.append(Expr.zero())
            
    
    def __neg__(self):
        return ExprList([-a for a in self.x])
    
    def __imul__(self, other):
        for i in range(len(self.x)):
            self.x[i] *= other
        return self
    
    def __mul__(self, other):
        r = self.copy()
        r *= other
        return r
    
    def __itruediv__(self, other):
        for i in range(len(self.x)):
            self.x[i] /= other
        return self
    
    def __truediv__(self, other):
        r = self.copy()
        r /= other
        return r
    
    def __iadd__(self, other):
        if isinstance(other, ExprList):
            r = []
            for i in range(len(other.x)):
                if i < len(self.x):
                    self.x[i] += other.x[i]
                else:
                    self.x.append(other.x[i].copy())
            return self
        
        for i in range(len(self.x)):
            self.x[i] += other
        return self
    
    def __add__(self, other):
        r = self.copy()
        r += other
        return r
    
    def __radd__(self, other):
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
    
    def __abs__(self):
        return eabs(self)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, key):
        r = self.x[key]
        if isinstance(r, list):
            return ExprList(r)
        return r
    
    def __setitem__(self, key, item):
        self.x[key] = item
    
    def __and__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return ExprList([a & b for a, b in zip(self.x, other.x)])
        return ExprList([a & other for a in self.x])
    
    def __or__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return ExprList([a | b for a, b in zip(self.x, other.x)])
        return ExprList([a | other for a in self.x])
    
    def __rand__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return ExprList([b & a for a, b in zip(self.x, other.x)])
        return ExprList([other & a for a in self.x])
    
    def __ror__(self, other):
        if isinstance(other, CompList) or isinstance(other, ExprList):
            return ExprList([b | a for a, b in zip(self.x, other.x)])
        return ExprList([other | a for a in self.x])
    
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
    
    
    def record_to(self, index):
        for a in self.x:
            a.record_to(index)
            
    def isregtermpresent(self):
        for a in self.x:
            if a.isregtermpresent():
                return True
        return False
    
    def tostring(self, style = 0, tosort = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        style = iutil.convert_str_style(style)
        nlstr = "\n"
        if style & PsiOpts.STR_STYLE_LATEX:
            nlstr = "\\\\\n"
            
        r = ""
        add_braket = True
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += "ExprList([ "
            add_braket = False
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += "\\left\\[\\begin{array}{l}\n"
                add_braket = False
            else:
                r += "\\[ "
        else:
            r += "[ "
        
        for i, a in enumerate(self.x):
            if i:
                if style & PsiOpts.STR_STYLE_LATEX:
                    if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                        r += nlstr
                    else:
                        r += ", "
                else:
                    r += ", "
            r += a.tostring(style = style, tosort = tosort)
            
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += " ])"
        elif style & PsiOpts.STR_STYLE_LATEX:
            if style & PsiOpts.STR_STYLE_LATEX_ARRAY:
                r += "\\end{array}\\right\\]"
            else:
                r += " \\]"
        else:
            r += " ]"
        
        return r
    
    
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"])
    
    def __repr__(self):
        return self.tostring(PsiOpts.STR_STYLE_PSITIP)
        

class ProofObj:
    def __init__(self, steps):
        self.steps = steps
    
    def empty():
        return ProofObj([])
    
    def copy(self):
        return ProofObj([(x.copy(), list(d), c) for x, d, c in self.steps])
    
    def from_region(x, c = ""):
        return ProofObj([(x.copy(), [], c)])
    
    def __iadd__(self, other):
        n = len(self.steps)
        self.steps += [(x.copy(), [di + n for di in d], c) for x, d, c in other.steps]
        return self
    
    def __add__(self, other):
        r = self.copy()
        r += other
        return r
    
    def clear(self):
        self.steps = []
          
    def tostring(self, style = 0, prev = None):
        """Convert to string. 
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        style = iutil.convert_str_style(style)
        r = ""
        start_n = 0
        if prev is not None:
            start_n = len(prev.steps)
        
        for i, (x, d, c) in enumerate(self.steps):
            if i > 0:
                r += "\n"
            r += "STEP #" + str(start_n + i)
            if c != "":
                r += " " + c
                
            if x is None or x.isuniverse():
                r += "\n"
            else:
                r += ":\n"
                r += x.tostring(style = style)
            
            r += "\n"
        return r
    
        
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"])
    

# Generators

class igen:
    
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
    r = None
    for x in a:
        if r is None:
            r = x
        else:
            r &= x
    return r

def anyor(a):
    r = None
    for x in a:
        if r is None:
            r = x
        else:
            r |= x
    return r
    
def rv(*args, **kwargs):
    """Random variable"""
    r = Comp.empty()
    for a in args:
        r += Comp.rv(a)
    
    for key, value in kwargs.items():
        r.add_markers([(key, value)])
        
    return r
    
def rv_array(name, st, en):
    """Array of random variables"""
    return Comp.array(name, st, en)
    
def rv_list(name, st, en):
    """List of random variables"""
    return CompList.make(*(Comp.array(name, st, en)))
    
def real(*args):
    """Real variable"""
    if len(args) == 1:
        return Expr.real(args[0])
    return ExprList([Expr.real(a) for a in args])
    
def real_list(name, st, en):
    """List of random variables"""
    t = rv_array(name, st, en)
    return ExprList([Expr.real(a.name) for a in t.varlist])

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
        return Expr.H(x)
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
    r = Region.universe()
    for i in range(1, len(args)):
        r &= (Expr.Hc(args[i], args[0]) == 0) & (Expr.Hc(args[0], args[i]) == 0)
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
    SU = rv("SU")
    r = Region.universe()
    for a in args:
        r &= H(SU | a) == 0
    return (r & indep(*args).conditioned(SU)).exists(SU)

@fcn_list_to_list
def cardbd(x, n):
    """Return Region where the cardinality of x is upper bounded by n."""
    if n <= 1:
        return H(x) == 0
    V = rv_array("V", 0, n-1)
    r = Expr.H(V[n - 2]) == 0
    r2 = Region.universe()
    for i in range(0, n - 1):
        r2 &= Expr.Hc(V[i], V[i - 1] if i > 0 else x) == 0
        r |= Expr.Hc(V[i - 1] if i > 0 else x, V[i]) == 0
    r = r.implicated(r2, skip_simplify = True).forall(V)
    return r & (H(x) <= numpy.log2(n))


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
    
    X = rv_array("CX", 0, n)
    for i in range(n):
        X[i].add_markers([("disjoint", disjoint_id), ("symm", symm_id_x), ("symm_nonempty", 1)])
    Y = rv_array("CY", 0, m)
    Z = rv_array("CZ", 0, m)
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
    X = rv("DX")
    Y = rv("DY")
    Z = rv("DZ")
    W = rv("DW")
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
    
    X = rv_array("CX", 0, n)
    for i in range(n):
        X[i].add_markers([("disjoint", disjoint_id), ("symm", symm_id_x), ("symm_nonempty", 2)])
    U = rv("CU")
    V = rv("CV")
    Z = rv("CZ")
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
    A = rv("CA")
    B = rv("CB")
    C = rv("CC")
    D = rv("CD")
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
    X = rv("X")
    Z = rv_array("Z", 0, n)
    
    if coeff is None:
        coeff = 1.0 / (1.0 - (1.0 - 1.0 / n) ** n) / n
    else:
        coeff = coeff / n
    
    if cadd is None:
        cadd = 2.43
        
    r = iidseq(*Z) & (H(X | Z) == 0)
    r &= H(Z[0]) <= H(X) * coeff + cadd
    return r.exists(Z).forall(X)
    
    
def existence(f, numarg = 2, nonempty = False):
    """A region for the existence of the random variable f(X, Y).
    e.g. existence(meet), existence(mss)
    """
    T = rv_array("T", 0, numarg)
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
    U = rv("BIT")
    return Comp.rv_reg(U, H(U) == 1)
    

def exists_bit(n = 1):
    """There exists a random variable with entropy 1."""
    U = rv_array("BIT", 0, n)
    return (alland([H(x) == 1 for x in U]) & indep(*U)).exists(U)


@fcn_list_to_list
def emin(*args):
    """Return the minimum of the expressions."""
    R = real(iutil.fcn_name_maker("min", args, pname = "emin", lname = "\\min"))
    r = universe()
    for x in args:
        r &= R <= x
    return r.maximum(R)

@fcn_list_to_list
def emax(*args):
    """Return the maximum of the expressions."""
    R = real(iutil.fcn_name_maker("max", args, pname = "emax", lname = "\\max"))
    r = universe()
    for x in args:
        r &= R >= x
    return r.minimum(R)


@fcn_list_to_list
def eabs(x):
    """Absolute value of expression."""
    R = real(iutil.fcn_name_maker("abs", x))
    return ((R >= x) & (R >= -x)).minimum(R)

@fcn_list_to_list
def meet(*args):
    """Gacs-Korner common part. 
    Peter Gacs and Janos Korner. Common information is far less than mutual information.
    Problems of Control and Information Theory, 2(2):149-162, 1973.
    """
    U = rv(iutil.fcn_name_maker("meet", args))
    V = rv("V")
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
    
    U = rv(iutil.fcn_name_maker("mss", [x, y]))
    V = rv("V")
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
    U = rv(iutil.fcn_name_maker("sfrl", [x, y], pname = "sfrl_rv"))
    #U = rv(y.tostring(add_braket = True) + "%" + x.tostring(add_braket = True))
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
    U = rv(iutil.fcn_name_maker("esfrl", [x, y], pname = "esfrl_rv"))
    K = rv(iutil.fcn_name_maker("esfrl_K", [x, y], pname = "esfrl_rv_K"))
    r = (Expr.Hc(K, x + U) == 0) & (Expr.Hc(Y, U + K) == 0) & (Expr.I(x, U) == 0)
    if gap is not None:
        if not isinstance(gap, Expr):
            gap = Expr.const(gap)
        r &= Expr.H(K) <= Expr.I(x, y) + gap
    return Comp.rv_reg(U, r, reg_det = False), Comp.rv_reg(K, r, reg_det = False)


def copylem_rv(x, y):
    """Copy lemma: for any X, Y, there exists Z such that (X, Y) has the same
    distribution as (X, Z), and Y-X-Z forms a Markov chain.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    Randall Dougherty, Chris Freiling, and Kenneth Zeger. "Non-Shannon information 
    inequalities in four random variables." arXiv preprint arXiv:1104.3602 (2011).
    """
    U = rv(iutil.fcn_name_maker("copy", [x, y], pname = "copylem_rv"))
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
    U = rv("U")
    R = real(iutil.fcn_name_maker("K", x, pname = "gacs_korner", cropi = True))
    r = universe()
    for a in x.x:
        r &= Expr.Hc(U, a+x.z) == 0
    r &= R <= Expr.Hc(U, x.z)
    return r.exists(U).maximum(R)


@fcn_list_to_list
def wyner_ci(x):
    """Wyner's common information. 
    A. D. Wyner. The common information of two dependent random variables.
    IEEE Trans. Info. Theory, 21(2):163-179, 1975. 
    e.g. wyner_ci(X & Y & Z | W)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("J", x, pname = "wyner_ci", cropi = True))
    r = indep(*(x.x)).conditioned(U + x.z)
    r &= R >= Expr.Ic(U, sum(x.x), x.z)
    return r.exists(U).minimum(R)


@fcn_list_to_list
def exact_ci(x):
    """Common entropy (one-shot exact common information). 
    G. R. Kumar, C. T. Li, and A. El Gamal. Exact common information. In Information
    Theory (ISIT), 2014 IEEE International Symposium on, 161-165. IEEE, 2014. 
    e.g. exact_ci(X & Y & Z | W)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("G", x, pname = "exact_ci", cropi = True))
    r = indep(*(x.x)).conditioned(U + x.z)
    r &= R >= Expr.Hc(U, x.z)
    return r.exists(U).minimum(R)


@fcn_list_to_list
def H_nec(x):
    """Necessary conditional entropy. 
    Cuff, P. W., Permuter, H. H., & Cover, T. M. (2010). Coordination capacity.
    IEEE Transactions on Information Theory, 56(9), 4181-4206. 
    e.g. H_nec(X + Y | W)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("Hnec", x, pname = "H_nec", lname = "H^\\dagger", cropi = True))
    r = markov(x.z, U, x.x[0]) & (Expr.Hc(U, x.x[0]) == 0)
    r &= R >= Expr.Hc(U, x.z)
    return r.exists(U).minimum(R)


@fcn_list_to_list
def excess_fi(x, y):
    """Excess functional information. 
    Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
    applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978. 
    e.g. excess_fi(X, Y)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("Hnec", [x, y], pname = "H_nec", lname = "H^\\dagger"))
    r = indep(U, x)
    r &= R >= Expr.Hc(y, U) - Expr.I(x, y)
    return r.exists(U).minimum(R)


@fcn_list_to_list
def korner_graph_ent(x, y):
    """Korner graph entropy. 
    J. Korner, "Coding of an information source having ambiguous alphabet and the 
    entropy of graphs," in 6th Prague conference on information theory, 1973, pp. 411-425.
    C. T. Li and A. El Gamal, "Extended Gray-Wyner system with complementary 
    causal side information," IEEE Transactions on Information Theory 64.8 (2017): 5862-5878.
    e.g. korner_graph_ent(X, Y)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("korner_graph_ent", [x, y], lname = "H_K"))
    r = markov(U, x, y) & (Expr.Hc(x, y+U) == 0)
    r &= R >= Expr.I(x, U)
    return r.exists(U).minimum(R)


@fcn_list_to_list
def perfect_privacy(x, y):
    """Perfect privacy rate. 
    A. Makhdoumi, S. Salamatian, N. Fawaz, and M. Medard, "From the information bottleneck 
    to the privacy funnel," in Information Theory Workshop (ITW), 2014 IEEE, Nov 2014, pp. 501-505.
    S. Asoodeh, F. Alajaji, and T. Linder, "Notes on information-theoretic privacy," in Communication, 
    Control, and Computing (Allerton), 2014 52nd Annual Allerton Conference on, Sept 2014, pp. 1272-1278.
    e.g. perfect_privacy(X, Y)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("perfect_privacy", [x, y], lname = "g_0"))
    r = markov(x, y, U) & (Expr.I(x, U) == 0)
    r &= R <= Expr.I(y, U)
    return r.exists(U).maximum(R)


@fcn_list_to_list
def max_interaction_info(x, y):
    """Maximal interaction information.
    C. T. Li and A. El Gamal, "Extended Gray-Wyner system with complementary 
    causal side information," IEEE Transactions on Information Theory 64.8 (2017): 5862-5878.
    e.g. max_interaction_info(X, Y)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("max_interaction_info", [x, y], lname = "G_{NNI}"))
    r = Region.universe()
    r &= R <= Expr.Ic(x, y, U) - Expr.I(x, y)
    return r.exists(U).maximum(R)


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
    return r.exists(U).maximum(R)


@fcn_list_to_list
def symm_interaction_info(x, y):
    """Symmetric private interaction information.
    C. T. Li and A. El Gamal, "Extended Gray-Wyner system with complementary 
    causal side information," IEEE Transactions on Information Theory 64.8 (2017): 5862-5878.
    e.g. max_interaction_info(X, Y)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("symm_interaction_info", [x, y], lname = "G_{PPI}"))
    r = indep(x, U) & indep(y, U)
    r &= R <= Expr.Ic(x, y, U) - Expr.I(x, y)
    return r.exists(U).maximum(R)


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
    U = rv("U")
    R = real(iutil.fcn_name_maker("MEC", [x, y], pname = "minent_coupling", lname = "H_{couple}"))
    r = indep(U, x) & (Expr.Hc(y, x + U) == 0)
    r &= R >= Expr.H(U)
    return r.exists(U).minimum(R)


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
    return r.maximum(R)


@fcn_list_to_list
def intrinsic_mi(x):
    """Intrinsic mutual information. 
    U. Maurer and S. Wolf. "Unconditionally secure key agreement and the intrinsic 
    conditional information." IEEE Transactions on Information Theory 45.2 (1999): 499-514.
    e.g. intrinsic_mi(X & Y | Z)
    """
    U = rv("U")
    R = real(iutil.fcn_name_maker("IMI", x, pname = "intrinsic_mi", lname = "I_{intrinsic}", cropi = True))
    r = markov(sum(x.x), x.z, U) & (R >= mutual_dep(Term(x.x, U)))
    return r.exists(U).minimum(R)


def directed_info(x, y):
    """Directed information. 
    Massey, James. "Causality, feedback and directed information." Proc. Int. 
    Symp. Inf. Theory Applic.(ISITA-90). 1990.
    Parameters can be either Comp or CompList.
    """
    x = CompList.arg_convert(x)
    y = CompList.arg_convert(y)
    return sum(I(x.past_ns() & y | y.past()))


def comp_vector(*args):
    return CompList.make(igen.subset(args, minsize = 1))


def ent_vector(*args):
    """Entropy vector.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    """
    return H(comp_vector(*args))


def ent_cells(*args, minsize = 1):
    """Cells of the I-measure.
    Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
    IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.
    """
    allrv = sum(args)
    r = ExprList.empty()
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
    xs = rv_array(var_name, 0, n)
    cv = comp_vector(*xs)
    re = real_list(real_name, 1, 1 << n)
    return (re == H(cv)).exists(xs)




