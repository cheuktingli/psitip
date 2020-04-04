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
Version 1.01
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

Currently this library can only check for conic constraints, e.g.:
bool((I(X & Y) == 0) >> (H(X) * 2 + H(Y) <= H(X + Y) * 2))
that is, constant nonzero numbers can only appear as coefficients.

It CANNOT check for general affine/convex constraints, e.g.:
(H(X) <= 5) >> (I(X & Y) <= 5) (does not work)

"""

import itertools
import collections
import fractions
import warnings

try:
    import numpy
    import numpy.linalg
except ImportError:
    numpy = None

try:
    import scipy
    import scipy.sparse
    import scipy.optimize
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
                      
        lptype      : Linear programming mode
                      LinearProgType.H : Classical
                      LinearProgType.HC1BN : Bayesian Network optimization
                      
        verbose_auxsearch : Print auxiliary RV search progress
        
        verbose_lp        : Print linear programming parameters
        
        rename_char : Char appended when renaming auxiliary RV
        
        eps         : Epsilon for float comparison
    """
    
    STR_STYLE_STANDARD = 0
    STR_STYLE_PSITIP = 1
    
    SFRL_LEVEL_SINGLE = 1
    SFRL_LEVEL_MULTIPLE = 2
    
    settings = {
        "eps": 1e-10,
        "max_denom": 1000,
        "max_denom_mul": 12,
        "str_style": 0,
        "rename_char": "_",
        
        "solver": "scipy",
        "lptype": LinearProgType.HC1BN,
        "lp_bounded": False,
        "solver_scipy_maxsize": 12,
        
        "imp_noncircular": True,
        "imp_noncircular_allaux": False,
        "imp_simplify": True,
        "tensorize_simplify": False,
        "forall_multiuse": True,
        "forall_multiuse_numsave": 32,
        
        "verbose_lp": False,
        "verbose_auxsearch": False,
        "verbose_auxsearch_step": False,
        "verbose_auxsearch_result": False,
        "verbose_auxsearch_cache": False,
        "verbose_auxsearch_step_cached": False,
        "verbose_subset": False,
        "verbose_sfrl": False,
        "verbose_flatten": False,
        "verbose_eliminate_toreal": False,
        
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
                    
        elif key == "verbose_auxsearch_all":
            d["verbose_auxsearch"] = value
            d["verbose_auxsearch_step"] = value
            d["verbose_auxsearch_result"] = value
            
        else:
            d[key] = value
    
    def set_setting(**kwargs):
        for key, value in kwargs.items():
            PsiOpts.set_setting_dict(PsiOpts.settings, key, value)
    
    def get_setting(key, defaultval = None):
        if key in PsiOpts.settings:
            return PsiOpts.settings[key]
        return defaultval
            
    def __init__(self, **kwargs):
        self.cur_settings = PsiOpts.settings.copy()
        for key, value in kwargs.items():
            PsiOpts.set_setting_dict(self.cur_settings, key, value)
    
    def __enter__(self):
        PsiOpts.settings, self.cur_settings = self.cur_settings, PsiOpts.settings
        return PsiOpts.settings
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        PsiOpts.settings, self.cur_settings = self.cur_settings, PsiOpts.settings
    
    
class IUtil:
    """Common utilities
    """
    
    
    solver_list = ["pulp.glpk", "pyomo.glpk", "pulp.cbc", "scipy"]
    pulp_solver = None
    pulp_solvers = {}
    
    
    def float_tostr(x):
        if abs(x) <= 1e-10:
            return "0"
        elif abs(x - round(x)) <= 1e-10:
            return str(int(round(x)))
        else:
            if x > 0:
                return "(" + str(fractions.Fraction(x).limit_denominator(
                        PsiOpts.settings["max_denom"])) + ")"
            else:
                return "-(" + str(fractions.Fraction(-x).limit_denominator(
                        PsiOpts.settings["max_denom"])) + ")"

    def float_snap(x):
        t = float(fractions.Fraction(x).limit_denominator(PsiOpts.settings["max_denom"]))
        if abs(x - t) <= PsiOpts.settings["eps"]:
            return t
        return x

    def gcd(a, b):
        while b > 0:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return (a // IUtil.gcd(a, b)) * b
    
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
    
    def str_python_multiline(s):
        s = str(s)
        return "(\"" + "\\n\"\n\"".join(s.split("\n")) + "\")"
    
    def hash_short(s):
        s = str(s)
        return hash(s) % 99991
        
    def get_solver(psolver = None):
        csolver_list = [PsiOpts.settings["solver"]] + IUtil.solver_list
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
            return IUtil.pulp_solver
        
        if copt in IUtil.pulp_solvers:
            return IUtil.pulp_solvers[copt]
        
        r = None
        if copt == "GLPK":
            r = pulp.solvers.GLPK(msg = 0)
        elif copt == "CBC" or copt == "PULP_CBC_CMD":
            r = pulp.solvers.PULP_CBC_CMD()
        
        IUtil.pulp_solvers[copt] = r
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
                        r += IUtil.list_tostr(x[i], tuple_delim, list_delim, inden + 2)[inden + 1:]
                    else:
                        r += list_delim + "\n" + IUtil.list_tostr(x[i], tuple_delim, list_delim, inden + 2)
                r += " ]"
                return r
            else:
                r += "[" + list_delim.join([IUtil.list_tostr(a, tuple_delim, list_delim, 0) for a in x]) + "]"
                return r
        elif isinstance(x, tuple):
            r += "(" + tuple_delim.join([IUtil.list_tostr(a, tuple_delim, list_delim, 0) for a in x]) + ")"
            return r
        r += str(x)
        return r
    
    def list_tostr_std(x):
        return IUtil.list_tostr(x, tuple_delim = ": ", list_delim = "; ")

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
    
    def __init__(self, vartype, name, reg = None, reg_det = False):
        self.vartype = vartype
        self.name = name
        self.reg = reg
        self.reg_det = reg_det
        
    def rv(name):
        return IVar(IVarType.RV, name)
        
    def real(name):
        return IVar(IVarType.REAL, name)
    
    def tostring(self):
        return self.name
    
    def __str__(self):
        return self.tostring()
        
    def __eq__(self, other):
        return self.name == other.name
        
    def copy(self):
        if self.reg is None:
            return IVar(self.vartype, self.name, None, self.reg_det)
        else:
            return IVar(self.vartype, self.name, self.reg.copy(), self.reg_det)

    
class Comp:
    """Compound random variable or real variable
    """
    
    def __init__(self, varlist):
        self.varlist = varlist
        
    def empty():
        return Comp([])
        
    def rv(name):
        return Comp([IVar(IVarType.RV, name)])
        
    def rv_reg(a, reg, reg_det = False):
        return Comp([IVar(IVarType.RV, str(a), reg.copy(), reg_det)])
        
    def real(name):
        return Comp([IVar(IVarType.REAL, name)])
    
    def array(name, st, en):
        return Comp([IVar(IVarType.RV, name + str(i)) for i in range(st, en)])
        
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
        
    def addvar(self, x):
        if x in self.varlist:
            return
        self.varlist.append(x.copy())
        
    def removevar(self, x):
        self.varlist = [a for a in self.varlist if a != x]
        
    def ispresent(self, x):
        
        for a in self.varlist:
            if a.reg is not None and a.reg.ispresent(x):
                return True
                
        if isinstance(x, Comp):
            for y in x.varlist:
                if y in self.varlist:
                    return True
            return False
        
        return x in self.varlist
        
    def ispresent_shallow(self, x):
                
        if isinstance(x, Comp):
            for y in x.varlist:
                if y in self.varlist:
                    return True
            return False
        
        return x in self.varlist
        
    def __iadd__(self, other):
        for i in range(len(other.varlist)):
            self.addvar(other.varlist[i])
        return self
        
    def __add__(self, other):
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
        
        
    def __ge__(self, other):
        return self.super_of(other)
         
    def __le__(self, other):
        return other.super_of(self)
    
    def __eq__(self, other):
        return self.super_of(other) and other.super_of(self)
    
    def __gt__(self, other):
        return self.super_of(other) and not other.super_of(self)
    
    def __lt__(self, other):
        return other.super_of(self) and not self.super_of(other)
    
    def tostring(self, style = 0, tosort = False, add_braket = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        namelist = [a.name for a in self.varlist]
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
        
    def get_type(self):
        if len(self.varlist) == 0:
            return IVarType.NIL
        return self.varlist[0].vartype
    
    
    def __and__(self, other):
        return Term.H(self) & Term.H(other)
    
    def __or__(self, other):
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
        for (name0, name1) in namemap.items():
            self.rename_var(name0, name1)
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
        
    def isnonneg(self):
        if self.get_type() == TermType.IC:
            return len(self.x) <= 2
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
        
    
    def tostring(self, style = 0, tosort = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
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
        (viss, viso) = self.match_x(other)
        if -1 in viso:
            return False
        return True
        
        
    def try_iadd(self, other):
        
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
        if isinstance(other, Comp):
            other = Term.H(other)
        return Term([a.copy() for a in self.x] + [a.copy() for a in other.x], self.z + other.z)
        
    
    def __or__(self, other):
        if isinstance(other, Comp):
            other = Term.H(other)
        return Term([a.copy() for a in self.x], self.z + other.allcomp())
        
        
            
    def ispresent(self, x):
        """Return whether any variable in x appears here"""
        if isinstance(x, IVar):
            x = Comp([x])
        
        if self.get_type() == TermType.REGION:
            if self.reg.ispresent(x):
                return True
            
        for y in x.varlist:
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
        for (name0, name1) in namemap.items():
            self.rename_var(name0, name1)
        return self
    
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound)"""
        if self.get_type() == TermType.REGION:
            self.reg.substitute(v0, v1)
        for a in self.x:
            a.substitute(v0, v1)
        self.z.substitute(v0, v1)
        
class Expr:
    """An expression
    """
    
    def __init__(self, terms):
        self.terms = terms
    
        
    def copy(self):
        return Expr([(a.copy(), c) for (a, c) in self.terms])
        
    
    def fromcomp(x):
        return Expr([(Term.fromcomp(x), 1.0)])
    
    def fromterm(x):
        return Expr([(x.copy(), 1.0)])
        
    def __iadd__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        self.terms += [(a.copy(), c) for (a, c) in other.terms]
        return self
        
    def __add__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if isinstance(other, Expr):
            return Expr([(a.copy(), c) for (a, c) in self.terms]
            + [(a.copy(), c) for (a, c) in other.terms])
        return self.copy()
        
    def __radd__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        if isinstance(other, Expr):
            return Expr([(a.copy(), c) for (a, c) in other.terms]
            + [(a.copy(), c) for (a, c) in self.terms])
        return self.copy()
        
    def __neg__(self):
        return Expr([(a.copy(), -c) for (a, c) in self.terms])
        
    def __isub__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        self.terms += [(a.copy(), -c) for (a, c) in other.terms]
        return self
        
    def __sub__(self, other):
        if isinstance(other, Comp):
            other = Expr.fromcomp(other)
        return Expr([(a.copy(), c) for (a, c) in self.terms]
        + [(a.copy(), -c) for (a, c) in other.terms])
        
    def __imul__(self, other):
        self.terms = [(a, c * other) for (a, c) in self.terms]
        return self
        
    def __mul__(self, other):
        return Expr([(a.copy(), c * other) for (a, c) in self.terms])
        
    def __rmul__(self, other):
        return Expr([(a.copy(), c * other) for (a, c) in self.terms])
        
    def __itruediv__(self, other):
        self.terms = [(a, c / other) for (a, c) in self.terms]
        return self
        
    def record_to(self, index):
        for (a, c) in self.terms:
            a.record_to(index)
    
        
    def size(self):
        return len(self.terms)
        
    def iszero(self):
        """Whether the expression is zero"""
        return len(self.terms) == 0
        
    def setzero(self):
        """Set expression to zero"""
        self.terms = []
        
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
    
    def zero():
        """The constant zero expression"""
        return Expr([])
    
    def H(x):
        """Entropy"""
        return Expr([(Term.H(x), 1.0)])
        
    def I(x, y):
        """Mutual information"""
        return Expr([(Term.I(x, y), 1.0)])
        
    def Hc(x, z):
        """Conditional entropy"""
        return Expr([(Term.Hc(x, z), 1.0)])
        
    def Ic(x, y, z):
        """Conditional mutual information"""
        return Expr([(Term.Ic(x, y, z), 1.0)])
        
        
    def real(name):
        """Real variable"""
        return Expr([(Term([Comp.real(name)], Comp.empty()), 1.0)])
    
    
    def tostring(self, style = 0, tosort = False):
        """Convert to string
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        
        termlist = self.terms
        if tosort:
            termlist = sorted(termlist, key=lambda a: (-round(a[1] * 1000.0), 
                                   a[0].tostring(style = style, tosort = tosort)))
        
        r = ""
        first = True
        for (a, c) in termlist:
            if abs(c) < PsiOpts.settings["eps"]:
                continue
            if c > 0.0 and not first:
                r += "+"
            if abs(c - 1.0) < PsiOpts.settings["eps"]:
                pass
            elif abs(c + 1.0) < PsiOpts.settings["eps"]:
                r += "-"
            else:
                r += IUtil.float_tostr(c)
                if style == PsiOpts.STR_STYLE_PSITIP:
                    r += "*"
            r += a.tostring(style = style, tosort = tosort)
            first = False
            
        if r == "":
            return "0"
        return r
        
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"])
        
    def sortIc(self):
        def sortkey(a):
            x = a[0]
            if x.isic2():
                return x.size()
            else:
                return 100000
            
        self.terms.sort(key=sortkey)
        
    def __le__(self, other):
        if isinstance(other, Expr):
            return Region([other - self], [], Comp.empty(), Comp.empty(), Comp.empty())
        else:
            return Region([-self], [], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __ge__(self, other):
        if isinstance(other, Expr):
            return Region([self - other], [], Comp.empty(), Comp.empty(), Comp.empty())
        else:
            return Region([self.copy()], [], Comp.empty(), Comp.empty(), Comp.empty())
            
    def __eq__(self, other):
        if isinstance(other, Expr):
            return Region([], [self - other], Comp.empty(), Comp.empty(), Comp.empty())
        else:
            return Region([], [self.copy()], Comp.empty(), Comp.empty(), Comp.empty())
        
    
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
    
    def remove_term(self, b):
        """Remove Term b in place"""
        self.terms = [(a, c) for (a, c) in self.terms if a != b]
        return self
    
    def removed_term(self, b):
        """Remove Term b, return Expr after removal"""
        return Expr([(a, c) for (a, c) in self.terms if a != b])
        
    
    def coeff_sum(self):
        """Sum of coefficients"""
        return sum([c for (a, c) in self.terms])
    
    def simplify_mul(self, mul_allowed = 0):
        if mul_allowed > 0:
            max_denom = PsiOpts.settings["max_denom"]
            max_denom_mul = PsiOpts.settings["max_denom_mul"]
            denom = 1
            for (a, c) in self.terms:
                denom = IUtil.lcm(fractions.Fraction(c).limit_denominator(
                    max_denom).denominator, denom)
                if denom > max_denom_mul:
                    break
                
            if denom > 0 and denom <= max_denom_mul:
                if mul_allowed >= 2:
                    if self.coeff_sum() < 0:
                        denom = -denom
                if len([(a, c) for (a, c) in self.terms if abs(c * denom - round(c * denom)) > PsiOpts.settings["eps"]]) == 0:
                    self.terms = [(a, IUtil.float_snap(c * denom)) for (a, c) in self.terms]
            
    
    def simplify(self, reg = None):
        """Simplify the expression in place"""
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
        
        #self.terms = [(a, IUtil.float_snap(c)) for (a, c) in self.terms 
        #              if abs(c) > PsiOpts.settings["eps"] and not a.iszero()]
            
        return self

    
    def simplified(self, reg = None):
        """Simplify the expression, return simplified expression"""
        r = self.copy()
        r.simplify(reg)
        return r
    
    def get_ratio(self, other):
        """Try dividing self by other, return None if self is not scalar multiple of other"""
        es = self.simplified()
        eo = other.simplified()
        
        if es.iszero():
            return 0.0
        if eo.iszero():
            return None
        
        if len(es.terms) != len(eo.terms):
            return None
            
        rmax = -1e12
        rmin = 1e12
        
        for i in range(len(es.terms)):
            found = False
            for j in range(len(eo.terms)):
                if abs(eo.terms[j][1]) > PsiOpts.settings["eps"] and es.terms[i][0] == eo.terms[j][0]:
                    cr = es.terms[i][1] / eo.terms[j][1]
                    rmax = max(rmax, cr)
                    rmin = min(rmin, cr)
                    eo.terms[j] = (Term.zero(), 0.0)
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
            return self.get_ratio(other)
        return Expr([(a.copy(), c / other) for (a, c) in self.terms])
        

    def ispresent(self, x):
        """Return whether any variable in x appears here"""
        for (a, c) in self.terms:
            if a.ispresent(x):
                return True
        return False
        
    def rename_var(self, name0, name1):
        for (a, c) in self.terms:
            a.rename_var(name0, name1)
        
    def rename_map(self, namemap):
        """Rename according to name map
        """
        for (name0, name1) in namemap.items():
            self.rename_var(name0, name1)
        return self

    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), in place"""
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
        return self

    def conditioned(self, b):
        """Condition on random variable b, return result"""
        r = self.copy()
        r.condition(b)
        return r

    
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
        
    
    def tostring_eqn(self, eqnstr, style = 0, tosort = False, lhsvar = None):
        lhs = self
        rhs = Expr.zero()
        if lhsvar is not None:
            lhs, rhs = self.split_lhs(lhsvar)
            rhs *= -1.0
            if lhs.iszero():
                lhs, rhs = rhs, lhs
                eqnstr = IUtil.reverse_eqnstr(eqnstr)
            if lhs.coeff_sum() < 0 or (abs(lhs.coeff_sum()) <= PsiOpts.settings["eps"] and rhs.coeff_sum() < 0):
                lhs *= -1.0
                rhs *= -1.0
                eqnstr = IUtil.reverse_eqnstr(eqnstr)
        
        return (lhs.tostring(style = style, tosort = tosort) + " "
        + eqnstr + " " + rhs.tostring(style = style, tosort = tosort))
        
        
class BayesNet:
    """Bayesian network"""
    
    def __init__(self):
        self.index = IVarIndex()
        self.parent = []
        self.child = []
        
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
        
    def add_edge(self, x, y):
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
        
        for i in range(n):
            cnparent[i] = len(self.parent[i])
            if cnparent[i] == 0:
                cstack.append(i)
                
        while len(cstack) > 0:
            i = cstack.pop()
            r.record(self.index.comprv[i])
            for j in self.child[i]:
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
            nvis = IUtil.bitcount(vis)
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
                        nedge = IUtil.bitcount(zk) + dp[vis | (1 << i)]
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
        
    def tostring(self):
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
        
class SparseMat:
    """List of lists sparse matrix. Do NOT use directly"""
    
    def __init__(self, width):
        self.width = width
        self.x = []
    
    def addrow(self):
        self.x.append([])
        
    def poprow(self):
        self.x.pop()
        
    def add_last_row(self, j, c):
        self.x[len(self.x) - 1].append((j, c))
        
    def simplify_last_row(self):
        i = len(self.x) - 1
        self.x[i].sort()
        t = self.x[i]
        self.x[i] = []
        cj = -1
        cc = 0.0
        for (j, c) in t:
            if j == cj:
                cc += c
            else:
                if abs(cc) > PsiOpts.settings["eps"]:
                    self.x[i].append((cj, cc))
                cj = j
                cc = c
        if abs(cc) > PsiOpts.settings["eps"]:
            self.x[i].append((cj, cc))
                
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
        
    def mapcol(self, m):
        for i in range(len(self.x)):
            for k in range(len(self.x[i])):
                j2 = m[self.x[i][k][0]]
                if j2 < 0:
                    return False
                self.x[i][k] = (j2, self.x[i][k][1])
        return True
        
    def tolil(self):
        r = scipy.sparse.lil_matrix((len(self.x), self.width))
        for i in range(len(self.x)):
            for (j, c) in self.x[i]:
                r[i, j] += c
        return r

    
class LinearProg:
    """A linear programming instance. Do NOT use directly"""
    
    def __init__(self, index, lptype, bnet = None):
        self.index = index
        self.lptype = lptype
        self.nvar = 0
        self.nxvar = 0
        self.xvarid = []
        self.realshift = 0
        self.cellpos = []
        self.bnet = bnet
        self.lp_bounded = PsiOpts.settings["lp_bounded"]
        
        if self.lptype == LinearProgType.H:
            self.nvar = (1 << self.index.num_rv()) - 1 + self.index.num_real()
            self.realshift = (1 << self.index.num_rv()) - 1
            
        elif self.lptype == LinearProgType.HC1BN:
            n = self.index.num_rv()
            nbnet = bnet.index.num_rv()
            
            self.cellpos = [-1] * (1 << n)
            cpos = 0
            for i in range(n):
                for mask in range(1 << i):
                    maski = mask + (1 << i)
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
                    if self.cellpos[maski] < 0:
                        self.cellpos[maski] = cpos
                        cpos += 1
            self.realshift = cpos
            self.nvar = self.realshift + self.index.num_real()
            
        
        self.nxvar = self.nvar
        self.Au = SparseMat(self.nvar)
        self.Ae = SparseMat(self.nvar)
        
        self.solver = None
        if self.nvar <= PsiOpts.settings["solver_scipy_maxsize"]:
            self.solver = IUtil.get_solver("scipy")
        else:
            self.solver = IUtil.get_solver()
        
        self.solver_param = {}
        self.bu = []
        self.be = []
        
        self.icp = [[] for i in range(self.index.num_rv())]
        
        
    
    def addreal_id(self, A, k, c):
        A.add_last_row(self.realshift + k, c)
    
    def addH_mask(self, A, mask, c):
        if self.lptype == LinearProgType.H:
            A.add_last_row(mask - 1, c)
        elif self.lptype == LinearProgType.HC1BN:
            n = self.index.num_rv()
            for i in range(n):
                if mask & (1 << i) != 0:
                    A.add_last_row(self.cellpos[mask & ((1 << (i + 1)) - 1)], c)
    
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
                
    def addExpr_ge0(self, x):
        if x.size() == 0:
            return
        self.Au.addrow()
        self.addExpr(self.Au, -x)
        self.bu.append(0.0)
        
    def addExpr_eq0(self, x):
        if x.size() == 0:
            return
        self.Ae.addrow()
        self.addExpr(self.Ae, x)
        self.be.append(0.0)
        
        
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
        self.add_ent_ineq()
        
        PsiRec.num_lpprob += 1
        
        if self.lptype == LinearProgType.HC1BN:
            self.Au.unique_row()
            self.bu = [0.0] * len(self.Au.x)
        
        if self.lp_bounded:
            self.Au.addrow()
            for i in range(self.realshift):
                self.Au.add_last_row(i, 1.0)
            self.bu.append(1.0)
        
        cols = self.Au.nonzero_cols()
        coles = self.Ae.nonzero_cols()
        self.xvarid = [0] * self.nvar
        self.nxvar = 0
        for i in range(self.nvar):
            if cols[i] or coles[i]:
                self.xvarid[i] = self.nxvar
                self.nxvar += 1
            else:
                self.xvarid[i] = -1
                
        self.Au.mapcol(self.xvarid)
        self.Au.width = self.nxvar
        self.Ae.mapcol(self.xvarid)
        self.Ae.width = self.nxvar
        
        if self.solver == "scipy":
            self.solver_param["Aus"] = self.Au.tolil()
            self.solver_param["Aes"] = self.Ae.tolil()
            
        elif self.solver.startswith("pulp."):
            prob = pulp.LpProblem("lpentineq" + str(PsiRec.num_lpprob), pulp.LpMinimize)
            xvar = pulp.LpVariable.dicts("x", [str(i) for i in range(self.nxvar)])
            
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
                    
            self.solver_param["prob"] = prob
            self.solver_param["xvar"] = xvar
        
        elif self.solver.startswith("pyomo."):
            solver_opt = self.solver[self.solver.index(".") + 1 :]
            opt = SolverFactory(solver_opt)
            
            model = pyo.ConcreteModel()
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
            
        
    def get_region(self, toreal = None, toreal_only = False):
        if toreal is None:
            toreal = Comp.empty()
        n = self.index.num_rv()
        
        torealmask = self.index.get_mask(toreal)
        
        xvarinv = [0] * self.nxvar
        for i in range(self.nvar):
            if self.xvarid[i] >= 0:
                xvarinv[self.xvarid[i]] = i
        
        cellposinv = list(range(1, self.realshift + 1))
        if self.lptype == LinearProgType.HC1BN:
            for mask in range((1 << n) - 1, 0, -1):
                cellposinv[self.cellpos[mask]] = mask
        
        r = Region.universe()
        for x, sn in [(a, 1) for a in self.Au.x] + [(a, 0) for a in self.Ae.x]:
            expr = Expr.zero()
            toreal_present = False
            
            for (j2, c) in x:
                j = xvarinv[j2]
                
                if j >= self.realshift:
                    expr += Expr.real(self.index.compreal.varlist[j - self.realshift].name) * c
                    continue
                
                mask = cellposinv[j]
                termreal = (mask & torealmask) != 0
                toreal_present |= termreal
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
                
                if termreal:
                    expr += Expr.real("R_" + str(term)) * c
                else:
                    expr += Expr.fromterm(term) * c
            
            if toreal_present or not toreal_only:
                expr.simplify()
                
                if sn == 1:
                    r &= (expr <= 0)
                else:
                    r &= (expr == 0)
        
        return r
    
    
    def checkexpr_ge0(self, x):
        if self.lptype == LinearProgType.HC1BN:
            if x.isnonpos_ic2():
                if self.bnet.check_ic(x):
                    return True
                #return False
        
        verbose = PsiOpts.settings.get("verbose_lp", False)
        zero_cutoff = -1e-5
        res = None
        
        optobj = SparseMat(self.nvar)
        optobj.addrow()
        self.addExpr(optobj, x)
        optobj.simplify_last_row()
        
        if not optobj.mapcol(self.xvarid):
            return False
        
        optobj.width = self.nxvar
        
        #print(optobj.row_dense(0))
        #print(self.Aus.todense())
        
        c = optobj.row_dense(0)
        if len([x for x in c if abs(x) > PsiOpts.settings["eps"]]) == 0:
            return True
            
        if len(self.Au.x) == 0 and len(self.Ae.x) == 0:
            return False
            
        if verbose:
            print("LP nrv=" + str(self.index.num_rv()) + " nreal=" + str(self.index.num_real())
            + " nvar=" + str(self.Au.width) + "/" + str(self.nvar) + " nineq=" + str(len(self.Au.x))
            + " neq=" + str(len(self.Ae.x)) + " solver=" + self.solver)
        
        if self.solver == "scipy":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = scipy.optimize.linprog(c, self.solver_param["Aus"], self.bu, self.solver_param["Aes"], self.be, 
                                         bounds = (None, None), method = "interior-point", options={'sparse': True})
                                         
                if verbose:
                    print("  status=" + str(res.status) + " optval=" + str(res.fun))
                    
            return (res.status == 0 and res.fun >= zero_cutoff)
        
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
                res = prob.solve(IUtil.pulp_get_solver(self.solver))
                if verbose:
                    print("  status=" + pulp.LpStatus[res] + " optval=" + str(prob.objective.value()))
                if pulp.LpStatus[res] == "Optimal" and prob.objective.value() >= zero_cutoff:
                    return True
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
            if (res.solver.status == pyo.SolverStatus.ok 
                and res.solver.termination_condition == pyo.TerminationCondition.optimal
                and model.o() >= zero_cutoff):
                return True
            return False
        
        
    
    def checkexpr_eq0(self, x):
        return self.checkexpr_ge0(x) and self.checkexpr_ge0(-x)
    
    
    def checkexpr(self, x, sg):
        if sg == "==":
            return self.checkexpr_eq0(x)
        elif sg == ">=":
            return self.checkexpr_ge0(x)
        elif sg == "<=":
            return self.checkexpr_ge0(-x)
        else:
            return False
    
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
    
    def universe():
        return Region([], [], Comp.empty(), Comp.empty(), Comp.empty())
        
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
        
        
    def sum_entrywise(self, other):
        return Region([x + y for (x, y) in zip(self.exprs_ge, other.exprs_ge)],
                      [x + y for (x, y) in zip(self.exprs_eq, other.exprs_eq)], 
                        self.aux + other.aux, self.inp + other.inp, self.oup + other.oup,
                        [x + y for (x, y) in zip(self.exprs_gei, other.exprs_gei)],
                        [x + y for (x, y) in zip(self.exprs_eqi, other.exprs_eqi)],
                        self.auxi + other.auxi)

    def ispresent(self, x):
        """Return whether any variable in x appears here"""
        for z in self.exprs_ge:
            if z.ispresent(x):
                return True
        for z in self.exprs_ge:
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
        return not self.aux.isempty() or not self.auxi.isempty()
    
    def aux_clear(self):
        self.aux = Comp.empty()
        self.auxi = Comp.empty()
        
    def getaux(self):
        return self.aux.copy()
    
    def getauxi(self):
        return self.auxi.copy()
    
    def aux_avoid(self, reg):
            
        for a in reg.getauxi().varlist:
            reg.rename_avoid(self, a.name)
        for a in self.getaux().varlist:
            self.rename_avoid(reg, a.name)
        for a in reg.getaux().varlist:
            reg.rename_avoid(self, a.name)
        for a in self.getauxi().varlist:
            self.rename_avoid(reg, a.name)
    
    def aux_avoid_from(self, reg):
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
        if isinstance(other, RegionOp):
            return RegionOp(RegionType.INTER, [self.copy()]) & other
            
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
        if isinstance(other, RegionOp):
            return RegionOp(RegionType.INTER, [self.copy()]) & other
            
        r = self.copy()
        r &= other
        return r
    
        
    def __or__(self, other):
        return RegionOp(RegionType.UNION, [self.copy()]) | other
        
    def __ior__(self, other):
        return RegionOp(RegionType.UNION, [self.copy()]) | other
    
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
        
        if other.get_type() == RegionType.INTER:
            return RegionOp(RegionType.UNION, 
                            [self.implicated(x) for x in other.regs])
            
        if other.get_type() == RegionType.UNION:
            return RegionOp(RegionType.INTER, 
                            [self.implicated(x) for x in other.regs])
        
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
        return RegionOp(RegionType.INTER, 
                        [self.implicated(other), other.implicated(self)])
        
    
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
                r.append(cs2)
        
        if len(r) == 0:
            return Region.universe()
        if len(r) == 1:
            return r[0]
        return RegionOp(RegionType.UNION, r)
            
    
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
        return RegionOp(RegionType.INTER, r)
            
        
    
    def corners(self, w):
        """Return union of regions corresponding to corner points of the real variables in w"""
        terms = []
        for (a, c) in w.terms:
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
        
        return RegionOp(RegionType.UNION, r)
        
    
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
        newvar = Comp.rv(ivar.name)
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
            for (a, c) in x.terms:
                rvs = a.allcomprv_shallow()
                for b in rvs.varlist:
                    if b.reg is not None:
                        return True
                if a.get_type() == TermType.REGION:
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
        for x in self.exprs_ge + self.exprs_eq + self.exprs_gei + self.exprs_eqi:
            for (a, c) in x.terms:
                rvs = a.allcomprv_shallow()
                for b in rvs.varlist:
                    if b.reg is not None:
                        s = str(b)
                        if not (s in cmap):
                            cmap[s] = b
                            if recur:
                                b.reg.regtermmap(cmap, recur)
                        
                if a.get_type() == TermType.REGION:
                    s = str(a)
                    if not (s in cmap):
                        cmap[s] = a
                        if recur:
                            a.reg.regtermmap(cmap, recur)
        
    def flatten(self):
        
        verbose = PsiOpts.settings.get("verbose_flatten", False)
        cs = self
        
        did = False
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
                
                if verbose:
                    print("=========     to     ========")
                    print(cs)
                    
                break
        
        if did:
            return cs.flatten()
        return cs
    
    def flattened(self):
        r = self.copy()
        r = r.flatten()
        return r.simplify_quick()
                
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
        tmpstr += str(IUtil.hash_short(self))
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
    
    def init_prog(self, index, lptype = None):
        prog = None
        if lptype is None:
            lptype = PsiOpts.settings["lptype"]
        if lptype == LinearProgType.H:
            prog = LinearProg(index, LinearProgType.H)
        elif lptype == LinearProgType.HC1BN:
            bnet = self.get_bayesnet_imp(skip_simplify = True)
            kindex = bnet.index.copy()
            kindex.add_varindex(index)
            prog = LinearProg(kindex, LinearProgType.HC1BN, bnet)
        
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
    
    def implies_ineq_quick(self, expr, sg):
        """Return whether self implies expr >= 0 or expr == 0, without linear programming"""
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
            
    def implies_ineq_prog(self, index, progs, expr, sg):
        if self.implies_ineq_quick(expr, sg):
            return True
        if len(progs) == 0:
            progs.append(self.init_prog(index))
        if progs[0].checkexpr(expr, sg):
            return True
        return False
        
    
    def check_plain(self, skip_simplify = False):
        """Return whether implication is true"""
        verbose_subset = PsiOpts.settings.get("verbose_subset", False)
        
        cs = self
        if not skip_simplify:
            cs = self.simplified_quick(zero_group = True)
            
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
    
    
    def check_getaux_inplace(self, must_include = None, single_include = None, hint_pair = None, hint_aux = None):
        """Return whether implication is true, with auxiliary search result"""
        verbose = PsiOpts.settings.get("verbose_auxsearch", False)
        verbose_step = PsiOpts.settings.get("verbose_auxsearch_step", False)
        verbose_result = PsiOpts.settings.get("verbose_auxsearch_result", False)
        verbose_cache = PsiOpts.settings.get("verbose_auxsearch_cache", False)
        verbose_step_cached = PsiOpts.settings.get("verbose_auxsearch_step_cached", False)
        
        if must_include is None:
            must_include = Comp.empty()
        
        noncircular = PsiOpts.settings["imp_noncircular"]
        noncircular_allaux = PsiOpts.settings["imp_noncircular_allaux"]
        #noncircular_skipfail = True
        
        forall_multiuse = PsiOpts.settings["forall_multiuse"]
        forall_multiuse_numsave = PsiOpts.settings["forall_multiuse_numsave"]
        
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
                tcmask = 0
                tcmask_pair = 0
                for j in range(m):
                    if tc.ispresent(ccomp[j]):
                        tcmask |= 1 << j
                        if comppair[j] < 0:
                            tcmask_pair |= 1 << j
                    elif comppair[j] >= 0 and tc.ispresent(ccomp[comppair[j]]):
                        tcmask_pair |= 1 << j
                        
                for i in range(n):
                    if taux.ispresent(auxcomp[i]):
                        setflag |= tcmask << (m * i)
                    elif auxpair[i] >= 0 and taux.ispresent(auxcomp[auxpair[i]]):
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
                    print(IUtil.strpad(hs, 8, ": " + str(eqvs[i][j][0]) + " " + str(eqvs[i][j][1]) + " 0" + eqvsnstr))
                    
            print("========= variables =========")
            print(ccomp)
            print("========= auxiliary =========")
            #print(auxcomp)
            print(str(auxcond) + " ; " + str(auxcomp - auxcond))
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
                csetflag = (setflag >> (m * i)) & ((1 << m) - 1)
                ccor = Comp.empty()
                cset = Comp.empty()
                for j in range(m):
                    if cflag & (1 << j) != 0:
                        ccor += ccomp[j]
                    if csetflag & (1 << j) != 0:
                        cset += ccomp[j]
                print(str(auxcomp.varlist[i]) + "  :  " + str(ccor) 
                    + ("   Fix: " + str(cset) if not cset.isempty() else ""))
        
        
        auxcache = [{} for i in range(n)]
        
        if verbose:
            print("========= progress: =========")
        
        if n_cond == 0:
            for (x, sg) in eqvs[eqvs_emptyid]:
                if not cs.implies_ineq_prog(index, progs, x, sg):
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
        
        maxprogress = [-1, 0, -1]
        flipflaglen = 0
        numfail = [0]
        numclear = [0]
        
        def clear_cache(mini, maxi):
            if verbose_cache:
                print("========= cache clear: " + str(mini) + " - " + str(maxi) + " =========")
            progs[:] = []
            auxcache[mini:maxi] = [{} for i in range(mini, maxi)]
            
            for i in range(mini * 2, eqvs_range):
                for j in range(len(eqvs[i])):
                    if eqvpresflag[i][j] & ((1 << maxi) - (1 << mini)) != 0:
                        eqvsncache[i][j] = [[], []]
                        eqvflagcache[i][j] = {}
                # eqvsncache[i] = [[[], []] for j in range(len(eqvs[i]))]
                # eqvflagcache[i] = [{} for j in range(len(eqvs[i]))]
                
        
        def build_region(i, allflag, allownew):
            numclear[0] += 1
            
            cs_added_changed = False
            prev_csstr = cs.tostring(tosort = True)
            
            if i > 0 and i >= n_cond and forall_multiuse:
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
                        cs_added.simplify_quick(zero_group = True)
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
                
        
        build_region(0, 0, False)
        
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
                            
                            tres = cs.implies_ineq_prog(index, progs, x2, sg)
                            computed = True
                            
                            eqvsn = eqvsns[i2][ieq]
                            if eqvsn == 0:
                                eqvflagcache[i2][ieq][auxflagpres] = tres
                            else:
                                eqvsncache[i2][ieq][1 if tres else 0].append(auxflagpres)
                        
                        if not tres:
                            if verbose_step and (verbose_step_cached or computed):
                                numfail[0] += 1
                                print(IUtil.strpad("; ".join([str(auxlist[i3]) for i3 in range(i)]),
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
                    if not cs.implies_ineq_prog(index, progs, x, sg):
                        if verbose_step:
                            numfail[0] += 1
                            print(IUtil.strpad("; ".join([str(auxlist[i3]) for i3 in range(i)]),
                               26, " S=" + str(cursizepass) + ",T=" + str(stepsize)
                               + ",L=" + str(flipflaglen) + ",#" + str(numfail[0]), 18, " " + str(x) + " " + sg + " 0"))
                        allflagcache[i][allflag] = True
                        
                        eqvsid[eqvs_emptyid].pop(ieqid)
                        eqvsid[eqvs_emptyid].insert(0, ieq)
                        return False
                
                    
            if i == n:
                return True
            
                
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
                    
                    if (auxflag[i] & singleflag) != 0 and IUtil.bitcount(auxflag[i]) > 1:
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
                        
                    if check_recur(i+1, size - tsize, stepsize, allflag + (auxflag[i] << (m * i))):
                        return True
                 
            return False
        
        maxsize = m * n - IUtil.bitcount(setflag)
        size = 0
        stepsize = 0
        while True:
            cursizepass = size
            prevprogress = maxprogress[0]
            if check_recur(0, size, stepsize, 0):
                
                if verbose or verbose_result:
                    print("========= success =========")
                    print("========= final region =========")
                    print(cs.imp_flipped())
                    print("========== aux  =========")
                    for i in range(n):
                        print(IUtil.strpad(str(auxcomp.varlist[i]), 6, ": " + str(auxlist[i])))
                        
                namelist = [auxcomp.varlist[i].name for i in range(n)]
                res = []
                for i in range(n):
                    i2 = namelist.index(self.aux.varlist[i].name)
                    cval = auxlist[i2]
                    if forall_multiuse and i2 < n_cond and len(condflagadded_true) > 0:
                        cval = [ccomp.from_mask(x >> (m * i2)) for x in condflagadded_true]
                        if len(cval) == 1:
                            cval = cval[0]
                    res.append((Comp([self.aux.varlist[i]]), cval))
                return res
            
            if size >= maxsize:
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
            
        return None
    
        
    def add_sfrl_imp(self, x, y, gap = None):
        ccomp = self.allcomprv() - self.aux
        
        newvar = Comp.rv(self.name_avoid(y.tostring(add_braket = True) + "%" + x.tostring(add_braket = True)))
        self.exprs_gei.append(-Expr.I(x, newvar))
        self.exprs_gei.append(-Expr.Hc(y, x + newvar))
        others = ccomp - x - y
        if not others.isempty():
            self.exprs_gei.append(-Expr.Ic(newvar, others, x + y))
        if gap is not None:
            self.exprs_gei.append(gap.copy() - Expr.Ic(x, newvar, y))
            
        self.auxi += newvar
        
        return newvar
    
    def add_sfrl(self, x, y, gap = None):
        self.imp_flip()
        r = self.add_sfrl_imp(x, y, gap)
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
                    csfrl += cs.add_sfrl_imp(sfrlx, sfrly, gap)
                
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
        
        pcs = self.copy()
        cs = pcs.flatten()
        
        if cs is not pcs:
            return cs.check_getaux(hint_pair, hint_aux)
        
        cs.simplify_quick(zero_group = True)
        cs.split()
        # cs.aux_addprefix()
        
        res = None
        
        sfrl_level = PsiOpts.settings["sfrl_level"]
        
        res = cs.check_getaux_inplace(hint_pair = hint_pair, hint_aux = hint_aux)
        
        if res is None and sfrl_level > 0:
            res = cs.check_getaux_sfrl(sfrl_minsize = 1, hint_pair = hint_pair, hint_aux = hint_aux)
            
        if res is None:
            return None
        return res
        #return [(Comp([self.aux.varlist[i]]) if i < len(self.aux.varlist) else res[i][0], res[i][1]) for i in range(len(res))]
    
        
    def check(self):
        """Return whether implication is true"""
        if self.isplain():
            return self.check_plain()
    
        return self.check_getaux() is not None
        
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
        
    def allcomprv(self):
        index = IVarIndex()
        self.record_to(index)
        return index.comprv
        
    def allcompreal(self):
        index = IVarIndex()
        self.record_to(index)
        return index.compreal
    
    def allcomprv_noaux(self):
        return self.allcomprv() - self.getaux() - self.getauxi()
        
    def convexify(self, q = None):
        """Convexify by a time sharing RV q, in place"""
        qname = "Q"
        if q is not None:
            qname = str(q)
        qname = self.name_avoid(qname)
        q = Comp.rv(qname)
        
        allcomp = self.allcomprv()
        
        self.condition(q)
        self.exprs_eq.append(Expr.Ic(q, allcomp - self.aux - self.auxi - self.inp - self.oup, self.inp))
        self.aux += q
        return self
        
    def convexified(self, q = None):
        """Convexify by a time sharing RV q, return result"""
        r = self.copy()
        r.convexify(q)
        return r
        
    def imp_convexified(self, q = None):
        """Intersect all time sharing RV q"""
        r = self.imp_flipped()
        r.convexify(q)
        r.imp_flip()
        return r
    
    def isconvex(self):
        """Check whether region is convex
        False return value does NOT necessarily mean region is not convex
        """
        return self.convexified().implies(self)
        #return ((self + self) / 2).implies(self)
        
    def simplify_quick(self, reg = None, zero_group = False):
        """Simplify a region in place, without linear programming
        Optional argument reg with constraints assumed to be true
        zero_group = True: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        for x in self.exprs_ge:
            x.simplify(reg)
        for x in self.exprs_eq:
            x.simplify(reg)
        
        
        did = True
        while did:
            did = False
            for i in range(len(self.exprs_ge)):
                if not self.exprs_ge[i].iszero():
                    for j in range(i):
                        if not self.exprs_ge[j].iszero():
                            ratio = self.exprs_ge[i].get_ratio(self.exprs_ge[j])
                            if ratio is None:
                                continue
                            if ratio > PsiOpts.settings["eps"]:
                                self.exprs_ge[i] = Expr.zero()
                                did = True
                                break
                            elif ratio < -PsiOpts.settings["eps"]:
                                self.exprs_eq.append(self.exprs_ge[i])
                                self.exprs_ge[i] = Expr.zero()
                                self.exprs_ge[j] = Expr.zero()
                                did = True
                                break
                            
            
            for i in range(len(self.exprs_ge)):
                if not self.exprs_ge[i].iszero():
                    for j in range(len(self.exprs_eq)):
                        if not self.exprs_eq[j].iszero():
                            ratio = self.exprs_ge[i].get_ratio(self.exprs_eq[j])
                            if ratio is None:
                                continue
                            self.exprs_ge[i] = Expr.zero()
                            did = True
                            break
            
            
            for i in range(len(self.exprs_eq)):
                if not self.exprs_eq[i].iszero():
                    for j in range(i):
                        if not self.exprs_eq[j].iszero():
                            ratio = self.exprs_eq[i].get_ratio(self.exprs_eq[j])
                            if ratio is None:
                                continue
                            self.exprs_eq[i] = Expr.zero()
                            did = True
                            break
                            
            self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
            self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        
        for i in range(len(self.exprs_ge)):
            if self.exprs_ge[i].isnonneg():
                self.exprs_ge[i] = Expr.zero()
        
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
             
        if zero_group:
            pass
        else:
            for i in range(len(self.exprs_ge)):
                if self.exprs_ge[i].isnonpos():
                    for (a, c) in self.exprs_ge[i].terms:
                        self.exprs_eq.append(Expr.fromterm(a))
                    self.exprs_ge[i] = Expr.zero()
                    
            for i in range(len(self.exprs_eq)):
                if self.exprs_eq[i].isnonpos() or self.exprs_eq[i].isnonneg():
                    for (a, c) in self.exprs_eq[i].terms:
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
        
        
    def simplify_noredundant(self, reg = None):
        if reg is None:
            reg = Region.universe()
        
        #if self.isregtermpresent():
        #    return self
        
        for i in range(len(self.exprs_ge)):
            t = self.exprs_ge[i]
            self.exprs_ge[i] = Expr.zero()
            if not ((self.imp_intersection_noaux() & reg) <= (t >= 0)).check_plain():
                self.exprs_ge[i] = t
        
        self.exprs_ge = [x for x in self.exprs_ge if not x.iszero()]
        
        for i in range(len(self.exprs_eq)):
            t = self.exprs_eq[i]
            self.exprs_eq[i] = Expr.zero()
            if not ((self.imp_intersection_noaux() & reg) <= (t == 0)).check_plain():
                self.exprs_eq[i] = t
        
        self.exprs_eq = [x for x in self.exprs_eq if not x.iszero()]
        
        for i in range(len(self.exprs_gei)):
            t = self.exprs_gei[i]
            self.exprs_gei[i] = Expr.zero()
            if not ((self.imp_flippedonly_noaux() & reg) <= (t >= 0)).check_plain():
                self.exprs_gei[i] = t
        
        self.exprs_gei = [x for x in self.exprs_gei if not x.iszero()]
        
        for i in range(len(self.exprs_eqi)):
            t = self.exprs_eqi[i]
            self.exprs_eqi[i] = Expr.zero()
            if not ((self.imp_flippedonly_noaux() & reg) <= (t == 0)).check_plain():
                self.exprs_eqi[i] = t
        
        self.exprs_eqi = [x for x in self.exprs_eqi if not x.iszero()]
        
        if False:
            if self.imp_present():
                t = self.imp_flippedonly()
                t.simplify_noredundant(reg)
                self.exprs_gei = t.exprs_ge
                self.exprs_eqi = t.exprs_eq
            
        return self
    
        
    def simplify_imp(self, reg = None):
        if not self.imp_present():
            return
        
    def simplify(self, reg = None, zero_group = False):
        """Simplify a region in place
        Optional argument reg with constraints assumed to be true
        zero_group = True: group all nonnegative terms as a single inequality
        """
        
        if reg is None:
            reg = Region.universe()
            
        self.simplify_quick(reg, zero_group)
        
        self.simplify_noredundant(reg)
        
        return self
    
    
    def simplified_quick(self, reg = None, zero_group = False):
        """Returns the simplified region
        Optional argument reg with constraints assumed to be true
        zero_group = True: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        r = self.copy()
        r.simplify_quick(reg, zero_group)
        return r
    
    def simplified(self, reg = None, zero_group = False):
        """Returns the simplified region
        Optional argument reg with constraints assumed to be true
        zero_group = True: group all nonnegative terms as a single inequality
        """
        if reg is None:
            reg = Region.universe()
        r = self.copy()
        r.simplify(reg, zero_group)
        return r
    
    def get_bayesnet(self, skip_simplify = False):
        """Return a Bayesian network containing the conditional independence
        conditions in this region
        """
        cs = self
        if not skip_simplify:
            cs = self.simplified_quick(zero_group = True)
        icexpr = Expr.zero()
        for x in cs.exprs_ge:
            if x.isnonpos():
                icexpr += x
        for x in cs.exprs_eq:
            if x.isnonpos():
                icexpr += x
            elif x.isnonneg():
                icexpr -= x
        return BayesNet.from_ic(icexpr).tsorted()
    
    def get_bayesnet_imp(self, skip_simplify = False):
        """Return a Bayesian network containing the conditional independence
        conditions in this region
        """
        cs = self
        if not skip_simplify:
            cs = self.simplified_quick(zero_group = True)
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
        
        
        
    def eliminate_term(self, w):
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
            
        
    def eliminate_toreal(self, w):
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
            self.eliminate_term(Term.fromcomp(Comp([a])))
            self.simplify_quick()
            if verbose:
                print("========== elim " + str(a) + " =========")
                print(self)
            
        return self
        
    def eliminate(self, w, reg = None, toreal = False):
        """Fourier-Motzkin elimination, in place. 
        w is the Expr object with the real variables to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        self.simplify_quick(reg)
        
        if isinstance(w, Comp):
            w = Expr.H(w)
        
        for (a, c) in w.terms:
            if a.get_type() == TermType.REAL:
                self.eliminate_term(a)
                self.simplify(reg)
            elif a.get_type() == TermType.IC:
                if toreal:
                    self.eliminate_toreal(a.allcomp())
                    self.simplify(reg)
                else:
                    self.aux += a.allcomp()
                
        return self
        
    def eliminate_quick(self, w, reg = None, toreal = False):
        """Fourier-Motzkin elimination, in place. 
        w is the Expr object with the real variables to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        self.simplify_quick(reg)
        
        if isinstance(w, Comp):
            w = Expr.H(w)
        
        for (a, c) in w.terms:
            if a.get_type() == TermType.REAL:
                self.eliminate_term(a)
                self.simplify_quick(reg)
            elif a.get_type() == TermType.IC:
                if toreal:
                    self.eliminate_toreal(a.allcomp())
                    self.simplify_quick(reg)
                else:
                    self.aux += a.allcomp()
                
        return self
        
    def eliminated(self, w, reg = None, toreal = False):
        """Fourier-Motzkin elimination, return region after elimination. 
        w is the Expr object with the real variable to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        r = self.copy()
        r.eliminate(w, reg, toreal)
        return r
        
    def eliminated_quick(self, w, reg = None, toreal = False):
        """Fourier-Motzkin elimination, return region after elimination. 
        w is the Expr object with the real variable to eliminate. 
        If w contains random variables, they will be treated as auxiliary RV.
        """
        r = self.copy()
        r.eliminate_quick(w, reg, toreal)
        return r
        
    def exists(self, w, reg = None, toreal = False):
        """Alias of eliminated
        """
        r = self.copy()
        r.eliminate(w, reg, toreal)
        return r
        
    def exists_quick(self, w, reg = None, toreal = False):
        """Alias of eliminated_quick
        """
        r = self.copy()
        r.eliminate_quick(w, reg, toreal)
        return r
        
    def forall(self, w, reg = None, toreal = False):
        """Region of intersection for all variable w
        """
        r = self.imp_flipped()
        r.eliminate(w, reg, toreal)
        r.imp_flip()
        return r
        
    def forall_quick(self, w, reg = None, toreal = False):
        """Region of intersection for all variable w
        """
        r = self.imp_flipped()
        r.eliminate_quick(w, reg, toreal)
        r.imp_flip()
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
        if other.get_type() != RegionType.NORMAL:
            return other.sum_minkowski(self)
        
        cs = self.copy()
        co = other.copy()
        param_real = cs.allcompreal() + co.allcompreal()
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
        
    def rename_map(self, namemap):
        """Rename according to name map.
        """
        for (name0, name1) in namemap.items():
            self.rename_var(name0, name1)
        return self
        
    def tensorize(self, reg_subset = None, chan_cond = None, nature = None, timeshare = False, hint_aux = None):
        """Check whether region tensorizes, return auxiliary RVs if tensorizes. 
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
        
        param_rv = index.comprv - self.aux - self.auxi
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
            
            if PsiOpts.settings["tensorize_simplify"]:
                rsum = (self & r2).eliminated(param_real_expr)
            else:
                rsum = (self & r2).eliminated_quick(param_real_expr)
        
        rsum.aux = Comp.empty()
        for i in range(max(len(self.aux), len(r2.aux))):
            if i < len(self.aux):
                rsum.aux += self.aux[i]
            if i < len(r2.aux):
                rsum.aux += r2.aux[i]
                
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
        
        
        return rx.implies_getaux(rsum, hint_pair = hint_pair, hint_aux = hint_aux)
        
        
    def check_converse(self, reg_subset = None, chan_cond = None, nature = None, hint_aux = None):
        """Check whether self is the capacity region of the operational region. 
        reg_subset, return auxiliary RVs if true. 
        chan_cond : The condition on the channel (e.g. degraded broadcast channel).
        """
        return self.tensorize(reg_subset, chan_cond, nature, True, hint_aux = hint_aux)
        
    def __xor__(self, other):
        return self.eliminated(other)
        
    def __ixor__(self, other):
        return self.eliminate(other)
        
        
    def __bool__(self):
        return self.check()
    
    def tostring(self, style = 0, tosort = False, lhsvar = None, inden = 0):
        """Convert to string. 
        Parameters:
            style   : Style of string conversion
                      STR_STYLE_STANDARD : I(X,Y;Z|W)
                      STR_STYLE_PSITIP : I(X+Y&Z|W)
        """
        
        r = ""
        
        if self.imp_present():
            r += self.imp_flippedonly().tostring(style = style, tosort = tosort, lhsvar = lhsvar, inden = inden)
            if style == PsiOpts.STR_STYLE_PSITIP:
                r += " >>\n"
            else:
                r += " ->\n"
        
        eqnlist = ([x.tostring_eqn(">=", style = style, tosort = tosort, lhsvar = lhsvar) for x in self.exprs_ge]
        + [x.tostring_eqn("==", style = style, tosort = tosort, lhsvar = lhsvar) for x in self.exprs_eq])
        if tosort:
            eqnlist = sorted(eqnlist, key=lambda a: (len(a), a))
        
        first = True
        
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += " " * inden + "("
        else:
            r += " " * inden + "{"
        
        for x in eqnlist:
            if style == PsiOpts.STR_STYLE_PSITIP:
                if first:
                    r += " ( "
                else:
                    r += "\n" + " " * inden + " &( "
            else:
                if first:
                    r += " "
                else:
                    r += ",\n" + " " * inden + "  "
            
            r += x
            
            if style == PsiOpts.STR_STYLE_PSITIP:
                r += " )"
                
            first = False
            
        
        if not self.aux.isempty():
            if style == PsiOpts.STR_STYLE_PSITIP:
                r += " ^ "
            else:
                r += " | "
            r += self.aux.tostring(style = style, tosort = tosort)
            
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += " )"
        else:
            r += " }"
        return r
        
    def __str__(self):
        return self.tostring(PsiOpts.settings["str_style"])
        

class RegionOp(Region):
    """A region which is the union/intersection of a list of regions."""
    
    def __init__(self, rtype, regs):
        self.rtype = rtype
        self.regs = regs
    
    def get_type(self):
        return self.rtype
    
    def copy(self):
        return RegionOp(self.rtype, [x.copy() for x in self.regs])
    
    def imp_flip(self):
        if self.get_type() == RegionType.INTER:
            self.rtype = RegionType.UNION
        elif self.get_type() == RegionType.UNION:
            self.rtype = RegionType.INTER
        for x in self.regs:
            x.imp_flip()
        return self
    
    def imp_flipped(self):
        r = self.copy()
        r.imp_flip()
        return r
    
    def universe_type(rtype):
        return RegionOp(rtype, [Region.universe()])
    
    def ispresent(self, x):
        """Return whether any variable in x appears here."""
        for x in self.regs:
            if x.ispresent(x):
                return True
        return False
    
    def rename_var(self, name0, name1):
        for x in self.regs:
            x.rename_var(name0, name1)

        
    def getaux(self):
        r = Comp.empty()
        for x in self.regs:
            r += x.getaux()
        return r
    
    def getauxi(self):
        r = Comp.empty()
        for x in self.regs:
            r += x.getauxi()
        return r
    
    
    def substitute(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), in place."""
        for x in self.regs:
            x.substitute(v0, v1)
        return self
    
    def substitute_aux(self, v0, v1):
        """Substitute variable v0 by v1 (v1 can be compound), and remove auxiliary v0, in place."""
        for x in self.regs:
            x.substitute_aux(v0, v1)
        return self

    def remove_present(self, v):
        for x in self.regs:
            x.remove_present(v)
    
    def condition(self, b):
        """Condition on random variable b, in place."""
        for x in self.regs:
            x.condition(b)
        return self
        
    def record_to(self, index):
        for x in self.regs:
            x.record_to(index)

    def pack_self(self):
        self.regs = [self.copy()]
        
        
    def iand_norename(self, other):
        for i in range(len(self.regs)):
            self.regs[i] = self.regs[i].iand_norename(other)
        return self
    
    def __iand__(self, other):
        if other.get_type() == RegionType.NORMAL:
            for i in range(len(self.regs)):
                self.regs[i] &= other
            return self
            
        if other.get_type() != RegionType.INTER:
            other = RegionOp(RegionType.INTER, [other])
        
        if self.get_type() != RegionType.INTER:
            self.pack_self()
            self.rtype = RegionType.INTER
        
        self.regs += [x.copy() for x in other.regs]
        return self
        
        
    def __and__(self, other):
        r = self.copy()
        r &= other
        return r
    
        
    def __ior__(self, other):
        if other.get_type() != RegionType.UNION:
            other = RegionOp(RegionType.UNION, [other])
        
        if self.get_type() != RegionType.UNION:
            self.pack_self()
            self.rtype = RegionType.UNION
        
        self.regs += [x.copy() for x in other.regs]
        return self
        
        
    def __or__(self, other):
        r = self.copy()
        r |= other
        return r
    
    
    def __imul__(self, other):
        for i in range(len(self.regs)):
            self.regs[i] *= other
        return self
    
    def sum_minkowski(self, other):
        if self.get_type() == RegionType.UNION:
            return RegionOp(RegionType.UNION, [a.sum_minkowski(other) for a in self.regs])
        if other.get_type() == RegionType.UNION:
            return RegionOp(RegionType.UNION, [self.sum_minkowski(a) for a in other.regs])
        
        # The following are technically wrong
        if self.get_type() == RegionType.INTER:
            return RegionOp(RegionType.INTER, [a.sum_minkowski(other) for a in self.regs])
        if other.get_type() == RegionType.INTER:
            return RegionOp(RegionType.INTER, [self.sum_minkowski(a) for a in other.regs])
        return self.copy()
    
    def implicate_entrywise(self, other, skip_simplify = False):
        tregs = []
        for x in self.regs:
            if isinstance(x, RegionOp):
                x.implicate(other, skip_simplify)
                tregs.append(x)
            else:
                tregs.append(x.implicated(other, skip_simplify))
        self.regs = tregs
        return self
    
    def implicate(self, other, skip_simplify = False):
        if self.get_type() == RegionType.INTER:
            self.implicate_entrywise(other, skip_simplify)
            return self
        
        if other.get_type() == RegionType.UNION:
            tregs = []
            for y in other.regs:
                tregs.append(self.implicated(y, skip_simplify))
            self.rtype = RegionType.INTER
            self.regs = tregs
            return self
            
        if other.get_type() == RegionType.INTER:
            for y in other.regs:
                self.implicate(y, skip_simplify)
            return self
            
        if self.get_type() == RegionType.UNION:
            self.implicate_entrywise(other, skip_simplify)
            return self
        
        return self
    
    def implicated(self, other, skip_simplify = False):
        r = self.copy()
        r.implicate(other, skip_simplify)
        return r
    
    def flatten(self):
        for i in range(len(self.regs)):
            self.regs[i] = self.regs[i].flatten()
        return self
    
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
        return self.simplified_quick().check_getaux_op_inplace(hint_pair, hint_aux)
        
    def check(self):
        """Return whether implication is true."""
        return self.check_getaux() is not None  
            
    def implies_getaux(self, other, hint_pair = None, hint_aux = None):
        """Whether self implies other, with auxiliary search result."""
        return (self <= other).check_getaux(hint_pair, hint_aux)
        
    def convexify(self, q = None):
        for x in self.regs:
            x.convexify(q)
    
    def distribute(self):
        """Expand to a single union layer.
        """
        if self.get_type() == RegionType.UNION:
            tregs = []
            for x in self.regs:
                if isinstance(x, RegionOp):
                    x.distribute()
                if x.get_type() == RegionType.UNION:
                    tregs += x.regs
                else:
                    tregs.append(x)
            self.regs = tregs
            return self
        
        if self.get_type() == RegionType.INTER:
            tregs = [Region.universe()]
            for x in self.regs:
                if isinstance(x, RegionOp):
                    x.distribute()
                if x.get_type() == RegionType.UNION:
                    tregs2 = []
                    for y in x.regs:
                        for a in tregs:
                            tregs2.append(a & y)
                    tregs = tregs2
                else:
                    tregs = [a & x for a in tregs]
            self.regs = tregs
            return self
        
        return self
            
        
    def simplify_quick(self, reg = None, zero_group = False):
        """Simplify a region in place, without linear programming. 
        Optional argument reg with constraints assumed to be true. 
        zero_group = True: group all nonnegative terms as a single inequality.
        """
        #self.distribute()
            
        for x in self.regs:
            x.simplify_quick(reg, zero_group)
        
        if self.get_type() == RegionType.INTER or self.get_type() == RegionType.UNION:
            tregs = []
            for x in self.regs:
                if x.get_type() == self.get_type():
                    tregs += x.regs
                else:
                    tregs.append(x)
            self.regs = tregs
            
        return self
        
        
    def simplify(self, reg = None, zero_group = False):
        """Simplify a region in place. 
        Optional argument reg with constraints assumed to be true. 
        zero_group = True: group all nonnegative terms as a single inequality.
        """
        #self.distribute()
        for x in self.regs:
            x.simplify(reg, zero_group)
        return self
        
    def eliminate(self, w, reg = None, toreal = False):
        for x in self.regs:
            x.eliminate(w, reg, toreal)
        
    def eliminate_quick(self, w, reg = None, toreal = False):
        for x in self.regs:
            x.eliminate_quick(w, reg, toreal)
        
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
        
        r = ""
        interstr = ""
        if self.get_type() == RegionType.UNION:
            if style == PsiOpts.STR_STYLE_PSITIP:
                interstr = "|"
            else:
                interstr = "OR"
        if self.get_type() == RegionType.INTER:
            if style == PsiOpts.STR_STYLE_PSITIP:
                interstr = "&"
            else:
                interstr = "AND"
        
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += " " * inden + "(\n"
        else:
            r += " " * inden + "{\n"
        
        r += ("\n" + " " * inden + " " + interstr + "\n").join([
                x.tostring(style = style, tosort = tosort, lhsvar = lhsvar, inden = inden + 2) 
                for x in self.regs])
        if style == PsiOpts.STR_STYLE_PSITIP:
            r += "\n" + " " * inden + ")"
        else:
            r += "\n" + " " * inden + "}"
        return r
            
            
# Shortcuts
    
def rv(*args):
    """Random variable"""
    r = Comp.empty()
    for a in args:
        r += Comp.rv(a)
    return r
    
def rv_array(name, st, en):
    """Random variable"""
    return Comp.array(name, st, en)
    
def real(*args):
    """Real variable"""
    if len(args) == 1:
        return Expr.real(args[0])
    return tuple([Expr.real(a) for a in args])

def empty():
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
    if isinstance(x, Comp):
        return Expr.H(x)
    return Expr([(x.copy(), 1.0)])
    
def Hc(x, z):
    """Conditional entropy. 
    Hc(X, Z) is the same as H(X | Z)
    """
    return Expr.Hc(x, z)

def Ic(x, y, z):
    """Conditional mutual information. 
    Ic(X, Y, Z) is the same as I(X & Y | Z)
    """
    return Expr.Ic(x, y, z)

def indep(*args):
    """Return Region where the arguments are independent."""
    r = Region.universe()
    for i in range(1, len(args)):
        r &= Expr.I(sum(args[:i]),args[i]) == 0
    return r

def indep_across(*args):
    """Take several arrays, return Region where entries are independent across dimension."""
    n = max([a.size() for a in args])
    vec = [sum([a[i] for a in args if i < a.size()]) for i in range(n)]
    return indep(*vec)

def equiv(*args):
    """Return Region where the arguments contain the same information."""
    r = Region.universe()
    for i in range(1, len(args)):
        r &= (Expr.Hc(args[i], args[0]) == 0) & (Expr.Hc(args[0], args[i]) == 0)
    return r

def markov(*args):
    """Return Region where the arguments form a Markov chain."""
    r = Region.universe()
    for i in range(2, len(args)):
        r &= Expr.Ic(sum(args[:i-1]),args[i],args[i-1]) == 0
    return r

def emin(*args):
    """Return the minimum of the expressions."""
    R = real("min(" + ",".join([str(x) for x in args]) + ")")
    r = universe()
    for x in args:
        r &= R <= x
    return r.maximum(R)

def emax(*args):
    """Return the maximum of the expressions."""
    R = real("max(" + ",".join([str(x) for x in args]) + ")")
    r = universe()
    for x in args:
        r &= R >= x
    return r.minimum(R)


def meet(*args):
    """Gacs-Korner common part. 
    Peter Gacs and Janos Korner. Common information is far less than mutual information.
    Problems of Control and Information Theory, 2(2):149-162, 1973.
    """
    U = rv("^".join([a.tostring(add_braket = True) for a in args]))
    V = rv("V")
    r = Region.universe()
    r2 = Region.universe()
    for a in args:
        r &= Expr.Hc(U, a) == 0
        r2 &= (Expr.Hc(V, a) == 0)
    r = (r & (Expr.Hc(V, U) == 0)).implicated(r2, skip_simplify = True).forall(V)
    return Comp.rv_reg(U, r, reg_det = True)


def mss(x, y):
    """Minimal sufficient statistic of x about y."""
    
    U = rv("MSS(" + str(x) + ";" + str(y) + ")")
    V = rv("V")
    r = (Expr.Hc(U, x) == 0) & (Expr.Ic(x, y, U) == 0)
    r2 = (Expr.Hc(V, x) == 0) & (Expr.Ic(x, y, V) == 0)
    r = (r & (Expr.Hc(U, V) == 0)).implicated(r2, skip_simplify = True).forall(V)
    return Comp.rv_reg(U, r, reg_det = True)


def sfrl(x, y, gap = None):
    """Strong functional representation lemma. 
    Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
    applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978.
    """
    U = rv(y.tostring(add_braket = True) + "%" + x.tostring(add_braket = True))
    r = (Expr.Hc(y, x + U) == 0) & (Expr.I(x, U) == 0)
    if gap is not None:
        r &= Expr.Ic(x, U, y) <= gap
    return Comp.rv_reg(U, r, reg_det = False)


def total_corr(x):
    """Total correlation. 
    Watanabe S (1960). Information theoretical analysis of multivariate correlation, 
    IBM Journal of Research and Development 4, 66-82. 
    e.g. total_corr(X & Y & Z | W)
    """
    if isinstance(x, Comp):
        return Expr.H(x)
    return sum([Expr.Hc(a, x.z) for a in x.x]) - Expr.Hc(sum(x.x), x.z)


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


def gacs_korner(x):
    """Gacs-Korner common information. 
    Peter Gacs and Janos Korner. Common information is far less than mutual information.
    Problems of Control and Information Theory, 2(2):149-162, 1973. 
    e.g. gacs_korner(X & Y & Z | W)
    """
    U = rv("U")
    R = real("GK" + str(x)[1:])
    r = universe()
    for a in x.x:
        r &= Expr.Hc(U, a+x.z) == 0
    r &= R <= Expr.Hc(U, x.z)
    return r.exists(U).maximum(R)


def wyner_ci(x):
    """Wyner's common information. 
    A. D. Wyner. The common information of two dependent random variables.
    IEEE Trans. Info. Theory, 21(2):163-179, 1975. 
    e.g. wyner_ci(X & Y & Z | W)
    """
    U = rv("U")
    R = real("WCI" + str(x)[1:])
    r = indep(*(x.x)).conditioned(U + x.z)
    r &= R >= Expr.Ic(U, sum(x.x), x.z)
    return r.exists(U).minimum(R)


def exact_ci(x):
    """Common entropy (one-shot exact common information). 
    G. R. Kumar, C. T. Li, and A. El Gamal. Exact common information. In Information
    Theory (ISIT), 2014 IEEE International Symposium on, 161-165. IEEE, 2014. 
    e.g. exact_ci(X & Y & Z | W)
    """
    U = rv("U")
    R = real("ECI" + str(x)[1:])
    r = indep(*(x.x)).conditioned(U + x.z)
    r &= R >= Expr.Hc(U, x.z)
    return r.exists(U).minimum(R)


def H_nec(x):
    """Necessary conditional entropy. 
    Cuff, P. W., Permuter, H. H., & Cover, T. M. (2010). Coordination capacity.
    IEEE Transactions on Information Theory, 56(9), 4181-4206. 
    e.g. H_nec(X + Y | W)
    """
    U = rv("U")
    R = real("Hnec" + str(x)[1:])
    r = markov(x.z, U, x.x[0]) & (Expr.Hc(U, x.x[0]) == 0)
    r &= R >= Expr.Hc(U, x.z)
    return r.exists(U).minimum(R)


def excess_fi(x, y):
    """Excess functional information. 
    Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
    applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978. 
    e.g. excess_fi(X, Y)
    """
    U = rv("U")
    R = real("EFI(" + str(x) + ";" + str(y) + ")")
    r = indep(U, x)
    r &= R >= Expr.Hc(y, U) - Expr.I(x, y)
    return r.exists(U).minimum(R)


def mutual_dep(x):
    """Mutual dependence. 
    Csiszar, Imre, and Prakash Narayan. "Secrecy capacities for multiple terminals." 
    IEEE Transactions on Information Theory 50, no. 12 (2004): 3047-3061.
    """
    n = len(x.x)
    if n <= 2:
        return I(x)
    R = real("MD" + str(x)[1:])
    Hall = Expr.Hc(sum(x.x), x.z)
    r = universe()
    for part in IUtil.enum_partition(n):
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


def intrinsic_mi(x):
    """Intrinsic mutual information. 
    U. Maurer and S. Wolf. "Unconditionally secure key agreement and the intrinsic 
    conditional information." IEEE Transactions on Information Theory 45.2 (1999): 499-514.
    e.g. intrinsic_mi(X & Y | Z)
    """
    U = rv("U")
    R = real("IMI" + str(x)[1:])
    r = markov(sum(x.x), x.z, U) & (R >= mutual_dep(Term(x.x, U)))
    return r.exists(U).minimum(R)

