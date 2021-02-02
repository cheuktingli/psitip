Psitip
======

Python Symbolic Information Theoretic Inequality Prover

Psitip is a computer algebra system for information theory written in Python. Random variables, expressions and regions are objects in Python that can be manipulated easily. Moreover, it supports a versatile deduction system for automated theorem proving. Psitip supports features such as:

- Proving linear information inequalities via the linear programming method by Zhang and Yeung (see `References`_), which was implemented in the ITIP software developed by Yeung and Yan ( http://user-www.ie.cuhk.edu.hk/~ITIP/ ). 

- `Automated inner and outer bounds`_ in network information theory (e.g. see the automated proof of Gel'fand-Pinsker theorem in the example below).

- `Numerical optimization`_ over distributions, and evaluation of rate regions involving auxiliary random variables.

- `Fourier-Motzkin elimination`_.

- `Discover inequalities`_ via the convex hull method for polyhedron projection [Lassez-Lassez 1991].

- Non-Shannon-type inequalities.

- Drawing `Information diagrams`_.

- User-defined information quantities (e.g. see Wyner's CI and Gács-Körner CI in the example below). 

- `Bayesian network optimization`_. Psitip is optimized for random variables following a Bayesian network structure, which can greatly improve performance.


Examples:

.. code-block:: python

    from psitip import *
    PsiOpts.setting(solver = "pyomo.glpk")

    X, Y, Z, W, U, M, S = rv("X", "Y", "Z", "W", "U", "M", "S") # declare random variables


    # ******** Basics ********

    print(bool(H(X) + I(Y & Z | X) >= I(Y & Z))) # H(X)+I(Y;Z|X)>=I(Y;Z) is True
    print(markov(X, Y, Z).implies(H(X | Y) <= H(X | Z))) # returns True
    print((H(X) == I(X & Y) + H(X | Y+Z)).implies(markov(X, Y, Z))) # returns True
    print((H(X+Y) - H(X) - H(Y)).simplified()) # gives -I(Y;X)

    # Defining Wyner's common information [Wyner 1975]
    wci = markov(X, U, Y).exists(U).minimum(I(U & X+Y))

    # Defining Gács-Körner common information [Gács-Körner 1973]
    gkci = ((H(U|X) == 0) & (H(U|Y) == 0)).exists(U).maximum(H(U))

    print(bool(emin(H(X), H(Y)) >= wci)) # min(H(X),H(Y)) >= Wyner is True
    print(bool(wci >= I(X & Y))) # Wyner >= I(X;Y) is True
    print(bool(I(X & Y) >= gkci)) # I(X;Y) >= Gács-Körner is True

    # The meet or Gács-Körner common part [Gács-Körner 1973] between X and Y
    # is a function of the GK common part between X and (Y,Z)
    print(bool(H(meet(X, Y) | meet(X, Y + Z)) == 0)) # returns True

    # The condition "there exists Y independent of (X,Z) such that 
    #  X-Y-Z forms a Markov chain" can be simplified to "X,Z independent"
    print((markov(X, Y, Z) & indep(X+Z, Y)).exists(Y).simplified()) # gives I(X;Z)==0



    # ******** Automated inner and outer bound for network information theory ********

    # Channel with noncausal state information at encoder
    R = real("R")
    model = CodingModel()
    model.add_node(M+S, X) # Encoder produces X given message M and state S
    model.add_edge(X+S, Y) # Channel output Y depends on X, S
    model.add_node(Y, M)   # Decoder produces M given Y
    model.set_rate(M, R)   # Rate of M is R

    # Automatic inner bound via [Lee-Chung 2015], recovers [Gel'fand-Pinsker 1980]
    r = model.get_inner()
    print(r)

    # Automated converse proof, print auxiliary random variables
    print((model.get_outer() >> r).check_getaux())



    # ******** Non-Shannon-type Inequalities ********

    # Zhang-Yeung inequality [Zhang-Yeung 1998] cannot be proved by Shannon-type inequalities
    print(bool(2*I(Z&W) <= I(X&Y) + I(X & Z+W) + 3*I(Z&W | X) + I(Z&W | Y))) # returns False

    # Using copy lemma [Zhang-Yeung 1998], [Dougherty-Freiling-Zeger 2011]
    with copylem().assumed():
        
        # Prove Zhang-Yeung inequality
        print(bool(2*I(Z&W) <= I(X&Y) + I(X & Z+W) + 3*I(Z&W | X) + I(Z&W | Y))) # returns True


    # State the copy lemma
    r = eqdist([X, Y, U], [X, Y, Z]) & markov(Z+W, X+Y, U)

    # Automatically discover non-Shannon-type inequalities using copy lemma
    print(r.discover(mi_cells(X, Y, Z, W)))


About
~~~~~

Author: Cheuk Ting Li ( https://www.ie.cuhk.edu.hk/people/ctli.shtml ). The source code of Psitip is released under the GNU General Public License v3.0 ( https://www.gnu.org/licenses/gpl-3.0.html ). The author would like to thank Raymond W. Yeung and Chandra Nair for their invaluable comments.

If you find Psitip useful in your research, please consider citing it as:

\C. T. Li, "An Automated Theorem Proving Framework for Information-Theoretic Results," arXiv preprint, available: https://arxiv.org/pdf/2101.12370.pdf , 2021.


WARNING
~~~~~~~

This program comes with ABSOLUTELY NO WARRANTY. This program is a work in progress, and bugs are likely to exist. The deduction system is incomplete, meaning that it may fail to prove true statements (as expected in most automated deduction programs). On the other hand, declaring false statements to be true should be less common. If you encounter a false accept in Psitip, please let the author know.


Installation
~~~~~~~~~~~~

Download `psitip.py <https://raw.githubusercontent.com/cheuktingli/psitip/master/psitip.py>`_ and place it in the same directory as your code, or open an IPython shell in the same directory as psitip.py. The file `test.py <https://raw.githubusercontent.com/cheuktingli/psitip/master/test.py>`_ contains examples of usages of Psitip. Use :code:`from psitip import *` in your code to import all functions in psitip.

Python 3 and numpy are required to run psitip. It also requires at least one of the following for sparse linear programming:

- **Pyomo** (https://github.com/Pyomo/pyomo). Recommended. Requires GLPK (installed separately) or another solver.
- **PuLP** (https://github.com/coin-or/pulp). Can use GLPK (installed separately), CBC (https://github.com/coin-or/Cbc , provided with PuLP, not recommended) or another solver.
- **GLPK** (https://www.gnu.org/software/glpk/). Recommended. An external solver to be used with PuLP or Pyomo. Can be installed using Conda (see https://anaconda.org/conda-forge/glpk ).
- **SciPy** (https://www.scipy.org/). Not recommended for problems with more than 8 random variables.

See the Solver section for details.


Other optional dependencies:

- **Pycddlib** (https://github.com/mcmtroffaes/pycddlib/), a Python wrapper for Komei Fukuda's cddlib (https://people.inf.ethz.ch/fukudak/cdd_home/). Needed only for the convex hull method for polyhedron projection.
- **PyTorch** (https://pytorch.org/). Needed only for numerical optimization over probability distributions.
- **Matplotlib** (https://matplotlib.org/). Required for drawing information diagrams.
- **Graphviz** (https://graphviz.org/). A Python binding of Graphviz is required for drawing Bayesian networks and communication network model.




Solver
~~~~~~

The default solver is Scipy, though it is highly recommended to switch to another solver, e.g.:

.. code-block:: python

    from psitip import *
    PsiOpts.setting(solver = "pulp.glpk")
    PsiOpts.setting(solver = "pyomo.glpk")
    PsiOpts.setting(solver = "pulp.cbc") # Not recommended

PuLP supports a wide range of solvers (see https://coin-or.github.io/pulp/technical/solvers.html ). Use the following line to set the solver to any supported solver:

.. code-block:: python

    PsiOpts.setting(pulp_solver = pulp.solvers.GLPK(msg = 0)) # Or another solver

For Pyomo (see https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers ), use the following line (replace ??? with the desired solver):

.. code-block:: python

    PsiOpts.setting(solver = "pyomo.???")

WARNING: It is possible for inaccuracies in the solver to result in wrong output in Psitip. Try switching to another solver if a problem is encountered.


Basics
~~~~~~

The following classes and functions are in the :code:`psitip` module. Use :code:`from psitip import *` to avoid having to type :code:`psitip.something` every time you use one of these functions.

- **Random variables** are declared as :code:`X = rv("X")`. The name "X" passed to "rv" must be unique. Variables with the same name are treated as being the same. The return value is a :code:`Comp` object (compound random variable).

 - As a shorthand, you may declare multiple random variables in the same line as :code:`X, Y = rv("X", "Y")`.

- The joint random variable (X,Y) is expressed as :code:`X + Y` (a :code:`Comp` object).

- **Entropy** H(X) is expressed as :code:`H(X)`. **Conditional entropy** H(X|Y) is expressed as :code:`H(X | Y)`. **Conditional mutual information** I(X;Y|Z) is expressed as :code:`I(X & Y | Z)`. The return values are :code:`Expr` objects (expressions).

- **Real variables** are declared as :code:`a = real("a")`. The return value is an :code:`Expr` object (expression).

- Expressions can be added and subtracted with each other, and multiplied and divided by scalars, e.g. :code:`I(X + Y & Z) * 3 - a * 4`.
 
 - While Psitip can handle affine expressions like :code:`H(X) + 1` (i.e., adding or subtracting a constant), affine expressions are unrecommended as they are prone to numerical error in the solver.

 - While expressions can be multiplied and divided by each other (e.g. :code:`H(X) * H(Y)`), most symbolic capabilities are limited to linear and affine expressions. **Numerical only:** non-affine expressions can be used in concrete models, and support automated gradient for numerical optimization tasks, but do not support most symbolic capabilities for automated deduction.

 - We can take power (e.g. :code:`H(X) ** H(Y)`) and logarithm (using the :code:`elog` function, e.g. :code:`elog(H(X) + H(Y))`) of expressions. **Numerical only:** non-affine expressions can be used in concrete models, and support automated gradient for numerical optimization tasks, but do not support most symbolic capabilities for automated deduction.

- When two expressions are compared (using :code:`<=`, :code:`>=` or :code:`==`), the return value is a :code:`Region` object (not a :code:`bool`). The :code:`Region` object represents the set of distributions where the condition is satisfied. E.g. :code:`I(X & Y) == 0`, :code:`H(X | Y) <= H(Z) + a`.
 
 - While Psitip can handle general affine and half-space constraints like :code:`H(X) <= 1` (i.e., comparing an expression with a nonzero constant, or comparing affine expressions), they are unrecommended as they are prone to numerical error in the solver.
 
 - While Psitip can handle strict inequalities like :code:`H(X) > H(Y)`, strict inequalities are unrecommended as they are prone to numerical error in the solver.

- The **intersection** of two regions (i.e., the region where the conditions in both regions are satisfied) can be obtained using the ":code:`&`" operator. E.g. :code:`(I(X & Y) == 0) & (H(X | Y) <= H(Z) + a)`.

 - To build complicated regions, it is often convenient to declare :code:`r = universe()` (:code:`universe()` is the region without constraints), and add constraints to :code:`r` by, e.g., :code:`r &= I(X & Y) == 0`.

- The **union** of two regions can be obtained using the ":code:`|`" operator. E.g. :code:`(I(X & Y) == 0) | (H(X | Y) <= H(Z) + a)`. (Note that the return value is a :code:`RegionOp` object, a subclass of :code:`Region`.)

- The **complement** of a region can be obtained using the ":code:`~`" operator. E.g. :code:`~(H(X | Y) <= H(Z) + a)`. (Note that the return value is a :code:`RegionOp` object, a subclass of :code:`Region`.)

- The **Minkowski sum** of two regions (with respect to their real variables) can be obtained using the ":code:`+`" operator.

- A region object can be converted to :code:`bool`, returning whether the conditions in the region can be proved to be true (using Shannon-type inequalities). E.g. :code:`bool(H(X) >= I(X & Y))`.

- **Logical implication**. To test whether the conditions in region :code:`r1` implies the conditions in region :code:`r2` (i.e., whether :code:`r1` is a subset of :code:`r2`), use :code:`r1.implies(r2)` (which returns :code:`bool`). E.g. :code:`(I(X & Y) == 0).implies(H(X + Y) == H(X) + H(Y))`.

- The constraint that X, Y, Z are **mutually independent** is expressed as :code:`indep(X, Y, Z)` (a :code:`Region` object). The function :code:`indep` can take any number of arguments.

 - The constraint that X, Y, Z are mutually conditionally independent given W is expressed as :code:`indep(X, Y, Z).conditioned(W)`.

- The constraint that X, Y, Z forms a **Markov chain** is expressed as :code:`markov(X, Y, Z)` (a :code:`Region` object). The function :code:`markov` can take any number of arguments.

- The constraint that X, Y, Z are **informationally equivalent** (i.e., contain the same information) is expressed as :code:`equiv(X, Y, Z)` (a :code:`Region` object). The function :code:`equiv` can take any number of arguments. Note that :code:`equiv(X, Y)` is the same as :code:`(H(X|Y) == 0) & (H(Y|X) == 0)`.

- The :code:`rv_seq` method constructs an array of random variables. For example, :code:`X = rv_seq("X", 10)` gives a :code:`Comp` object consisting of X0, X1, ..., X9.

 - An array can be used by itself to represent the joint random variable of the variables in the array. For example, :code:`H(X)` gives H(X0,...,X9).

 - An array can be indexed using :code:`X[i]` (returns a :code:`Comp` object). The slice notation in Python also works, e.g., :code:`X[5:-1]` gives X5,X6,X7,X8 (a :code:`Comp` object).

 - The region where the random variables in the array are mutually independent can be given by :code:`indep(*X)`. The region where the random variables form a Markov chain can be given by :code:`markov(*X)`. 

- :code:`Expr` and :code:`Region` objects have a :code:`simplify()` method, which simplify the expression/region in place. The :code:`simplified()` method returns the simplified expression/region without modifying the object. For example, :code:`(H(X+Y) - H(X) - H(Y)).simplified()` gives :code:`-I(Y & X)`.

 - Note that calling :code:`Region.simplify()` can take some time for the detection of redundant constraints. Use :code:`Region.simplify_quick()` instead to skip this step.

- Use :code:`str(x)` to convert :code:`x` (a :code:`Comp`, :code:`Expr` or :code:`Region` object) to string. The :code:`tostring` method of :code:`Comp`, :code:`Expr` and :code:`Region` provides more options. For example, :code:`r.tostring(tosort = True, lhsvar = R)` converts the region :code:`r` to string, sorting all terms and constraints, and putting the real variable :code:`R` to the left hand side of all expressions (and the rest to the right).



Advanced
~~~~~~~~

- **Existential quantification** is represented by the :code:`exists` method of :code:`Region` (which returns a :code:`Region`). For example, the condition "there exists auxiliary random variable U such that R <= I(U;Y) - I(U;S) and U-(X,S)-Y forms a Markov chain" (as in Gelfand-Pinsker theorem) is represented by:

  .. code-block:: python

    ((R <= I(U & Y) - I(U & S)) & markov(U, X+S, Y)).exists(U) 

 - Calling :code:`exists` on real variables will cause the variable to be eliminated by Fourier-Motzkin elimination (see Fourier-Motzkin elimination section). Currently, calling :code:`exists` on real variables for a region obtained from material implication is not supported.

 - Calling :code:`exists` on random variables will cause the variable to be marked as auxiliary (dummy).

 - Calling :code:`exists` on random variables with the option :code:`toreal = True` will cause all information quantities about the random variables to be treated as real variables, and eliminated using Fourier-Motzkin elimination. Those random variables will be absent in the resultant region (not even as auxiliary random variables). E.g.:

  .. code-block:: python

    (indep(X+Z, Y) & markov(X, Y, Z)).exists(Y, toreal = True)

  gives :code:`{ I(Z;X) == 0 }`. Note that using :code:`toreal = True` can be extremely slow if the number of random variables is more than 5, and may cause false accepts (i.e., declaring a false inequality to be true) since only Shannon-type inequalities are enforced.

- **Material implication** between :code:`Region` is denoted by the operator :code:`>>`, which returns a :code:`Region` object. The region :code:`r1 >> r2` represents the condition that :code:`r2` is true whenever :code:`r1` is true. Note that :code:`r1 >> r2` is equivalent to :code:`~r1 | r2`, and :code:`r1.implies(r2)` is equivalent to :code:`bool(r1 >> r2)`.

 - **Material equivalence** is denoted by the operator :code:`==`, which returns a :code:`Region` object. The region :code:`r1 == r2` represents the condition that :code:`r2` is true if and only if :code:`r1` is true.

- **Universal quantification** is represented by the :code:`forall` method of :code:`Region` (which returns a :code:`Region`). This is usually called after the implication operator :code:`>>`. For example, the condition "for all U such that U-X-(Y1,Y2) forms a Markov chain, we have I(U;Y1) >= I(U;Y2)" (less noisy broadcast channel [Körner-Marton 1975]) is represented by:

  .. code-block:: python

    (markov(U,X,Y1+Y2) >> (I(U & Y1) >= I(U & Y2))).forall(U)

 - Currently, calling :code:`forall` on real variables is not supported.

- Existential/universal quantification over marginal distributions is represented by the :code:`marginal_exists` or :code:`marginal_forall` method of :code:`Region`. This is usually used in channel coding settings where only the marginal distribution of the input can be altered (but not the channel). This is sometimes followed by the :code:`convexified()` (use :code:`convexified(forall = True)` for :code:`marginal_forall`) method to add a time sharing random variable, for example, for the less noisy broadcast channel:

  .. code-block:: python

    (markov(U,X,Y1+Y2) >> (I(U & Y1) >= I(U & Y2))
        ).forall(U).marginal_forall(X).convexified(forall = True)


- The function call :code:`r.substituted(x, y)` (where :code:`r` is an :code:`Expr` or :code:`Region`, and :code:`x`, :code:`y` are either both :code:`Comp` or both :code:`Expr`) returns an expression/region where all appearances of :code:`x` in :code:`r` are replaced by :code:`y`.

 - Call :code:`substituted_aux` instead of :code:`substituted` to stop treating :code:`x` as an auxiliary in the region :code:`r` (useful in substituting a known value of an auxiliary).

- **Minimization / maximization** over an expression subject to the constraints in a region is represented by the :code:`minimum` / :code:`maximum` method of :code:`Region` respectively (which returns an :code:`Expr` object). This method usually follows an :code:`exists` call to mark the dummy variables in the optimization. For example, Wyner's common information [Wyner 1975] is represented by:

  .. code-block:: python

    markov(X, U, Y).exists(U).minimum(I(U & X+Y))

- It is simple to define new information quantities. For example, to define the information bottleneck [Tishby-Pereira-Bialek 1999]:

  .. code-block:: python

    def info_bot(X, Y, t):
        U = rv("U")
        return (markov(U, X, Y) & (I(Y & U) >= t)).exists(U).minimum(I(X & U))
    
    X, Y = rv("X", "Y")
    t1, t2 = real("t1", "t2")
    # Check that info bottleneck is non-decreasing
    print(bool((t1 <= t2) >> (info_bot(X, Y, t1) <= info_bot(X, Y, t2)))) # True
    

- The **minimum / maximum** of two (or more) :code:`Expr` objects is represented by the :code:`emin` / :code:`emax` function respectively. For example, :code:`bool(emin(H(X), H(Y)) >= I(X & Y))` returns True.

- The **absolute value** of an :code:`Expr` object is represented by the :code:`abs` function. For example, :code:`bool(abs(H(X) - H(Y)) <= H(X) + H(Y))` returns True.

- While one can check the conditions in :code:`r` (a :code:`Region` object) by calling :code:`bool(r)`, to also obtain the auxiliary random variables, instead call :code:`r.check_getaux()`, which returns a list of pairs of :code:`Comp` objects that gives the auxiliary random variable assignments (returns None if :code:`bool(r)` is False). For example:

  .. code-block:: python

    (markov(X, U, Y).exists(U).minimum(I(U & X+Y)) <= H(X)).check_getaux()

  returns :code:`[(U, X)]`.

 - If branching is required (e.g. for union of regions), :code:`check_getaux` may give a list of lists of pairs, where each list represents a branch. For example:

  .. code-block:: python

    (markov(X, U, Y).exists(U).minimum(I(U & X+Y))
        <= emin(H(X),H(Y))).check_getaux()

  returns :code:`[[(U, X)], [(U, X+Y)], [(U, Y)]]`.

- The **meet** or **Gács-Körner common part** [Gács-Körner 1973] between X and Y is denoted as :code:`meet(X, Y)` (a :code:`Comp` object).

- The **minimal sufficient statistic** of X about Y is denoted as :code:`mss(X, Y)` (a :code:`Comp` object).

- The random variable given by the **strong functional representation lemma** [Li-El Gamal 2018] applied on X, Y (:code:`Comp` objects) with a gap term logg (:code:`Expr` object) is denoted as :code:`sfrl_rv(X, Y, logg)` (a :code:`Comp` object). If the gap term is omitted, this will be the ordinary functional representation lemma [El Gamal-Kim 2011].



Information diagrams
~~~~~~~~~~~~~~~~~~~~

The :code:`venn` method of :code:`Comp`, :code:`Expr` and :code:`Region` draws the information diagram of that object. The :code:`venn` method takes any number of arguments (:code:`Comp`, :code:`Expr` or :code:`Region`) which are drawn together. For :code:`Region.venn`, only the nonzero cells of the region will be drawn (the others are in black). The ordering of the random variables is decided by the first :code:`Comp` argument (or automatically if no :code:`Comp` argument is supplied). To draw a Gray-coded table instead of a Venn diagram, use :code:`table` instead of :code:`venn`. The methods :code:`venn` and :code:`table` take a :code:`style` argument, which is a string with the following options (multiple options are separated by ","):

- :code:`blend`: Blend the colors in overlapping areas. Default for :code:`venn`.

- :code:`hatch`: Use hatch instead of fill.

- :code:`pm`: Use +/- instead of numbers.

- :code:`notext`: Hide the numbers.

- :code:`nosign`: Hide the signs of each cell (+/-) on the bottom of each cell.

- :code:`nolegend`: Hide the legends.

Example:

.. code-block:: python

    from psitip import *
    X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
    (X+Y+Z).venn(H(X), H(Y) - H(Z))

.. image:: https://raw.githubusercontent.com/cheuktingli/psitip/master/doc/img/Figure_1.png



.. code-block:: python

    (markov(X, Y, Z, W) & (H(W | Z) == 0)).venn(H(X), I(Y & W), style = "hatch,pm")

.. image:: https://raw.githubusercontent.com/cheuktingli/psitip/master/doc/img/Figure_2.png



.. code-block:: python

    # Entropy, total correlation [Watanabe 1960] and dual total correlation [Han 1978]
    # use Branko Grunbaum's Venn diagram for 5 variables
    (X+Y+Z+W+U).venn(H(X+Y+Z+W+U), total_corr(X&Y&Z&W&U), 
                    dual_total_corr(X&Y&Z&W&U), style = "nolegend")

.. image:: https://raw.githubusercontent.com/cheuktingli/psitip/master/doc/img/Figure_3.png



Numerical optimization
~~~~~~~~~~~~~~~~~~~~~~

Psitip supports numerical optimization on distributions of random variables. While :code:`Comp` are abstract random variables without information on their distributions, you can use a :code:`ConcModel` object (concrete model) to assign joint distributions to random variables.

**Caution:** In order to use the numerical functions of Psitip, the cardinality of random variables must be specified using :code:`set_card`, e.g. :code:`X = rv("X").set_card(2)`. For numerical optimization, add the line :code:`PsiOpts.setting(istorch = True)` at the beginning to enable PyTorch.


Concrete distributions
----------------------

A (joint/conditional) distribution is stored as a :code:`ConcDist` (concrete distribution) object. It is constructed as :code:`ConcDist(a, num_in)`, where :code:`a` is the probability table (a :code:`numpy.array` or :code:`torch.Tensor`), and :code:`num_in` is the number of random variables to be conditioned on. For example, if X -> Y is a Z-channel, P(Y|X) can be represented as :code:`ConcDist(array([[1.0, 0.0], [0.1, 0.9]]), num_in = 1)`. Note that for P(Y[0],...,Y[m-1] | X[0],...,X[n-1]), the number of dimensions of :code:`a` is n+m, where the first n dimensions correspond to X[0],...,X[n-1], and the remaining m dimensions correspond to Y[0],...,Y[m-1].

- If :code:`p` is P(Y|X), and :code:`q` is P(Z|X), then P(Y,Z|X) (assuming Y,Z are conditionally independent given X) is :code:`p * q`.

- If :code:`p` is P(Y|X), and :code:`q` is P(Z|Y), then P(Z|X) is :code:`p @ q`.

- If :code:`p` is P(Y|X), and :code:`q` is P(Z|Y), then P(Y,Z|X) is :code:`p.semidirect(q)`.

- If :code:`p` is P(Y0,...,Y5|X), then P(Y2,Y4|X) is :code:`p.marginal(2,4)`.

- If :code:`p` is P(Y|X), then P(Y|X=x) is :code:`p.given(x)`.

- If :code:`p` is P(X), then E[f(X)] is :code:`p.mean(f)`. :code:`f` is a function, :code:`numpy.array` or :code:`torch.Tensor`. If f is a function, the number of arguments must match the number of dimensions (random variables) of the joint distribution. If f is an array or tensor, shape must match the shape of the distribution.

 - In both :code:`given` and :code:`mean`, the values of X are assumed to range from 0 to the cardinality of X minus 1. If X does not take these values, manual conversion is needed between the values of X and indices between 0 and the cardinality of X minus 1. 


Concrete model
--------------

Letting :code:`P = ConcModel()`, we have the following operations:

- :code:`P[X]` for a random variable (:code:`Comp`) :code:`X` gives the distribution of X (:code:`ConcDist`). Use :code:`P[X] = p` to set the distribution of X (where :code:`p` is :code:`ConcDist`, :code:`numpy.array` or :code:`torch.Tensor`). Use :code:`P[X+Y | Z+W]` for the conditional distribution P(X,Y|Z,W).

 - Random variables must be added to the model in the order they are generated. E.g., :code:`P[X] = p1; P[Y|X] = p2; P[Z|Y] = p3`. If Z is added as :code:`P[Z|Y] = p3`, it is assumed to be conditionally independent of all previously added random variables given Y.

 - Use :code:`P[Y|X] = "var"` to specify that P(Y|X) is a variable that can be optimized over. Use :code:`P[Y|X] = "var_rand"` to randomize its initial value (otherwise the initial value is uniform, which may not be desirable for some optimization tasks).

- :code:`P[a]` for an expression (:code:`Expr`) :code:`a` gives the value of :code:`a` under the distribution in :code:`P`. E.g. :code:`P[I(X & Y) - H(Z | Y)]`.

 - Note that :code:`P[a]` is read-only except when :code:`a` is a single real variable. In that case, :code:`P[a]=1.0` sets the value of the real variable to 1.0. Use :code:`P[a]=ConcReal(1.0, lbound = 0.0, ubound = 10.0, isvar = True)` to set :code:`a` to be a variable that can be optimized over, with lower bound lbound and upper bound ubound.

- :code:`P[r]` for a region (:code:`Region`) :code:`r` gives the truth value of the conditions in :code:`r`.

- :code:`P.venn()` draws the information diagram of the random variables.

- :code:`P.graph()` gives the Bayesian network of the random variables as a Graphviz graph.


Useful functions
----------------

Letting :code:`X, Y, Z = rv("X", "Y", "Z")`,

- :code:`X.prob(x)` (an :code:`Expr` object) gives the probability P(X=x). For joint probability, :code:`(X+Y).prob(x, y)` gives P(X=x, Y=y).

 - :code:`X.pmf()` gives the whole probability vector (an :code:`ExprArray` object). :code:`(X+Y+Z).pmf()` gives the probability tensor. :code:`(X|Y).pmf()` gives the transition matrix. :code:`ExprArray` objects support basic numpy-array-like operations such as +, -, \*, @, dot, transpose, trace, diag, reshape.

 - Note that :code:`X.prob(x)` gives an abstract expression (:code:`Expr`). To evaluate it on a concrete model :code:`P`, use :code:`P[X.prob(x)]` as mentioned in the `Concrete model`_ section. This can also be used on :code:`ExprArray`, e.g. :code:`P[X.pmf()]` is the same as :code:`P[X]`.

- :code:`X.mean(f)` (an :code:`Expr` object) gives the expectation E[f(X)]. For joint probability, :code:`(X+Y).mean(f)` gives E[f(X, Y)]. The parameter :code:`f` follows the same requirements as :code:`ConcDist.mean` above.

- For other functions e.g. divergence, Rényi entropy, maximal correlation, varentropy, see `Real-valued information quantities`_ and `Real-valued information quantities (numerical only)`_.

- For general user-defined functions, use :code:`Expr.fcn` to wrap any function mapping a :code:`ConcModel` to a number as an :code:`Expr`. E.g. the Hamming distortion is given by :code:`Expr.fcn(lambda P: P[X+Y].mean(lambda x, y: float(x != y)))`. For optimization using PyTorch, the return value should be a scalar :code:`torch.Tensor` with gradient information.


Optimization
------------

The function :code:`ConcModel.minimize(expr, vs, reg)` (or :code:`maximize`) takes 3 arguments: :code:`expr` (:code:`Expr` object) is the optimization objective, :code:`vs` (:code:`ConcDist`, :code:`ConcReal`, or a list of these objects) specifies the variables to be optimized over, and :code:`reg` (:code:`Region` object, optional) specifies the constraints. General functions (not only linear combinations of entropy) may be used in :code:`expr` and :code:`reg` using :code:`Expr.fcn` (see `Useful functions`_).

- Use :code:`PsiOpts.setting(opt_optimizer = ???)` to choose the optimization method. The default algorithm is :code:`"SLSQP"` via :code:`scipy.optimize` [Kraft 1988], which is suitable for convex problems (e.g. channel capacity, rate distortion). Other choices are :code:`"sgd"` (gradient descent) and :code:`"adam"` [Kingma 2014] via PyTorch. 

- Use :code:`PsiOpts.setting(opt_basinhopping = True)` to enable basin hopping [Wales-Doye 1997] for nonconvex problems (e.g. problems involving auxiliary random variables).

 - Use :code:`PsiOpts.setting(opt_num_hop = 50)` to set the number of hops for basin hopping.

- Use :code:`PsiOpts.setting(opt_num_iter = 100)` to set the number of iterations. Use :code:`PsiOpts.setting(opt_num_iter_mul = 2)` to multiply to the number of iterations.

- Use :code:`PsiOpts.setting(opt_num_points = 10)` to set the number of random initial points to try.

- Use :code:`PsiOpts.setting(opt_aux_card = 3)` to set the default cardinality of the auxiliary random variables where :code:`set_card` has not been called.

- Use :code:`PsiOpts.setting(verbose_opt = True)` and :code:`PsiOpts.setting(verbose_opt_step = True)` to display steps.


Example 1: Channel coding, finding optimal input distribution
-------------------------------------------------------------

.. code-block:: python

    # ********** Channel input distribution optimization **********

    import numpy
    import scipy
    import torch
    from psitip import *
    PsiOpts.setting(solver = "pyomo.glpk")
    PsiOpts.setting(istorch = True)       # Enable pytorch

    # Enable pytorch for numerical optimization
    PsiOpts.setting(istorch = True)

    # X, Y are binary RVs (cardinality = 2)
    X, Y = rv("X", "Y").set_card(2)

    # P is the underlying distribution of random variables
    P = ConcModel()

    # Distribution of X is Bernoulli(0.7)
    P[X] = [0.3, 0.7]

    # X -> Y is binary symmetric channel with crossover 0.2
    P[Y | X] = [[0.8, 0.2], [0.2, 0.8]]

    print(P[Y])        # print distribution of Y
    print(P[I(X & Y)]) # print I(X;Y)

    # Declare that the distribution of X is a variable in optimization
    P[X] = "var"

    # Maximize I(X;Y) over variable P[X]
    P.maximize(I(X & Y), P[X])

    print(P[I(X & Y)]) # print optimal I(X;Y)
    print(P[X])        # print distribution of X attaining optimum
    P.venn()           # draw information diagram


Example 2: Lossy source coding, rate distortion
-----------------------------------------------

.. code-block:: python

    # ********** Rate distortion **********

    import numpy
    import scipy
    import torch
    from psitip import *
    PsiOpts.setting(solver = "pyomo.glpk")
    PsiOpts.setting(istorch = True)       # Enable pytorch

    # X, Y are binary RVs (cardinality = 2)
    X, Y = rv("X", "Y").set_card(2)

    # P is the underlying distribution of random variables
    P = ConcModel()

    # Distribution of X is Bernoulli(0.7)
    P[X] = [0.3, 0.7]

    # Declare that P[Y | X] is a variable in optimization
    P[Y | X] = "var"

    # Hamming distortion function is the mean of the function 1{x != y}
    # over the distribution P(X,Y). We demonstrate 4 methods to specify it:
    # Method 1: Use the mean function
    dist = (X+Y).mean(lambda x, y: float(x != y))

    # Method 2: Distortion = P(X=0,Y=1) + P(X=1,Y=0)
    # dist = (X+Y).prob(0, 1) + (X+Y).prob(1, 0)

    # Method 3: Use "pmf" to obtain probability matrix (ExprArray object)
    # and take 1 - trace
    # dist = 1 - (X+Y).pmf().trace()

    # Method 4: Use Expr.fcn to wrap any function
    # mapping a ConcModel to a number as an Expr
    # dist = Expr.fcn(lambda P: P[X+Y][0, 1] + P[X+Y][1, 0])

    # Minimize I(X;Y) over P[Y | X], under constraint dist <= 0.1
    P.minimize(I(X & Y), P[Y | X], dist <= 0.1)

    print(P[I(X & Y)])       # print optimal I(X;Y)
    print(P[Y | X].given(0)) # print P[Y | X=0] attaining optimum
    print(P[Y | X].given(1)) # print P[Y | X=1] attaining optimum
    print(P[dist])           # print distortion
    P.venn()                 # draw information diagram



Example 3: Finding the most informative bit
-------------------------------------------

.. code-block:: python

    # ********** Finding the most informative bit **********
    # Kumar and Courtade, "Which boolean functions are 
    # most informative?", ISIT 2013
    # Given X1,...,Xn i.i.d. fair bits, and Y1,...,Yn produced by passing 
    # X1,...,Xn through a memoryless BSC, the problem is to find a binary
    # function F(X1,...,Xn) that maximizes I(F;Y)

    import numpy
    import scipy
    import torch
    from psitip import *
    PsiOpts.setting(solver = "pyomo.glpk")
    PsiOpts.setting(istorch = True)       # Enable pytorch
    # PsiOpts.setting(verbose_opt = True) # Uncomment to display steps
    # PsiOpts.setting(verbose_opt_step = True)

    n = 3
    a = 0.1

    # X, Y are array of bits (cardinality = 2)
    X = rv_seq("X", n).set_card(2)
    Y = rv_seq("Y", n).set_card(2)

    # F is a binary random variable
    F = rv("F").set_card(2)

    # P is the underlying distribution of random variables
    P = ConcModel()

    # Add random variables to the model in the order they are generated
    for x, y in zip(X, Y):
        
        # P(x) is Bernoulli(1/2)
        P[x] = ConcDist.bit()
        
        # P(y|x) is BSC with crossover a
        P[y | x] = ConcDist.bsc(a)

    # P(F|X) is the variable we optimize over
    P[F | X] = "var_rand"

    # Maximize I(F1,F2,F3 ; Y1,Y2,Y3)
    # The default setting is not suitable for nonconvex optimization
    print(P.maximize(I(F & Y), P[F | X]))
    print(P[F | X])
    print(P[I(F & Y)])

    # Switch to basin-hopping for nonconvex optimization
    PsiOpts.setting(opt_basinhopping = True)
    PsiOpts.setting(opt_num_iter_mul = 2) # double the number of iterations

    # "timer = 60000" sets time limit 60000ms for code within the block
    with PsiOpts(timer = 60000):
        print(P.maximize(I(F & Y), P[F | X]))
    print(P[F | X])
    print(P[I(F & Y)])




Automated inner and outer bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Psitip supports automated achievability and converse proofs in network information theory. The achievability part uses the general coding theorem for network information theory in [Lee-Chung 2015], whereas the converse part follows the general strategy of identifying auxiliaries using past and future random variables pioneered by Gallager [Gallager 1974], using Csiszár sum identity [Körner-Marton 1977], [Csiszár-Körner 1978].

A setting in network information theory is represented by a :code:`CodingModel` object. To specify a setting, use the following three functions (here we let :code:`model = CodingModel()`):

- :code:`model.add_node(M, X)` specifies that there is an encoder/decoder which observes M (a :code:`Comp` object) and outputs X (:code:`Comp`).

 - For causal observation, use the argument :code:`rv_in_causal`. E.g. :code:`model.add_node(M+S, X, rv_in_causal = S)` means that the encoder produces Xi using only M,S1,...,Si.

 - Passing the argument :code:`ndec_mode = "min"` to :code:`add_node` instructs the algorithm to avoid using simultaneous nonunique decoding. The argument :code:`ndec_mode = "max"` instructs the algorithm to use simultaneous nonunique decoding whenever possible. The default is to try all possibilities and output the inner bound as the union, which can be quite slow.

- :code:`model.add_edge(X, Y)` specifies that Y (:code:`Comp`) is produced by a channel with input X (:code:`Comp`). The random variable Y is conditionally independent of all previously added random variables given X, and hence edges are also needed between correlated sources.

 - **Caution.** Random variables must be added in the order they are generated in the setting (e.g. channel outputs after channel inputs, decoders after encoders).

- :code:`model.set_rate(M, R)` specifies that M (:code:`Comp`) is a message with rate R (:code:`Expr`).

 - **Caution.** :code:`model.set_rate` must be called **after** all calls of :code:`model.add_node` and :code:`model.add_edge`.

After a setting is specified, call:

- :code:`model.get_inner()` to obtain an inner bound (:code:`Region`).

 - Use :code:`model.get_inner(convexify = True)` instead to convexify the region using a time sharing random variable.

 - If this is taking too long, use the option :code:`ndec_mode = "min"` for :code:`model.add_node` mentioned before.

- :code:`model.get_outer()` to obtain an outer bound (:code:`Region`). 

 - Note that the outer bound includes all past/future random variables, and is not simplified. Though this is useful for checking other outer bounds. For example, :code:`(model.get_outer() >> r).check_getaux()` checks whether :code:`r` is an outer bound (by checking whether the outer bound implies :code:`r`), and if so, outputs the choices of auxiliaries for the proof. If :code:`r` is an inner bound, this checks whether :code:`r` is tight.

- :code:`model.graph()` to obtain a graphical representation of the setting (Graphviz graph).


An example (channel with noncausal state information at encoder) is given at the beginning. More examples:


Example 1: Degraded broadcast channel
-------------------------------------

.. code-block:: python

    # ********** Degraded broadcast channel **********

    import numpy
    import scipy
    import torch
    import matplotlib.pyplot as plt
    from psitip import *
    PsiOpts.setting(solver = "pyomo.glpk")

    X, Y, Z, M1, M2 = rv("X", "Y", "Z", "M1", "M2")
    R1, R2 = real("R1", "R2")

    model = CodingModel()
    model.add_node(M1+M2, X)  # Encoder maps M1,M2 to X
    model.add_edge(X, Y)      # Channel X -> Y -> Z
    model.add_edge(Y, Z)
    model.add_node(Y, M1)     # Decoder1 maps Y to M1
    model.add_node(Z, M2)     # Decoder2 maps Z to M2
    model.set_rate(M1, R1)    # Rate of M1 is R1
    model.set_rate(M2, R2)    # Rate of M2 is R2
    # display(model.graph())  # Draw the model

    r = model.get_inner()     # Get inner bound, recovers superposition region 
    print(r)                  # [Bergmans 1973], [Gallager 1974]
    # display(r.graph())      # Draw Bayesian network of RVs

    r_out = model.get_outer() # Get outer bound

    # Check outer bound implies inner bound and output auxiliaries for proof
    print((r_out >> r).check_getaux())


    # *** Plot capacity region for Z-channel ***

    PsiOpts.setting(istorch = True)   # Enable pytorch
    PsiOpts.setting(opt_aux_card = 3) # Default cardinality for auxiliary
    X.set_card(2)                     # X,Y,Z have cardinality 2
    Y.set_card(2)
    Z.set_card(2)
    P = ConcModel()
    P[X] = "var"                      # Optimize over P(X)
    P[R1] = "var"                     # Optimize over R1,R2
    P[R2] = "var"
    P[Y|X] = [[1.0, 0.0], [0.2, 0.8]] # X->Y is a Z-channel
    P[Z|Y] = [[0.8, 0.2], [0.0, 1.0]] # Y->Z is a Z-channel

    lams = numpy.linspace(0.5, 1, 10)
    R1s = []
    R2s = []
    for lam in lams:
        # Maximize lambda sum-rate over P(X),R1,R2 subject to inner bound
        P.maximize(R1*(1-lam) + R2*lam, [P[X], R1, R2], r)
        R1s.append(float(P[R1]))
        R2s.append(float(P[R2]))
        
    plt.figure()
    plt.plot(R1s, R2s)  # Plot capacity region
    plt.show()


Example 2: Lossy source coding with side information at decoder
---------------------------------------------------------------

.. code-block:: python

    # ********** Wyner-Ziv theorem [Wyner-Ziv 1976] **********

    from psitip import *
    PsiOpts.setting(solver = "pyomo.glpk")

    X, Y, Z, M = rv("X", "Y", "Z", "M")
    R = real("R")

    model = CodingModel()
    model.add_edge(X, Y)      # X and Y are correlated
    model.add_node(X, M)      # Encoder observes X, produces M
    model.add_node(M+Y, Z)    # Decoder observes M,Y, produces Z
    # model.add_node(M+Y, Z, rv_in_causal = Y) # Use this instead if 
                                              # Y observed causally
    model.set_rate(M, R)      # The rate of M is R

    r = model.get_inner()     # Get inner bound, recovers Wyner-Ziv
    print(r)
    r_out = model.get_outer() # Get outer bound
    print((r_out >> r).check_getaux()) # Tightness, output auxiliaries




Fourier-Motzkin elimination
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`exists` method of :code:`Region` with real variable arguments performs Fourier-Motzkin elimination over those variables, for example:

.. code-block:: python

    from psitip import *
    PsiOpts.setting(solver = "pyomo.glpk")

    # Fourier-Motzkin elimination for Marton's inner bound with common message
    # [Marton 1979], [Liang-Kramer 2007]
    R0, R1, R2, R10, R20, Rs = real("R0", "R1", "R2", "R10", "R20", "Rs")
    U0, U1, U2, Y1, Y2 = rv("U0", "U1", "U2", "Y1", "Y2")
    
    r = universe()
    r &= R0 >= 0
    r &= R1 >= 0
    r &= R2 >= 0
    r &= R10 >= 0
    r &= R10 <= R1
    r &= R20 >= 0
    r &= R20 <= R2
    r &= Rs >= 0
    
    r &= R0 + R20 + R1 + Rs <= I(U0 + U1 & Y1)
    r &= R1 - R10 + Rs <= I(U1 & Y1 | U0)
    r &= R0 + R10 + R2 - Rs <= I(U0 + U2 & Y2) - I(U1 & U2 | U0)
    r &= R0 + R10 + R2 <= I(U0 + U2 & Y2)
    r &= R2 - R20 - Rs <= I(U2 & Y2 | U0) - I(U1 & U2 | U0)
    r &= R2 - R20 <= I(U2 & Y2 | U0)
    
    r = r.exists(R10+R20+Rs+U0+U1+U2)
    print(r)


Discover inequalities
~~~~~~~~~~~~~~~~~~~~~

The :code:`discover` method of :code:`Region` accepts a list of variables of interest (:code:`Comp` or :code:`Expr`), and automatically discover inequalities among those variables implied by the region. It either uses the convex hull method for polyhedron projection [Lassez-Lassez 1991], or trial and error in case the region is a :code:`RegionOp` object. For example:

.. code-block:: python

    from psitip import *

    PsiOpts.setting(solver = "pyomo.glpk")

    X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")

    K = gacs_korner(X&Y)
    J = wyner_ci(X&Y)
    G = exact_ci(X&Y)

    RK, RJ, RG = real("RK", "RJ", "RG")

    # Automatically discover relationship between different notions of common information
    # Gives RK >= 0, RG >= RJ, RG <= H(X), RG <= H(Y), RK <= I(X;Y), RJ >= I(X;Y)
    print(universe().discover([(RK, K), (RJ, J), (RG, G), X, Y], maxsize = 2))


    # State the copy lemma [Zhang-Yeung 1998], [Dougherty-Freiling-Zeger 2011]
    r = eqdist([X, Y, U], [X, Y, Z]) & markov(Z+W, X+Y, U)

    # Automatically discover non-Shannon-type inequalities using copy lemma
    # Gives 2I(X;Y|Z,W)+I(X;Z|Y,W)+I(Y;Z|X,W)+I(Z;W|X,Y)+I(X;Y;W|Z)+2I(X;Z;W|Y)+2I(Y;Z;W|X) >= 0, etc
    print(r.discover(mi_cells(X, Y, Z, W)))



Bayesian network optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian network optimization is turned on by default. It builds a Bayesian network automatically using the given conditional independence conditions, so as to reduce the dimension of the linear programming problem. The speed up is significant when the Bayesian network is sparse, for instance, when the variables form a Markov chain:

.. code-block:: python

    X = rv_seq("X", 0, 9)
    print(bool(markov(*X) >> (I(X[0] & X[8]) <= H(X[4]))))

Nevertheless, building the Bayesian network can take some time. If your problem does not admit a sparse Bayesian network structure, you may turn off this optimization by:

.. code-block:: python

    PsiOpts.setting(lptype = "H")

The :code:`get_bayesnet` method of :code:`Region` returns a :code:`BayesNet` object (a Bayesian network) that can be deduced by the conditional independence conditions in the region. The :code:`check_ic` method of :code:`BayesNet` checks whether an expression containing conditional mutual information terms is always zero. The :code:`get_region` method of :code:`BayesNet` returns the :code:`Region` corresponding to the network. The :code:`graph` method of :code:`BayesNet` draws the Bayesian network (as a Graphviz graph). E.g.:

.. code-block:: python

    ((I(X&Y|Z) == 0) & (I(U&X+Z|Y) <= 0)).get_bayesnet().check_ic(I(X&U|Z))
    ((I(X&Y|Z) == 0) & (I(U&X+Z|Y) <= 0)).get_bayesnet().get_region()


Built-in functions
~~~~~~~~~~~~~~~~~~

There are several built-in information functions listed below. While they can be defined by the user easily (see the source code for their definitions), they are provided for convenience.

Theorems
--------

The following are true statements (:code:`Region` objects) that allow Psitip to prove results not provable by Shannon-type inequalities (at the expense of longer computation time). They can either be used in the context manager (e.g. :code:`with sfrl(logg).assumed():`), or directly (e.g. sfrl().implies(excess_fi(X, Y) <= H(X | Y))).

- **Strong functional representation lemma** [Li-El Gamal 2018] is given by :code:`sfrl(logg)`. It states that for any random variables (X, Y), there exists random variable Z independent of X such that Y is a function of (X, Z), and I(X;Z|Y) <= log(I(X;Y) + 1) + 4. The "log(I(X;Y) + 1) + 4" term is usually represented by the real variable :code:`logg = real("logg")` (which is the argument of :code:`sfrl(logg)`). Omitting the :code:`logg` argument gives the original functional representation lemma [El Gamal-Kim 2011]. For example:

  .. code-block:: python

    R = real("R") # declare real variable
    logg = real("logg")

    # Channel with state information at encoder, lower bound
    r_op = ((R <= I(M & Y)) & indep(M,S) & markov(M, X+S, Y)
            & (R >= 0)).exists(M).marginal_exists(X)
    
    # Gelfand-Pinsker theorem [Gel'fand-Pinsker 1980]
    r = ((R <= I(U & Y) - I(U & S)) & markov(U, X+S, Y)
            & (R >= 0)).exists(U).marginal_exists(X)
    
    # Using strong functional representation lemma
    with sfrl(logg).assumed():
        
        # Automated achievability proof of Gelfand-Pinsker theorem
        print(r.implies(r_op.relaxed(R, logg * 5))) # returns True

 - Note that writing :code:`with sfrl(logg).assumed():` allows SFRL to be used only once. To allow it to be used twice, write :code:`with (sfrl(logg) & sfrl(logg)).assumed():`.

- **Copy lemma** [Zhang-Yeung 1998], [Dougherty-Freiling-Zeger 2011] is given by :code:`copylem(n, m)`. It states that for any random variables X_1,...,X_n,Y_1,...,Y_m, there exists Z_1,...,Z_m such that (X_1,...,X_n,Y_1,...,Y_m) has the same distribution as (X_1,...,X_n,Z_1,...,Z_m) (only equalities of entropies are enforced in Psitip), and (Y_1,...,Y_m)-(X_1,...,X_n)-(Z_1,...,Z_m) forms a Markov chain. The default values of n, m are 2, 1 respectively. For example:

  .. code-block:: python

    # Using copy lemma
    with copylem().assumed():
        
        # Prove Zhang-Yeung inequality
        print(bool(2*I(Z&W) <= I(X&Y) + I(X & Z+W) + 3*I(Z&W | X) + I(Z&W | Y))) # returns True

- **Double Markov property** [Csiszar-Körner 2011] is given by :code:`dblmarkov()`. It states that if X-Y-Z and Y-X-Z are Markov chains, then there exists W that is a function of X, a function of Y, and (X,Y)-W-Z is Markov chain. For example:

  .. code-block:: python
  
    # Using double Markov property
    with dblmarkov().assumed():
        aux = ((markov(X, Y, Z) & markov(Y, X, Z))
            >> (H(mss(X, Z) | mss(Y, Z)) == 0)).check_getaux()
        print(iutil.list_tostr_std(aux))
        
        aux = ((markov(X, Y, Z) & markov(Y, X, Z))
            >> markov(X+Y, meet(X, Y), Z)).check_getaux()
        print(iutil.list_tostr_std(aux))

- The approximate infinite divisibility of information [Li 2020] is given by :code:`ainfdiv(n)`.

- The non-Shannon inequality in [Makarychev-Makarychev-Romashchenko-Vereshchagin 2002] is given by :code:`mmrv_thm(n)`.

- The non-Shannon inequalities in four variables in [Zhang-Yeung 1998] and [Dougherty-Freiling-Zeger 2006] is given by :code:`zydfz_thm()`.

- **Existence of meet and minimal sufficient statistics** is given by :code:`existence(meet)` and :code:`existence(mss)` respectively.


Conditions
----------

The following are conditions (:code:`Region` objects) on the random variable arguments.

- **Mutual independence** is expressed as :code:`indep(X, Y, Z)`. The function :code:`indep` can take any number of arguments. For random sequence :code:`X = rv_seq("X", 5)`, the mutual independence condition can be expressed as :code:`indep(*X)`.

- **Markov chain** is expressed as :code:`markov(X, Y, Z)`. The function :code:`markov` can take any number of arguments. For random sequence :code:`X = rv_seq("X", 5)`, the Markov chain condition can be expressed as :code:`markov(*X)`.

- **Informational equivalence** (i.e., containing the same information) is expressed as :code:`equiv(X, Y, Z)`. The function :code:`equiv` can take any number of arguments. Note that :code:`equiv(X, Y)` is the same as :code:`(H(X|Y) == 0) & (H(Y|X) == 0)`.

- **Same distribution**. The condition that (X,Y) has the same distribution as (Z,W) is expressed as :code:`eqdist([X, Y], [Z, W])`. The function :code:`eqdist` can take any number of arguments (that are all lists). Note that only equalities of entropies are enforced (i.e., H(X)=H(Z), H(Y)=H(W), H(X,Y)=H(Z,W)).

- **Exchangeability** is expressed as :code:`exchangeable(X, Y, Z)`. The function :code:`exchangeable` can take any number of arguments. For random sequence :code:`X = rv_seq("X", 5)`, the condition that it is an exchangeable sequence of random variables can be expressed as :code:`exchangeable(*X)`. Note that only equalities of entropies are enforced.

- **IID sequence** is expressed as :code:`iidseq(X, Y, Z)`. The function :code:`iidseq` can take any number of arguments. For random sequence :code:`X = rv_seq("X", 5)`, the condition that it is an IID sequence of random variables can be expressed as :code:`iidseq(*X)`. Note that only equalities of entropies are enforced.


Random variables
----------------

The following are :code:`Comp` objects (random-variable-valued functions).

- **Meet** or **Gács-Körner common part** [Gács-Körner 1973] between X and Y is denoted as :code:`meet(X, Y)` (a :code:`Comp` object).

- **Minimal sufficient statistic** of X about Y is denoted as :code:`mss(X, Y)` (a :code:`Comp` object).

- The random variable given by the **strong functional representation lemma** [Li-El Gamal 2018] applied on X, Y (:code:`Comp` objects) with a gap term logg (:code:`Expr` object) is denoted as :code:`sfrl_rv(X, Y, logg)` (a :code:`Comp` object). If the gap term is omitted, this will be the ordinary functional representation lemma [El Gamal-Kim 2011].


Real-valued information quantities
----------------------------------

The following are :code:`Expr` objects (real-valued functions).

- **Gács-Körner common information** [Gács-Körner 1973] is given by :code:`gacs_korner(X & Y)`. The multivariate conditional version can be obtained by :code:`gacs_korner(X & Y & Z | W)`. The following tests return True:

  .. code-block:: python

    # Definition
    print(bool(gacs_korner(X & Y) == 
        ((H(U|X) == 0) & (H(U|Y) == 0)).exists(U).maximum(H(U))))
    print(bool(gacs_korner(X & Y) == H(meet(X, Y))))

    # Properties
    print(bool(markov(X, Y, Z) >> (gacs_korner(X & Y) >= gacs_korner(X & Z))))
    print(bool(indep(X, Y, Z) >> (gacs_korner(X+Z & Y+Z) == H(Z))))
    print(bool(indep(X+Y, Z+W) >> 
        (gacs_korner(X & Y) + gacs_korner(Z & W) <= gacs_korner(X+Z & Y+W))))

- **Wyner's common information** [Wyner 1975] is given by :code:`wyner_ci(X & Y)`. The multivariate conditional version can be obtained by :code:`wyner_ci(X & Y & Z | W)`. The following tests return True:

  .. code-block:: python

    # Definition
    print(bool(wyner_ci(X & Y) == markov(X, U, Y).exists(U).minimum(I(U & X+Y))))

    # Properties
    print(bool(markov(X, Y, Z) >> (wyner_ci(X & Y) >= wyner_ci(X & Z))))
    print(bool(indep(X, Y, Z) >> (wyner_ci(X+Z & Y+Z) == H(Z))))
    print(bool(indep(X+Y, Z+W) >> 
        (wyner_ci(X & Y) + wyner_ci(Z & W) <= wyner_ci(X+Z & Y+W))))
    print(bool(indep(X+Y, Z+W) >> 
        (wyner_ci(X & Y) + wyner_ci(Z & W) >= wyner_ci(X+Z & Y+W))))

- **Common entropy** (or one-shot exact common information) [Kumar-Li-El Gamal 2014] is given by :code:`exact_ci(X & Y)`. The multivariate conditional version can be obtained by :code:`exact_ci(X & Y & Z | W)`. The following tests return True:

  .. code-block:: python

    # Definition
    print(bool(exact_ci(X & Y) == markov(X, U, Y).exists(U).minimum(H(U))))

    # Properties
    print(bool(markov(X, Y, Z) >> (exact_ci(X & Y) >= exact_ci(X & Z))))
    print(bool(indep(X, Y, Z) >> (exact_ci(X+Z & Y+Z) == H(Z))))
    print(bool(indep(X+Y, Z+W) >> 
        (exact_ci(X & Y) + exact_ci(Z & W) >= exact_ci(X+Z & Y+W))))

- **Total correlation** [Watanabe 1960] is given by :code:`total_corr(X & Y & Z)`. The conditional version can be obtained by :code:`total_corr(X & Y & Z | W)`. The following test returns True:

  .. code-block:: python

    # By definition
    print(bool(total_corr(X & Y & Z) == H(X) + H(Y) + H(Z) - H(X+Y+Z)))

- **Dual total correlation** [Han 1978] is given by :code:`dual_total_corr(X & Y & Z)`. The conditional version can be obtained by :code:`dual_total_corr(X & Y & Z | W)`. The following test returns True:

  .. code-block:: python

    # By definition
    print(bool(dual_total_corr(X & Y & Z) == 
        H(X+Y+Z) - H(X|Y+Z) - H(Y|X+Z) - H(Z|X+Y)))

- **Multivariate mutual information** [McGill 1954] is simply given by :code:`I(X & Y & Z) == I(X & Y) - I(X & Y | Z)`. The conditional version can be obtained by :code:`I(X & Y & Z | W)`.

- **Mutual dependence** [Csiszar-Narayan 2004] is given by :code:`mutual_dep(X & Y & Z)`. The conditional version can be obtained by :code:`mutual_dep(X & Y & Z | W)`. The following tests return True:

  .. code-block:: python

    # By definition
    print(bool(mutual_dep(X & Y & Z) == 
        emin(I(X+Y & Z), I(X+Z & Y), I(Y+Z & X), total_corr(X & Y & Z) / 2)))

    # Properties
    print(bool(mutual_dep(X & Y & Z) <= total_corr(X & Y & Z) / 2))
    print(bool(mutual_dep(X & Y & Z) <= dual_total_corr(X & Y & Z)))
    print(bool(markov(X, Y, Z) >> 
        (mutual_dep(X & Y & Z) == emin(I(X & Y), I(Y & Z)))))

- **Intrinsic mutual information** [Maurer-Wolf 1999] is given by :code:`intrinsic_mi(X & Y | Z)`. The following tests return True:

  .. code-block:: python

    # Definition
    print(bool(intrinsic_mi(X & Y | Z) == markov(X+Y, Z, U).exists(U).minimum(I(X & Y | U))))

    # Properties
    print(bool(intrinsic_mi(X & Y | Z) <= I(X & Y | Z)))

- **Necessary conditional entropy** [Cuff-Permuter-Cover 2010] is given by :code:`H_nec(Y | X)`.

- **Excess functional information** [Li-El Gamal 2018] is given by :code:`excess_fi(X, Y)`.

- The entropy of the **minimum entropy coupling** of the distributions p_{Y|X=x} is given by :code:`minent_coupling(X, Y)` ([Vidyasagar 2012], [Painsky et al. 2013], [Kovacevic et al. 2015], [Kocaoglu et al. 2017], [Cicalese et al. 2019], [Li 2020]).

- **Directed information** [Massey 1990] is given by :code:`directed_info(X, Y, Z)`. The arguments :code:`X, Y, Z` are either :code:`CompArray` or lists of :code:`Comp`.

- **Entropy vector** [Zhang-Yeung 1998] is given by :code:`ent_vector(*X)` (where :code:`X` is a random sequence of length n e.g. :code:`X = rv_seq("X", n)`). The return value is an :code:`ExprArray` of length 2^n-1.


Real-valued information quantities (numerical only)
---------------------------------------------------

The following are :code:`Expr` objects (real-valued functions) with limited symbolic capabilities. They are mostly used with :code:`ConcModel` for numerical optimization (they support automated gradient).

- **Renyi entropy** [Renyi 1961] is given by :code:`renyi(X, order)`. The argument :code:`X` can be a :code:`Comp` or :code:`ConcDist`.

- **Maximal correlation** [Hirschfeld 1935], [Gebelein 1941], [Renyi 1959] is given by :code:`maxcorr(X & Y)`.

- **Divergence** is given by :code:`divergence(X, Y, mode)`. The arguments :code:`X,Y` can be :code:`Comp` or :code:`ConcDist`. Choices of :code:`mode` are :code:`"kl"` for Kullback-Leibler divergence, "tv" for total variation distance, "chi2" for chi-squared divergence, "hellinger" for Hellinger distance [Hellinger 1909] and "js" for Jensen-Shannon divergence.

- **Varentropy** and **dispersion** [Kontoyiannis-Verdu 2013], [Polyanskiy-Poor-Verdu 2010] are given by :code:`varent(X)` and :code:`varent(X & Y)`.


Options
~~~~~~~

There are two ways to set options. One can set an option globally using:

.. code-block:: python

    PsiOpts.setting(option = value)

or locally within a :code:`with` block using context manager:

.. code-block:: python

    with PsiOpts(option = value):
        # do something here

Some of the options are:

- :code:`ent_base` : The base of logarithm for entropy. Default is 2.

- :code:`truth` : Specify a region that is assumed to be true in all deductions. For example, use :code:`truth = sfrl(logg)` to assume the strong functional representation lemma with logarithmic gap given by :code:`logg = real("logg")`. Default is None.

- :code:`truth_add` : Add another assumption (:code:`Region` object) to :code:`truth`.

- :code:`solver` : The solver used (e.g. :code:`"pulp.glpk"`, :code:`"pyomo.glpk"`, :code:`"pulp.cbc"`, :code:`"scipy"`).

- :code:`solver_scipy_maxsize` : For linear programming problems with number of variables less than or equal to this value, the scipy solver will be used (regardless of the :code:`solver` option). This can lead to significant speed-up for small problems. Default is -1 (disabled).

- :code:`lptype` : Values are :code:`"HC1BN"` (Bayesian network optimization, default) or :code:`"H"` (no optimization).

- :code:`lp_bounded` : Set to True to add an upper bound (given by the option :code:`lp_ubound`) on the joint entropy of all random variables (so the linear program is always bounded). Default is False.

- :code:`lp_ubound` : The value of the upper bound for :code:`lp_bounded`. Default is :code:`1e3`. It should be set to a value larger than all affine constants in the problem.

- :code:`lp_eps` : Strict inequalities in the constraints like :code:`H(X) > H(Y)` are replaced by :code:`H(X) >= H(Y) + lp_eps`. Default is :code:`1e-3`. It should be set to a value smaller than all affine constants in the problem.

- :code:`lp_eps_obj` : Strict inequalities in the objective (region to be proved) like :code:`H(X) > H(Y)` are replaced by :code:`H(X) >= H(Y) + lp_eps_obj`. Default is :code:`1e-4`. It should be set to a value smaller than :code:`lp_eps`.

- :code:`lp_zero_cutoff` : An optimal value larger than :code:`lp_zero_cutoff` is considered nonnegative in a linear program. Default is :code:`-1e-5`. It should be set to a value smaller than all affine constants in the problem.

- :code:`auxsearch_leaveone` : Set to True to handle case decomposition in auxiliary search. Default is False.

- :code:`forall_multiuse` : Set to False to only allow one value for variables with universal quantification. Default is True. Note that if this option is True, then the auxiliary search result for variables with universal quantification will be meaningless.

- :code:`str_style` : The style of string conversion :code:`str(x)` and verbose output. Values are :code:`"standard"` (e.g. :code:`3I(X,Y;Z|W)-H(X) >= 0`, default), :code:`"code"` (e.g. :code:`3*I(X+Y&Z|W)-H(X) >= 0`, consistent with the Psitip syntax so the output can be copied back to the code), or :code:`"latex"` (e.g. :code:`3I(X,Y;Z|W)-H(X) \ge 0`, for LaTeX equations).

- :code:`verbose_lp` : Set to True to output linear programming problem sizes and results. Default is False.

- :code:`verbose_lp_cons` : Set to True to output the constraints in the linear program. Default is False. For example:

  .. code-block:: python

    with PsiOpts(lptype = "H", verbose_lp = True, verbose_lp_cons = True):
        bool(H(X) * 2 >= I(X & Y))

 gives::

    ============ LP constraints ============
    { H(X,Y)-H(Y) >= 0,
      H(X,Y)-H(X) >= 0,
      H(X)+H(Y)-H(X,Y) >= 0 }
    ============  LP objective  ============
    -H(X)+H(Y)-H(X,Y)
    ========================================
    LP nrv=2 nreal=0 nvar=3/3 nineq=3 neq=0 solver=pyomo.glpk
      status=Optimal optval=0.0


- :code:`verbose_auxsearch` : Set to True to output each problem of auxiliary random variable searching. Default is False.

- :code:`verbose_auxsearch_step` : Set to True to output each step in auxiliary searching. Default is False.

- :code:`verbose_auxsearch_result` : Set to True to output the final result of auxiliary searching. Default is False.

- :code:`verbose_auxsearch_all` : Set to True to turn on :code:`verbose_auxsearch`, :code:`verbose_auxsearch_step` and :code:`verbose_auxsearch_result`.

- :code:`verbose_auxsearch_cache` : Set to True to output each event in which the cache of auxiliary searching is discarded. Default is False.

- :code:`verbose_subset` : Set to True to output each implication problem. Default is False.

- :code:`verbose_sfrl` : Set to True to output strong functional representation lemma searching steps. Default is False.

- :code:`verbose_flatten` : Set to True to output progress in unfolding user-defined information quantities. Default is False.

- :code:`verbose_eliminate_toreal` : Set to True to output progress in eliminating random variables using the :code:`toreal = True` option. Default is False.


License
~~~~~~~

The source code of Psitip is released under the GNU General Public License v3.0 ( https://www.gnu.org/licenses/gpl-3.0.html ).

This program comes with ABSOLUTELY NO WARRANTY.


Contact
~~~~~~~

Please contact Cheuk Ting Li ( https://www.ie.cuhk.edu.hk/people/ctli.shtml ) for any feedback.


References
~~~~~~~~~~

The general method of using linear programming for solving information 
theoretic inequality is based on the following work:

- \R. W. Yeung, "A new outlook on Shannon's information measures," IEEE Trans. Inform. Theory, vol. 37, pp. 466-474, May 1991.

- \R. W. Yeung, "A framework for linear information inequalities," IEEE Trans. Inform. Theory, vol. 43, pp. 1924-1934, Nov 1997.

- \Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities," IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.

There are several other pieces of software based on the linear programming approach in ITIP, for example, `Xitip <http://xitip.epfl.ch/>`_, `FME-IT <http://www.ee.bgu.ac.il/~fmeit/index.html>`_, `Minitip <https://github.com/lcsirmaz/minitip>`_, and `Citip <https://github.com/coldfix/Citip>`_.

We remark that there is a Python package for discrete information theory called dit ( https://github.com/dit/dit ), which contains a collection of numerical optimization algorithms for information theory. Though it is not for proving information theoretic results.


Convex hull method for polyhedron projection:

- \C. Lassez and J.-L. Lassez, Quantifier elimination for conjunctions of linear constraints via a convex hull algorithm, IBM Research Report, T.J. Watson Research Center, RC 16779 (1991)


General coding theorem for network information theory:

- Si-Hyeon Lee, and Sae-Young Chung. "A unified approach for network information theory." 2015 IEEE International Symposium on Information Theory (ISIT). IEEE, 2015.


Optimization algorithms:

- Kraft, D. A software package for sequential quadratic programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace Center – Institute for Flight Mechanics, Koln, Germany.

- Wales, David J.; Doye, Jonathan P. K. (1997). "Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms". The Journal of Physical Chemistry A. 101 (28): 5111-5116.

- Hestenes, M. R. (1969). "Multiplier and gradient methods". Journal of Optimization Theory and Applications. 4 (5): 303-320.

- Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).


Results used as examples above:

- Peter Gács and Janos Körner. Common information is far less than mutual information.Problems of Control and Information Theory, 2(2):149-162, 1973.

- \A. D. Wyner. The common information of two dependent random variables. IEEE Trans. Info. Theory, 21(2):163-179, 1975.

- \S. I. Gel'fand and M. S. Pinsker, "Coding for channel with random parameters," Probl. Contr. and Inf. Theory, vol. 9, no. 1, pp. 19-31, 1980.

- Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978.

- \K. Marton, "A coding theorem for the discrete memoryless broadcast channel," IEEE  Transactions on Information Theory, vol. 25, no. 3, pp. 306-311, May 1979.

- \Y. Liang and G. Kramer, "Rate regions for relay broadcast channels," IEEE Transactions on Information Theory, vol. 53, no. 10, pp. 3517-3535, Oct 2007.

- Bergmans, P. "Random coding theorem for broadcast channels with degraded components." IEEE Transactions on Information Theory 19.2 (1973): 197-207.

- Gallager, Robert G. "Capacity and coding for degraded broadcast channels." Problemy  Peredachi Informatsii 10.3 (1974): 3-14.

- \J. Körner and K. Marton, Comparison of two noisy channels, Topics in Inform. Theory (ed. by I. Csiszar and P. Elias), Keszthely, Hungary (August, 1975), 411-423.

- El Gamal, Abbas, and Young-Han Kim. Network information theory. Cambridge University Press, 2011.

- Watanabe S (1960). Information theoretical analysis of multivariate correlation, IBM Journal of Research and Development 4, 66-82. 

- Han T. S. (1978). Nonnegative entropy measures of multivariate symmetric correlations, Information and Control 36, 133-156. 

- McGill, W. (1954). "Multivariate information transmission". Psychometrika. 19 (1): 97-116.

- Csiszar, Imre, and Prakash Narayan. "Secrecy capacities for multiple terminals." IEEE Transactions on Information Theory 50, no. 12 (2004): 3047-3061.

- Tishby, Naftali, Pereira, Fernando C., Bialek, William (1999). The Information Bottleneck Method. The 37th annual Allerton Conference on Communication, Control, and Computing. pp. 368-377.

- \U. Maurer and S. Wolf. "Unconditionally secure key agreement and the intrinsic conditional information." IEEE Transactions on Information Theory 45.2 (1999): 499-514.

- Wyner, Aaron, and Jacob Ziv. "The rate-distortion function for source coding with side information at the decoder." IEEE Transactions on information Theory 22.1 (1976): 1-10.

- Randall Dougherty, Chris Freiling, and Kenneth Zeger. "Non-Shannon information inequalities in four random variables." arXiv preprint arXiv:1104.3602 (2011).

- Imre Csiszar and Janos Körner. Information theory: coding theorems for discrete memoryless systems. Cambridge University Press, 2011.

- Makarychev, K., Makarychev, Y., Romashchenko, A., & Vereshchagin, N. (2002). A new class of non-Shannon-type inequalities for entropies. Communications in Information and Systems, 2(2), 147-166.

- Randall Dougherty, Christopher Freiling, and Kenneth Zeger. "Six new non-Shannon information inequalities." 2006 IEEE International Symposium on Information Theory. IEEE, 2006.

- \M. Vidyasagar, "A metric between probability distributions on finite sets of different cardinalities and applications to order reduction," IEEE Transactions on Automatic Control, vol. 57, no. 10, pp. 2464-2477, 2012.

- \A. Painsky, S. Rosset, and M. Feder, "Memoryless representation of Markov processes," in 2013 IEEE International Symposium on Information Theory. IEEE, 2013, pp. 2294-298.

- \M. Kovacevic, I. Stanojevic, and V. Senk, "On the entropy of couplings," Information and Computation, vol. 242, pp. 369-382, 2015.

- \M. Kocaoglu, A. G. Dimakis, S. Vishwanath, and B. Hassibi, "Entropic causal inference," in Thirty-First AAAI Conference on Artificial Intelligence, 2017.

- \F. Cicalese, L. Gargano, and U. Vaccaro, "Minimum-entropy couplings and their applications," IEEE Transactions on Information Theory, vol. 65, no. 6, pp. 3436-3451, 2019.

- \C. T. Li, "Efficient Approximate Minimum Entropy Coupling of Multiple Probability Distributions," arXiv preprint https://arxiv.org/abs/2006.07955 , 2020.

- \C. T. Li, "Infinite Divisibility of Information," arXiv preprint https://arxiv.org/abs/2008.06092 , 2020.

- \J. Körner and K. Marton, "Images of a set via two channels and their role in multi-user communication," IEEE Transactions on Information Theory, vol. 23, no. 6, pp. 751–761, 1977.

- \I. Csiszár and J. Körner, "Broadcast channels with confidential messages," IEEE transactions on information theory, vol. 24, no. 3, pp. 339–348, 1978.

- Kumar and Courtade, "Which boolean functions are most informative?", ISIT 2013.

- Massey, James. "Causality, feedback and directed information." Proc. Int. Symp. Inf. Theory Applic.(ISITA-90). 1990.

- Renyi, Alfred (1961). "On measures of information and entropy". Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability 1960. pp. 547-561.

- \H. O. Hirschfeld, "A connection between correlation and contingency," in Mathematical Proceedings of the Cambridge Philosophical Society, vol. 31, no. 04. Cambridge Univ Press, 1935, pp. 520-524.

- \H. Gebelein, "Das statistische problem der korrelation als variations-und eigenwertproblem und sein zusammenhang mit der ausgleichsrechnung," ZAMM-Journal of Applied Mathematics and Mechanics/Zeitschrift fur Angewandte Mathematik und Mechanik, vol. 21, no. 6, pp. 364-379, 1941.

- \A. Renyi, "On measures of dependence," Acta mathematica hungarica, vol. 10, no. 3, pp. 441-451, 1959.

- Kontoyiannis, Ioannis, and Sergio Verdu. "Optimal lossless compression: Source varentropy and dispersion." 2013 IEEE International Symposium on Information Theory. IEEE, 2013.

- Polyanskiy, Yury, H. Vincent Poor, and Sergio Verdu. "Channel coding rate in the finite blocklength regime." IEEE Transactions on Information Theory 56.5 (2010): 2307-2359.

- Hellinger, Ernst (1909), "Neue Begründung der Theorie quadratischer Formen von unendlichvielen Veränderlichen", Journal für die reine und angewandte Mathematik, 136: 210–271.
