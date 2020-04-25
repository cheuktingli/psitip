Psitip
======

Python Symbolic Information Theoretic Inequality Prover

The ITIP software was developed by Raymond W. Yeung and Ying-On Yan
( http://user-www.ie.cuhk.edu.hk/~ITIP/ ). There are several pieces of software based on the linear programming approach in ITIP, for example, `Xitip <http://xitip.epfl.ch/>`_, `FME-IT <http://www.ee.bgu.ac.il/~fmeit/index.html>`_, `Minitip <https://github.com/lcsirmaz/minitip>`_, and `Citip <https://github.com/coldfix/Citip>`_. Psitip also uses the linear programming approach (see References section), but is otherwise unrelated to the aforementioned projects.

Psitip is unique in the sense that it is a Python module tightly integrated with the Python syntax, benefitting from its operator overloading support. Random variables, expressions and regions are objects in Python that can be manipulated easily. Moreover, it supports a versatile deduction system for automated theorem proving. For example:

.. code-block:: python

    from psitip import *
    PsiOpts.set_setting(solver = "pulp.glpk")
    
    X, Y, Z, W, U, M, S = rv("X", "Y", "Z", "W", "U", "M", "S") # declare random variables
    print(bool(H(X) + I(Y & Z | X) >= I(Y & Z))) # H(X)+I(Y;Z|X)>=I(Y;Z) is True
    print(markov(X, Y, Z).implies(H(X | Y) <= H(X | Z))) # returns True
    print((H(X) == I(X & Y) + H(X | Y+Z)).implies(markov(X, Y, Z))) # returns True
    print((H(X+Y) - H(X) - H(Y)).simplified()) # gives -I(Y;X)

    # Defining Wyner's common information [Wyner 1975]
    wci = markov(X, U, Y).exists(U).minimum(I(U & X+Y))

    # Defining Gacs-Korner common information [Gacs-Korner 1973]
    gkci = ((H(U|X) == 0) & (H(U|Y) == 0)).exists(U).maximum(H(U))

    print(bool(emin(H(X), H(Y)) >= wci)) # min(H(X),H(Y)) >= Wyner is True
    print(bool(wci >= I(X & Y))) # Wyner >= I(X;Y) is True
    print(bool(I(X & Y) >= gkci)) # I(X;Y) >= Gacs-Korner is True

    # The meet or Gacs-Korner common part [Gacs-Korner 1973] between X and Y
    # is a function of the GK common part between X and (Y,Z)
    print(bool(H(meet(X, Y) | meet(X, Y + Z)) == 0)) # returns True
    
    # If either Y is independent of (X,Z), or Z is independent of (X,Y),
    # then the minimal sufficient statistic of (X,Y) about (X,Z) is X
    print((indep(Y, X + Z) | indep(Z, X + Y)).implies(
        equiv(mss(X + Y, X + Z), X))) # returns True

    # The condition "there exists Y independent of (X,Z) such that 
    #  X-Y-Z forms a Markov chain" can be simplified to "X,Z independent"
    print((indep(X+Z, Y) & markov(X, Y, Z)).exists(Y, toreal = True))
    # the above gives { I(Z;X) == 0 }


    R = real("R") # declare real variable
    logg = real("logg")

    # Channel with state information at encoder, lower bound
    r_op = ((R <= I(M & Y)) & indep(M,S) & markov(M, X+S, Y)
            & (R >= 0)).exists(M).marginal_exists(X)
    
    # Gelfand-Pinsker theorem [Gel'fand-Pinsker 1980]
    r = ((R <= I(U & Y) - I(U & S)) & markov(U, X+S, Y)
            & (R >= 0)).exists(U).marginal_exists(X)
    
    # Using strong functional representation lemma [Li-El Gamal 2018]
    with PsiOpts(truth = sfrl(logg)):
        
        # Automated achievability proof of Gelfand-Pinsker theorem
        print(r.implies(r_op.relaxed(R, logg * 5))) # returns True
    
    # Automated converse proof of Gelfand-Pinsker theorem
    aux = r.check_converse(r_op, nature = S)

    # Print auxiliary RVs
    for (a, b) in aux:
        print(str(a) + " : " + str(b))


    # Zhang-Yeung inequality [Zhang-Yeung 1998] cannot be proved by Shannon-type inequalities
    print(bool(2*I(Z&W) <= I(X&Y) + I(X & Z+W) + 3*I(Z&W | X) + I(Z&W | Y))) # returns False
    
    # Using copy lemma [Zhang-Yeung 1998], [Dougherty-Freiling-Zeger 2011]
    with PsiOpts(truth = copylem()):
        
        # Prove Zhang-Yeung inequality
        print(bool(2*I(Z&W) <= I(X&Y) + I(X & Z+W) + 3*I(Z&W | X) + I(Z&W | Y))) # returns True


Psitip supports advanced features such as automated converse proofs (e.g. Gelfand-Pinsker theorem above), Fourier-Motzkin elimination, non-Shannon-type inequalities, automated search for auxiliary random variables, automated checking of whether a region tensorizes, and user-defined information quantities (e.g. Wyner's CI and Gacs-Korner CI above). Psitip is optimized for random variables following a Bayesian network structure, which can greatly improve performance.


About
~~~~~

Author: Cheuk Ting Li ( https://www.ie.cuhk.edu.hk/people/ctli.shtml ). The source code of Psitip is released under the GNU General Public License v3.0 ( https://www.gnu.org/licenses/gpl-3.0.html ).

The author would like to thank Raymond W. Yeung and Chandra Nair for their invaluable comments.


WARNING
~~~~~~~

This program comes with ABSOLUTELY NO WARRANTY. This section lists some known limitations of Psitip.

This program may produce false rejects (i.e., returning False when checking a true inequality) in scenarios involving:

- True non-Shannon-type inequalities. Nevertheless, Psitip supports the strong functional representation lemma and the copy lemma, which allows it to prove some results that cannot be proved using Shannon-type inequalities.

- Union of regions. While Psitip will attempt to perform deduction in this case, unions are generally much harder to handle than intersection.

- A lower (resp. upper) bound on the maximum (resp. minimum) of two expressions, by extension of the limitation about union.

- Auxiliary random variable searching. Psitip can only identify auxiliaries that are joint random variables of existing random variables (as in Gallager converse), and those produced by the strong functional representation lemma (if :code:`sfrl(logg)` is assumed).

- Numerical errors in the linear programming solver.


This program may produce false accepts (i.e., declaring a false inequality to be true) in scenarios involving:

- Existential quantification on random variables with the option :code:`toreal = True` (see Advanced section).

- Numerical errors in the linear programming solver.

As in most automated deduction programs, false rejects (i.e., failure to prove a true statement) are commonplace and should be expected. On the other hand, false accepts should be less common. If you encounter a false accept in Psitip outside of the aforementioned cases, please let me know.


Installation
~~~~~~~~~~~~

Download `psitip.py <https://raw.githubusercontent.com/cheuktingli/psitip/master/psitip.py>`_ and place it in the same directory as your code, or open an IPython shell in the same directory as psitip.py. The file `test.py <https://raw.githubusercontent.com/cheuktingli/psitip/master/test.py>`_ contains examples of usages of Psitip. Use :code:`from psitip import *` in your code to import all functions in psitip.

Python 3 and numpy are required to run psitip. It also requires at least one of the following for sparse linear programming:

- **PuLP** (https://github.com/coin-or/pulp). Recommended. Can use GLPK (installed separately), CBC (https://github.com/coin-or/Cbc , provided with PuLP, not recommended) or another solver.
- **Pyomo** (https://github.com/Pyomo/pyomo). Also requires GLPK or another solver.
- **GLPK** (https://www.gnu.org/software/glpk/). Recommended. An external solver to be used with PuLP or Pyomo. Can be installed using Conda (see https://anaconda.org/conda-forge/glpk ).
- **SciPy** (https://www.scipy.org/). Not recommended for problems with more than 8 random variables.

See the Solver section for details.



Solver
~~~~~~

The default solver is Scipy, though it is highly recommended to switch to another solver, e.g.:

.. code-block:: python

    from psitip import *
    PsiOpts.set_setting(solver = "pulp.glpk")
    PsiOpts.set_setting(solver = "pyomo.glpk")
    PsiOpts.set_setting(solver = "pulp.cbc") # Not recommended

PuLP supports a wide range of solvers (see https://coin-or.github.io/pulp/technical/solvers.html ). Use the following line to set the solver to any supported solver:

.. code-block:: python

    PsiOpts.set_setting(pulp_solver = pulp.solvers.GLPK(msg = 0)) # Or another solver

For Pyomo (see https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers ), use the following line (replace ??? with the desired solver):

.. code-block:: python

    PsiOpts.set_setting(solver = "pyomo.???")

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

 - Expressions CANNOT be multiplied with each other. :code:`H(X) * H(Y)` is invalid.
 
 - While Psitip can handle affine expressions like :code:`H(X) + 1` (i.e., adding or subtracting a constant), affine expressions are highly unrecommended as they are prone to numerical error in the solver.

- When two expressions are compared (using :code:`<=`, :code:`>=` or :code:`==`), the return value is a :code:`Region` object (not a :code:`bool`). The :code:`Region` object represents the set of distributions where the condition is satisfied. E.g. :code:`I(X & Y) == 0`, :code:`H(X | Y) <= H(Z) + a`.
 
 - While Psitip can handle general affine and half-space constraints like :code:`H(X) <= 1` (i.e., comparing an expression with a nonzero constant, or comparing affine expressions), they are highly unrecommended as they are prone to numerical error in the solver.
 
 - While Psitip can handle strict inequalities like :code:`H(X) > H(Y)`, strict inequalities are highly unrecommended as they are prone to numerical error in the solver.

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

- The :code:`rv_array` method constructs an array of random variables. For example, :code:`X = rv_array("X", 0, 10)` gives a :code:`Comp` object consisting of X0, X1, ..., X9.

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

- **Material implication** between :code:`Region` is denoted by the operator :code:`>>`, which returns a :code:`Region` object. The region :code:`r1 >> r2` represents the condition that :code:`r2` is true whenever :code:`r1` is true. Note that :code:`r1.implies(r2)` is equivalent to :code:`bool(r1 >> r2)`.

 - **Material equivalence** is denoted by the operator :code:`==`, which returns a :code:`Region` object. The region :code:`r1 == r2` represents the condition that :code:`r2` is true if and only if :code:`r1` is true.

- **Universal quantification** is represented by the :code:`forall` method of :code:`Region` (which returns a :code:`Region`). This is usually called after the implication operator :code:`>>`. For example, the condition "for all U such that U-X-(Y1,Y2) forms a Markov chain, we have I(U;Y1) >= I(U;Y2)" (less noisy broadcast channel [Korner-Marton 1975]) is represented by:

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

- The **meet** or **Gacs-Korner common part** [Gacs-Korner 1973] between X and Y is denoted as :code:`meet(X, Y)` (a :code:`Comp` object).

- The **minimal sufficient statistic** of X about Y is denoted as :code:`mss(X, Y)` (a :code:`Comp` object).

- The random variable given by the **strong functional representation lemma** [Li-El Gamal 2018] applied on X, Y (:code:`Comp` objects) with a gap term logg (:code:`Expr` object) is denoted as :code:`sfrl_rv(X, Y, logg)` (a :code:`Comp` object). If the gap term is omitted, this will be the ordinary functional representation lemma [El Gamal-Kim 2011].


Fourier-Motzkin elimination
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`exists` method of :code:`Region` with real variable arguments performs Fourier-Motzkin elimination over those variables, for example:

.. code-block:: python

    from psitip import *
    PsiOpts.set_setting(solver = "pulp.glpk")

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
    
    region_str = r.exists(R10+R20+Rs+U0+U1+U2).tostring(
        tosort = True, lhsvar = R0+R1+R2)
    print(region_str)


Automated Converse Proof
~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`check_converse` method of :code:`Region` attempts to prove the two-letter version of the converse by identifying auxiliary random variables. In the call:

.. code-block:: python

    r.check_converse(r_op, chan_cond = c, nature = S)

- :code:`r` is the :code:`Region` object specifying the outer bound to be proved.

- :code:`r_op` is the :code:`Region` object specifying the operational region of the setting. While the operational region is usually defined for n-letter, you only need to specify the single-letter version. The :code:`check_converse` call automatically computes the product of two copies of :code:`r_op`. Basically, :code:`check_converse` checks whether the two-letter :code:`r_op` is a subset of the two-letter :code:`r`.

- :code:`chan_cond = c` specifies that the condition :code:`c` (a :code:`Region` object) must be satisfied by each of the two copies of the channel (e.g. more capable broadcast channel).

- :code:`nature = S` specifies that the random variables in :code:`S` (a :code:`Comp` object) are independent across the two copies of the channel (e.g. the state of a channel).

- The return value of :code:`check_converse` is a list of pairs of :code:`Comp` objects, where the first of each pair is the auxiliary, and the second of each pair is its assignment. The return value is :code:`None` if a valid auxiliary assignment cannot be found.

- For channel coding settings, call :code:`marginal_exists` on :code:`r` and :code:`r_op` to mark the input random variables of the channel. The input random variables will not be assumed to be independent across the two copies of the channel. All other random variables in the channel will be assumed to be conditionally independent across the two copies of the channel given the input random variables.

- For source coding settings, call :code:`kernel_exists` on :code:`r` and :code:`r_op` to mark the output random variables.

- If you know for sure the auxiliary :code:`U` must contain the random variable :code:`M2`, call :code:`check_converse` with the argument :code:`hint_aux = [(U, M2)]`. Add more entries to the list if needed (e.g. :code:`hint_aux = [(U2, M2), (U3, M3)]`). This may speed up the search.

All the above also applies to the :code:`tensorize` method of :code:`Region` (e.g. :code:`r.tensorize(chan_cond = c, nature = S)`), which checks whether the two-letter version of :code:`r` is a subset of the single-letter version (except that there is no argument :code:`r_op`).

The following code demonstrates its use in proving that superposition coding is optimal for more capable broadcast channel (this program can take several minutes):

.. code-block:: python

    from psitip import *
    PsiOpts.set_setting(solver = "pulp.glpk")

    R1, R2 = real("R1", "R2")
    U, X, Y1, Y2, M1, M2 = rv("U", "X", "Y1", "Y2", "M1", "M2")
    
    # Broadcast channel operational region
    r_op = ((R1 <= I(M1 & Y1)) & (R2 <= I(M2 & Y2)) & indep(M1,M2) & markov(M1+M2, X, Y1+Y2)
            & (R1 >= 0) & (R2 >= 0)).exists(M1+M2).marginal_exists(X)
    
    # Superposition coding region [Bergmans 1973], [Gallager 1974]
    r = ((R2 <= I(U & Y2)) & (R1 + R2 <= I(X & Y1 | U) + I(U & Y2)) & (R1 + R2 <= I(X & Y1))
                & markov(U, X, Y1+Y2) & (R1 >= 0) & (R2 >= 0)).exists(U).marginal_exists(X)
    
    # More capable [Korner-Marton 1975]
    # Reads: For all marginal distr. of X, I(X & Y1) >= I(X & Y2)
    c_mc = (I(X & Y1) >= I(X & Y2)).marginal_forall(X).convexified(forall = True)

    # Attempt to prove converse assuming more capable
    aux = r.check_converse(r_op, chan_cond = c_mc)
    
    # Print auxiliary RVs
    for (a, b) in aux:
        print(str(a) + " : " + str(b))


The following code demonstrates its use in proving the converse part in Wyner-Ziv theorem for source coding with side information at the decoder [Wyner-Ziv 1976] (note that Psitip does not support distortion constraints):

.. code-block:: python

    from psitip import *
    PsiOpts.set_setting(solver = "pulp.glpk")

    R = real("R")
    U, X, Y, Z, M = rv("U", "X", "Y", "Z", "M")
    
    # Lossy source coding with side information at decoder, upper bound
    r_op = ((R >= I(M & X)) & markov(M, X, Y) & markov(X, M+Y, Z)
            ).exists(M).kernel_exists(Z)
    
    # Wyner-Ziv theorem [Wyner-Ziv 1976]
    r = ((R >= I(X & U | Y)) & markov(U, X, Y) & markov(X, U+Y, Z)
            ).exists(U).kernel_exists(Z)
    
    # Automated converse proof
    aux = r.check_converse(r_op)
    
    # Print auxiliary RVs
    for (a, b) in aux:
        print(str(a) + " : " + str(b))


Bayesian network optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian network optimization is turned on by default. It builds a Bayesian network automatically using the given conditional independence conditions, so as to reduce the dimension of the linear programming problem. The speed up is significant when the Bayesian network is sparse, for instance, when the variables form a Markov chain:

.. code-block:: python

    X = rv_array("X", 0, 9)
    print(bool(markov(*X) >> (I(X[0] & X[8]) <= H(X[4]))))

Nevertheless, building the Bayesian network can take some time. If your problem does not admit a sparse Bayesian network structure, you may turn off this optimization by:

.. code-block:: python

    PsiOpts.set_setting(lptype = "H")

The :code:`get_bayesnet` method of :code:`Region` returns a :code:`BayesNet` object (a Bayesian network) that can be deduced by the conditional independence conditions in the region. The :code:`check_ic` method of :code:`BayesNet` checks whether an expression containing conditional mutual information terms is always zero, e.g.:

.. code-block:: python

    ((I(X&Y|Z) == 0) & (I(U&X+Z|Y) <= 0)).get_bayesnet().check_ic(I(X&U|Z))


Built-in functions
~~~~~~~~~~~~~~~~~~

There are several built-in information functions listed below. While they can be defined by the user easily (see the source code for their definitions), they are provided for convenience.

Theorems
--------

The following are true statements (:code:`Region` objects) that allow Psitip to prove results not provable by Shannon-type inequalities (at the expense of longer computation time). They can either be used in the context manager (e.g. :code:`with PsiOpts(truth = sfrl(logg)):`, see Options section), or directly (e.g. sfrl().implies(excess_fi(X, Y) <= H(X | Y))).

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
    with PsiOpts(truth = sfrl(logg)):
        
        # Automated achievability proof of Gelfand-Pinsker theorem
        print(r.implies(r_op.relaxed(R, logg * 5))) # returns True

 - Note that writing :code:`with PsiOpts(truth = sfrl(logg)):` allows SFRL to be used only once. To allow it to be used twice, write :code:`with PsiOpts(truth = sfrl(logg) & sfrl(logg)):`.

- **Copy lemma** [Zhang-Yeung 1998], [Dougherty-Freiling-Zeger 2011] is given by :code:`copylem(n, m)`. It states that for any random variables X_1,...,X_n,Y_1,...,Y_m, there exists Z_1,...,Z_m such that (X_1,...,X_n,Y_1,...,Y_m) has the same distribution as (X_1,...,X_n,Z_1,...,Z_m) (only equalities of entropies are enforced in Psitip), and (Y_1,...,Y_m)-(X_1,...,X_n)-(Z_1,...,Z_m) forms a Markov chain. The default values of n, m are 2, 1 respectively. For example:

  .. code-block:: python

    # Using copy lemma
    with PsiOpts(truth = copylem()):
        
        # Prove Zhang-Yeung inequality
        print(bool(2*I(Z&W) <= I(X&Y) + I(X & Z+W) + 3*I(Z&W | X) + I(Z&W | Y))) # returns True

- **Double Markov property** [Csiszar-Korner 2011] is given by :code:`dblmarkov()`. It states that if X-Y-Z and Y-X-Z are Markov chains, then there exists W that is a function of X, a function of Y, and (X,Y)-W-Z is Markov chain. For example:

  .. code-block:: python
  
    # Using double Markov property
    with PsiOpts(truth = dblmarkov()):
        aux = ((markov(X, Y, Z) & markov(Y, X, Z))
            >> (H(mss(X, Z) | mss(Y, Z)) == 0)).check_getaux()
        print(IUtil.list_tostr_std(aux))
        
        aux = ((markov(X, Y, Z) & markov(Y, X, Z))
            >> markov(X+Y, meet(X, Y), Z)).check_getaux()
        print(IUtil.list_tostr_std(aux))

- **Existence of meet and minimal sufficient statistics** is given by :code:`existence(meet)` and :code:`existence(mss)` respectively.

Conditions
----------

The following are conditions (:code:`Region` objects) on the random variable arguments.

- **Mutual independence** is expressed as :code:`indep(X, Y, Z)`. The function :code:`indep` can take any number of arguments. For random sequence :code:`X = rv_array("X", 0, 5)`, the mutual independence condition can be expressed as :code:`indep(*X)`.

- **Markov chain** is expressed as :code:`markov(X, Y, Z)`. The function :code:`markov` can take any number of arguments. For random sequence :code:`X = rv_array("X", 0, 5)`, the Markov chain condition can be expressed as :code:`markov(*X)`.

- **Informational equivalence** (i.e., containing the same information) is expressed as :code:`equiv(X, Y, Z)`. The function :code:`equiv` can take any number of arguments. Note that :code:`equiv(X, Y)` is the same as :code:`(H(X|Y) == 0) & (H(Y|X) == 0)`.

- **Same distribution**. The condition that (X,Y) has the same distribution as (Z,W) is expressed as :code:`eqdist([X, Y], [Z, W])`. The function :code:`eqdist` can take any number of arguments (that are all lists). Note that only equalities of entropies are enforced (i.e., H(X)=H(Z), H(Y)=H(W), H(X,Y)=H(Z,W)).

- **Exchangeability** is expressed as :code:`exchangeable(X, Y, Z)`. The function :code:`exchangeable` can take any number of arguments. For random sequence :code:`X = rv_array("X", 0, 5)`, the condition that it is an exchangeable sequence of random variables can be expressed as :code:`exchangeable(*X)`. Note that only equalities of entropies are enforced.

- **IID sequence** is expressed as :code:`iidseq(X, Y, Z)`. The function :code:`iidseq` can take any number of arguments. For random sequence :code:`X = rv_array("X", 0, 5)`, the condition that it is an IID sequence of random variables can be expressed as :code:`iidseq(*X)`. Note that only equalities of entropies are enforced.


Random variables
----------------

The following are :code:`Comp` objects (random-variable-valued functions).

- **Meet** or **Gacs-Korner common part** [Gacs-Korner 1973] between X and Y is denoted as :code:`meet(X, Y)` (a :code:`Comp` object).

- **Minimal sufficient statistic** of X about Y is denoted as :code:`mss(X, Y)` (a :code:`Comp` object).

- The random variable given by the **strong functional representation lemma** [Li-El Gamal 2018] applied on X, Y (:code:`Comp` objects) with a gap term logg (:code:`Expr` object) is denoted as :code:`sfrl_rv(X, Y, logg)` (a :code:`Comp` object). If the gap term is omitted, this will be the ordinary functional representation lemma [El Gamal-Kim 2011].


Real-valued information quantities
----------------------------------

The following are :code:`Expr` objects (real-valued functions).

- **Gacs-Korner common information** [Gacs-Korner 1973] is given by :code:`gacs_korner(X & Y)`. The multivariate conditional version can be obtained by :code:`gacs_korner(X & Y & Z | W)`. The following tests return True:

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



Options
~~~~~~~

There are two ways to set options. One can set an option globally using:

.. code-block:: python

    PsiOpts.set_setting(option = value)

or locally within a :code:`with` block using context manager:

.. code-block:: python

    with PsiOpts(option = value):
        # do something here

Some of the options are:

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

- :code:`str_style` : The style of string conversion :code:`str(x)` and verbose output. Values are :code:`PsiOpts.STR_STYLE_STANDARD` (e.g. :code:`3I(X,Y;Z|W)-H(X)`, default) or :code:`PsiOpts.STR_STYLE_PSITIP` (e.g. :code:`3*I(X+Y&Z|W)-H(X)`, consistent with the Psitip syntax so the output can be copied back to the code).

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
    LP nrv=2 nreal=0 nvar=3/3 nineq=3 neq=0 solver=pulp.glpk
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


Results used as examples above:

- Peter Gacs and Janos Korner. Common information is far less than mutual information.Problems of Control and Information Theory, 2(2):149-162, 1973.

- \A. D. Wyner. The common information of two dependent random variables. IEEE Trans. Info. Theory, 21(2):163-179, 1975.

- \S. I. Gel'fand and M. S. Pinsker, "Coding for channel with random parameters," Probl. Contr. and Inf. Theory, vol. 9, no. 1, pp. 19-31, 1980.

- Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978.

- \K. Marton, "A coding theorem for the discrete memoryless broadcast channel," IEEE  Transactions on Information Theory, vol. 25, no. 3, pp. 306-311, May 1979.

- \Y. Liang and G. Kramer, "Rate regions for relay broadcast channels," IEEE Transactions on Information Theory, vol. 53, no. 10, pp. 3517-3535, Oct 2007.

- Bergmans, P. "Random coding theorem for broadcast channels with degraded components." IEEE Transactions on Information Theory 19.2 (1973): 197-207.

- Gallager, Robert G. "Capacity and coding for degraded broadcast channels." Problemy  Peredachi Informatsii 10.3 (1974): 3-14.

- \J. Korner and K. Marton, Comparison of two noisy channels, Topics in Inform. Theory (ed. by I. Csiszar and P. Elias), Keszthely, Hungary (August, 1975), 411-423.

- El Gamal, Abbas, and Young-Han Kim. Network information theory. Cambridge University Press, 2011.

- Watanabe S (1960). Information theoretical analysis of multivariate correlation, IBM Journal of Research and Development 4, 66-82. 

- Han T. S. (1978). Nonnegative entropy measures of multivariate symmetric correlations, Information and Control 36, 133-156. 

- McGill, W. (1954). "Multivariate information transmission". Psychometrika. 19 (1): 97-116.

- Csiszar, Imre, and Prakash Narayan. "Secrecy capacities for multiple terminals." IEEE Transactions on Information Theory 50, no. 12 (2004): 3047-3061.

- Tishby, Naftali, Pereira, Fernando C., Bialek, William (1999). The Information Bottleneck Method. The 37th annual Allerton Conference on Communication, Control, and Computing. pp. 368-377.

- \U. Maurer and S. Wolf. "Unconditionally secure key agreement and the intrinsic conditional information." IEEE Transactions on Information Theory 45.2 (1999): 499-514.

- Wyner, Aaron, and Jacob Ziv. "The rate-distortion function for source coding with side information at the decoder." IEEE Transactions on information Theory 22.1 (1976): 1-10.

- Randall Dougherty, Chris Freiling, and Kenneth Zeger. "Non-Shannon information inequalities in four random variables." arXiv preprint arXiv:1104.3602 (2011).

- Imre Csiszar and Janos Korner. Information theory: coding theorems for discrete memoryless systems. Cambridge University Press, 2011.
