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
Examples and tests 
Copyright (C) 2020  Cheuk Ting Li

References for the examples used are provided at the end.
"""

from psitip import *
import unittest
import time

class TestPsitip(unittest.TestCase):
    
    def test_marton(self):
        
        # Fourier-Motzkin elimination for Marton's inner bound with common message
        # [Marton 1979], [Liang-Kramer 2007]
        R0, R1, R2, R10, R20, Rs = real("R0", "R1", "R2", "R10", "R20", "Rs")
        U0, U1, U2, X, Y1, Y2 = rv("U0", "U1", "U2", "X", "Y1", "Y2")
        
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
        
        r &= markov(U0+U1+U2, X, Y1+Y2)
        
        region_str = r.exists(R10+R20+Rs+U0+U1+U2).tostring(tosort = True, lhsvar = R0+R1+R2)
        print(region_str)
        self.assertEqual(region_str, 
           ("{ R0 >= 0,\n"
            "  R1 >= 0,\n"
            "  R2 >= 0,\n"
            "  R0+R1 <= I(U0,U1;Y1),\n"
            "  R0+R2 <= I(U0,U2;Y2),\n"
            "  R0+R1+R2 <= I(U0,U1;Y1)+I(U2;Y2|U0)-I(U1;U2|U0),\n"
            "  R0+R1+R2 <= I(U0,U2;Y2)+I(U1;Y1|U0)-I(U1;U2|U0),\n"
            "  2R0+R1+R2 <= I(U0,U1;Y1)+I(U0,U2;Y2)-I(U1;U2|U0),\n"
            "  I(U0,U1,U2;Y1,Y2|X) == 0,\n"
            "  I(U1;U2|U0) <= I(U1;Y1|U0)+I(U2;Y2|U0) } , exists U0,U1,U2")
            )
        
        
    def test_implication(self):
        
        X, Y1, Y2, U = rv("X", "Y1", "Y2", "U")
        
        # Degraded broadcast channel [Cover 1972]
        r_deg = markov(X, Y1, Y2)
        
        # Less noisy  [Korner-Marton 1975]
        # Reads: For all marginal distr. of X, for all U, markov(U,X,Y1+Y2) implies I(U & Y1) >= I(U & Y2)
        r_ln = (markov(U,X,Y1+Y2) >> (I(U & Y1) >= I(U & Y2))).forall(U).marginal_forall(X).convexified(forall = True)
        
        # More capable  [Korner-Marton 1975]
        # Reads: For all marginal distr. of X, I(X & Y1) >= I(X & Y2)
        r_mc = (I(X & Y1) >= I(X & Y2)).marginal_forall(X).convexified(forall = True)
        
        self.assertTrue(r_deg.implies(r_ln))
        self.assertTrue(r_ln.implies(r_mc))
        self.assertFalse(r_ln.implies(r_deg))
        self.assertFalse(r_mc.implies(r_ln))
        
        
    def test_commoninfo(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        # Properties of Gacs-Korner common information [Gacs-Korner 1973], 
        # Wyner's common information [Wyner 1975] and 
        # Common entropy (one-shot exact common information) [Kumar-Li-El Gamal 2014]
        
        # Comparisons
        self.assertTrue(emin(H(X), H(Y)) >= exact_ci(X & Y))
        self.assertTrue(exact_ci(X & Y) >= wyner_ci(X & Y))
        self.assertTrue(wyner_ci(X & Y) >= I(X & Y))
        self.assertTrue(I(X & Y) >= gacs_korner(X & Y))
        
        self.assertFalse(emin(H(X), H(Y)) <= exact_ci(X & Y))
        self.assertFalse(exact_ci(X & Y) <= wyner_ci(X & Y))
        self.assertFalse(wyner_ci(X & Y) <= I(X & Y))
        self.assertFalse(I(X & Y) <= gacs_korner(X & Y))
        
        # Data processing
        r = markov(X, Y, Z)
        
        self.assertTrue(r >> (exact_ci(X & Y) >= exact_ci(X & Z)))
        self.assertTrue(r >> (wyner_ci(X & Y) >= wyner_ci(X & Z)))
        self.assertTrue(r >> (gacs_korner(X & Y) >= gacs_korner(X & Z)))
        
        self.assertFalse(r >> (exact_ci(X & Y) <= exact_ci(X & Z)))
        self.assertFalse(r >> (wyner_ci(X & Y) <= wyner_ci(X & Z)))
        self.assertFalse(r >> (gacs_korner(X & Y) <= gacs_korner(X & Z)))
        
        # Tensorize
        r = indep(X + Y, Z + W)
        
        self.assertTrue(r >> (exact_ci(X & Y) + exact_ci(Z & W) >= exact_ci(X + Z & Y + W)))
        self.assertFalse(r >> (exact_ci(X & Y) + exact_ci(Z & W) <= exact_ci(X + Z & Y + W)))
        self.assertTrue(r >> (wyner_ci(X & Y) + wyner_ci(Z & W) >= wyner_ci(X + Z & Y + W)))
        self.assertTrue(r >> (wyner_ci(X & Y) + wyner_ci(Z & W) <= wyner_ci(X + Z & Y + W)))
        # self.assertTrue(r >> (gacs_korner(X & Y) + gacs_korner(Z & W) >= gacs_korner(X + Z & Y + W)))
        self.assertTrue(r >> (gacs_korner(X & Y) + gacs_korner(Z & W) <= gacs_korner(X + Z & Y + W)))
        
        # Independent parts
        r = indep(X, Y, Z)
        self.assertTrue(r >> (exact_ci(X+Z & Y+Z) == H(Z)))
        self.assertTrue(r >> (wyner_ci(X+Z & Y+Z) == H(Z)))
        self.assertTrue(r >> (gacs_korner(X+Z & Y+Z) == H(Z)))
        
        # Symmetry
        self.assertTrue(exact_ci(X & Y) == exact_ci(Y & X))
        self.assertTrue(wyner_ci(X & Y) == wyner_ci(Y & X))
        self.assertTrue(gacs_korner(X & Y) == gacs_korner(Y & X))
        
        # Independence
        self.assertTrue((exact_ci(X & Y) == 0) == indep(X, Y))
        self.assertTrue((wyner_ci(X & Y) == 0) == indep(X, Y))
        self.assertFalse((gacs_korner(X & Y) == 0) == indep(X, Y))
        
        
    def test_commoninfo_def(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        # Defining Wyner's common information etc. explicitly
        eci = markov(X, U, Y).minimum(H(U), U)
        wci = markov(X, U, Y).minimum(I(U & X+Y), U)
        gkci = ((H(U|X) == 0) & (H(U|Y) == 0)).maximum(H(U), U)
        
        self.assertTrue(exact_ci(X & Y) == eci)
        self.assertTrue(wyner_ci(X & Y) == wci)
        self.assertTrue(gacs_korner(X & Y) == gkci)
        self.assertTrue(gacs_korner(X & Y) == H(meet(X, Y)))
        
        # Comparisons
        self.assertTrue(emax(H(X), H(Y)) >= eci)
        self.assertTrue(eci >= wci)
        self.assertTrue(wci >= I(X & Y))
        self.assertTrue(I(X & Y) >= gkci)
        
        self.assertFalse(emax(H(X), H(Y)) <= eci)
        self.assertFalse(eci <= wci)
        self.assertFalse(wci <= I(X & Y))
        self.assertFalse(I(X & Y) <= gkci)
        
        # self.assertTrue((H(X) >= eci) & (H(Y) >= eci)) # This currently fails
        
        
    def test_commoninfo_discover(self):
        
        X, Y = rv("X", "Y")
        
        K = gacs_korner(X&Y)
        J = wyner_ci(X&Y)
        G = exact_ci(X&Y)
        
        RK, RJ, RG = real("RK", "RJ", "RG")
        
        # Automatically discover relationship between these quantities
        region_str = str(universe().discover([(RK, K), (RJ, J), (RG, G), X, Y], maxsize = 2))
        print("test_commoninfo_discover")
        print(region_str)
        
        self.assertEqual(region_str, 
           ("{ RK >= 0,\n"
            "  RJ <= RG,\n"
            "  RG <= H(X),\n"
            "  RG <= H(Y),\n"
            "  RK <= I(X;Y),\n"
            "  RJ >= I(X;Y) }")
            )
        
    def test_new_rv(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        # Gacs-Korner common part [Gacs-Korner 1973]
        K = meet(X, Y)
        self.assertTrue(H(K | X) == 0)
        self.assertTrue(((H(U | X) == 0) & (H(U | Y) == 0)) >> (H(U | K) == 0))
        self.assertFalse(H(X | K) == 0)
        self.assertFalse((H(U | X) == 0) >> (H(U | K) == 0))
        self.assertTrue(indep(X, Y) >> (H(K) == 0))
        self.assertTrue(equiv(meet(X+Y, Y), Y))
        self.assertTrue(H(meet(X+Y,Y+Z)) >= H(Y))
        self.assertTrue(H(Y | meet(X + Y, Y + Z)) == 0)
        self.assertTrue(H(meet(X, Y) | meet(X, Y + Z)) == 0)
        self.assertTrue(indep(X, Y, Z) >> equiv(meet(X+Y,Y+Z), Y))
        self.assertFalse(indep(X, Y, Z) << equiv(meet(X+Y,Y+Z), Y))
        
        # Minimal sufficient statistic of X about Y
        S = mss(X, Y)
        self.assertTrue(H(S | X) == 0)
        self.assertTrue(markov(X, S, Y))
        self.assertTrue(((H(U | X) == 0) & markov(X, U, Y)) >> (H(S | U) == 0))
        self.assertTrue(indep(X, Y) >> (H(S) == 0))
        self.assertTrue(equiv(mss(X+Y, Y), Y))
        self.assertTrue(equiv(mss(X, X+Y), X))
        self.assertTrue((indep(Y, X + Z) | indep(Z, X + Y)).implies(equiv(mss(X + Y, X + Z), X)))
        self.assertTrue(H(meet(X, Y) | mss(X, Y)) == 0)
        
        # Strong functional representation lemma [Li-El Gamal 2018]
        logg = real("logg")
        F = sfrl_rv(X, Y, logg)
        self.assertTrue(indep(X, F))
        self.assertTrue(H(Y | X+F) == 0)
        self.assertTrue(H(Y | F) <= I(X & Y) + logg)
        self.assertTrue(markov(F, X+Y, Z))
        
        
        
    def test_csie_region(self):
        
        R = real("R")
        U, X, Y, S, M = rv("U", "X", "Y", "S", "M")
        logg = real("logg")
        
        # Channel with state information at encoder, lower bound
        r_op = ((R <= I(M & Y)) & indep(M,S) & markov(M, X+S, Y)
             & (R >= 0)).exists(M).marginal_exists(X)
        
        # Gelfand-Pinsker theorem [Gel'fand-Pinsker 1980]
        r = ((R <= I(U & Y) - I(U & S)) & markov(U, X+S, Y)
             & (R >= 0)).exists(U).marginal_exists(X)
        
        # Strong functional representation lemma with logarithmic gap [Li-El Gamal 2018]
        with sfrl(logg).assumed():
            
            # Achievability: GP region implies operational region relaxed by log gap
            self.assertTrue(r >> r_op.relaxed(R, logg * 5))
        
        # Converse: Specify that S comes from nature, and are independent across time
        aux = r.check_converse(r_op, nature = S)
            
        print("test_csie_region")
        print(iutil.list_tostr_std(aux))
        self.assertEqual(str(aux[0][1]), "M,Y_")
        self.assertEqual(str(aux[1][1]), "M,S")
        
        
    def test_wz_region(self):
        
        R = real("R")
        U, X, Y, Z, M = rv("U", "X", "Y", "Z", "M")
        
        # Lossy source coding with side information at decoder, upper bound
        r_op = ((R >= I(M & X)) & markov(M, X, Y) & markov(X, M+Y, Z)
             ).exists(M).kernel_exists(Z)
        
        # Wyner-Ziv theorem [Wyner-Ziv 1976]
        r = ((R >= I(X & U | Y)) & markov(U, X, Y) & markov(X, U+Y, Z)
             ).exists(U).kernel_exists(Z)
        
        # Converse
        aux = r.check_converse(r_op)
        
        print("test_wz_region")
        print(iutil.list_tostr_std(aux))
        self.assertTrue(aux is not None)
        #self.assertEqual(str(aux[0][1]), "M,Y_")
        #self.assertEqual(str(aux[1][1]), "M,Y")
        
        
        
    def test_dbc_region(self):
        
        R1, R2 = real("R1", "R2")
        U, X, Y1, Y2, M1, M2 = rv("U", "X", "Y1", "Y2", "M1", "M2")
        
        # Broadcast channel operational region
        r_op = ((R1 <= I(M1 & Y1)) & (R2 <= I(M2 & Y2)) & (H(X | M1+M2) == 0) & indep(M1,M2) & markov(M1+M2, X, Y1+Y2)
             & (R1 >= 0) & (R2 >= 0)).exists(M1+M2).marginal_exists(X)
        
        # Superposition coding region [Bergmans 1973], [Gallager 1974]
        r = ((R2 <= I(U & Y2)) & (R1 + R2 <= I(X & Y1 | U) + I(U & Y2)) & (R1 + R2 <= I(X & Y1))
                    & markov(U, X, Y1+Y2) & (R1 >= 0) & (R2 >= 0)).exists(U).marginal_exists(X)
        #r2 = ((R1 <= I(X & Y1 | U)) & (R2 <= I(U & Y2)) & (R1 + R2 <= I(X & Y1))
        #            & markov(U, X, Y1+Y2) & (R1 >= 0) & (R2 >= 0)).exists(U).marginal_exists(X)
        
        # Degraded broadcast channel [Cover 1972]
        c_dg = markov(X, Y1, Y2)
        
        # More capable [Korner-Marton 1975]
        # Reads: For all marginal distr. of X, I(X & Y1) >= I(X & Y2)
        c_mc = (I(X & Y1) >= I(X & Y2)).marginal_forall(X).convexified(forall = True)
        
        print("test_dbc_region")
        
        if True:
            # Attempt to tensorize assuming degraded
            aux = r.tensorize(chan_cond = c_dg)
            
            # Print auxiliary RVs
            print(iutil.list_tostr_std(aux))
                
            self.assertTrue(aux is not None)
            #self.assertEqual(str(aux[0][1]), "U")
            #self.assertEqual(str(aux[1][1]), "U,Y1")
        
        if True:
            # Attempt to prove converse assuming degraded
            aux = r.check_converse(reg_subset = r_op, chan_cond = c_dg)
            
            # Print auxiliary RVs
            print(iutil.list_tostr_std(aux))
                
            self.assertTrue(aux is not None)
            #self.assertEqual(str(aux[0][1]), "M2")
            #self.assertEqual(str(aux[1][1]), "M2,Y1")
            
        if True:
            # Attempt to prove converse assuming more capable
            aux = r.check_converse(reg_subset = r_op, chan_cond = c_mc)
            
            # Print auxiliary RVs
            print(iutil.list_tostr_std(aux))
                
            self.assertTrue(aux is not None)
            #self.assertEqual(str(aux[0][1]), "M1,Y1_")
            #self.assertEqual(str(aux[1][1]), "M1,Y2")
            #self.assertEqual(str(aux[2][1]), "M2,Y1_")
            #self.assertEqual(str(aux[3][1]), "M2,Y2")

        
        
    def test_gw_region(self):
        
        R0, R1, R2 = real("R0", "R1", "R2")
        X1, X2, U = rv("X1", "X2", "U")
        
        # Gray-Wyner region [Gray-Wyner 1974]
        r = ((R0 >= I(X1 + X2 & U)) & (R1 >= H(X1 | U))
                    & (R2 >= H(X2 | U))).exists(U)
        
        # Attempt to tensorize
        aux = r.tensorize()
            
        print("test_gw_region")
        print(iutil.list_tostr_std(aux))
        self.assertTrue(aux is not None)
        
        self.assertEqual(str(aux[0][1]), "U_1")
        self.assertEqual(str(aux[1][1]), "U_1,X1,X2")
        
        
    def test_nonshannon(self):
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
    
        print("test_nonshannon")
        # Zhang-Yeung inequality [Zhang-Yeung 1998] cannot be proved by Shannon-type inequalities
        self.assertFalse(2 * I(Z & W) <= I(X & Y) + I(X & Z+W) + 3 * I(Z & W | X) + I(Z & W | Y))
        
        # Using copy lemma [Zhang-Yeung 1998], [Dougherty-Freiling-Zeger 2011]
        with copylem().assumed():
            
            # Prove Zhang-Yeung inequality
            aux = (2 * I(Z & W) <= I(X & Y) + I(X & Z+W) + 3 * I(Z & W | X) + I(Z & W | Y)).check_getaux()
            self.assertTrue(aux is not None)
            print(iutil.list_tostr_std(aux))
            
            self.assertEqual(str(aux[0][1]), "X")
            self.assertEqual(str(aux[1][1]), "Z")
            self.assertEqual(str(aux[2][1]), "W")
        
        
    def test_dblmarkov(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        print("test_dblmarkov")
        
        # Using double Markov property [Csiszar-Korner 2011]
        with dblmarkov().assumed():
            aux = ((markov(X, Y, Z) & markov(Y, X, Z))
                >> (H(mss(X, Z) | mss(Y, Z)) == 0)).check_getaux()
            self.assertTrue(aux is not None)
            print(iutil.list_tostr_std(aux))
            
            aux = ((markov(X, Y, Z) & markov(Y, X, Z))
                >> markov(X+Y, meet(X, Y), Z)).check_getaux()
            self.assertTrue(aux is not None)
            print(iutil.list_tostr_std(aux))
            
            aux = ((markov(X, Y, Z) & markov(Y, X, Z) & markov(Y, Z, X))
                >> (indep(X, Y, Z).conditioned(U)
                  & (H(U|X)==0) & (H(U|Y)==0) & (H(U|Z)==0)).exists(U)).check_getaux()
            self.assertTrue(aux is not None)
            print(iutil.list_tostr_std(aux))
            
        
    def test_eliminate_toreal(self):
        
        X, Y, Z, W, U, V = rv("X", "Y", "Z", "W", "U", "V")
        IX, IY, IXY = real("IX", "IY", "IXY")
        
        self.assertEqual(str(
            (markov(X, Y, Z) & indep(X+Z, Y)).exists(Y, toreal = True)
            ), "I(X;Z) == 0")
        
        # Mutual information region outer bound [Li-El Gamal 2017]
        self.assertEqual((
                (I(X & U) == IX) & (I(Y & U) == IY) & (I(X+Y & U) == IXY)
        ).exists(U, toreal = True).tostring(tosort = True, lhsvar = IX+IY+IXY),
            ("{ IX >= 0,\n"
            "  IY >= 0,\n"
            "  IX <= IXY,\n"
            "  IY <= IXY,\n"
            "  IXY-IX <= H(Y|X),\n"
            "  IXY-IY <= H(X|Y),\n"
            "  IX+IY-IXY <= I(X;Y) }")
        )
        
        self.assertEqual(str(
            (markov(X, Y, U, W) & (H(U) == 0)).exists(U, toreal = True)),
            ("I(X,Y;W) == 0")
        )
        
        # This currently fails
        # print(((H(X | U) == 0) >> (H(Y | U) == 0)).forall(U, toreal = True))
        
        
    def test_convex(self):
        
        R1, R2 = real("R1", "R2")
        
        X, X1, X2 = rv("X", "X1", "X2")
        Y, Y1, Y2 = rv("Y", "Y1", "Y2")
        U, U1, U2 = rv("U", "U1", "U2")
        Q = rv("Q")
        
        # Multiple access channel region without time sharing [Ahlswede 1971], [Liao 1972]
        r = ((R1 <= I(X1 & Y | X2)) & (R2 <= I(X2 & Y | X1))
            & (R1 + R2 <= I(X1 + X2 & Y)) & (I(X1 & X2) == 0)).marginal_exists(X1 + X2)
        self.assertFalse(r.isconvex())
        
        # Multiple access channel region with time sharing
        r = ((R1 <= I(X1 & Y | X2 + Q)) & (R2 <= I(X2 & Y | X1 + Q))
            & (R1 + R2 <= I(X1 + X2 & Y | Q)) & (I(X1 & X2 | Q) == 0)).exists(Q).marginal_exists(X1 + X2)
        self.assertTrue(r.isconvex())
        
        # Degraded broadcast channel region [Bergmans 1973], [Gallager 1974]
        r = ((R1 <= I(X & Y1 | U)) & (R2 <= I(U & Y2))
            & (R1 + R2 <= I(X & Y1))).exists(U).marginal_exists(X)
        self.assertTrue(r.isconvex())
        
        # Berger-Tung inner bound without time sharing [Berger 1978], [Tung 1978]
        r = ((R1 >= I(X1 & U1 | U2)) & (R2 >= I(X2 & U2 | U1))
            & (R1 + R2 >= I(X1 + X2 & U1 + U2))
            & (I(U1 & X2 + U2 | X1) == 0) & (I(U2 & X1 + U1 | X2) == 0)).kernel_exists(U1+U2)
        self.assertFalse(r.isconvex())
        
        # Berger-Tung inner bound with time sharing
        r = ((R1 >= I(X1 & U1 | U2 + Q)) & (R2 >= I(X2 & U2 | U1 + Q))
            & (R1 + R2 >= I(X1 + X2 & U1 + U2 | Q))
            & (I(U1 & X2 + U2 | X1 + Q) == 0) & (I(U2 & X1 + U1 | X2 + Q) == 0)).exists(Q).kernel_exists(U1+U2)
        self.assertTrue(r.isconvex())
        
        # Berger-Tung outer bound
        r = ((R1 >= I(X1 + X2 & U1 | U2)) & (R2 >= I(X1 + X2 & U2 | U1))
            & (R1 + R2 >= I(X1 + X2 & U1 + U2))
            & (I(U1 & X2 | X1) == 0) & (I(U2 & X1 | X2) == 0)).kernel_exists(U1+U2)
        # self.assertTrue(r.isconvex())  # This test currently fails
        
        
    def test_other_quantities(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        # Necessary conditional entropy [Cuff-Permuter-Cover 2010]
        self.assertTrue(H_nec(Y | X) <= H(Y | X))
        
        # Excess functional information [Li-El Gamal 2018]
        self.assertTrue(excess_fi(X, Y) <= H(Y | X))
        
        # The following two requires functional representation lemma to prove
        with sfrl().assumed():
            # with PsiOpts(flatten_distribute = False):
            self.assertTrue(excess_fi(X, Y) <= exact_ci(X & Y) - I(X & Y))
            self.assertTrue(excess_fi(X, Y) <= H(X | Y))
        
        # Independent parts
        r = indep(X, Y, Z)
        self.assertTrue(r >> (excess_fi(X+Z, Y+Z) == 0))
        self.assertTrue(r >> (H_nec(X+Z | Y+Z) == 0))
        
        # Total correlation [Watanabe 1960]
        self.assertEqual(str(total_corr(X & Y & Z+W | U)), 
        "H(X|U)+H(Y|U)+H(Z,W|U)-H(X,Y,Z,W|U)")
        self.assertTrue(total_corr(X & Y & Z) == H(X) + H(Y) + H(Z) - H(X+Y+Z))
        
        # Dual total correlation [Han 1978]
        self.assertEqual(str(dual_total_corr(X & Y & Z+W | U)), 
        "H(X,Y,Z,W|U)-H(X|Y,Z,W,U)-H(Y|X,Z,W,U)-H(Z,W|X,Y,U)")
        self.assertTrue(dual_total_corr(X & Y & Z) == H(X+Y+Z) - H(X|Y+Z) - H(Y|X+Z) - H(Z|X+Y))
        
        # Mutual dependence [Csiszar-Narayan 2004]
        self.assertTrue(mutual_dep(X & Y & Z) == emin(I(X+Y & Z), I(X+Z & Y), I(Y+Z & X), total_corr(X & Y & Z) / 2))
        self.assertTrue(mutual_dep(X & Y & Z) <= total_corr(X & Y & Z) / 2)
        self.assertTrue(mutual_dep(X & Y & Z) <= dual_total_corr(X & Y & Z))
        self.assertTrue(markov(X, Y, Z) >> 
                        (mutual_dep(X & Y & Z) == emin(I(X & Y), I(Y & Z))))
        
        # Intrinsic mutual information [Maurer-Wolf 1999]
        self.assertTrue(intrinsic_mi(X & Y | Z) == markov(X+Y, Z, U).minimum(I(X & Y | U), U))
        self.assertTrue(intrinsic_mi(X & Y | Z) <= I(X & Y | Z))
        self.assertFalse(intrinsic_mi(X & Y | Z) >= I(X & Y | Z))
        
        
        # Information bottleneck [Tishby-Pereira-Bialek 1999]
        def info_bot(X, Y, t):
            U = rv("U")
            return (markov(U, X, Y) & (I(Y & U) >= t)).exists(U).minimum(I(X & U))
        
        t1, t2 = real("t1", "t2")
        self.assertTrue((t1 <= t2) >> (info_bot(X, Y, t1) <= info_bot(X, Y, t2)))
        
        
    def test_exists(self):
        X, Y, Z, W, U, V = rv("X", "Y", "Z", "W", "U", "V")
        
        self.assertFalse((H(X) > 0).exists(X))
        self.assertTrue((H(Y)==1) >> (H(X)==1).exists(X))
        self.assertFalse((H(Y)==1) >> (H(X)==2).exists(X))
        
        # Assume there exists a random variable with entropy 1
        with exists_bit().assumed():
            self.assertTrue((H(X) > 0).exists(X))
            self.assertTrue(((H(X) > 0) & indep(X, Y)).exists(X))
            self.assertFalse(((H(X) > 0) & indep(X, X)).exists(X))
            self.assertTrue((~(H(X) <= 0)).exists(X))
            self.assertTrue((~(H(X) <= 0) & indep(X, Y)).exists(X))
            self.assertFalse((~(H(X) <= 0) & indep(X, X)).exists(X))
            
            self.assertTrue((~(H(X) <= 0) & ~(H(Y) <= 0)).exists(X + Y))
            self.assertFalse((~(H(X) <= 0) & ~(H(Y) <= 0) & indep(X, Y)).exists(X + Y))
            self.assertFalse((~(H(X) <= 0) & ~(H(Y | X) <= 0)).exists(X + Y))
            
        # Assume there exists two random variables with entropy 1
        with (exists_bit() ** 2).assumed():
            self.assertTrue((~(H(X) <= 0) & ~(H(Y) <= 0) & indep(X, Y)).exists(X + Y))
            self.assertTrue((~(H(X) <= 0) & ~(H(Y | X) <= 0)).exists(X + Y))
        
        
    def test_sfrl(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        with PsiOpts(sfrl = "frl"):
            self.assertTrue(((I(X & U) == 0) & (H(Y | X + U) == 0)).exists(U))
            
    
    def test_ratio(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        self.assertEqual(H(X) * 2 / (I(X & Y) + H(X | Y)), 2.0)
        self.assertEqual((H(X) + H(Y) - H(X+Y) - I(X&Y)) / (I(X & Y) + H(X | Y)), 0.0)
        # self.assertEqual((H(X) + H(Y)) / (H(X) * 3 - H(X) * 3), None)
        # self.assertEqual(H(X) * 2 / I(X & Y), None)
        self.assertEqual((H(X) * 2 + H(Z) * 6) / (H(Z) * 3 + I(X & Y) + H(X | Y)), 2.0)
        # self.assertEqual((H(X) * 2 + H(Z) * 6) / (H(Z) * 2 + I(X & Y) + H(X | Y)), None)
        # self.assertEqual((H(X) * 2 + H(Z) * 6) / (H(Z) * 3 + I(X & Y) + H(X | Y) + H(W)), None)
        # self.assertEqual((H(X) * 2 + H(Z) * 6 + H(W)) / (H(Z) * 3 + I(X & Y) + H(X | Y)), None)
    
    
    def test_bayesnet(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        if False:
            print(str(((I(X&Y|Z) == 0) & (I(U&X+Z|Y) <= 0)).get_bayesnet()))
            print(str(((I(X&Y|Z) == 0) & (I(U&X+Z|Y) <= 0)).get_bayesnet().get_ic()))
            print(str((I((W+X) & (Y+U) | Z) == 0).get_bayesnet()))
            print(str((I((W+X) & (Y+U) | Z) == 0).get_bayesnet().get_ic()))
            print(str(((I(X&Y) == 0) & (I(X+Y&W|Z) <= 0)).get_bayesnet()))
            print(str(((I(X&Y) == 0) & (I(X+Y&W|Z) <= 0)).get_bayesnet().get_ic()))
            
        self.assertTrue(((I(X&Y|Z) == 0) & (I(U&X+Z|Y) <= 0)).get_bayesnet().check_ic(I(X&U|Z)))
        self.assertFalse(((I(X&Y|Z) == 0) & (I(U&X+Z|Y) <= 0)).get_bayesnet().check_ic(I(X&Y|U)))
        self.assertTrue((I((W+X) & (Y+U) | Z) == 0).get_bayesnet().check_ic(I(X&U|Z)))
        self.assertFalse((I((W+X) & (Y+U) | Z) == 0).get_bayesnet().check_ic(I(X&U)))
        self.assertTrue(((I(X&Y) == 0) & (I(X+Y&W|Z) <= 0)).get_bayesnet().check_ic(I(X&Y)))
        self.assertTrue(((I(X&Y) == 0) & (I(X+Y&W|Z) <= 0)).get_bayesnet().check_ic(I(X&W|Z)))
        self.assertFalse(((I(X&Y) == 0) & (I(X+Y&W|Z) <= 0)).get_bayesnet().check_ic(I(X&Y|W)))
        
        b = BayesNet([(X, Z), (Y, Z), (Z, W, U)])
        b.set_fcn(Z+W)
        self.assertFalse(b.check_ic(H(U|X)))
        self.assertTrue(b.check_ic(H(W|Z)))
        self.assertFalse(b.check_ic(H(W|X)))
        self.assertTrue(b.check_ic(H(W|X+Y)))
        self.assertTrue(b.check_ic(I(X & Y)))
        self.assertTrue(b.check_ic(I(W & W|X+Y)))
        self.assertTrue(b.check_ic(I(W & X+W|X+Y)))
        self.assertTrue(b.check_ic(I(W & W+Z|X+Y)))
        self.assertTrue(b.check_ic(I(W & Z|X+Y)))
        self.assertTrue(b.check_ic(I(U & Z|X+Y)))
        self.assertFalse(b.check_ic(I(X & Y | Z)))
        self.assertFalse(b.check_ic(I(U & U | Z)))
        self.assertTrue(b.check_ic(I(X & U | W)))


    def test_markov(self):
        
        X = rv_seq("X", 0, 9)
        self.assertTrue(markov(*X) >> markov(X[0], X[4], X[8]))
        self.assertTrue(markov(*X) >> (I(X[0] & X[8]) <= H(X[4])))
        
        
    def test_regionop(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        print("test_regionop")
        aux = (H(X | U) == 0).exists(U).forall(X).check_getaux()
        self.assertTrue(aux is not None)
        print(iutil.list_tostr_std(aux))
        
        aux = (H(X | U) == 0).forall(X).exists(U).check_getaux()
        self.assertFalse(aux is not None)
        
        aux = (H(U | X) == 0).exists(U).forall(X).check_getaux()
        self.assertTrue(aux is not None)
        print(iutil.list_tostr_std(aux))
        
        aux = (H(U | X) == 0).forall(X).exists(U).check_getaux()
        self.assertTrue(aux is not None)
        print(iutil.list_tostr_std(aux))
        
    
    def test_nested_implication(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        # If H(U|X)=0 implies H(U|Y)=0, then H(X|Y)=0
        self.assertTrue(((H(U | X) == 0) >> (H(U | Y) == 0)).forall(U) >> (H(X | Y) == 0))
        self.assertTrue(((H(U | X) == 0) >> (H(U | Y) == 0)).forall(U) << (H(X | Y) == 0))
        self.assertFalse(((H(U | X) == 0) >> (H(U | Y) == 0)).forall(U) >> (H(Y | X) == 0))
        self.assertFalse(((H(U | X) == 0) >> (H(U | Y) == 0)).forall(U) << (H(Y | X) == 0))
        self.assertFalse(((H(U | X) == 0) >> (H(U | Y) == 0)).forall(U) >> (H(Z | Y) == 0))
        self.assertFalse(((H(U | X) == 0) >> (H(U | Y) == 0)).forall(U) << (H(Z | Y) == 0))
        
        self.assertTrue((H(X | U) == 0).forall(U) >> (H(X) == 0))
        # self.assertTrue((H(U | X) == 0).forall(U) >> (H(X) == 0))
        
        self.assertTrue(((H(U | X) == 0) >> (I(Y & Z | U) == 0)).forall(U)
            >> ((I(Y & Z) == 0) & (I(Y & Z | X) == 0)))
        self.assertTrue(((H(U | X) == 0) >> (I(Y & Z | U) == 0)).forall(U)
            >> (I(Y & Z & X) == 0))
        
        self.assertFalse(((H(U | X) == 0) >> (H(U | X) == 0)).forall(U) >> (H(Y | X) == 0))
        self.assertTrue(((H(U | X) == 0) >> (I(U & Y) == 0)).forall(U) >> 
                        (((I(U & Y) == 0) >> (H(U | Z) == 0)).forall(U) >> (H(X | Z) == 0)))
        self.assertTrue(((I(U & Y) == 0) >> (H(U | Z) == 0)).forall(U) >> 
                        (((H(U | X) == 0) >> (I(U & Y) == 0)).forall(U) >> (H(X | Z) == 0)))
        
    
            
    def test_union(self):
            
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        if False:
            print(((H(X) >= H(Z)) | (H(Y) >= H(Z))) >> (H(X+Y) >= H(Z)))
            print((((H(X) >= H(Z)) | (H(Y) >= H(Z))) >> (H(X+Y) >= H(Z))).simplified())
            print(((H(X) >= H(Z)) | (H(Y) >= H(Z))) >> ((H(X) >= H(Z)) | (H(Y) >= H(Z))))
            print((((H(X) >= H(Z)) | (H(Y) >= H(Z))) >> ((H(X) >= H(Z)) | (H(Y) >= H(Z)))).simplified_quick())
            print((((H(X) >= H(Z)) | (H(Y) >= H(Z))) >> ((H(X) >= H(Z)) | (H(Y) >= H(Z)))).simplified())
            
        self.assertTrue(((H(X) >= H(Z)) | (H(Y) >= H(Z))) >> (H(X+Y) >= H(Z)))
        self.assertTrue(((H(X) >= H(Z)) | (H(Y) >= H(Z))) >> ((H(X) >= H(Z)) | (H(Y) >= H(Z))))
        self.assertTrue((H(X) >= H(Z)) >> ((H(X) >= H(Z)) | (H(Y) >= H(Z))))
        
        self.assertTrue(emax(H(X), H(Y)) >= H(X))
        self.assertFalse(emax(H(X), H(Y)) <= H(X))
        self.assertTrue(emin(H(X), H(Y)) <= H(X))
        self.assertFalse(emin(H(X), H(Y)) >= H(X))
        self.assertTrue(emax(H(X), H(Y)) >= emin(H(X), H(Y)))
        self.assertFalse(emax(H(X), H(Y)) <= emin(H(X), H(Y)))
        self.assertFalse(emin(H(X), H(Y)) <= emin(H(X), H(Y), H(Z)))
        self.assertTrue(emin(H(X), H(Y)) >= emin(H(X), H(Y), H(Z)))
        
        self.assertTrue((emin(H(X), H(Y)) == 0) >> (I(X&Y) == 0))
        self.assertTrue((emax(H(X), H(Y)) == 0) >> (I(X&Y) == 0))
        #self.assertTrue(((emin(H(X), H(Y)) <= 0) & (emin(H(Z), H(W)) <= 0))
        #    >> (indep(X,Z) | indep(X,W) | indep(Y,Z) | indep(Y,W))) # This currently fails

        self.assertTrue(emax(H(X), emax(H(Y),H(Z))) >= H(Y))
        self.assertTrue(emax(H(X), emax(H(Y),H(Z))) >= emax(H(X), H(Y)))
        self.assertFalse(emax(H(X), emax(H(Y),H(Z))) <= emax(H(X), H(Y)))
            
            
    def test_regterm(self):
            
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        R = real("R")
        a = ((R <= H(X)) & (R <= H(Y))).maximum(R, R, reg_outer = ((R <= H(X)+H(Z)) & (R <= H(Y))))
        self.assertTrue(a <= H(X)+H(Z))
        self.assertFalse(a <= H(X))
        b = ((R >= H(X)) & (R >= H(Y))).minimum(R, R, reg_outer = ((R >= I(X&Z)) & (R <= H(Y))))
        self.assertTrue(b >= I(X&Z))
        self.assertFalse(b >= H(X))
        self.assertFalse(b <= I(X&Z)+H(Y))
        self.assertTrue(b <= H(X)+H(Y))


    def test_reals(self):
        
        r = universe()
        a, b, c, d = real("a", "b", "c", "d")
        
        # Set r to be the region of (a,b,c,d) satisfying a>=b, c<=b, d==c
        r &= a >= b
        r &= c <= b
        r &= d == c
        
        # Test always true statements
        # Note that if right hand side is a number, it can only be 0
        self.assertTrue(a * 0 == 0)
        self.assertTrue(b - b == 0)
        self.assertFalse(c * 2 == 0)
        
        # Use the syntax "r >> (...)" to test if r implies ...
        self.assertTrue(r >> (b >= c))
        self.assertTrue(r >> (a >= d))
        self.assertTrue(r >> (c == d))
        self.assertTrue(r >> (c >= d))
        self.assertTrue(r >> (c <= d))
        self.assertFalse(r >> (a <= c))
        self.assertFalse(r >> (a >= 0))
    
    
    def test_mixed(self):
        
        r = universe()
        a = real("a")
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        r &= H(X) >= a
        r &= a >= H(Y)
        
        # Test always true statements (r is ignored)
        self.assertTrue(H(X) >= 0)
        self.assertTrue(I(X & Y) - H(X) <= 0)
        self.assertTrue(I(X & Y) == H(X) + H(Y) - H(X + Y))
        self.assertTrue(I(X & Y) == H(X) - H(X | Y))
        self.assertFalse(H(X) >= H(Y))
        self.assertFalse(a >= 0)
        
        # Use the syntax "r >> (...)" to test if r implies ...
        self.assertTrue(r >> (H(X) >= H(Y)))
        self.assertTrue(r >> (a >= 0))
        self.assertTrue(r >> (H(X | Y) >= H(Y | X)))
        self.assertTrue(r >> (a >= I(X & Y)))
        self.assertTrue(r >> (a >= I(X & Y | X)))
        self.assertFalse(r >> (a <= I(X & Y)))
        
        self.assertTrue(((H(X|Z+W)==0) & (H(Y|Z) == 0)) >> (H(X+Y|Z+W) == 0))
        self.assertFalse(((H(X|Z+W)==0) & (H(Y|Z) == 0)) >> (H(X+Y|Z) == 0))
        self.assertFalse(((H(X|Z+W)==0) & (H(Y|Z) == 0)) >> (H(X+Y|W) == 0))
        self.assertTrue(((H(X|Y)==0) & (H(Y|Z+W) == 0)) >> (H(X+Y|Z+W) == 0))
        self.assertTrue(((H(X|Y)==0) & (H(Y|Z+W) == 0)) >> (H(X|Z+W) == 0))
        self.assertFalse(((H(X|Y)==0) & (H(Y|Z+W) == 0)) >> (H(X|W) == 0))
        self.assertFalse(((H(X|Y)==0) & (H(Y|Z+W) == 0)) >> (H(X|U) == 0))
    
    
    def test_simplify(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        self.assertEqual(str(I(X & Y+Z & Z+Y & Z+Y+X).simplified()), "I(X;Z,Y)")
        self.assertTrue(I(X & Y+Z & Z+Y & Z+Y+X) == I(X & Y+Z & Z+Y & Z+Y+X).simplified())
        
        self.assertEqual(str(I(X & Y+Z & Z+Y | Z+X).simplified()), "0")
        self.assertTrue(I(X & Y+Z & Z+Y | Z+X) == I(X & Y+Z & Z+Y | Z+X).simplified())
        
        self.assertEqual(str((I(X & Y+Z & Z+Y) + H(X) + H(Y|Z+Y) + I(Z+Y & X) - H(X)).simplified()), "2I(X;Z,Y)")
        self.assertTrue((I(X & Y+Z & Z+Y) + H(X) + H(Y|Z+Y) + I(Z+Y & X) - H(X))
        == (I(X & Y+Z & Z+Y) + H(X) + H(Y|Z+Y) + I(Z+Y & X) - H(X)).simplified())
        
        self.assertEqual(str((H(X | Y + W) - I(X & U | W) * 2 + I(Y & X | W) + H(X | W)).simplified()), "2H(X|W,U)")
        self.assertTrue((H(X | Y + W) - I(X & U | W) * 2 + I(Y & X | W) + H(X | W))
        == (H(X | Y + W) - I(X & U | W) * 2 + I(Y & X | W) + H(X | W)).simplified())
        
        self.assertEqual(str((H(X|Y+Z) + I(X&Y|Z)).simplified()), "H(X|Z)")
        self.assertEqual(str((H(X|Y+Z) + H(Y|Z)).simplified()), "H(X,Y|Z)")
        self.assertEqual(str((H(X|Z) - I(X&Y|Z)).simplified()), "H(X|Z,Y)")
        self.assertEqual(str((H(X|Z) - H(X|Y+Z)).simplified()), "I(X;Y|Z)")
        self.assertEqual(str((H(X+Y|Z) - H(Y|Z)).simplified()), "H(X|Z,Y)")
        self.assertEqual(str((H(X+Y|Z) - H(Y|X+Z)).simplified()), "H(X|Z)")
        
        self.assertEqual(str((I(X&W|Y+Z) + I(X&Y&W|Z)).simplified()), "I(X;W|Z)")
        self.assertEqual(str((I(X&W|Y+Z) + I(Y&W|Z)).simplified()), "I(X,Y;W|Z)")
        self.assertEqual(str((I(X&W|Z) - I(X&Y&W|Z)).simplified()), "I(X;W|Z,Y)")
        self.assertEqual(str((I(X&W|Z) - I(X&W|Y+Z)).simplified()), "I(X;W;Y|Z)")
        self.assertEqual(str((I(X+Y&W|Z) - I(Y&W|Z)).simplified()), "I(X;W|Z,Y)")
        self.assertEqual(str((I(X+Y&W|Z) - I(Y&W|X+Z)).simplified()), "I(X;W|Z)")
        
        self.assertEqual(str((H(X+Y) - H(X) - H(Y)).simplified()), "-I(Y;X)")
        self.assertTrue((H(X+Y) - H(X) - H(Y)) == (H(X+Y) - H(X) - H(Y)).simplified())
        
        self.assertEqual(str(((H(X) >= 0) & (I(X & Y | Z) + H(W) >= 0)).simplified()), "")
        
        self.assertEqual(str(((I(Z & W | X) + I(W & U) <= 0) & (H(X) + I(Y & U) == 0)
            & (-2*H(U) - 3*I(Y & W) == 0)).simplified()),
            ("{ H(X) == 0,\n"
            "  H(U) == 0,\n"
            "  I(Y;W) == 0,\n"
            "  I(Z;W) == 0 }"))
    
        # self.assertEqual(str(((I(Z & W | X) + I(W & U) <= 0) & (H(X) + I(Y & U) == 0)
        #     & (-2*H(U) - 3*I(Y & W) == 0)).simplified(zero_group = 2)),
        #     "{ I(W;U)+I(Y;U)+I(Y;W)+I(Z;W|X)+H(X)+H(U) <= 0 }")
        
        self.assertEqual(str(((H(X | Y) == 0) & (I(X & Y) <= 0)).simplified()),
            "H(X) == 0")
    
        self.assertEqual(str(((H(X) == H(X|Y)) & (I(X & Z) == 0) & (I(Z & Y) == 0) & (H(X+Y+Z) == H(X+Y)+H(Z))).simplified()),
                         "{ I(X;Y) == 0,\n  I(Z;X,Y) == 0 }")
        self.assertTrue(((I(X & Y) == 0) & (I(X & Z) == 0) & (I(Z & Y) == 0) & (I(X+Y & Z) == 0))
        == ((I(X & Y) == 0) & (I(X & Z) == 0) & (I(Z & Y) == 0) & (I(X+Y & Z) == 0)).simplified())
        
        self.assertEqual(str(((H(X) >= H(Y)) & (H(Y) >= H(X))).simplified()), "H(X) == H(Y)")
        self.assertEqual(str(((H(X) >= H(Y)) & (H(Y) == H(X))).simplified()), "H(Y) == H(X)")
        self.assertEqual(str(((H(X) >= H(Y)) & (H(Y) >= H(X)) & (H(X) <= H(X)) & (H(X) <= H(Y))).simplified()), "H(X) == H(Y)")
        self.assertEqual(str(((H(X) >= H(Y)) & (H(Y) >= H(X)) & (H(X) <= H(X)) & (H(X) <= H(Y)) & (H(Y) <= H(Z)) & (H(Y) == H(Z))).simplified()),
                         "{ H(Y) == H(Z),\n  H(X) == H(Y) }")
    
        
    def test_eliminate(self):
        
        R = real("R")
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        self.assertEqual(str(
                (((R >= H(Y)) & (R <= H(Z)))).exists(R)),
                "H(Y) <= H(Z)")
        
        self.assertEqual(str(
                (((R >= H(Y)) & (R <= H(Z)) & (R == H(W)))).exists(R)),
                ("{ H(W) <= H(Z),\n"
                "  H(Y) <= H(W) }"))
        
        self.assertEqual(str(
                ((R == H(X)) >> ((R >= H(Y)) & (R <= H(Z)) & (R == H(W)))).forall(R)),
                ("{ H(X) <= H(Z),\n"
                "  H(Y) <= H(X),\n"
                "  H(X) == H(W) }"))
        
        
    
    def test_sum(self):
        
        R, R1, R2 = real("R", "R1", "R2")
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        r = (R1 >= 0) & (R2 >= 0) & (R1 <= H(X)) & (R2 <= H(Y))
        r2 = (R1 >= 0) & (R2 >= 0) & (R1 + R2 <= H(Z))
        
        self.assertEqual(str(r * 2),
            ("{ R1 >= 0,\n"
            "  R2 >= 0,\n"
            "  R1 <= 2H(X),\n"
            "  R2 <= 2H(Y) }"))
        self.assertEqual(str(r + r),
            ("{ R1 >= 0,\n"
            "  R1 <= 2H(X),\n"
            "  R2 >= 0,\n"
            "  R2 <= 2H(Y) }"))
        self.assertEqual(str(r + r2),
            ("{ R1 >= 0,\n"
            "  R2 >= 0,\n"
            "  R1+R2 <= H(Z)+H(X)+H(Y),\n"
            "  R1 <= H(Z)+H(X),\n"
            "  R2 <= H(Z)+H(Y) }"))
        self.assertEqual(str(r + (r | r2)),
            ("{\n"
            "  { R1 >= 0,\n"
            "    R1 <= 2H(X),\n"
            "    R2 >= 0,\n"
            "    R2 <= 2H(Y) }\n"
            " OR\n"
            "  { R1 >= 0,\n"
            "    R2 >= 0,\n"
            "    R1 <= H(X)+H(Z),\n"
            "    R2 <= H(Y)+H(Z),\n"
            "    R1+R2 <= H(Y)+H(X)+H(Z) }\n"
            "}"))
        
        
    def test_cases(self):
        
        R, R2 = real("R", "R2")
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        print("test_cases")
        
        self.assertTrue((R >= 0) | (R <= 0))
        self.assertTrue((R >= 0) | ~(R >= 0))
        self.assertTrue((R == 0) | ~(R <= 0) | ~(R >= 0))
        self.assertFalse(~(R <= 0) | ~(R >= 0))
        self.assertFalse(~(R == 0) | ~(R >= 0))
        self.assertTrue(~(R == 0) | (R >= 0))
        self.assertFalse((R == 0) | ~(R >= 0))
        self.assertTrue((H(X) >= H(Y)) | (H(Y) >= H(X)))
        self.assertFalse((H(X) >= H(Y)) | ((H(Y) >= H(X)) & (H(Y) >= H(Z))))
        
        
        with PsiOpts(auxsearch_leaveone = True):
            
            self.assertTrue((R >= 0) | (R <= 0))
            
            self.assertTrue((R >= 0) | ((R <= 0) & (R2 <= R)) | ((R <= 0) & (R2 >= R)))
            
            aux = ((I(X&Y|U) >= I(X&Y|Z)) & (I(X&Y|U) >= I(X&Y))).exists(U).check_getaux()
            self.assertTrue(aux is not None)
            print(iutil.list_tostr_std(aux))
            
            aux = RegionOp.inter([(I(X&Y|U) >= I(X&Y|Z)) & (I(X&Y|U) >= I(X&Y))]).exists(U).check_getaux()
            self.assertTrue(aux is not None)
            print(iutil.list_tostr_std(aux))
            
            aux = (((H(U) == H(X)) & (H(X) >= H(Y))) | ((H(U) == H(X+Y)) & (H(Y) >= H(X)))).exists(U).check_getaux()
            self.assertTrue(aux is not None)
            print(iutil.list_tostr_std(aux))
            
            aux = (((H(U) == H(X)) & (H(X) >= H(Y))) | ((H(U) == H(X+Y)) & (H(X) >= H(Y)))).exists(U).check_getaux()
            self.assertFalse(aux is not None)
            
            
    def test_abs(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        self.assertTrue(abs(H(X)) >= 0)
        self.assertTrue(abs(H(X) - H(Y)) >= 0)
        self.assertFalse(abs(H(X) - H(Y)) <= 0)
        self.assertFalse(abs(H(X) - H(Y)) <= H(X))
        self.assertTrue(abs(H(X) - H(Y)) <= H(X) + H(Y))
        self.assertTrue(abs(H(X) - H(Y)) >= H(X) - H(Y))
        self.assertTrue(abs(H(X) - H(Y)) <= abs(H(X) + H(Y)))
        
        
    def test_symmetric(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        
        self.assertTrue(indep(X, Y, Z, W).issymmetric(X+Y+Z+W))
        self.assertTrue(indep(X, Y, Z+W).issymmetric(X+Y))
        self.assertFalse(indep(X, Y, Z+W).issymmetric(X+Y+Z))
        self.assertFalse(markov(X, Y, Z).issymmetric(X+Y+Z))
        self.assertTrue(markov(X, Y, Z).issymmetric(X+Z))
        
    
    def test_affine(self):
        
        X, Y, Z, W, U = rv("X", "Y", "Z", "W", "U")
        #PsiOpts.set_setting(lp_eps = 0.00001)
        
        self.assertTrue(H(X) + 7 - 5 >= 0)
        self.assertTrue(((H(X) >= 5) & (H(Y) <= 2)) >> (H(X | Y) >= 3))
        self.assertTrue(((H(X) >= 5) & (H(X) <= 4)) >> (H(Y) >= 3))
        self.assertTrue(((H(X) >= 5) & (H(Y) <= 2)).isfeasible())
        self.assertFalse(((H(X) >= 5) & (H(X) <= 4)).isfeasible())
        self.assertTrue(((H(X) >= 5) & (H(X) < 6)).isfeasible())
        self.assertFalse(((H(X) >= 5) & (H(X) < 5)).isfeasible())
        self.assertTrue((H(X) != H(Y)).isfeasible())
        self.assertFalse((H(X) != H(X)).isfeasible())
        self.assertTrue((H(X) * 2 > 1) >> (H(X) > 0.5))
        self.assertTrue((H(X) * 0.5 > 1) >> (H(X) > 2))
        self.assertTrue((H(X) > 0.5) >> (H(X) * 2 > 1))
        self.assertTrue((H(X) > 2) >> (H(X) * 0.5 > 1))
        self.assertFalse((H(X) * 2 >= 1) >> (H(X) > 0.5))
        self.assertFalse((H(X) * 0.5 >= 1) >> (H(X) > 2))
        self.assertFalse((H(X) >= 0.5) >> (H(X) * 2 > 1))
        self.assertFalse((H(X) >= 2) >> (H(X) * 0.5 > 1))
        
        
    def test_condindep(self):
        
        A, B, C, D = rv("A", "B", "C", "D")
        
        # [Studeny 1989]
        s1 = (I(A&B|C+D)==0) & (I(C&D|A)==0) & (I(C&D|B)==0) & (I(A&B)==0)
        s2 = (I(C&D|A+B)==0) & (I(A&B|C)==0) & (I(A&B|D)==0) & (I(C&D)==0)
        
        self.assertTrue(s1 >> s2)
        self.assertTrue(s2 >> s1)
        
    
    def test_evalrepr(self):
        X, Y, Z, U, V, W = rv("X, Y, Z, U, V, W")
        
        print("test_evalrepr")
        
        r = markov(X, Y, Z) & markov(U, V, W)
        print(repr(r))
        self.assertTrue(r.equiv(eval(repr(r))))
        
        r = markov(X, Y, Z) & markov(U, V, W) & indep(X+Y+Z, U+V+W)
        print(repr(r))
        self.assertTrue(r.equiv(eval(repr(r))))
        
        r = indep(X+Y, Z, U+V+W)
        print(repr(r))
        self.assertTrue(r.equiv(eval(repr(r))))
        
        r = markov(X+Y, Z, U+V+W)
        print(repr(r))
        self.assertTrue(r.equiv(eval(repr(r))))
        
        r = markov(X+Y, Z, U, V+W)
        print(repr(r))
        self.assertTrue(r.equiv(eval(repr(r))))
        
        r = markov(X, Y, Z) & markov(U, V, W) & indep(X+Y+Z, U+V+W) & (H(X+U) == 1)
        print(repr(r))
        self.assertTrue(r.equiv(eval(repr(r))))
        
        r = markov(X, Y, Z) & markov(U, V, W) & indep(X+Y+Z, U+V+W) & (H(X|U) == 0)
        print(repr(r))
        self.assertTrue(r.equiv(eval(repr(r))))
        
        r = markov(X+Y, Z, V+W) & indep(X, U+W)
        print(repr(r))
        self.assertTrue(r.equiv(eval(repr(r))))
    
    
    def test_simplify_aux(self):
        X, Y, Z, U, V, W = rv("X, Y, Z, U, V, W")
        R = real("R")
        
        print("test_simplify_aux")

        self.assertEqual(repr(((R <= I(U+V & Y)) & indep(U+V, X)).exists(U+V).simplified()),
            '( ( R <= I(U&Y) )\n &( indep(U, X) ) ).exists(U)')

        self.assertEqual(repr(((R <= I(U+V & Y)) & indep(U+V, X) & indep(U, Z)).exists(U+V).simplified()),
            '( ( R <= I(V&Y) )\n &( indep(V, X) ) ).exists(V)')

        r = ((H(U|X)==1) & (H(U|Y)==2) & (H(V|X)==3) & (H(V|Y)==4) 
            & markov(X, U, Y) & markov(X, V, Y) & markov(U, X+Y, V)).exists(U+V)

        self.assertEqual(repr(r.simplified(level = 9)),
            '( ( H(U|X) <= 1 )\n &( H(U|Y) == 1+H(U|X) )\n &( markov(X, U, Y) ) ).exists(U)')

    
    def test_performance(self):
        
        ls = [(3, 6), (6, 2), (9, 1)]
        
        for solver in ["pulp.glpk", "pyomo.glpk", "scipy"]:
            for (l, nit) in ls:
                if solver == "scipy" and l > 6:
                    continue
                start = time.time()
                for it in range(nit):
                    X = rv_seq("X", 0, l)
                    r = universe()
                    for i in range(l - 1):
                        r &= H(X[i + 1] | X[i]) >= H(X[i])
                    with PsiOpts(solver = solver, solver_scipy_maxsize = -1):
                        self.assertTrue(r >> (H(X[l-1]) >= H(X[0])))
                        self.assertFalse(r >> (H(X[0]) >= H(X[l-1])))
                end = time.time()
                print(solver + ", " + str(l) + ", " + str((end - start) * 1.0 / nit))
        
        

# Run tests
#PsiOpts.set_setting(solver = "pulp.glpk")
PsiOpts.set_setting(solver = "pyomo.glpk")
#PsiOpts.set_setting(pulp_solver = pulp.solvers.GUROBI(mip = False, msg = 0))

PsiOpts.set_setting(str_style = PsiOpts.STR_STYLE_STANDARD)

unittest.main(verbosity=2)


"""
References:

Cover, Thomas. "Broadcast channels." IEEE Transactions on Information Theory 18.1 (1972): 2-14.

J. Korner and K. Marton, Comparison of two noisy channels, Topics in Inform. 
Theory (ed. by I. Csiszar and P. Elias), Keszthely, Hungary (August, 1975), 411-423.

K. Marton, "A coding theorem for the discrete memoryless broadcast channel," IEEE 
Transactions on Information Theory, vol. 25, no. 3, pp. 306-311, May 1979.

Y. Liang and G. Kramer, "Rate regions for relay broadcast channels," IEEE Transactions
on Information Theory, vol. 53, no. 10, pp. 3517-3535, Oct 2007.

R. Ahlswede, "Multi-way communication channels," in 2nd Int. Symp. Inform. Theory, 
Tsahkadsor, Armenian SSR, 1971, pp. 23-52.

H. Liao, "Multiple access channels," Ph.D. dissertation, University of Hawaii, Honolulu, HI, 1972.

Bergmans, P. "Random coding theorem for broadcast channels with degraded components."
IEEE Transactions on Information Theory 19.2 (1973): 197-207.

Gallager, Robert G. "Capacity and coding for degraded broadcast channels." Problemy
Peredachi Informatsii 10.3 (1974): 3-14.

T. Berger, "Multiterminal source coding," in The Information Theory Approach to 
Communications, G. Longo, Ed. New York: Springer-Verlag, 1978,
pp. 171-231.

S.-Y. Tung, "Multiterminal source coding," Ph.D. dissertation, Cornell University, 
Ithaca, NY, 1978.

Peter Gacs and Janos Korner. Common information is far less than mutual information.
Problems of Control and Information Theory, 2(2):149-162, 1973.

A. D. Wyner. The common information of two dependent random variables.
IEEE Trans. Info. Theory, 21(2):163-179, 1975.

G. R. Kumar, C. T. Li, and A. El Gamal. Exact common information. In Information
Theory (ISIT), 2014 IEEE International Symposium on, 161-165. IEEE, 2014.

Cuff, P. W., Permuter, H. H., & Cover, T. M. (2010). Coordination capacity.
IEEE Transactions on Information Theory, 56(9), 4181-4206.

Li, C. T., & El Gamal, A. (2018). Strong functional representation lemma and
applications to coding theorems. IEEE Trans. Info. Theory, 64(11), 6967-6978.

S. I. Gel'fand and M. S. Pinsker, "Coding for channel with random parameters,"
Probl. Contr. and Inf. Theory, vol. 9, no. 1, pp. 19-31, 1980.

Gray, R. M., and A. D. Wyner. "Source coding for a simple network." Bell System 
Technical Journal 53.9 (1974): 1681-1721.

Wyner, Aaron, and Jacob Ziv. "The rate-distortion function for source coding with 
side information at the decoder." IEEE Transactions on information Theory 22.1 (1976): 1-10.

Watanabe S (1960). Information theoretical analysis of multivariate correlation, 
IBM Journal of Research and Development 4, 66-82.

Han T. S. (1978). Nonnegative entropy measures of multivariate symmetric 
correlations, Information and Control 36, 133-156.

Li, Cheuk Ting, and Abbas El Gamal. "Extended Gray-Wyner system with complementary 
causal side information." IEEE Transactions on Information Theory 64, no. 8 (2017): 5862-5878.

Csiszar, Imre, and Prakash Narayan. "Secrecy capacities for multiple terminals." 
IEEE Transactions on Information Theory 50, no. 12 (2004): 3047-3061.

Tishby, Naftali, Pereira, Fernando C., Bialek, William (1999). The Information 
Bottleneck Method. The 37th annual Allerton Conference on Communication, Control, 
and Computing. pp. 368-377.

U. Maurer and S. Wolf. "Unconditionally secure key agreement and the intrinsic 
conditional information." IEEE Transactions on Information Theory 45.2 (1999): 499-514.

Z. Zhang and R. W. Yeung, "On characterization of entropy function via information inequalities,"
IEEE Trans. Inform. Theory, vol. 44, pp. 1440-1452, Jul 1998.

Randall Dougherty, Chris Freiling, and Kenneth Zeger. "Non-Shannon information 
inequalities in four random variables." arXiv preprint arXiv:1104.3602 (2011).

Randall Dougherty, Chris Freiling, and Kenneth Zeger. "Six new non-Shannon 
information inequalities." 2006 IEEE International Symposium on Information Theory. IEEE, 2006.

Imre Csiszar and Janos Korner. Information theory: coding theorems for 
discrete memoryless systems. Cambridge University Press, 2011.

M. Studeny, Multiinformation and the problem of characterization of conditional 
independence relations, Problems of Control and Information Theory 18 (1989) 3-16.

"""

