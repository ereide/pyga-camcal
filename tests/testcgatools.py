import unittest

import clifford as cl
from clifford import g3c
from numpy import pi, e
import numpy as np

from scipy.sparse.linalg.matfuncs import _sinch as sinch


from clifford import MultiVector

from pygacal.common.cgatools import (  Sandwich, Dilator, Translator, Reflector, 
                        inversion, Rotor, Transversor, I3, I5, 
                        VectorEquality, Distance, ga_log, ga_exp, MVEqual, Meet, 
                        extractBivectorParameters_complicated, ga_exp_complicated, one)

from pygacal.geometry import createRandomBivector, createRandomVector, createRandomPoints

from pygacal.geometry.lines import createLine
from pygacal.geometry.planes import createPlane

layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])

np.random.seed(2512)



def AssertMVEqual(actual, expected,  rtol = 1e-5, atol = 1e-6, verbose = False):
    assert(MVEqual(actual, expected, rtol, atol, verbose))


def AssertMVUnEqual(actual, expected,  rtol = 1e-5, atol = 1e-6, verbose = False):
    assert(not MVEqual(actual, expected,  rtol, atol, verbose))


class TestCGAOperators(unittest.TestCase):
    def testDilator(self):
        x = 2*e1 + 3* e2 + 4*e3
        X = up(x)        
        assert(down(Sandwich(X, Dilator(0.1))) == x * 0.1)
    
    def testTranslation(self):
        x = 2*e1 + 3* e2 + 4*e3
        X = up(x)
        a = 2 * e1 + e3
        assert(down(Sandwich(X, Translator(a))) == x + a)

    def testRotation(self):
        x = 2*e1 + 3* e2 + 4*e3
        X = up(x)
        actual = down(Sandwich(X, Rotor(e12, pi/2)))
        expected = (-3.0)*e1 + 2.0*e2  + 4.0 * e3
        assert(actual == expected)
    
    def testInversion(self):
        x = 2*e1 + 3* e2 + 4*e3
        X = up(x)
        assert(down(inversion(X)) * x == 1)
    
    def testDistance(self):
        a = e1
        b = e2
        A, B = up(a), up(b)
        assert(Distance(A, B) == np.sqrt(2))

    def testMeet(self):
        A, B, C, D = createRandomPoints(N = 4, scale = 50)
        L = createLine(A, B)
        L2 = createLine(A, C)
        P1 = createPlane(A, B, C)
        P2 = createPlane(A, B, D)
        L_actual = Meet(P1, P2)
        assert(MVEqual(L, L_actual))

        #Plane to line
        Q = (ninf ^ A).normal()
        P3 = A ^ C ^ D ^ ninf
        Q_actual = Meet(P3, L).normal() #How do we define order/direction? 
        assert(MVEqual(Q, Q_actual))

    def testAssertEqual(self):
        verbose = False
        a = createRandomBivector()
        b = a + 0.01
        a2 = b - 0.01
        c = a + 1
        d = c - a
        AssertMVEqual(a, a2, verbose = verbose)
        AssertMVUnEqual(a, b, verbose = verbose)
        AssertMVEqual(d, 1, verbose = verbose)

    def testLogarithm(self):
        verbose = False

        if verbose:
            print("\nTest Logarithms and exponents")

        phi = 0.5                         #Rotation amount
        P = (e12 + 2*e23 + 3*e13).normal()  #Rotation Plane 
        P_n = P*I3 

        t = 2.73 * e1 + 3.14*e2             #Translation vector
        t_nor = (P_n | t) * P_n             #Decomposition into normal component
        t_par = t - t_nor                   #Decomposition into paralel component
        assert(t_par + t_nor == t)

        if verbose:
            print("P     = ", P)
            print("phi   = ", phi)
            print("t     = ", t)
            print("t_nor = ", t_nor)
            print("t_par = ", t_par)
            print("")

        assert(P|t_nor == 0) #Normal to P
        assert(P^t_nor != 0) #Normal to P
        assert(P|t_par != 0) #Parallel to P
        assert(P^t_par == 0) #Parallel to P
        assert(P*t != 0)     #Non zero product

        R_expected = (np.cos(phi) + (np.sin(phi) * P))*(1 + (t_nor*ninf)) + np.sinc(phi/np.pi)*t_par * ninf
        B_expected = phi * P + t*ninf

        R_exponential = np.exp(B_expected)

        R_actual = ga_exp(B_expected, verbose = verbose)
        B_new = ga_log(R_expected, verbose = verbose)
        R_ga = ga_exp(B_new)


        if verbose:
            print("R_old        ", R_expected)
            print("R_expected   ", R_actual)
            print("R_exponential", R_exponential) 
            print("R_ga         ", R_ga)
            print("B_new        ", B_new)
            print("B_expected   ", B_expected)


        #Rotor properties
        AssertMVEqual(R_expected * ~R_expected, 1, verbose = verbose)
        AssertMVEqual(R_ga * ~R_ga, 1, verbose = verbose)

        #Equalities
        AssertMVEqual(R_actual, R_expected, verbose = verbose)
        AssertMVEqual(R_exponential, R_expected, verbose = verbose)
        AssertMVEqual(B_new, B_expected, verbose = verbose)
        AssertMVEqual(R_ga, R_expected, verbose = verbose)

        N = 100
        #Random bivectors to test this as well
        for i in range(N):
            B = createRandomBivector()
            AssertMVEqual(B, ga_log(ga_exp(B, verbose = verbose), verbose = verbose), verbose = verbose)


    def testComplicatedLogarithm(self):
        verbose = True

        if verbose:
            print("\nTest Complicated Logarithms and exponents")

        phi = 0.2                            #Rotation amount
        P = (e12 + 2*e23 + 3*e13).normal()  #Rotation Plane 
        P_n = P*I3 

        #t = 0
        
        t = 2.73 * e1 + 3.14*e2             #Translation vector
        t_nor = (P_n | t) * P_n             #Decomposition into normal component
        t_par = t - t_nor                   #Decomposition into paralel component

        omega = 0.1

        assert(t_par + t_nor == t)

        if verbose:
            print("P     = ", P)
            print("phi   = ", phi)
            print("t     = ", t)
            print("t_nor = ", t_nor)
            print("t_par = ", t_par)
            print("omega = ", omega)
            print("")

        """
        assert(P|t_nor      == 0) #Normal to P
        assert(P^t_nor      != 0) #Normal to P
        assert(P|t_par      != 0) #Parallel to P
        assert(P^t_par      == 0) #Parallel to P
        assert(P*t          != 0) #Non zero product
        assert(t_par|t_nor  == 0) #Non zero product
        """

        B_expected = (phi * P) + (t*ninf) + (omega * E0)

        k = (omega * omega + phi * phi)

        R_expected = (np.cos(phi) + np.sin(phi) * P)*(np.cosh(omega) + np.sinh(omega) * E0 + sinch(omega) * t_nor*ninf)

        if (k > 0):
            R_expected += 1/k* ( (-omega * np.sin(phi) * np.cosh(omega) + phi * np.cos(phi) * np.sinh(omega)) * P 
                               + ( omega * np.cos(phi) * np.sinh(omega) + phi * np.sin(phi) * np.cosh(omega))) * t_par * ninf

        else:
            R_expected += t_par * ninf

        phi_test, P_test, t_nor_test, t_par_test, omega_test = extractBivectorParameters_complicated(B_expected)
        B_actual = phi_test * P_test + (t_nor_test + t_par_test)*ninf + omega_test * E0

        #Testing some basic properties of the extraction
        AssertMVEqual(phi*(P * ~P), phi*one, verbose = False)
        AssertMVEqual(phi*P, phi*P_test, verbose = False)        


        R_exponential = np.exp(B_expected)

        R_actual    = ga_exp_complicated(B_expected, verbose = verbose)
        #B_new       = ga_log(R_expected, verbose = verbose)
        #R_ga        = ga_exp(B_new)


        if verbose:
            print("R_expected    ", R_expected)
            print("R_actual      ", R_actual)
            print("R_exponential ", R_exponential) 
            #print("R_ga         ", R_ga)
            #print("B_new        ", B_new)
            print("B_expected   ", B_expected)
            print()

        #BivectorExtraction
        AssertMVEqual(B_actual, B_expected, verbose = verbose)
    

        AssertMVEqual(R_expected * ~R_expected, one, verbose = verbose)


        #Rotor properties
        AssertMVEqual(R_actual * ~R_actual, one, verbose = verbose)

        #Only an approximation
        AssertMVEqual(R_exponential * ~R_exponential, one, verbose = verbose)

        #AssertMVEqual(R_expected * ~R_expected, 1, verbose = verbose)
        #AssertMVEqual(R_ga * ~R_ga, 1, verbose = verbose)

        #Equalities
        #AssertMVEqual(R_actual, R_expected, verbose = verbose)
        AssertMVEqual(R_exponential, R_actual, rtol = 1e-2, atol = 1e-3, verbose = verbose)
        #AssertMVEqual(B_new, B_expected, verbose = verbose)
        #AssertMVEqual(R_ga, R_expected, verbose = verbose)

        #N = 100
        #Random bivectors to test this as well
        #for i in range(N):
        #    B = createRandomBivector()
        #    AssertMVEqual(B, ga_log(ga_exp(B, verbose = verbose), verbose = verbose), verbose = verbose)


if __name__ == "__main__":
    unittest.main()
