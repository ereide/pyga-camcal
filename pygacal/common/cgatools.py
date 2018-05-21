
from clifford import MultiVector
from clifford import g3c
from numpy import pi, e
import numpy as np

from scipy.sparse.linalg.matfuncs import _sinch as sinch


layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])

I3 = e123
I5 = e12345
one = e1 * e1 #Util to distinguish MV from scalar


def Sandwich(X, T):
    return T * X * ~T

def Dilator(alpha):
    "TODO: Change to unitvector"
    D = (1 + E0 * (1 - alpha)/(1 + alpha))
    return D 

def Translator(a):
    T = 1 + 0.5 * ninf * a
    return T

def Reflector(b):
    return b

def inversion(X):
    return Sandwich(X, ep)

def Distance(A, B):
    return np.sqrt(abs(-2 * homo(A) | homo(B)))

def Rotor(B, alpha):
    return np.cos(-alpha/2.) + B.normal() * np.sin(-alpha/2)

def Transversor(a):
    return 1 - no * a 

def VectorEquality(a, b):
    return abs(a - b) < 1E-8

def anticommuter(A, B):
    #TODO: Check this
    return 0.5*(A * B + B * A)

def commuter(A, B):
    #TODO: Check this
    return 0.5*(A * B - B * A)

def Meet(A, B):
    "Simple linear meet -> fails for degenerate cases"  
    return ((A.dual() ^ B.dual()).dual()).normal() #TODO: Something is wrong

def extractBivectorParameters(B):
    """
    B must be a general rotation object on the format $ \phi*P + t n_{\inf}} $
    """


    t = (B | ep)
    phiP = B - (t*ninf)
    phi = np.sqrt(-float(phiP * phiP))

    if abs(phi) > 0:
        P = phiP/phi
        P_n = P * I3 

        t_nor = (t | P_n)*P_n       #NORMAL
        t_par = t - t_nor           #PARALEL

    else:
        P = 0
        t_nor = t
        t_par = 0

    return phi, P, t_par, t_nor

def ga_exp(B, verbose = False):
    """
    
    This is an expansion of the exponential of a general rotation and translation

    Presented by R. Wareham (Applications of CGA)

    B must be a general rotation object on the format $ \phi*P + t n_{\inf}} $

    $P$ is a 2 blade in real space.

    $$\phi \in \mathcal{R}, P^2=-1, t \in \mathcal{R}^n $$

    WARNING: DOES NOT COMMUTE exp(A + B) != exp(A)*exp(B)

    """

    phi, P, t_par, t_nor = extractBivectorParameters(B)



    if verbose:
        print("_t", t)          #OK
        print("_phi", phi)      #OK
        print("_t_nor", t_nor)  #OK
        print("_t_par", t_par)  #OK
        print("_P", P)


    #Notice: np.sinc(pi * x)/(pi x)
    R = (np.cos(phi) + np.sin(phi) * P)*(1 + t_nor*ninf) + np.sinc(phi/np.pi)*t_par * ninf
    return R

def extractBivectorParameters_complicated(B):
    """
    B must be a general rotation object on the format $ \phi*P + t n_{\inf}} $
    """


    omega = float(B | E0)

    phi, P, t_par, t_nor = extractBivectorParameters(B - omega * E0)
    return phi, P, t_par, t_nor, omega

def ga_exp_complicated(B, verbose = False):
    """
    B must be a general rotation object on the format $ \phi*P + t n_{\inf}} + \omega N $

    $N = e_{+} e_{-}$
    """


    phi, P, t_par, t_nor, omega = extractBivectorParameters_complicated(B)

    omega2 = omega * omega
    phi2   = phi   * phi

    k = omega2 + phi2

    R = (np.cos(phi) + np.sin(phi) * P)*(np.cosh(omega) + np.sinh(omega) * e45 + sinch(omega) * t_nor*ninf) 

    if (k > 0):
        R += 1/k * ( (-omega * np.sin(phi) * np.cosh(omega) + phi * np.cos(phi) * np.sinh(omega)) * P 
                   + ( omega * np.cos(phi) * np.sinh(omega) + phi * np.sin(phi) * np.cosh(omega))    ) * t_par * ninf

    else:
        #TODO: Should use a taylor expansion of the first terms here
        R+= t_par * ninf

    return R

    


def extractRotorComponents(R, verbose = False):
    phi = np.arccos(float(R[0]))             #scalar
    phi2 = phi * phi                  #scalar
    #Notice: np.sinc(pi * x)/(pi x)
    phi_sinc = np.sinc(phi/np.pi)             #scalar

    phiP = ((R(2)*ninf)|ep)/(phi_sinc)
    t_normal_n = -((phiP * R(4))/(phi2 * phi_sinc))
    t_perpendicular_n = -(phiP * (phiP * R(2))(2))/(phi2 * phi_sinc)


    if verbose:
        print("phiP             ", phiP)                  #
        print("P                ", phiP.normal())

        print("t_normal_n       ", t_normal_n)          #
        print("t_perpendicular_n", t_perpendicular_n)   #
        print("")

    return phiP, t_normal_n, t_perpendicular_n


def ga_log(R, verbose = False):
    """
    R must be a rotation and translation rotor. grades in [0, 2, 4]

    Presented by R. Wareham (Applications of CGA)

    WARNING: DOES NOT COMMUTE log(A * B) != log(A) + log(B)
    """

    phiP, t_normal_n, t_perpendicular_n = extractRotorComponents(R, verbose)

    return phiP + t_normal_n + t_perpendicular_n

def MVEqual(actual, expected, rtol = 1e-6, atol = 1e-6,  verbose = False):
    if verbose:
        print("Assert Equal  ")
        print("expected     =", expected)
        print("actual       =", actual)

    if isinstance(actual , MultiVector):
        actual = actual.value

    if isinstance(expected , MultiVector):
        expected = expected.value

    if isinstance(expected, (float, int)):
        val = expected
        expected = np.zeros(actual.shape)
        expected[0] = val
    
    #Scale invariant
    return np.all(abs(actual - expected)< (abs(actual)*rtol + atol)) #or np.all(abs(actual + expected)< (abs(actual)*rtol + atol))

def MVto3DVec(a):
    if a.grades() != [1] and a.grades() != []:
        pass
        #TODO: 
        #print("Warning: wrong grade")
        #raise ValueError()
    
    x, y, z = float((a|e1)[0]),float((a|e2)[0]), float((a|e3)[0])
    return np.array([x, y, z])

def VectoMV(a):
    return a[0] * e1 + a[1]*e2 + a[2]*e3
