from clifford import g3c
import numpy as np

from pygacal.common.cgatools import Sandwich, Dilator, Translator, Reflector, inversion, Rotor, Transversor, I3, I5, VectorEquality, anticommuter, one

from clifford_tools.numerical.g3c.core import ga_exp

#Defining variables
layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])


def parameterRotor(a, c, ma,  mc):
    warnings.warn("This function is no longer used", DeprecationWarning)
    #R = (ma - mc)
    #B = (ma^mc).normal()
    #cos_a = float(ma | mc)
    #R = np.sqrt(0.5*(1 + cos_a)) + np.sqrt(-0.5*(1 - cos_a)) * B
    R = 1 - mc*ma
    return (Translator(c) * R * Translator(-a)).normal()    
    

#Randomness:
def createRandomVector(scale = 1):
    """
    Creates a vector where each entry has mean zero and standard deviation sigma

    """
    def rand():
        #-1 to 1
        return 2*(np.random.rand(3) - 0.5)

    a, b, c = rand()

    vec =  a * e1 + b * e2 + c*e3
    return vec * scale

def createRandomVectors(N, scale = 1):
    vecs = [createRandomVector(scale) for i in range(N)]
    return vecs


def createRandomNoiseVector(sigma):
    """
    Creates a vector where each entry has mean zero and standard deviation sigma
    """
    a, b, c = np.random.normal(scale=sigma, size=3)
    return a * e1 + b * e2 + c * e3



def createRandomPoint(scale = 1):
    vec = createRandomVector(scale)
    return up(vec)

def createRandomBivector():
    """
    Creates a random bivector on the form postulated by R. Wareham

    $$ B =  ab + c*n_{\inf}$$ where $a, b, c \in \mathcal(R)^3$
    """

    a = createRandomVector()
    c = createRandomVector()

    return a*I3 + c*ninf

def createRandomNoiseBivector(sigma_R, sigma_T):
    """
    Creates a random bivector on the form postulated by R. Wareham

    $$ B =  ab + c*n_{\inf}$$ where $a, b, c \in \mathcal(R)^3$
    """

    P = createRandomVector().normal() * I3
    phi = np.random.normal(scale = sigma_R)
    t = createRandomNoiseVector(sigma=sigma_T)

    return one * (phi * P + t*ninf)

def createRandomNoiseInplane(sigma_R, sigma_T):
    """
    Creates a random bivector in GA2 instead of GA3

    $$ B =  ab + c*n_{\inf}$$ where $a, b, c \in \mathcal(R)^3$
    """

    P = e12
    phi = np.random.normal(scale = sigma_R*sigma_T)
    a, b = np.random.normal(scale=sigma_T, size=2)
    t = a*e1 + b*e2
    return phi * P + t*ninf

def createRandomPoints(N = 1, scale = 1):
    points = [createRandomPoint(scale) for i in range(N)]
    return points

def perturbeObject(obj, sigma_T, sigma_R):
    B = createRandomNoiseBivector(sigma_R, sigma_T) 
    M = ga_exp(B) * one 
    return M * obj * ~M

def perturbeObjectInplane(obj, sigma_T, sigma_R):
    B = createRandomNoiseInplane(sigma_R, sigma_T) 
    M = ga_exp(B) * one
    return M * obj * ~M

