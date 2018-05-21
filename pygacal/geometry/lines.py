from clifford import g3c

import numpy as np
import warnings

from pygacal.common.cgatools import (  Sandwich, Dilator, Translator, 
                        Reflector, inversion, Rotor, 
                        Transversor, I3, I5, 
                        VectorEquality, anticommuter, MVto3DVec)
                        
from pygacal.geometry import   (createRandomBivector, createRandomNoiseBivector, 
                        createRandomNoiseVector, createRandomPoint, 
                        createRandomPoints, createRandomVector, 
                        createRandomVectors, perturbeObject)

layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])



def RotorLine2Line(L1, L2):
    """
    Joan Lasenby's Rotor implementation

    .. math:: 
        K = 2 + L_1 L_2 + L_2 L_1  
        X = (1 - \frac{(K)_4 }{2*(K)_0})
        L2 = X (1 + L_2 L_1)


    Parameters
    ----------
    L1 : MultiVector
        A 3-blade representation of line 1 in CGA
    L2 : MultiVector
        A 3-blade representation of line 1 in CGA


    Returns
    -------
    
    R : MultiVector
        A rotor taking line 1 to line 2


    """
    L21 = L2 * L1   #hack -> L12 = L21
    K = 2 + 2*anticommuter(L1, L2)   #Should actually be 2 + L12 + L21
    beta = K(4)
    alpha = 2 * float(K[0])
    R = ((1 - beta/alpha)*(1 + L21))/np.sqrt(alpha/2) #TODO: Check teh normalization factor
    return R

def RotorLine2LineSafe(L1, L2):
    """
    TODO: Hopefully this shouldn't be needed. Need some better way of doing this
    """
    if float((2 + 2*anticommuter(L1, L2))[0]) < float((2 + 2*anticommuter(L1, -L2))[0]):
        return RotorLine2Line(L1, -L2)
    else:
        return RotorLine2Line(L1, L2)

def RotorLine2LineThroughO(L1, L2):
    """
    This function finds the rotor that converts L1 into L2 by translating L1 to the origin, 
    reflecting it s.t. it is parallel with L2 and translating it out to L2. 
    """

    a, ma = findLineParams(L1)
    c, mc = findLineParams(L2)

    return parameterRotor(a, c, ma, mc)



def createLine(A, B):
    """
    Given G4,1 null vector representations of A = F(a), B = F(b) we compute the line going through these points
    """

    return (A^B^ninf).normal()


def findLineParams(line):
    """
    We find the vector from $O$ to the closest point on the line $a$, and then the unit vector defining the direction of the line $\hat(m)_a$. 

    $$ \hat(m)_a = L \cdot E_0 $$

    $$ a = ((L ^ n_0)\cdot E_0) \cdot \hat(m)_a $$ 
    """

    #Normalize the line
    line = line.normal()
    ma = (line | E0)
    a = -((line ^ no)|E0)|ma 
    return a, ma


def extract2Dparams(Limg_d, f = 1):
    """
    Translates line paralell to the e12 plane to the origin and extracts the lines paramaters 

    Returns 2 2-D vectors representing the closest distance from the origin and the direction of the line.

    Asserts that the line was indeed parallel to the plane as assumed up to a reasonable approximation.

    TODO: Under development

    """
    line2D = Sandwich(Limg_d, Translator(-e3*f))
    a, ma = findLineParams(line2D)
    a_vec = MVto3DVec(a)
    ma_vec = MVto3DVec(ma)
    
    #assert(a_vec[2] < 1e-5 and  ma_vec[2] < 1e-5)
    if not (abs(a_vec[2]) < 1e-5 and  abs(ma_vec[2]) < 1e-5):
        print(a_vec[2], ma_vec[2])

    return a_vec[:2], ma_vec[:2]    


def createPointsOnLine(line, N_points):
    """
    Creates a list of points on a line. 

    ans[0] = a             #closest point to origin
    m = unitvector along the line

    ans[1:3] = a + (1 + m |a|*[1, 10, 100])

    """
    a, m = findLineParams(line)
    a_mag = np.linalg.norm(a)
    return [a + a_mag*m * 10**k for k in range(N_points)]



def createRandomLines(n = 1, scale = 1):
    vectors = createRandomPoints(n*2, scale)
    lines = []
    for i in range(n):
        lines.append(createLine(vectors[i], vectors[i + n]))

    return lines

def createRandomRotationNoise(sigma):
    """
    Generates a random rotion vector that rotates a point in an arbitrary direction
    alpha radians. Alpha is normally distributed with variance sigma
    """

    alpha = np.random.normal(scale=sigma)
    a, b, c = np.random.rand(3)

    B = (a * e1 + b * e2 + c * e3)*I3
    return Rotor(B, alpha)

def noisyRotor(a, c, ma, mc, sigma_R, sigma_T):
    """
    Disturbes the transformation defined by a, c, ma, mc with small deviations in rotation and translation

    Params
    a - distance to line 1
    c - distance to line 2
    ma - direction of line 1
    mc - direction of line 2
    sigma_R - std of disturbance in rotation 
    sigma_T - std of disturbance in translation    

    """
    e_c = createRandomNoiseVector(sigma_T)
    R_e = createRandomRotationNoise(sigma_R)
    R = 1 - mc*ma
    R_T = (Translator(c + e_c) * R_e * R * Translator(-a)).normal()    
    return R_T



def createNoisyLineSet(R_real, sigma_R, sigma_T, N = 1, scale = 1):
    """
    Creates a set of 2-tuples with the base line and the converted and disturbed line.
    """

    lineSet = []
    lines = createRandomLines(N, scale = scale)
    for line in lines:
        line = line
        newline = R_real * line * ~R_real
        lineSet.append((line, perturbeObject(newline, sigma_T, sigma_R)))
    return lineSet




def createNoisySetThroughOrigin(a, c, ma, mc, sigma_R, sigma_T, N = 1):
    """
    Creates a set of 2-tuples with the base line and the converted and disturbed line.
    """

    warnings.warn("This function deprecated in favour of createNoisySet", DeprecationWarning)

    lineSet = []
    lines = createRandomLines(N)
    for line in lines:
        R_T = noisyRotor(a, c, ma, mc, sigma_R, sigma_T)
        newline = (R_T * line * ~R_T)
        lineSet.append((line, newline))
    return lineSet


