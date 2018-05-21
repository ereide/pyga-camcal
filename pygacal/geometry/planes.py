from clifford import g3c
from numpy import pi, e
import numpy as np

from pygacal.common.cgatools import (  Sandwich, Dilator, Translator, 
                        Reflector, inversion, Rotor, 
                        Transversor, I3, I5, 
                        VectorEquality, anticommuter, MVto3DVec, VectoMV)
                        
from pygacal.geometry import   (createRandomBivector, createRandomNoiseBivector, 
                        createRandomNoiseVector, createRandomPoint, 
                        createRandomPoints, createRandomVector, 
                        createRandomVectors, perturbeObject)

layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])



def findPlaneParams(plane):
    """
    We find the vector from $O$ to the closest point on the line $a$, and then the unit vector defining the direction of the line $\hat(m)_a$. 

    #proposed by Rich Wareham
    $$ \Phi^* = d * n_{\inf} + \hat{n}$$

    """

    #Normalize the line
    plane = plane.normal()
    dualplane = plane.dual()
    d = float(dualplane | ep)
    n = (dualplane - d*ninf)*d
    return n


def createPlane(A, B, C):
    """
    Given G4,1 null vector representations of A = F(a), B = F(b) and C = F(c) we compute the line going through these points
    """

    return (A^B^C^ninf).normal()

def createRandomPlanes(N = 1, scale = 1):
    vectors = createRandomPoints(N*3, scale)
    planes = []
    for i in range(N):
        planes.append(createPlane(vectors[i], vectors[i + N], vectors[i + 2*N]))
    return planes    
 

def createNoisyPlaneSet(R_real, sigma_R, sigma_T, N = 1, scale = 1):
    """
    Creates a set of 2-tuples with the base line and the converted and disturbed line.
    """

    planeSet = []
    planes = createRandomPlanes(N, scale = scale)
    for plane in planes:
        newplane = R_real * plane * ~R_real
        planeSet.append((plane, perturbeObject(newplane, sigma_T, sigma_R)))

    return planeSet

def RotorPlane2Plane(P1, P2):
    return (1 - P2 * P1).normal()



def RotorPlane2PlaneSafe(P1, P2):
    """
    TODO: Hopefully this shouldn't be needed. Need some better way of doing this
    """
    if float((2 - 2*anticommuter(P1, P2))[0]) < float((2 - 2*anticommuter(P1, -P2))[0]):
        return RotorPlane2Plane(P1, -P2)
    else:
        return RotorPlane2Plane(P1, P2)


def createPointsOnPlane(plane):
    """
    Creates a list of points on a line. 

    ans[0] = a             #closest point to origin
    m = unitvector along the line

    ans[1:3] = a + (1 + m |a|*[1, 10, 100])

    """
    norm = np.linalg.norm
    a  = MVto3DVec(findPlaneParams(plane))
    a_mag = norm(a)
    m1 = np.array([a[1], -a[0], 0])
    m2 = np.cross(m1, a)

    m1 = VectoMV(a_mag * m1/norm(m1))
    m2 = VectoMV(a_mag * m2/norm(m2))
    a  = VectoMV(a)
    return [a, a + m1, a + m2, a + 100 *(m1 + m2), a + 100 *(m1 - m2)]




if __name__ == '__main__':
    pass
