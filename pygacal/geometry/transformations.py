from clifford import MultiVector
from clifford import g3c
from numpy import pi, e
import numpy as np

from pygacal.common.cgatools import Translator, Rotor, MVto3DVec, MVEqual, VectoMV
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])

#Useful transformations

def rotation_to_matrix(theta):
    alpha, beta, gamma = theta

    cy = np.cos(alpha)
    sy = np.sin(alpha)
    X = np.matrix([[1, 0, 0],
                   [0, cy, -sy],
                   [0, sy, cy]])
    

    cp = np.cos(beta)
    sp = np.sin(beta)
    Y = np.matrix([[cp,  0, sp],
                   [0,   1, 0],
                   [-sp, 0, cp]])

    cr = np.cos(gamma)
    sr = np.sin(gamma)
    Z = np.matrix([[cr, -sr, 0],
                   [sr, cr, 0],
                   [0, 0, 1]])
    
    R = np.dot(Z, np.dot(Y, X))

    return R


def rotation_to_rotor(theta):
    alpha, beta, gamma = theta
    return Rotor(e12, gamma) * Rotor(-e13, beta) * Rotor(e23, alpha) 

def translation_to_rotor(translation):
    x, y, z = translation
    t = x*e1 + y*e2 + z*e3
    return Translator(t)

def matrix_to_rotation(R_mat):
    #From http://planning.cs.uiuc.edu/node103.html
    alpha = np.arctan2(R_mat[2, 1], R_mat[2, 2])
    beta  = np.arctan2(-R_mat[2, 0],np.sqrt(R_mat[2, 1]**2 + R_mat[2, 2]**2))
    gamma = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    return np.array([alpha, beta, gamma])
    
def matrix_to_translation(R):
    return R[:3, 3]

def full_matrix_to_param(R_mat):
    theta = matrix_to_rotation(R_mat)
    translation = matrix_to_translation(R_mat)
    return theta, translation
    
def parameters_to_versor(theta, translation):
    T = translation_to_rotor(translation)
    R = rotation_to_rotor(theta)
    return T*R

#Modified
#From wikipedia: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
"""
def quaternion_to_rotation(quat):
    w, x, y, z = quat
    ysqr = y * y
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    Z = -np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    X = -np.arctan2(t3, t4)
    
    return np.array([X, Y, Z])
"""

def rotation_to_quaternion(theta):
    c = np.cos(-theta/2)
    s = np.sin(-theta/2)
    
    quat = np.zeros(4)
    
    quat[0] = c[2]*c[1]*c[0] - s[2]*s[1]*s[0]
    quat[1] = c[2]*c[1]*s[0] + s[2]*s[1]*c[0]
    quat[2] = c[2]*s[1]*c[0] - s[2]*c[1]*s[0]
    quat[3] = s[2]*c[1]*c[0] + c[2]*s[1]*s[0]
    return quat 

def quaternion_to_rotation(quat):
    w, x, y, z = quat
    
    t0 = 2*(w*z - x*y)
    t1 = w*w + x*x - y*y - z*z 
    Z  = -np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y + z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y  = -np.arcsin(t2)
    
    t3 = +2.0 * (w * x - y * z)
    t4 =  w*w - x*x - y*y + z*z 
    X  = -np.arctan2(t3, t4)
    
    return np.array([X, Y, Z])


def rotor_to_quaternion(rotor):
    quat = np.zeros(4)
    quat[0] = rotor[0]
    quat[1] = rotor[(2, 3)]
    quat[2] = -rotor[(1, 3)]  
    quat[3] = rotor[(1, 2)]
    return quat 


def rotor_to_translation(T):
    return MVto3DVec(ep | T)

def versor_decomposition(V):
    B_R = V[(1, 2)]*e12 + V[(1, 3)]*e13 + V[(2, 3)]*e23  
    R = V(0) + B_R
    T = (V * ~R)
    return T, R

def versor_to_param(V):
    T, R = versor_decomposition(V)
    theta = quaternion_to_rotation(rotor_to_quaternion(R))
    t     = rotor_to_translation(T)*2
    return theta, t
    
    
def full_projection_matrix(theta, translation):
    ans = np.zeros((4, 4))
    ans[:3, :3] = rotation_to_matrix(theta)
    ans[:3, 3] = translation
    ans[3, 3] = 1
    return ans
    
    
def versors_to_projection(V):
    return full_projection_matrix(*versor_to_param(V))

def projection_to_versor(P):
    theta = matrix_to_rotation(P) 
    t     = matrix_to_translation(P)
    return parameters_to_versor(theta, t)


def vec_to_homogenous(x):
    return np.concatenate((x, np.array([1])))

def homogenous_to_vec(X):
    X = np.array(X)
    return X[:-1]/X[-1]

