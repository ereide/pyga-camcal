from pygacal.common.cgatools import *
from pygacal.geometry import *
from pygacal.geometry.lines import *
from pygacal.geometry.transformations import * 
from pygacal.rotation.mapping import BivectorLineMapping
from pygacal.rotation import minimizeError
from pygacal.rotation.costfunction import rotorAbsCostFunction

import matplotlib.pyplot as plt
import scipy.stats as stats

import time

import numpy as np

def quaternion_mult(q,r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                         x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                         x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)

def bivector_difference_rotor(R_true, R_test):
    B_true = ga_log(R_true)
    B_test = ga_log(R_test) 
    return ga_exp(B_test - B_true)

def quat_conjugate(quat):
    return np.array([quat[0],-1*quat[1],-1*quat[2],-1*quat[3]])

def cross_product(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

def dual_quat_line(x_1, x_2):
    n = x_2 - x_1
    m = np.cross(x_1, x_2)
    return n, m

def D_matrix(a, ad, b, bd):
    D = np.zeros((6, 8))
    D[:3, 0]   = a - b
    D[:3, 1:4] = cross_product(a + b)
    D[3:, 0]   = ad - bd
    D[3:, 1:4] = cross_product(ad + bd)
    D[3:, 4]   = a - b
    D[3:, 5:] = cross_product(a + b)

    return D

def estimate_dual_quaternions(traininglinesets):
    t0 = time.time()
    N = len(traininglinesets)
    C = np.zeros((N*6, 8))

    for i in range(len(traininglinesets)):
        linepair = traininglinesets[i]
        L1 = linepair[0]
        L2 = linepair[1]    
        a, ma = findLineParams(L1)
        b, mb = findLineParams(L2)
            
        x11 = MVto3DVec(a)
        x12 = MVto3DVec(a) + MVto3DVec(ma)
        x21 = MVto3DVec(b)
        x22 = MVto3DVec(b) + MVto3DVec(mb)
        
        a, ad = dual_quat_line(x11, x12)
        b, bd = dual_quat_line(x21, x22)   
        
        C[6*i:6*(i + 1), :] = D_matrix(a, ad, b, bd)
        
        

    U, S, VH = np.linalg.svd(C)

    V = VH.transpose()

    v7 = V[:, 6]
    v8 = V[:, 7]

    u1 = v7[:4]
    v1 = v7[4:]

    u2 = v8[:4]
    v2 = v8[4:]

    a = np.dot(u1, v1)
    b = np.dot(u1, v2) + np.dot(v1, u2)
    c = np.dot(u2, v2)

    mu_1 = (-b + np.sqrt(b*b - 4 * a * c))/(2 * a)
    mu_2 = (-b - np.sqrt(b*b - 4 * a * c))/(2 * a)

    a = np.dot(u1, u1)
    b = 2 * np.dot(u1, u2) 
    c = np.dot(u2, u2)

    beta_1 = 1.0 / np.sqrt(mu_1*mu_1 * a + mu_2 * b + c)
    beta_2 = 1.0 / np.sqrt(mu_2*mu_2 * a + mu_2 * b + c)

    alpha_1 = mu_1 * beta_1
    alpha_2 = mu_2 * beta_2

    beta = min(beta_1, beta_2)
    mu = ([mu_1, mu_2])[np.argmin([[beta_1, beta_2]])]
    alpha = mu * beta

    r  = alpha * v7 + beta * v8
    #r_1 = alpha_1 * v7 + beta_1 * v8
    #r_2 = alpha_2 * v7 + beta_2 * v8

    #r = [r_1, r_2][np.argmax([np.linalg.norm(r_1),np.linalg.norm(r_2)])]

    t_diff_edu = time.time() - t0

    #print(t_diff_edu)

    quat = r[:4]

    angles = quaternion_to_rotation(quat) 
    
    dual_quat = r[4:]

    t = -2 * quaternion_mult(dual_quat, quat_conjugate(quat))

    translation = t[1:]

    return parameters_to_versor(angles, translation)


def rotation_cost(R):
    rotation = (R - 1)
    return float((rotation*~rotation)[0])

def translation_cost(R):
    translation = R | ep
    return float((translation*~translation)[0])

def generate_data():
    seed = 31

    N = 20
    

    for sigma in [0.0001, 0.0005, 0.0008, 0.001, 0.005, 0.008, 0.01, 0.05, 0.08, 0.1]:
        #for N in [10, 15, 20, 50, 70, 100, 150, 200]:
        for _ in range(10):
            line1, line2 = createRandomLines(2)
            R_true = RotorLine2Line(line1, line2)

            traininglinesets = createNoisyLineSet(R_true, sigma, sigma, N)
            try:
                R_eduardo = estimate_dual_quaternions(traininglinesets)
                Rc_eduardo = bivector_difference_rotor(R_true, R_eduardo)

                x0 = BivectorLineMapping.inverserotorconversion(R_eduardo)

                R_min, nit_boost = minimizeError(traininglinesets, mapping = BivectorLineMapping, x0 = None)   
                Rc_min = bivector_difference_rotor(R_true, R_min)

                print("%.4f, %d, %.4e, %.4e, %.4e, %.4e" % (sigma, N, rotation_cost(Rc_eduardo), translation_cost(Rc_eduardo),  
                                                            rotation_cost(Rc_min), translation_cost(Rc_min)))

            except:
                print()
                continue

def plot_data():
    data = np.genfromtxt('data_n20.txt', delimiter=',')

    fontsize = 22

    x  = np.log10(data[:, 0])
    y_ed_rot  = np.log10(data[:, 2])
    y_c_rot  = np.log10(data[:, 4])

    a_1, b_1     = stats.linregress(x, y_ed_rot)[:2]
    
    a_2, b_2     = stats.linregress(x, y_c_rot)[:2]

    y_ed_tra  = np.log10(data[:, 3])
    y_c_tra   = np.log10(data[:, 5])

    left  = -4.5
    right = -0.5


    plt.plot(x, y_ed_rot, 'ro', x, y_c_rot, 'bx')
    plt.plot([left, right], [left*a_1 + b_1, right*a_1 + b_1], 'r-')
    plt.plot([left, right], [left*a_2 + b_2, right*a_2 + b_2], 'b-')

    plt.legend(["Dual quaternions", "Rotor cost function"], loc = 2, fontsize = fontsize)
    plt.xlabel("$ \\log_{10}(\\sigma) $", fontsize = fontsize)
    plt.ylabel("$ \\log_{10}(e_r) $", fontsize = fontsize)
    plt.xlim(left = -4.5, right = -0.5)
    plt.show()

    a_1, b_1     = stats.linregress(x, y_ed_tra)[:2]   
    a_2, b_2     = stats.linregress(x, y_c_tra)[:2]


    plt.figure()
    plt.plot(x, y_ed_tra, 'ro', x, y_c_tra, 'bx')
    plt.plot([left, right], [left*a_1 + b_1, right*a_1 + b_1], 'r-')
    plt.plot([left, right], [left*a_2 + b_2, right*a_2 + b_2], 'b-')

    plt.legend(["Dual quaternions", "Rotor cost function"], loc = 2, fontsize = fontsize)
    plt.xlabel("$ \\log_{10}(\\sigma) $", fontsize = fontsize)
    plt.ylabel("$ \\log_{10}(e_t) $", fontsize = fontsize)
    plt.xlim(left = -4.5, right = -0.5)
    plt.show()


if __name__ == "__main__":
    #generate_data()
    plot_data()


