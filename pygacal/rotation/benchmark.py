import unittest

import os
import time

from numpy import pi, e
import numpy as np


import clifford as cl
from clifford import g3c

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




from pygacal.common.cgatools import Sandwich, Dilator, Translator, Reflector, inversion, Rotor, Transversor, I3, I5, VectorEquality, ga_log, ga_exp
from pygacal.common.plotting import Plot3D

#TODO: Explicit imports
from pygacal.rotation import minimizeError
from pygacal.rotation.mapping import BivectorLineMapping
from pygacal.rotation.costfunction import linePointCostMetric, rotorAbsCostFunction

from pygacal.geometry.planes import *
from pygacal.geometry.lines import *


#from clifford_tools.common.g3c.core import RotorLine2Line, RotorPlane2Plane, ga_exp


layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])

def benchmarkMinimizeError(R_real, trainingdata, validationdata, N = None, fileout = None, 
                            mapping = BivectorLineMapping, 
                            verificationfunction = linePointCostMetric):
    """
    A function to benchmark the error using a given mapping
    """
    #To allow for both writing to std out and a file
    if fileout:
        outfile = open(fileout, 'a')
        def fileprint(string):
            outfile.write(string + "\n")
    else:
        def fileprint(string):
            print(string)

    t0 = time.time()

    costfunction = mapping.costfunction

    #Finding the cost if we used the actual rotor used to generate the matrix
    
    x0 = mapping.startValue()
    R_start = mapping.rotorconversion(x0)
    if N is None:
        N = len(trainingdata)

    

    R_min, nit = minimizeError(trainingdata, mapping = mapping, x0 = x0)   

    
    realtrainingcost = costfunction(R_real, trainingdata)
    fileprint("Real training cost is %s" % str(realtrainingcost))  

    realvalidationcost = costfunction(R_real, validationdata)
    fileprint("Real validation cost is %s" % str(realvalidationcost))  



    fileprint("")
    initialtrainingcost = costfunction(R_start, trainingdata)
    fileprint("initial training cost %f" % initialtrainingcost)

    initialvalidationcost = costfunction(R_start, validationdata)
    fileprint("initial validation cost %f" % initialvalidationcost)



    fileprint("")
    minimumtrainingcost = costfunction(R_min, trainingdata)
    fileprint("minimized training cost %f" % minimumtrainingcost)

    minimumvalidationcost = costfunction(R_min, validationdata)
    fileprint("minimized validation cost = %f" % minimumvalidationcost)
    


    fileprint("")
    fileprint("Costfunction invariant point cost %s" % str(verificationfunction(R_min, R_real, 100)))

    R_real_norm = R_real/np.sign(float(R_real(0)))
    R_min_norm = R_min/np.sign(float(R_min(0)))

    fileprint("")
    fileprint("R_real= %s" %  str(R_real_norm))
    fileprint("R_min = %s" %  str(R_min_norm))

    B_real = ga_log(R_real_norm) 
    B_min  = ga_log(R_min_norm)
    B_diff = B_real - B_min 

    fileprint("")
    fileprint("B_real= %s" %  str(B_real))
    fileprint("B_min = %s" %  str(B_min))

    R_diff = ga_exp(B_diff)
    diff_cost = rotorAbsCostFunction(R_diff)

    fileprint("")
    fileprint("B_diff  = %s" %  str(B_diff))
    fileprint("R_min   = %s" %  str(R_diff))
    fileprint("cost(R) = %s" %  str(diff_cost))
    

    t_end = time.time()

    fileprint("")
    fileprint("Running time for extracting best rotor for %d line pairs converging after %d iterations is %f s" % (N, nit, t_end - t0))
    fileprint("\n\n")

    if fileout:
        outfile.close()

    return realtrainingcost, minimumvalidationcost, R_min

def benchmarkMinimizeErrorParameters(translation_errors, rotation_errors, N_list, fileprint = False):
    if fileprint:
        timestring =time.strftime("%Y%m%d-%H%M%S")
        outfilenname = os.path.abspath("../benchmarkreports/parameters_%s.txt" % timestring)
    
        open(outfilenname, 'w').close()
        

    line1, line2 = createRandomLines(2)
    R_real = RotorLine2Line(line1, line2)

    for sigma_R in rotation_errors:
        for sigma_T in translation_errors:
            for N in N_list:
                print("Starting: sig_r = %f, sig_t = %f with N = %d"  % (sigma_R, sigma_T, N))

                trainingdata = createNoisyLineSet(R_real, sigma_R, sigma_T, N)
                validationdata = createNoisyLineSet(R_real, sigma_R, sigma_T, N)

                if fileprint:
                    outfile = open(outfilenname, 'a')
                    outfile.write("Training and validation sets created with sig_r = %f and sig_t = %f, N = %d \n" % (sigma_R, sigma_T, N))
                    outfile.close()

                if fileprint:
                    benchmarkMinimizeError(R_real, trainingdata, validationdata, fileout = outfilenname)
                else:
                    benchmarkMinimizeError(R_real, trainingdata, validationdata, fileout = None)
                    
    
    print("Complete")

def benchmarkParameterErrorPlot(N_list, sigma, mapping = BivectorLineMapping, show = False):

    sizeN = len(N_list)
    sizeS = len(sigma)
    shape = (sizeN, sizeS)
    x = np.zeros(shape)
    y = np.zeros(shape)
    realcost = np.zeros(shape)
    mincost =  np.zeros(shape)

    if show:
        timestring =time.strftime("%Y%m%d-%H%M%S")
        figname = os.path.abspath("../benchmarkreports/parameters_%s_%s.png" % (mapping.name, timestring))
        outfilenname = os.path.abspath("../benchmarkreports/parameters_%s_%s.txt" % (mapping.name, timestring))
        open(outfilenname, 'w').close()


    line1, line2 = createRandomLines(2)
    R_real = RotorLine2Line(line1, line2)

    for i in range(len(N_list)):
        for j in range(len(sigma)):
            N = N_list[i]
            sig = sigma[j]

            x[i, j] = np.log10(sig)
            y[i, j] = N

            trainingdata = createNoisyLineSet(R_real, sig, sig, N)
            validationdata = createNoisyLineSet(R_real, sig, sig, N)

            if show:
                outfile = open(outfilenname, 'a')
                outfile.write("Training and validation sets created with sig_r = %f and sig_t = %f, N = %d \n" % (sig, sig, N))
                outfile.close()
                realtrainingcost, minimumvalidationcost, R_min = benchmarkMinimizeError(R_real, trainingdata, validationdata, fileout = outfilenname)
                
            else:
                realtrainingcost, minimumvalidationcost, R_min = benchmarkMinimizeError(R_real, trainingdata, validationdata, fileout = None)
                
            mincost[i, j], realcost[i, j] = np.log10(realtrainingcost), np.log10(minimumvalidationcost)
            print("Starting: sig = %f with N = %d"  % (sig, N))


    fig = plt.figure()
    axes = Axes3D(fig)


    axes.plot_wireframe(x, y, realcost, colors = 'r')
    axes.plot_wireframe(x, y, mincost, colors = 'b')

    axes.xaxis.set_label("log(sigma)")
    axes.yaxis.set_label("N")
    axes.zaxis.set_label("log(cost)")


    plt.savefig(figname)
    if show:
        plt.show()


if __name__ == '__main__':
    pass