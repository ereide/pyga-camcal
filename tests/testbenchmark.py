import numpy as np

from pygacal.common.cgatools import *

from pygacal.rotation import minimizeError
from pygacal.rotation.benchmark import *
from pygacal.rotation.costfunction import *
from pygacal.rotation.mapping import (  RotorLineMapping, BivectorLineMapping, LinePropertyBivectorMapping, 
                                BivectorPlaneMapping, PlanePropertyBivectorMapping, 
                                BivectorLogCostLineMapping, BivectorLineEstimationMapping, BivectorLineMultMapping, 
                                BivectorWeightedLineMapping, BivectorLogSumLineMapping, BivectorSumLogLineMapping, 
                                ExtendedBivectorMapping)

from pygacal.geometry import perturbeObject
from pygacal.geometry.lines  import RotorLine2Line, createRandomLines
from pygacal.geometry.planes import RotorPlane2Plane, createRandomPlanes

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class TestBenchmark(unittest.TestCase):
    def testBenchmarkLinesMinimizeError(self):
        seed = 21
        sigma_T = 0.005
        sigma_R = 0.002
        N = 10

        line1, line2 = createRandomLines(2)
        R_real = RotorLine2Line(line1, line2)

        traininglinesets = createNoisyLineSet(R_real, sigma_R, sigma_T, N)
        validationlinesets = createNoisyLineSet(R_real, sigma_R, sigma_T, N)
        print("Training and validation sets created with sig_r = %f and sig_t = %f, N = %d" % (sigma_R, sigma_T, N))

        map_list = [BivectorLineMapping]

        for map_obj in map_list:
            np.random.seed(seed)
            print(map_obj.name)
            benchmarkMinimizeError(R_real, traininglinesets, validationlinesets, fileout = None, mapping = map_obj)

    def testBenchmarkPlanesMinimizeError(self):
        seed = 234
        sigma_T = 0.5
        sigma_R = 0.2
        N = 50 

        plane1, plane2 = createRandomPlanes(2, scale = 20)
        R_real = RotorPlane2Plane(plane1, plane2)

        trainingsets = createNoisyPlaneSet(R_real, sigma_R, sigma_T, N)
        validationsets = createNoisyPlaneSet(R_real, sigma_R, sigma_T, N)
        print("Training and validation sets created with sig_r = %f and sig_t = %f, N = %d" % (sigma_R, sigma_T, N))

        print("Training and validation sets created with sig_r = %f and sig_t = %f, N = %d" % (sigma_R, sigma_T, N))

        map_list = [BivectorPlaneMapping, PlanePropertyBivectorMapping]

        plot = Plot3D()
        plane = validationsets[0][0]
        plot.addPlane(R_real * plane * ~R_real, color='r')

        for map_obj in map_list:
            np.random.seed(seed)
            print(map_obj.name)
            _, _, R_min = benchmarkMinimizeError(R_real, trainingsets, validationsets, fileout = None, mapping = map_obj, verificationfunction = planePointCostMetric)

            plot.addPlane(R_min * plane * ~R_min, color = map_obj.color)

        plot.show(False)
        
    def testBenchmarkMinimizeErrorPlot(self):
        sigma_R = 0.005
        sigma_T = 0.007
        N_training = 30
        N_plot = 4

        line1, line2 = createRandomLines(2)
        R_real = RotorLine2Line(line1, line2)

        traininglinesets = createNoisyLineSet(R_real, sigma_R, sigma_T, N_training)
        R_min, nit = minimizeError(traininglinesets, RotorLineMapping)   

        validationlines = createRandomLines(N_plot)
        plot = Plot3D()

        for line in validationlines:
            line_real = R_real * line * ~R_real
            line_est = R_min * line * ~R_min

            plot.addLine(line_real)
            plot.addLine(line_est)

        timestring =time.strftime("%Y%m%d-%H%M%S")
        figname = "../benchmarkreports/plot_%s.png" %timestring
        plot.save(figname)

    def testBenchmarkMinimizeErrorParameters(self):
        print("\nWARNING: VERY SLOW")

        translation_errors  = [0.0001, 0.01]
        rotation_errors     = [0.0001]
        N_list = [5, 10]

        benchmarkMinimizeErrorParameters(translation_errors, rotation_errors, N_list)

    def testPlotParameterDependencies(self):
        print("\nWARNING: VERY SLOW")

        sigma = np.array([0.0001, 0.01, 1])
        N_list = np.array([3, 9, 15])

        benchmarkParameterErrorPlot(N_list = N_list, sigma=sigma, show = False)


    def testLinePointError(self):
        print("\nWARNING: SLOW")
        #Testing how using the distance from certain points on a line as a good external metric of how well we are performing.

        np.random.seed(10)

        sigma_R = 0.05 
        sigma_T = 0.07
        N = 20

        line1, line2 = createRandomLines(2)
        R_real = RotorLine2Line(line1, line2)

        traininglinesets = createNoisyLineSet(R_real, sigma_R, sigma_T, N)

        #Test various minimazation algorithms 
        R_rotor, nit         = minimizeError(traininglinesets, RotorLineMapping)
        R_bivector, nit      = minimizeError(traininglinesets, BivectorLineMapping)
        R_lineproduct, nit   = minimizeError(traininglinesets, LinePropertyBivectorMapping)
        R_dummy              = RotorLine2Line(traininglinesets[0][0], traininglinesets[0][1])  #comparing it to just taking the first one we find
        
        R_list = [R_rotor, R_bivector, R_lineproduct, R_dummy]

        costs = []

        N_val = 10
        N_points = 4

        for R in R_list:
            costs.append(linePointCostMetric(R, R_real, N_val = N_val))

        print("rotor_pointcost          ", costs[0])
        print("bivector_pointcost       ", costs[1])
        print("lineproduct_pointcost    ", costs[2])
        print("dummy_pointcost          ", costs[3])
                

    def testExtremeLineRotation(self):
        print("\nRunning testExtremeLineRotation")
        print("")
        np.random.seed(1)
        #Test extreme values
        sigma_R             = 0.1 
        sigma_T             = 1
        N_train             = 100
        N_val               = 20
        line_scale          = 1000
        translation_scale   = 1000

        line1, line2 = createRandomLines(2)
        a = createRandomVector(scale = translation_scale)
        print("Translated lineA by ", a)

        b = createRandomVector(scale = translation_scale)
        print("Translated lineB by ", b)

        T_a = Translator(a)
        T_b = Translator(b)

        #Move them far away from the origin
        lineA = T_a * line1 * ~T_a  
        lineB = T_b * line2 * ~T_b  

        R_real = RotorLine2Line(lineA, lineB)

        traininglinesets    = createNoisyLineSet(R_real, sigma_R, sigma_T, N_train, scale = line_scale)
        validationlinesets  = createNoisyLineSet(R_real, sigma_R, sigma_T, N_val,   scale = line_scale)

        mappingList = [BivectorLineMapping]

        x0 = BivectorLineMapping.inverserotorconversion(R_real) 
        x0 += np.array([0.1, 0.2, -0.1, 1, -0.1, 0.21])

        plot = Plot3D()

        testline = validationlinesets[0][0]
        plot.addLine(R_real * testline * ~R_real, color='r')

        for mapping in mappingList:


            t0 = time.time()
            print("Running %s" % mapping.name)

            #Test various minimazation algorithms 
            R_min, nit     = minimizeError(traininglinesets, mapping, x0 = x0)

            costfunction = mapping.costfunction

            #Finding the cost if we used the actual rotor used to generate the matrix
            realtrainingcost = costfunction(R_real, traininglinesets)
            print("Real training cost is %s" % str(realtrainingcost))  

            realvalidationcost = costfunction(R_real, validationlinesets)
            print("Real validation cost is %s" % str(realvalidationcost))  

            minimumtrainingcost = costfunction(R_min, traininglinesets)
            print("minimized training cost %f" % minimumtrainingcost)

            minimumvalidationcost = costfunction(R_min, validationlinesets)
            print("minimized validation cost = %f" % minimumvalidationcost)

            realpointcost = linePointCostMetric(R_real, R_min, 10)
            print("Averaged cost for points a: %f, a + m: %f, a + 10m: %f, a + 100m: %f, a + 1000m: %f" % tuple(realpointcost))

            print("")
            print("R_real= %s" % str(R_real/np.sign(float(R_real(0)))))
            print("R_min = %s" % str(R_min/np.sign(float(R_min(0)))))

            B_real = ga_log(R_real/np.sign(float(R_real(0))))
            B_min  = ga_log(R_min/np.sign(float(R_min(0))))

            print("B_real= %s" % str(B_real))
            print("B_min = %s" % str(B_min))

            print("")
            print("C(R(B_real - B_min)) = %f" % rotorAbsCostFunction(ga_exp(B_min - B_real)))


            plot.addLine(R_min * testline * ~R_min, color = mapping.color)

            t_end = time.time()

            print("")
            print("Running time for extracting best rotor for %d line pairs is %f s" % (N_train, t_end - t0))
            print("\n\n")
        plot.show(False)

    def testExtendedLineRotation(self):
        print("\nRunning testExtendedLineRotation")
        print("")
        np.random.seed(1)
        #Test extreme values
        sigma_R             = 0.1 
        sigma_T             = 1
        N_train             = 100
        N_val               = 20
        line_scale          = 40

        mapping = ExtendedBivectorMapping

        B_real = (0.21381*e12) + (0.64143*e13) + (2.73*e14) + (2.73*e15) + (0.42762*e23) + (3.14*e24) + (3.14*e25) + (0.8*e45)
        R_real = ga_exp_complicated(B_real)

        print(R_real)

        traininglinesets    = createNoisyLineSet(R_real, sigma_R, sigma_T, N_train, scale = line_scale)
        validationlinesets  = createNoisyLineSet(R_real, sigma_R, sigma_T, N_val,   scale = line_scale)


        plot = Plot3D()

        testline = validationlinesets[0][0]
        plot.addLine(R_real * testline * ~R_real, color='r')



        t0 = time.time()
        print("Running %s" % mapping.name)

        #Test various minimazation algorithms 
        R_min, nit     = minimizeError(traininglinesets, mapping, x0 = None)

        costfunction = mapping.costfunction

        #Finding the cost if we used the actual rotor used to generate the matrix
        realtrainingcost = costfunction(R_real, traininglinesets)
        print("Real training cost is %s" % str(realtrainingcost))  

        realvalidationcost = costfunction(R_real, validationlinesets)
        print("Real validation cost is %s" % str(realvalidationcost))  

        minimumtrainingcost = costfunction(R_min, traininglinesets)
        print("minimized training cost %f" % minimumtrainingcost)

        minimumvalidationcost = costfunction(R_min, validationlinesets)
        print("minimized validation cost = %f" % minimumvalidationcost)

        realpointcost = linePointCostMetric(R_real, R_min, 10)
        print("Averaged cost for points a: %f, a + m: %f, a + 10m: %f, a + 100m: %f, a + 1000m: %f" % tuple(realpointcost))

        print("")
        print("R_real= %s" % str(R_real/np.sign(float(R_real(0)))))
        print("R_min = %s" % str(R_min/np.sign(float(R_min(0)))))

        B_real = ga_log(R_real/np.sign(float(R_real(0))))
        B_min  = ga_log(R_min/np.sign(float(R_min(0))))

        print("B_real= %s" % str(B_real))
        print("B_min = %s" % str(B_min))

        print("")
        print("C(R(B_real - B_min)) = %f" % rotorAbsCostFunction(ga_exp(B_min - B_real)))


        plot.addLine(R_min * testline * ~R_min, color = mapping.color)

        t_end = time.time()

        print("")
        print("Running time for extracting best rotor for %d line pairs is %f s" % (N_train, t_end - t0))
        print("\n\n")
        
        plot.show(False)

    def testWeigthingFunction(self):
        print("\nRunning testWeigthingFunction")
        print("")
        #np.random.seed(5)
        #Test on some extreme values
        sigma_R = 0.01 
        sigma_T = 0.1
        N_train = 100
        N_val   = 20
        line_scale = 10
        translation_scale = 100

        line1, line2 = createRandomLines(2)
        a = createRandomVector(scale = translation_scale)
        print("Translated lineA by ", a)

        b = createRandomVector(scale = translation_scale)
        print("Translated lineB by ", b)

        T_a = Translator(a)
        T_b = Translator(b)

        #Move them far away from the origin
        lineA = T_a * line1 * ~T_a  
        lineB = T_b * line2 * ~T_b  

        R_real = RotorLine2Line(lineA, lineB)

        #x0 = BivectorWeightedLineMapping.inverserotorconversion(R_real) 

        #x0 += np.array([15, 21, -11, 0.1, -0.1, 0.21])
        #R_start = BivectorWeightedLineMapping.
        #R_start = BivectorWeightedLineMapping.rotorconversion(x0)

        traininglinesets    = createNoisyLineSet(R_real, sigma_R, sigma_T, N_train, scale = line_scale)
        validationlinesets  = createNoisyLineSet(R_real, sigma_R, sigma_T, N_val,   scale = line_scale)

        mapping = BivectorWeightedLineMapping

        weightList = [1e-4 , 1e-2, 1, 1e2, 1e4]
        #mappingList = [BivectorLineMapping, LinePropertyBivectorMapping, BivectorLogCostLineMapping]

        plot = Plot3D()

        testline = validationlinesets[0][0]
        plot.addLine(R_real * testline * ~R_real, color='r')

        for weight in weightList:


            t0 = time.time()
            print("Running %s" % mapping.name)
            print("Weight = %e" %weight)

            #Test various minimazation algorithms 
            mapping = BivectorWeightedLineMapping
            mapping.costfunction = sumWeightedLineSquaredErrorCost(weight)

            R_min, nit     = minimizeError(traininglinesets, mapping, x0 = None)

            #mapping.costfunction = logSumWeightedLineSquaredErrorCost(weight)

            #Finding the cost if we used the actual rotor used to generate the matrix
            realtrainingcost = mapping.costfunction(R_real, traininglinesets)
            print("Real training cost is %s" % str(realtrainingcost))  

            realvalidationcost = mapping.costfunction(R_real, validationlinesets)
            print("Real validation cost is %s" % str(realvalidationcost))  

            minimumtrainingcost = mapping.costfunction(R_min, traininglinesets)
            print("minimized training cost %f" % minimumtrainingcost)

            minimumvalidationcost = mapping.costfunction(R_min, validationlinesets)
            print("minimized validation cost = %f" % minimumvalidationcost)

            realpointcost = linePointCostMetric(R_real, R_min, 10)
            print("Averaged cost for points a: %f, a + m: %f, a + 10m: %f, a + 100m: %f, a + 1000m: %f" % tuple(realpointcost))

            print("")
            print("R_real= %s" % str(R_real/np.sign(float(R_real(0)))))
            print("R_min = %s" % str(R_min/np.sign(float(R_min(0)))))
            #print("R_start = %s" % str(R_start/np.sign(float(R_start(0)))))


            B_real  = ga_log(R_real/np.sign(float(R_real(0))))
            B_min   = ga_log(R_min/np.sign(float(R_min(0))))
            #B_start = ga_log(R_start/np.sign(float(R_start(0))))

            print("")
            print("B_real= %s"      % str(B_real))
            print("B_min = %s"      % str(B_min))
            #print("B_start = %s"    % str(B_start))


            print("")
            print("C(R(B_real - B_min)) = %f" % rotorAbsCostFunction(ga_exp(B_min - B_real)))

            plot.addLine(R_min * testline * ~R_min, color = mapping.color)

            t_end = time.time()

            print("")
            print("Running time for extracting best rotor for %d line pairs is %f s" % (N_train, t_end - t0))
            print("\n\n")


        plot.show(False)


    def testLineAveraging(self):
        seed = 1
        sigma_T = 0.005
        sigma_R = 0.002
        N = 100 

        mapping = BivectorLineMapping

        line_start, line_target = createRandomLines(2)
        R_real_min, N_int = minimizeError([(line_start, line_target)], mapping = BivectorLineMapping)
        R_real     = RotorLine2Line(line_start, line_target)

        print("R_real    ", R_real)
        print("R_real_min", R_real_min)
        print("B_real_min", ga_log(R_real_min))
        print("B_real    ", ga_log(R_real))

        print("L_real_min", R_real_min * line_start * ~R_real_min)


        traingingdata   = [line_start, [perturbeObject(line_target, sigma_R, sigma_T) for _ in range(N)]]
        validationdata  = [line_start, [perturbeObject(line_target, sigma_R, sigma_T) for _ in range(N)]]

        print("Training and validation sets created with sig_r = %f and sig_t = %f, N = %d" % (sigma_R, sigma_T, N))

        map_list = [BivectorLineEstimationMapping]

        for map_obj in map_list:
            np.random.seed(seed)
            print(map_obj.name)
            realtrainingcost, minimumvalidationcost, R_min = benchmarkMinimizeError(R_real, traingingdata, validationdata, N = N, fileout = None, mapping = map_obj)
            print("L_real   = ", line_target)
            print("L_min    = ", R_min*line_start*~R_min)
            print("L_example= ", validationdata[0])


def plotCostFunctionEffect():
    print("\nRunning plotCostFunctionEffect")
    print("")
    np.random.seed(1)


    N_train             = 20

    line_scale          = 10
    translation_scale   = 10

    #Test extreme values
    sigma_R             = 0.01 
    sigma_T             = 1

    line1, line2 = createRandomLines(2)
    a = createRandomVector(scale = translation_scale)
    print("Translated lineA by ", a)

    b = createRandomVector(scale = translation_scale)
    print("Translated lineB by ", b)

    T_a = Translator(a)
    T_b = Translator(b)

    #Move them far away from the origin
    lineA = T_a * line1 * ~T_a  
    lineB = T_b * line2 * ~T_b  

    R_real = RotorLine2Line(lineA, lineB)

    linesets     = createNoisyLineSet(R_real, sigma_R, sigma_T, N_train, scale = line_scale)

    mapping              = BivectorWeightedLineMapping
    mapping.costfunction = sumWeightedLineSquaredErrorCost(1./translation_scale)

    x0       = mapping.inverserotorconversion(R_real) 
    x_test   = x0[0]
    y_test   = x0[3]

    N_rot               = 50
    N_tran              = 50
    rot_range           = 0.4
    translation_range   = 10

    rotation    = np.linspace(-rot_range,         rot_range,         N_rot)
    translation = np.linspace(-translation_range, translation_range, N_tran)

    ans = np.zeros((N_rot, N_tran))

    for i, rot in enumerate(rotation):
        for j, tran in enumerate(translation):
            x0[0]       = x_test + tran
            x0[3]       = y_test + rot
            ans[i, j]   = np.log(mapping.costfunction(mapping.rotorconversion(x0), linesets))

    xv, yv = np.meshgrid(translation, rotation)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.set_xlabel("Translation error")
    ax.set_ylabel("Rotation error")
    ax.set_zlabel("log(objective function)")

    ax.plot_wireframe(xv, yv, ans)
    plt.show()        

if __name__ == '__main__':  
    unittest.main()
    #plotCostFunctionEffect()