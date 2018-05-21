import unittest

import clifford as cl
from clifford import g3c
from numpy import pi, e
import numpy as np
from clifford import MultiVector

from pygacal.common.cgatools import (  Sandwich, Dilator, Translator, Reflector, 
                        inversion, Rotor, Transversor, I3, I5, 
                        VectorEquality, Distance, ga_log, ga_exp, MVEqual, Meet)

from pygacal.geometry import createRandomBivector, createRandomVector, createRandomPoints, perturbeObject, perturbeObjectInplane
from pygacal.geometry.lines import createLine, createRandomLines
from pygacal.geometry.planes import createPlane


from pygacal.common.plotting import *


from pygacal.camera import SLAM
from pygacal.camera.projection import *


from pygacal.rotation import minimizeError
from pygacal.rotation.costfunction import sumImageFunction, sumImageThreeViewAllPairsCostFunction
from pygacal.rotation.mapping import (BivectorLineImageMapping, MultiViewLineImageMapping, 
                            bivectorToVecRepr, vecReprToBivector, ThreeViewLineImageMapping,
                            ExtendedBivectorLineImageMapping)
from pygacal.rotation.benchmark import benchmarkMinimizeError




layout = g3c.layout
locals().update(g3c.blades)

    
ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])

np.random.seed(2512)



class TestCamera(unittest.TestCase):
    def testMappingRepresentation(self):
        K = 5
        x = np.random.rand(K*6)
        x_test = np.zeros(K*6)
        for i in range(K):
            B  = vecReprToBivector(x[i*6:(i+1)*6])
            x_test[i*6:(i+1)*6] = bivectorToVecRepr(B)
        assert(all(x == x_test))


    def testLineProjection(self):
        A, B = createRandomPoints(2)
        R = ga_exp(createRandomBivector())
        L = createLine(A, B)
        L_img = projectLineToPlane(L, R)
        A_img = projectPointToPlane(A, R)
        B_img = projectPointToPlane(B, R)
        L_img_actual = createLine(A_img, B_img)
        assert(MVEqual(L_img, L_img_actual))

    def testImageCostFunction(self):
        O1 = up(0)
        B = 0.1 * e12 + 0.2*e13 + 0.1 *e23 + (3*e1 -1*e2 + 2*e3)*ninf
        R = ga_exp(B)
        N = 10

        lines = createRandomLines(N, scale = 30)
        cPlane1 = (ninf + e3)*I5   #Camera plane 1
        cPlane2 = R * cPlane1 * ~R

        lines_img_d   = [projectLineToPlane(line, R) for line in lines]

        assert(sumImageFunction(R, lines, lines_img_d) < 1e-20) #Very small 
        R_wrong = ga_exp(B * 0.5)
        assert(sumImageFunction(R_wrong, lines, lines_img_d) > 1e-5) #not very small
        

    def testRotationExtraction(self):
        np.random.seed(2)
        O1 = up(0)
        B = 0.1 * e12 + 0.2*e13 + 0.1 *e23 + (3*e1 -1*e2 + 2*e3)*ninf
        R = ga_exp(B)
        N = 10

        lines = createRandomLines(N, scale = 30)
        cPlane1 = (ninf + e3)*I5   #Camera plane 1
        cPlane2 = R * cPlane1 * ~R

        lines_img_d   = [projectLineToPlane(line, R) for line in lines]
        R_min, Nint = minimizeError((lines, lines_img_d), BivectorLineImageMapping, x0 = None)

        assert(MVEqual(R_min, R, rtol = 1e-2, atol = 1e-2,  verbose = False))    #Hard coded values. 
        assert(sumImageFunction(R, lines, lines_img_d) < sumImageFunction(R_min, lines, lines_img_d))
        
    def testNoisyRotationExtraction(self):
        verbose = False

        np.random.seed(2)
        O1 = up(0)
        B = 0.1 * e12 + 0.2*e13 + 0.1 *e23 + 1*(3*e1 -1*e2 + 2*e3)*ninf
        R = ga_exp(B)
        N = 10

        lines = createRandomLines(N, scale = 2)
        for i in range(len(lines)):
            lines[i] = Sandwich(lines[i], Translator(e3*3))

        sigma_R_model = 0.0001
        sigma_T_model = 0.0001
        lines_perturbed = [perturbeObject(line, sigma_T_model, sigma_R_model) for line in lines] #Model noise

        lines_img_d_real   = [projectLineToPlane(line, R) for line in lines]                     #Real lines

        sigma_R_image = 0.0001
        sigma_T_image = 0.0001        
        lines_img_d        = [perturbeObjectInplane(projectLineToPlane(line, R), sigma_R_image, sigma_T_image) for line in lines]

        #using our noisy model and the noisy image of them to estimate R
        R_min, Nint = minimizeError((lines_perturbed, lines_img_d), BivectorLineImageMapping, x0 = None) 

        if verbose:
            print("R:   ", R)
            print("R_min", R_min)

        assert(MVEqual(R_min, R, rtol = 1e-2, atol = 1e-2,  verbose = False))    #Hard coded values. 
        #Weird condition. But we hope to find a "better" solution than the true one for the data we see, but worse than the true projection
        assert(sumImageFunction(R, lines, lines_img_d) > sumImageFunction(R_min, lines, lines_img_d) > sumImageFunction(R, lines, lines_img_d_real)) 


    @unittest.skip("Slow")
    def testNoisyRotationExtendedExtraction(self):
        verbose = True

        np.random.seed(2)
        O1 = up(0)
        B = 0.1 * e12 + 0.2*e13 + 0.1 *e23 + 1*(3*e1 -1*e2 + 2*e3)*ninf + 0.1 * E0
        R = np.exp(B)
        N = 100

        lines = createRandomLines(N, scale = 2)
        for i in range(len(lines)):
            lines[i] = Sandwich(lines[i], Translator(e3*3))

        sigma_R_model = 0.001
        sigma_T_model = 0.001
        lines_perturbed = [perturbeObject(line, sigma_T_model, sigma_R_model) for line in lines] #Model noise

        lines_img_d_real   = [projectLineToPlane(line, R) for line in lines]                     #Real lines

        sigma_R_image = 0.001
        sigma_T_image = 0.001        
        lines_img_d        = [perturbeObjectInplane(projectLineToPlane(line, R), sigma_R_image, sigma_T_image) for line in lines]

        #using our noisy model and the noisy image of them to estimate R
        R_min, Nint = minimizeError((lines_perturbed, lines_img_d), ExtendedBivectorLineImageMapping , x0 = None) 

        if verbose:
            print("R:   ", R)
            print("R_min", R_min)

        assert(MVEqual(R_min, R, rtol = 1e-2, atol = 1e-2,  verbose = False))    #Hard coded values. 
        #Weird condition. But we hope to find a "better" solution than the true one for the data we see, but worse than the true projection
        assert(sumImageFunction(R, lines, lines_img_d) > sumImageFunction(R_min, lines, lines_img_d) > sumImageFunction(R, lines, lines_img_d_real)) 


    @unittest.skip("Slow")
    def testExtremeRotationExtraction(self):
        verbose = True

        np.random.seed(2)
        O1 = up(0)

        rot_scale       = 10
        tran_scale      = 10
        spread_scale    = 10

        B = 0.1 * e12 + 0.2*e13 + 0.1 *e23 + rot_scale*(3*e1 -1*e2 + 2*e3)*ninf
        R = ga_exp(B)
        N = 10

        lines = createRandomLines(N, scale = spread_scale)
        for i in range(len(lines)):
            lines[i] = Sandwich(lines[i], Translator(e3*tran_scale))

        sigma_R_model = 0.001
        sigma_T_model = 0.001
        lines_perturbed = [perturbeObject(line, sigma_T_model, sigma_R_model) for line in lines] #Model noise

        lines_img_d_real   = [projectLineToPlane(line, R) for line in lines]                     #Real lines

        sigma_R_image = 0.001
        sigma_T_image = 0.001      
        lines_img_d        = [perturbeObjectInplane(projectLineToPlane(line, R), sigma_R_image, sigma_T_image) for line in lines]

        mapping = BivectorLineImageMapping

        x0      = mapping.inverserotorconversion(R)
        x0[:3] += np.array([0.1, 0.9, -0.17]) 
        R_start = mapping.rotorconversion(x0)

        #using our noisy model and the noisy image of them to estimate R
        R_min, Nint = minimizeError((lines_perturbed, lines_img_d), mapping, x0 = x0) 

        if verbose:
            print("R:     ", R)
            print("R_min  ", R_min/np.sign(R_min[0]))
            print("R_start", R_start/np.sign(R_start[0]))

            print("")

            print("B:          ", B)
            print("B_min   - B ", ga_log(R_min/np.sign(R_min[0])) - B)
            print("B_start - B ", ga_log(R_start/np.sign(R_start[0])) - B)


        #assert(MVEqual(R_min, R, rtol = 1e-2, atol = 1e-2,  verbose = False))    #Hard coded values. 
        #Weird condition. But we hope to find a "better" solution than the true one for the data we see, but worse than the true projection
        #assert(sumImageFunction(R, lines, lines_img_d) > sumImageFunction(R_min, lines, lines_img_d) > sumImageFunction(R, lines, lines_img_d_real)) 



    def setUpMultiView(self, N_lines, K_imgs, sigma_R_image = 0.001, sigma_T_image = 0.001):
        
        #Define random rotations for our cameras
        R_list = [ga_exp(createRandomBivector()) for _ in range(K_imgs)]            

        #Create N lines
        lines = createRandomLines(N_lines, scale = 2)
        for i in range(len(lines)):
            lines[i] = Sandwich(lines[i], Translator(e3*4))

        
        #Create our noise free images for comparison
        lines_img_base_d_real   =  [projectLineToPlane(line, one) for line in lines]        #Real base lines
        lines_imgs_d_real       = [[projectLineToPlane(line, R_list[i]) for line in lines] for i in range(K_imgs)]
  


        #Create noisy images
        lines_img_base_d     =  [perturbeObjectInplane(projectLineToPlane(line, one)      , sigma_R_image, sigma_T_image) for line in lines]    
        lines_imgs_d         = [[perturbeObjectInplane(projectLineToPlane(line, R_list[i]), sigma_R_image, sigma_T_image) for line in lines] for i in range(K_imgs)]  

        i = 0
        #for j in range(len(lines_imgs_d[i])):
        #    print((R_list[i]*(-no ^ lines_imgs_d_real[i][j])* ~R_list[i]).normal())
        #    print(((R_list[i]*(-no)* ~R_list[i])^lines[j]).normal())
        #    print("")


        return R_list, lines, lines_img_base_d, lines_img_base_d_real , lines_imgs_d, lines_imgs_d_real

    @unittest.skip("Slow")
    def testMultiViewEstimation(self):
        self.multiViewEstimation(MultiViewLineImageMapping, 10, 2, scale = 0.1)

    @unittest.skip("Slow")
    def testThreeViewAllPairsEstimation(self):
        K = 2
        self.multiViewEstimation(ThreeViewLineImageMapping, 20, 2, scale = 0.01)


    def multiViewEstimation(self, mapping, N, K, scale = None):
        np.random.seed(1)
        
        R_list, lines, lines_img_base_d, lines_img_base_d_real , lines_imgs_d, lines_imgs_d_real = self.setUpMultiView(N, K, sigma_R_image = 0.01, sigma_T_image = 0.01)

        if scale:
            dx = np.random.normal(size=6*K, scale=scale)                              #Starting estimate is slightly off            
            x0 = mapping.inverserotorconversion(R_list) + dx      #Add the noise
            R_list_start = mapping.rotorconversion(x0)            #Recover rotor
        
        else:
            x0 = None
        
        print("No noise cost =", mapping.costfunction(R_list,        lines_img_base_d_real, lines_imgs_d_real,  verbose = False))
        print("Target cost   =", mapping.costfunction(R_list,        lines_img_base_d,      lines_imgs_d,       verbose = False))        
        print("Start cost    =", mapping.costfunction(R_list_start,  lines_img_base_d,      lines_imgs_d,       verbose = False))        


        #Computation
        args = (lines_img_base_d, lines_imgs_d)
        R_list_min, Nint = minimizeError(args, mapping, x0 = x0)

        #Output
        print("Nint = ", Nint)

        for i in range(K):
            print("")
            print("R_real : ", R_list[i])
            print("R_min  : ", R_list_min[i])
            print("R_start: ", R_list_start[i])


        print("")
        print("Start cost    =", mapping.costfunction(R_list_start,  lines_img_base_d,      lines_imgs_d, verbose = False))        
        print("Final cost    =", mapping.costfunction(R_list_min,    lines_img_base_d,      lines_imgs_d, verbose = False))        
        print("Target cost   =", mapping.costfunction(R_list,        lines_img_base_d,      lines_imgs_d, verbose = False))        
        print("No noise cost =", mapping.costfunction(R_list,        lines_img_base_d_real, lines_imgs_d_real, verbose = False))

    @unittest.skip("Incorrect")
    def testSLAM(self):
        np.random.seed(1)

        N = 10
        K = 4

        sigma_R_image = 0.0001
        sigma_T_image = 0.0001   
        R_list, lines, lines_img_base_d, lines_img_base_d_real , lines_imgs_d, lines_imgs_d_real = self.setUpMultiView(N, K, 
                                                                                                    sigma_R_image = sigma_R_image, 
                                                                                                    sigma_T_image = sigma_T_image)

        sigma_R_model = 0.005
        sigma_T_model = 0.003
        lines_model = [perturbeObject(line, sigma_T_model, sigma_R_model) for line in lines] #Model noise


        slam = SLAM(lines_model, lines_img_base_d, lines_imgs_d[:3], R_start = R_list[:3])

        #print(R_list[0])
        print("cost: ", slam.cost())
        slam.updateLocation()

        print(slam.R_estimate[0])
        print("Current cost: ", slam.cost(), "\n")

        for i in range(3):
            print("R_min    ", slam.R_estimate[i])
            print("R_real   ", R_list[i])
                
            #print("model_estimate", slam.model_estimate[i])
            #print("Real line     ", lines[i])

        slam.updateModel()
        print("Current cost: ", slam.cost(), "\n")
        
        for i in range(3):
            print("model_estimate", slam.model_estimate[i])
            print("Real line     ", lines[i])

        slam.addImage(lines_imgs_d[3])
        slam.updateLocation()
        print("Current cost: ", slam.cost(), "\n")

        for i in range(4):
            print("R_min    ", slam.R_estimate[i])
            print("R_real   ", R_list[i])
                

        slam.updateModel()
        print("Current cost: ", slam.cost(), "\n")
    
        for i in range(4):
            print("model_estimate", slam.model_estimate[i])
            print("Real line     ", lines[i])


      
    @unittest.skip("Incorrect")
    def testThreeViewAllPairsEstimation(self):
        np.random.seed(3)
        N = 5
        #Defining the external parameters
        B_A = (0.1 * e12 + 0.2*e13 + 0.1 *e23 + (3*e1 -1*e2 + 2*e3)*ninf)
        R_A = ga_exp(B_A)

        B_B = (-0.2 * e12 + -0.1*e13- 0.05 *e23 + (1*e1 +2*e2 - 3*e3)*ninf)
        R_B = ga_exp(B_B)

        dx = np.random.normal(size=12, scale=0.01)                    #Starting estimate is slightly off            
        x0 = MultiViewLineImageMapping.inverserotorconversion([R_A, R_B]) + dx
        R_A_start, R_B_start = MultiViewLineImageMapping.rotorconversion(x0)


        lines = createRandomLines(N, scale = 2)
        for i in range(len(lines)):
            lines[i] = Sandwich(lines[i], Translator(e3*10))

        lines_img_d_base_real   = [projectLineToPlane(line, one) for line in lines]        #Real lines A
        lines_img_d_A_real      = [projectLineToPlane(line, R_A) for line in lines]        #Real lines A
        lines_img_d_B_real      = [projectLineToPlane(line, R_B) for line in lines]        #Real lines A

        sigma_R_image = 0.00001
        sigma_T_image = 0.00001

        lines_img_base_d     = [perturbeObjectInplane(projectLineToPlane(line, one), sigma_R_image, sigma_T_image) for line in lines]      
        lines_img_A_d        = [perturbeObjectInplane(projectLineToPlane(line, R_A), sigma_R_image, sigma_T_image) for line in lines]
        lines_img_B_d        = [perturbeObjectInplane(projectLineToPlane(line, R_B), sigma_R_image, sigma_T_image) for line in lines]


        #print("No noise cost =", MultiViewLineImageMapping.costfunction(R_A,       R_B,        lines_img_d_base_real, lines_img_d_A_real, lines_img_d_B_real))


        #Computation
        lines_imgs_d = [lines_img_A_d, lines_img_B_d]
        args = (lines_img_base_d, )
        R_list, Nint = minimizeError(args ,MultiViewLineImageMapping, x0 = x0)

        R_A_min, R_B_min = R_list

        #Output
        print("Nint = ", Nint)

        print("")
        print("R_A_real : ", R_A)
        print("R_A_min  : ", R_A_min)
        print("R_A_start: ", R_A_start)

        print("")
        print("R_B_real:  ", R_B)
        print("R_B_min :  ", R_B_min)
        print("R_B_start: ", R_B_start)

        print("")
        print("Start cost    =", MultiViewLineImageMapping.costfunction(R_A_start, R_B_start,  lines_img_base_d,      lines_img_A_d, lines_img_B_d))        
        print("Final cost    =", MultiViewLineImageMapping.costfunction(R_A_min,   R_B_min,    lines_img_base_d,      lines_img_A_d, lines_img_B_d))        
        print("Target cost   =", MultiViewLineImageMapping.costfunction(R_A,       R_B,        lines_img_base_d,      lines_img_A_d, lines_img_B_d))        
        print("No noise cost =", MultiViewLineImageMapping.costfunction(R_A,       R_B,        lines_img_d_base_real, lines_img_d_A_real, lines_img_d_B_real))
    

    @unittest.skip("Slow")
    def testPlotProjections(self):
        np.random.seed(2)
        #A, B = createRandomPoints(2, 100) #Real points
        #L = createLine(A, B)              #Real line

        O1 = up(0)
        F1 = up(e3)             #Image origin   
        Q1 = up(e3 + e2)        #Defines image rotation  

        #O1 = up(3*e1 + 4*e2)
        #cPlane1 = createRandomPlane(2)

        B = 0.1 * e12 + 0.2*e13 + 0.1 *e23 + 1*(3*e1 -1*e2 + 2*e3)*ninf
        #x0 = np.array([0.54, 0.85, 0.29, 1*3.1, -1.4 * 1, 1*1.89]) #Close to the real answer
        N  = 10

        R = ga_exp(B)

        O2 = R * O1 * ~R   #O2
        F2 = R * F1 * ~R
        Q2 = R * Q1 * ~R

        cPlane1 = (ninf + e3)*I5   #Camera plane 1
        cPlane2 = R * cPlane1 * ~R


        lines = createRandomLines(N, scale = 2)
        for i in range(len(lines)):
            lines[i] = Sandwich(lines[i], Translator(e3*3))

        sigma_R_model = 0.01
        sigma_T_model = 0.05
        lines_perturbed = [perturbeObject(line, sigma_T_model, sigma_R_model) for line in lines] #Model noise

        lines_img_d_real   = [projectLineToPlane(line, R) for line in lines]        #Real lines

        sigma_R_image = 0.0001
        sigma_T_image = 0.0001        
        lines_img_d        = [perturbeObjectInplane(projectLineToPlane(line, R), sigma_R_image, sigma_T_image) for line in lines]

        print("")
        print("Inital cost", sumImageFunction(R, lines_perturbed, lines_img_d))
        print("R_real: ", R)
        R_min, Nint = minimizeError((lines_perturbed, lines_img_d), BivectorLineImageMapping, x0 = None)
        print("R_min:  ", R_min)
        print("Nint = ", Nint)
        print("Final cost= ", sumImageFunction(R_min, lines, lines_img_d))

        lines_img_d_min   = [projectLineToPlane(line, R_min) for line in lines]
        lines_img_d_model = [projectLineToPlane(line, R_min) for line in lines_perturbed]


        #Printing
        color_print = ['m', 'y', 'k']
        N_print = len(color_print)


        plot_img = Plot2D()

        for i in range(N_print):
            Limg = lines_img_d[i]
            Limg_min = lines_img_d_min[i]
            Limg_real = lines_img_d_real[i]
            Limg_model = lines_img_d_model[i]
            plot_img.plotLine2D(Limg_min, color = 'g')              #Green: estimate of the real line  (hidden)
            plot_img.plotLine2D(Limg_model, color = 'c')            #Cyan:  estimate of model line
            plot_img.plotLine2D(Limg, color = 'b')                  #Blue:  image (with image noise)
            plot_img.plotLine2D(Limg_real, color = color_print[i])  #Other: real line                      (hidden)



        plot = Plot3D()

        plot.configure(5)
        plot.addPoint(O1, color='r')
        plot.addPoint(O2, color='b')

        #plot.addPoint(F1, color='r')
        plot.addPoint(F2, color='b')

        #plot.addPoint(Q1, color='r')
        plot.addPoint(Q2, color='b')


        for i in range(N_print):
            L = lines[i]
            L_img = R*lines_img_d_real[i]*~R
            L_perturbed = lines_perturbed[i]

            plot.addLine(L_perturbed, color = 'c')
            plot.addLine(L_img, color = color_print[i])
            plot.addLine(L, color = color_print[i])

        #plot.addPlane(cPlane1, center = F1, color='r')
        plot.addPlane(cPlane2, center = F2, color='b')
        
        plot_img.show(block = False)
        plot.show(block = False)


def benchmarkImageCostFunction():
    np.random.seed(123)
    B = createRandomBivector()
    R_real = ga_exp(B)
    N = 10

    lines = createRandomLines(N, scale = 2)
    for i in range(len(lines)):
        lines[i] = Sandwich(lines[i], Translator(e3*3))

    sigma_R_model = 0.01
    sigma_T_model = 0.01
    lines_perturbed = [perturbeObject(line, sigma_T_model, sigma_R_model) for line in lines] #Model noise

    lines_img_d_real   = [projectLineToPlane(line, R_real) for line in lines]                     #Real lines

    sigma_R_image = 0.01
    sigma_T_image = 0.01        
    lines_img_d        = [perturbeObjectInplane(projectLineToPlane(line, R_real), sigma_R_image, sigma_T_image) for line in lines]

    traininglinesets = (lines_perturbed, lines_img_d)

    benchmarkMinimizeError(R_real, traininglinesets, traininglinesets, fileout = None, mapping = BivectorLineImageMapping)



def plotCostFunctionEffect():
    print("\nRunning plotCostFunctionEffect")
    print("")
    np.random.seed(1)
    #Test extreme values

    np.random.seed(2)
    O1 = up(0)

    rot_scale       = 1
    tran_scale      = 10
    spread_scale    = 10

    B = 0.1 * e12 + 0.2*e13 - 0.1 *e23 + rot_scale*(3*e1 -1*e2 + 2*e3)*ninf
    R = ga_exp(B)
    N = 10

    lines = createRandomLines(N, scale = spread_scale)
    for i in range(len(lines)):
        lines[i] = Sandwich(lines[i], Translator(e3*tran_scale))

    sigma_R_model = 0.01
    sigma_T_model = 0.1
    lines_perturbed    = [perturbeObject(line, sigma_T_model, sigma_R_model) for line in lines] #Model noise

    lines_img_d_real   = [projectLineToPlane(line, R) for line in lines]                     #Real lines

    sigma_R_image = 0.002
    sigma_T_image = 0.01      
    lines_img_d        = [perturbeObjectInplane(projectLineToPlane(line, R), sigma_R_image, sigma_T_image) for line in lines]

    mapping = BivectorLineImageMapping

    x0       = mapping.inverserotorconversion(R) 
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
            ans[i, j]   = np.log(mapping.costfunction(mapping.rotorconversion(x0), lines_perturbed, lines_img_d, O1))

    xv, yv = np.meshgrid(translation, rotation)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.set_xlabel("Translation error")
    ax.set_ylabel("Rotation error")
    ax.set_zlabel("log(objective function)")

    ax.plot_wireframe(xv, yv, ans)
    plt.show()        



if __name__ == "__main__":
    #plotCostFunctionEffect()
    unittest.main()
    #benchmarkImageCostFunction()