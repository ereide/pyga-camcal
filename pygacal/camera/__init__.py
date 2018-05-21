

from clifford import g3c
import numpy as np
import scipy.optimize as opt
from pygacal.rotation.costfunction import restrictedImageCostFunction, restrictedMultiViewImageCostFunction
from pygacal.rotation import minimizeError
from pygacal.rotation.mapping import BivectorLineImageMapping, BivectorLineMapping, LinePropertyBivectorMapping, BivectorLineEstimationMapping

from pygacal.common.cgatools import Sandwich, Dilator, Translator, Reflector, inversion, Rotor, Transversor, I3, I5, VectorEquality, anticommuter, ga_exp, Meet


#Defining variables
layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])

class SLAM(object):
    def __init__(self, model_estimate, lines_img_base, lines_imgs, R_start = None, mapping = BivectorLineImageMapping):
        self.mapping = mapping
        self.model_estimate = model_estimate
        self.lines_img_base = lines_img_base
        self.lines_imgs = lines_imgs
        assert(len(lines_imgs[0]) == len(model_estimate))

        if R_start is None:
            self.R_estimate = [None for _ in range(len(lines_imgs))]
        else:
            assert(len(R_start) == len(lines_imgs))
            self.R_estimate = R_start

    def cost(self):
        cost = sum([self.mapping.costfunction(self.R_estimate[i], self.model_estimate, self.lines_imgs[i]) for i in range(len(self.lines_imgs))])
        return cost/len(self.lines_imgs)


    def updateLocation(self):
        print("Update Location")
        for i in range(len(self.lines_imgs)):
            args = (self.model_estimate, self.lines_imgs[i])
            if (self.R_estimate[i] is None):
                x0 = None
            else:
                x0 = self.mapping.inverserotorconversion(self.R_estimate[i])

            R_min, N_int = minimizeError(args, self.mapping, x0 = x0)
            self.R_estimate[i] = R_min
            print("N_int = ", N_int)
        print("Complete: Update location")


    def addImage(self, lines_img_new, R_img_new = None):
        self.lines_imgs.append(lines_img_new)
        self.R_estimate.append(R_img_new)
    
    def improveLine(self, i, O1 = up(0)):
        line_guesses = []
        
        R_B = self.R_estimate[ 0 ]
        Line_A = self.lines_img_base[i]
        Line_B = self.lines_imgs[0][i]

        P_A = (O1 ^ Line_A).normal()
        P_B = (R_B * (O1 ^ Line_B) * ~R_B).normal()

        new_line = Meet(P_A, P_B)
        line_guesses.append(new_line) 

        for j in range(1, len(self.R_estimate)):
            R_A = self.R_estimate[j-1]
            R_B = self.R_estimate[ j ]

            Line_A = self.lines_imgs[j-1][i]
            Line_B = self.lines_imgs[ j ][i]

            P_A = (R_A * (O1 ^ Line_A) * ~R_A).normal()
            P_B = (R_B * (O1 ^ Line_B) * ~R_B).normal()

            new_line = Meet(P_A, P_B)
            line_guesses.append(new_line) 
        
        for guess in line_guesses:
            print("guess  ", guess)
        print("model  ", self.model_estimate[i], "\n")
        
        return self.averageLines(self.model_estimate[i], line_guesses)



    def averageLines(self, line_start, line_guesses):
        mapping = BivectorLineEstimationMapping
        args = [line_start, line_guesses]
        x0 = np.random.normal(0.01, size=6)
        R_min, Nint = minimizeError(args, mapping, x0 = x0)
        return R_min * line_start * ~R_min

    def updateModel(self):

        if any(self.R_estimate) is None:
            self.updateLocation()
        
        print("Update Model ")

        for i in range(len(self.model_estimate)):
            self.model_estimate[i] = self.improveLine(i)
        
        print("Complete: model update")

