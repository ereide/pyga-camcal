
import numpy as np

from pygacal.common.cgatools import *
from pygacal.geometry.lines import createLine

class Model():
    def __init__(self):
        self.set_lines()
        self.set_lines_ga()

    def set_lines(self):
        raise NotImplementedError()

    def set_lines_ga(self):
        self.lines_ga = {}
        for name, line in self.lines.items():
            a = VectoMV(line[0])
            A = up(a)
            b = VectoMV(line[1])
            B = up(b)
            
            line_GA = createLine(A, B)
            self.lines_ga[name] = line_GA
        return self.lines_ga

    def get_lines(self):
        return self.lines
    
    def get_lines_ga(self):
        return self.lines_ga


class ModelImage():
    def __init__(self):
        #Implement these:
        self.K = None
        self.distortion_coeff = None
        self.lines = None
        self.lines_img_ga = None
        self.img_name = None
        raise NotImplementedError()

    def get_image_name(self):
        return self.img_name

    def get_internal_calibration(self):
        return self.K

    def get_distortion_coeff(self):
        return self.distortion_coeff
    
    def get_lines(self):
        if(self.lines):
            return self.lines
        else:
            self.set_lines()
            return self.lines

    def get_lines_img_ga(self):
        if(self.lines_img_ga):
            return self.lines_img_ga
        else:
            self.get_lines()
            self.set_lines_img_ga()
            return self.lines_img_ga

    def set_lines_img_ga(self, focus = 1):
        K_inv = np.linalg.inv(self.K)
        self.lines_img_ga = {}
        for name, line in self.lines.items():        
            start = np.array(np.dot(K_inv, line[0])).reshape(-1)
            end   = np.array(np.dot(K_inv, line[1])).reshape(-1)
            
            start = start/start[2]
            end   = end/end[2]
            
            a = start[0] * e1 + start[1] *e2
            b = end[0] * e1   + end[1] * e2

            A, B = up(a), up(b)
            L_img = createLine(A, B)
                
            L_img_ga = Sandwich(createLine(A, B), Translator(focus * e3))

            self.lines_img_ga[name] = L_img_ga

        return self.lines_img_ga

    def set_lines(self):
        raise NotImplementedError()

