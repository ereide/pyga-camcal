import cv2

from matplotlib import pyplot as plt
import numpy as np
import time
import os

#   from pygacal import geometry, camera, rotation

from pygacal.geometry.transformations import *

from pygacal.rotation import minimizeError
from pygacal.rotation.mapping import BivectorLineImageMapping
#from clifford_tools.numerical.applications.vision.core import Camera 

from pygacal.camera.projection import projectLineToPlane
from pygacal.geometry.lines import extract2Dparams, createLine

from pygacal.common.cgatools import *

from os.path import join, abspath


from interface.camera_calibration import Model


def align_objects(old_model, new_model):
    #Flips all the lines that have the wrong orientation
    dict_img_old_ga = old_model.get_lines_ga()
    dict_img_new_ga = new_model.get_lines_ga() 

    for name in dict_img_old_ga.keys():
        if name in dict_img_new_ga:
            if float((dict_img_new_ga[name] * dict_img_old_ga[name])[0]) < 0:
                dict_img_new_ga[name] = -dict_img_new_ga[name]

    new_model.lines_ga = dict_img_new_ga
    return new_model

def calibrate(model, image_model):
    lines_ga        = []
    lines_img_ga    = []

    dict_lines_ga = model.get_lines_ga()

    for name, line in image_model.get_lines_img_ga().items():
        lines_img_ga.append(line)
        lines_ga.append(dict_lines_ga[name])


    t0 = time.clock()

    R_min, Nint = minimizeError((lines_ga, lines_img_ga), 
                                BivectorLineImageMapping, 
                                x0 = None)
    print("t = ", time.clock()- t0, "N_iter = ", Nint)

    return R_min


def create_model(image_model_left, R_left, image_model_right, R_right):
    
    lines_model_ga = {}

    C = up(0)

    dict_img_left_ga = image_model_left.get_lines_img_ga()
    dict_img_right_ga = image_model_right.get_lines_img_ga()

    for name in dict_img_left_ga.keys():
        if name != "front_out_line": #Due to an error, 

            line_left  = dict_img_left_ga[name]
            line_right = dict_img_right_ga[name]

            P_left  = R_left  * (C  ^ line_left) * ~R_left
            P_right = R_right * (C ^ line_right) * ~R_right

            model_line = Meet(P_left, P_right)

            lines_model_ga[name] = model_line

            

    class createdModel(Model):
        def __init__(self, **kwargs):
            self.lines_ga = lines_model_ga
            
    return createdModel()




def project_lines_matrix(model, image_model, P):
    def image_projection(x, R_mat, K):
        X = np.array(np.dot(K, np.dot(R_mat, x)[:3])).reshape(-1)
        return X[:2]/X[2]

    lines_ga        = []
    lines_img_ga    = []

    dict_lines_ga = model.get_lines_ga()

    for name, line in image_model.get_lines_img_ga().items():
        lines_img_ga.append(line)
        lines_ga.append(dict_lines_ga[name])



    K = image_model.get_internal_calibration()

    image_name = image_model.get_image_name()
    img        = cv2.imread(image_name, cv2.IMREAD_COLOR)


    for line in lines_ga:

        start = np.int64(image_projection(line[0], P, K))
        end = np.int64(image_projection(line[1], P, K))
        cv2.line(img, tuple(start), tuple(end), (255, 0, 0), 1)
    
    for line in lines_img_ga:
        start = line[0][:2]
        end = line[1][:2]
        cv2.line(img, tuple(start), tuple(end), (0, 255, 0), 1)

    cv2.imshow("Projection", img)
    cv2.waitKey(0)


def project_lines(model, image_model, R):
    
    lines_ga        = []
    lines_img_ga    = []

    dict_lines_ga = model.get_lines_ga()

    for name, line in image_model.get_lines_img_ga().items():
        if name in dict_lines_ga:
            lines_img_ga.append(line)
            lines_ga.append(dict_lines_ga[name])

    K = image_model.get_internal_calibration()

    image_name = image_model.get_image_name()
    img        = cv2.imread(image_name, cv2.IMREAD_COLOR)


    def to_homogenous_coords(vec2):
        ans = np.ones(3)
        ans[:2] = vec2
        return ans

    for line in lines_img_ga:
        size = 1 #Some scaling parameter 

                
        a, ma = extract2Dparams(line)
        
        A = a + size*ma
        B = a - size*ma
                        
        #Why is numpy so broken!! 
        A = np.reshape(np.array(np.dot(K, to_homogenous_coords(A))), (3, ))
        B = np.reshape(np.array(np.dot(K, to_homogenous_coords(B))), (3, ))

        start = tuple(np.int32(A[:2]))
        end   = tuple(np.int32(B[:2]))

        #Blue from image
        cv2.line(img, start, end, (255, 0, 0), 2)

    for line in lines_ga:
        size = 1 #Some scaling parameter 

        line_img_projected = projectLineToPlane(line, R)
                
        a, ma = extract2Dparams(line_img_projected)
        
        A = a + size*ma
        B = a - size*ma
                
        
        #Why is numpy so broken!! 
        A = np.reshape(np.array(np.dot(K, to_homogenous_coords(A))), (3, ))
        B = np.reshape(np.array(np.dot(K, to_homogenous_coords(B))), (3, ))

        start = tuple(np.int32(A[:2]))
        end   = tuple(np.int32(B[:2]))

        #Green from model
        cv2.line(img, start, end, (0, 255, 0), 2)



        
    cv2.imshow("Projection", img)
    cv2.waitKey(0)


def compare_model_lines(model_1, model_2, image_model, R):
    
    lines_ga_1        = []
    lines_ga_2        = []
    lines_img_ga    = []

    dict_lines_ga_1 = model_1.get_lines_ga()
    dict_lines_ga_2 = model_2.get_lines_ga()

    for name, line in image_model.get_lines_img_ga().items():
        if name in dict_lines_ga_1 and name in dict_lines_ga_2:
            lines_img_ga.append(line)
            lines_ga_1.append(dict_lines_ga_1[name])
            lines_ga_2.append(dict_lines_ga_2[name])

    K = image_model.get_internal_calibration()

    image_name = image_model.get_image_name()
    img        = cv2.imread(image_name, cv2.IMREAD_COLOR)


    def to_homogenous_coords(vec2):
        ans = np.ones(3)
        ans[:2] = vec2
        return ans

    for line in lines_img_ga:
        size = 1 #Some scaling parameter 

                
        a, ma = extract2Dparams(line)
        
        A = a + size*ma
        B = a - size*ma
                        
        #Why is numpy so broken!! 
        A = np.reshape(np.array(np.dot(K, to_homogenous_coords(A))), (3, ))
        B = np.reshape(np.array(np.dot(K, to_homogenous_coords(B))), (3, ))

        start = tuple(np.int32(A[:2]))
        end   = tuple(np.int32(B[:2]))

        #Blue from image
        cv2.line(img, start, end, (255, 0, 0), 2)

    for line in lines_ga_1:
        size = 1 #Some scaling parameter 

        line_img_projected = projectLineToPlane(line, R)
                
        a, ma = extract2Dparams(line_img_projected)
        
        A = a + size*ma
        B = a - size*ma
                
        
        #Why is numpy so broken!! 
        A = np.reshape(np.array(np.dot(K, to_homogenous_coords(A))), (3, ))
        B = np.reshape(np.array(np.dot(K, to_homogenous_coords(B))), (3, ))

        start = tuple(np.int32(A[:2]))
        end   = tuple(np.int32(B[:2]))

        #Green from model
        cv2.line(img, start, end, (0, 255, 0), 2)

    for line in lines_ga_2:
        size = 1 #Some scaling parameter 

        line_img_projected = projectLineToPlane(line, R)
                
        a, ma = extract2Dparams(line_img_projected)
        
        A = a + size*ma
        B = a - size*ma
                
        
        #Why is numpy so broken!! 
        A = np.reshape(np.array(np.dot(K, to_homogenous_coords(A))), (3, ))
        B = np.reshape(np.array(np.dot(K, to_homogenous_coords(B))), (3, ))

        start = tuple(np.int32(A[:2]))
        end   = tuple(np.int32(B[:2]))

        #Green from model
        cv2.line(img, start, end, (255, 0, 255), 2)


        
    cv2.imshow("Projection", img)
    cv2.waitKey(0)


