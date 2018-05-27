
import os
import numpy as np
from interface.camera_calibration import ModelImage


dir_path = os.path.dirname(os.path.realpath(__file__))

class RobertSquashCourtImage(ModelImage):
    def __init__(self):
        self.K = np.matrix([[524.79644775,   0. ,        293.28320922],
                            [  0. ,        523.37878418 ,226.37976338],
                            [  0.    ,       0.     ,      1.        ]])

        #Currently not used
        self.distortion_coeff = None
        self.lines = None
        self.lines_img_ga = None
        self.img_name = os.path.join(dir_path, "squash_robert_left_2.jpg")

        self.R = np.array([[ 0.98280887,  0.06995721,  0.17085878],
                           [ 0.0175361,  -0.95661948,  0.29081206],
                           [ 0.18379124, -0.28281647, -0.94140088]])

        self.T = np.array([-57.72239289,   1002.8903749,   10847.04675503])*0.001

        

    def set_lines(self):
        left_wall_down = np.array([130, 277, 1])
        right_wall_down = np.array([435, 273, 1])
        
        left_wall_back_down = np.array([67, 478, 1])
        right_wall_back_down =  np.array([639, 463, 1])
        
        left_wall_back_up    = np.array([0, 217, 1])
        right_wall_back_up   = np.array([634, 150, 1])
        
        front_wall_top_line_left = np.array([123.3, 23.6, 1])
        front_wall_top_line_right = np.array([471, 52, 1])
        
        front_wall_middle_line_left = np.array([126.5, 185, 1])
        front_wall_middle_line_right = np.array([447, 194, 1])
        
        front_wall_bot_line_left = np.array([128, 253, 1])
        front_wall_bot_line_right = np.array([439, 255, 1])
        
        # Define lines
        tint_line      = (front_wall_bot_line_left,    front_wall_bot_line_right)
        service_line   = (front_wall_middle_line_left, front_wall_middle_line_right)
        front_out_line = (front_wall_top_line_left,    front_wall_top_line_right)
        
        side_out_line_left  = (front_wall_top_line_left,  left_wall_back_up)
        side_out_line_right = (front_wall_top_line_right, right_wall_back_up)

        floor_line_front = (left_wall_down,  right_wall_down)
        
        floor_line_left  = (left_wall_down,  left_wall_back_down)
        floor_line_right = (right_wall_down, right_wall_back_down)

        vertical_line_left  = (left_wall_down,  front_wall_top_line_left)
        vertical_line_right = (right_wall_down, front_wall_top_line_right)
        
        court_lines = {"tint_line": tint_line, 
                        "service_line": service_line, 
                        "front_out_line": front_out_line, 
                        "side_out_line_left": side_out_line_left, 
                        "side_out_line_right": side_out_line_right}

        floor_lines = { "floor_line_front": floor_line_front, 
                        "floor_line_right": floor_line_right, 
                        "floor_line_left": floor_line_left}

        vertical_lines = {"vertical_line_left": vertical_line_left, 
                          "vertical_line_right": vertical_line_right}
        
        self.lines = {**court_lines, **floor_lines, **vertical_lines}
        return self.lines

    