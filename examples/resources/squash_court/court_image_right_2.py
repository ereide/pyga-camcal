
import os
import numpy as np
from interface.camera_calibration import ModelImage


dir_path = os.path.dirname(os.path.realpath(__file__))

class RobertSquashCourtImage(ModelImage):
    def __init__(self):
        self.K = np.matrix([[540.24969482,   0. ,        312.17905922],
                            [  0.   ,      540.30438232, 230.17860315],
                            [  0.  ,         0.     ,      1.        ]])

        #Currently not used
        self.distortion_coeff = None
        self.lines = None
        self.lines_img_ga = None
        self.img_name = os.path.join(dir_path, "squash_robert_right_2.jpg")

                
        self.R = np.matrix([[ 0.97695816,  0.03712448, -0.21017739],
                            [ 0.09340124,  -0.9598201,  0.26461592],
                            [-0.19190876, -0.27814951, -0.94117155]]) 

        self.T = np.matrix([194.64507859, 1126.70876979, 11039.97173116])*0.001

    def set_lines(self):
        left_wall_down  = np.array([169, 271, 1])
        right_wall_down = np.array([479, 295, 1])
        
        left_wall_back_down  = np.array([0, 418, 1])
        right_wall_back_down =  np.array([541, 479, 1])
        
        left_wall_back_up    = np.array([0, 122, 1])
        right_wall_back_up   = np.array([639, 225, 1])
        
        front_wall_top_line_left  = np.array([155, 43, 1])
        front_wall_top_line_right = np.array([508, 44.6, 1])
        
        front_wall_middle_line_left  = np.array([162, 189.4, 1])
        front_wall_middle_line_right = np.array([490, 205, 1])
        
        front_wall_bot_line_left  = np.array([168.4, 252, 1])
        front_wall_bot_line_right = np.array([482, 274, 1])
        
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