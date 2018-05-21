
import os
import numpy as np
from interface.camera_calibration import ModelImage


dir_path = os.path.dirname(os.path.realpath(__file__))

class RobertSquashCourtImage(ModelImage):
    def __init__(self):
        self.K = np.matrix([[-524.79644775,   0.,         293.28320922],
                            [  0.,         -523.37878418, 226.37976338],
                            [  0.,           0.,           1.        ]])

        #Currently not used
        self.distortion_coeff = None
        self.lines = None
        self.lines_img_ga = None
        self.img_name = os.path.join(dir_path, "squash_robert_left.jpg")

    def set_lines(self):
        left_wall_down = np.array([162, 309, 1])
        right_wall_down = np.array([473, 294.5, 1])
        
        
        left_wall_back_down = np.array([108, 500, 1])
        right_wall_back_down =  np.array([640, 434, 1])
        
        left_wall_back_up    = np.array([0, 273, 1])
        right_wall_back_up   = np.array([640, 136, 1])
        
        front_wall_top_line_left = np.array([145, 66, 1])
        front_wall_top_line_right = np.array([492, 66, 1])
        
        front_wall_middle_line_left = np.array([155.5, 220, 1])
        front_wall_middle_line_right = np.array([480, 210, 1])
        
        front_wall_bot_line_left = np.array([161, 286, 1])
        front_wall_bot_line_right = np.array([475.5, 273, 1])
        
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