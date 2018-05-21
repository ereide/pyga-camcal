
import os
import numpy as np
from interface.camera_calibration import ModelImage


dir_path = os.path.dirname(os.path.realpath(__file__))

class RobertSquashCourtImage(ModelImage):
    def __init__(self):
        self.K = np.matrix([[-1060.9659554663276, 0.0,                674.658928371776],
                            [0.0,                -1062.1279134612116, 351.3765083534052],
                            [0.0,                0.0,                1.0]])


        #Currently not used
        self.distortion_coeff = [0.07427872235876719, -0.1131201164597351, 0.007473077278327413, -0.017172865150964434, -0.09811870016822016]
        self.lines = None
        self.lines_img_ga = None
        self.img_name = os.path.join(dir_path, "squash_robert.jpg")

    def set_lines(self):
        left_wall_down = np.array([341, 507, 1])
        right_wall_down = np.array([960, 475, 1])
        
        
        left_wall_back_down = np.array([271, 695, 1])
        right_wall_back_down = np.array([1258, 702, 1])
        
        left_wall_back_up    = np.array([22, 330, 1])
        right_wall_back_up   = np.array([1266, 122, 1])
        
        front_wall_top_line_left = np.array([303, 5, 1])
        front_wall_top_line_right = np.array([1000, 17, 1])
        
        front_wall_middle_line_left = np.array([327, 324, 1])
        front_wall_middle_line_right = np.array([971, 304, 1])
        
        front_wall_bot_line_left = np.array([337, 460, 1])
        front_wall_bot_line_right = np.array([964, 433, 1])
        
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