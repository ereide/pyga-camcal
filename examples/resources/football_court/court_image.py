
import os
import numpy as np
from interface.camera_calibration import ModelImage


dir_path = os.path.dirname(os.path.realpath(__file__))

class FootballManCityCourtImage2(ModelImage):
    def __init__(self):
        alpha = 13000
        self.K = np.matrix([[alpha, 0.0,        800],
                            [0.0,   alpha,      400],
                            [0.0,   0.0,        1.0]])


        #Currently not used
        self.distortion_coeff = None
        self.lines = None
        self.lines_img_ga = None
        self.img_name = os.path.join(dir_path, "rgb_goal_line_MCity_2.png")

    def set_lines(self):
       
        goal_bot_left_corner  = np.array([162 , 466, 1])
        goal_bot_right_corner = np.array([932, 436, 1])
        
        goal_top_left_corner  = np.array([698 , 111, 1])
        goal_top_right_corner = np.array([1442, 110, 1])
        
        goal_left_back_corner   =  np.array([1,   141, 1])
        goal_right_back_corner  =  np.array([721, 37,  1])


        #keeper_box_back_left  = np.array([ keeper_box_width/2, 0, backline, 1])     
        keeper_box_back_right = np.array([1421, 419, 1])     
        
        #keeper_box_front_left    = np.array([ keeper_box_width/2, 0, backline - keeper_box_depth, 1])     
        keeper_box_front_right   = np.array([1592, 732, 1])   
        
        penalty_area_back_left   = np.array([ 1, 465, 1]) 
        penalty_area_back_right  = np.array([1592, 408, 1]) 
        
        
        #penalty_area_front_left  = 
        #penalty_area_front_right = 
        
        back_line = (penalty_area_back_left, penalty_area_back_right)
        
        goal_top_bar  = (goal_top_left_corner,  goal_top_right_corner)
        goal_left_bar = (goal_bot_left_corner,  goal_top_left_corner)
        goal_right_bar = (goal_bot_right_corner, goal_top_right_corner)
        
        goal_left_back_line = (goal_bot_left_corner, goal_left_back_corner)
        goal_right_back_line = (goal_bot_right_corner, goal_right_back_corner)

        #keeper_box_left_line  = (keeper_box_back_left, keeper_box_front_left) 
        keeper_box_right_line = (keeper_box_back_right, keeper_box_front_right)
        #keeper_box_front_line = (keeper_box_front_left, keeper_box_front_right)
        
        #penalty_box_left_line = (penalty_area_back_left, penalty_area_front_left)
        #penalty_box_right_line = (penalty_area_back_right, penalty_area_front_right)
        #penalty_box_front_line = (penalty_area_front_left, penalty_area_front_right)

        
        
        goal_lines        = {"back_line":           back_line, 
                            "goal_top_bar":         goal_top_bar, 
                            "goal_left_bar" :       goal_left_bar, 
                            "goal_right_bar":       goal_right_bar, 
                            "goal_left_back_line":  goal_left_back_line, 
                            "goal_right_back_line": goal_right_back_line} 
          
        keeper_box_lines  = {"keeper_box_right_line": keeper_box_right_line}
                            

        penalty_box_lines = {}
        
        self.lines = {**goal_lines, **keeper_box_lines, **penalty_box_lines}
    
    
        return self.lines
