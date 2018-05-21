
import numpy as np    

from interface.camera_calibration import Model

def yard2meter(yard):
    return 0.9144 * yard /10.

def feet2yard(feet):
    return feet / 3.0

class PenaltyAreaFootballModel(Model):
    """
    Origin is located at penalty point
    
    x = left
    y = up
    z = towards goal
    
    Homogenous coordinates
    
    Dimensions = m
    
    data: https://en.wikipedia.org/wiki/Football_pitch

    """

    def set_lines(self):
        backline = yard2meter(12) #Distance to the goal

        goal_height = yard2meter(feet2yard(8))
        goal_width  = yard2meter(8)

        goal_depth  = yard2meter(2) #unknown!!! ???

        keeper_box_width = yard2meter(2 * 6 + 8)
        keeper_box_depth = yard2meter(6)
        
        penalty_area_width = keeper_box_width + yard2meter(2 * 12)
        penalty_area_depth = yard2meter(18)
        
        goal_bot_left_corner  = np.array([ goal_width/2, 0, backline, 1])
        goal_bot_right_corner = np.array([-goal_width/2, 0, backline, 1])
        
        goal_top_left_corner  = np.array([ goal_width/2, goal_height, backline, 1])
        goal_top_right_corner = np.array([-goal_width/2, goal_height, backline, 1])
        
        goal_left_back_corner   =  np.array([ goal_width/2, 0, backline + goal_depth, 1])
        goal_right_back_corner  =  np.array([-goal_width/2, 0, backline + goal_depth, 1])

        keeper_box_back_left  = np.array([ keeper_box_width/2, 0, backline, 1])     
        keeper_box_back_right = np.array([-keeper_box_width/2, 0, backline, 1])     
        
        keeper_box_front_left    = np.array([ keeper_box_width/2, 0, backline - keeper_box_depth, 1])     
        keeper_box_front_right   = np.array([-keeper_box_width/2, 0, backline - keeper_box_depth, 1])   
        
        penalty_area_back_left   = np.array([ penalty_area_width/2, 0, backline, 1]) 
        penalty_area_back_right  = np.array([-penalty_area_width/2, 0, backline, 1]) 
        
        penalty_area_front_left  = np.array([ penalty_area_width/2, 0, backline - penalty_area_depth, 1]) 
        penalty_area_front_right = np.array([-penalty_area_width/2, 0, backline - penalty_area_depth, 1]) 
        
        back_line = (penalty_area_back_left, penalty_area_back_right)
        
        goal_top_bar  = (goal_top_left_corner,  goal_top_right_corner)
        goal_left_bar = (goal_bot_left_corner,  goal_top_left_corner)
        goal_right_bar = (goal_bot_right_corner, goal_top_right_corner)
        goal_left_back_line = (goal_bot_left_corner, goal_left_back_corner)
        goal_right_back_line = (goal_bot_right_corner, goal_right_back_corner)


        keeper_box_left_line  = (keeper_box_back_left, keeper_box_front_left) 
        keeper_box_right_line = (keeper_box_back_right, keeper_box_front_right)
        keeper_box_front_line = (keeper_box_front_left, keeper_box_front_right)
        
        penalty_box_left_line = (penalty_area_back_left, penalty_area_front_left)
        penalty_box_right_line = (penalty_area_back_right, penalty_area_front_right)
        penalty_box_front_line = (penalty_area_front_left, penalty_area_front_right)

        
        goal_lines        = {"back_line":           back_line, 
                            "goal_top_bar":         goal_top_bar, 
                            "goal_left_bar" :       goal_left_bar, 
                            "goal_right_bar":       goal_right_bar, 
                            "goal_left_back_line":  goal_left_back_line, 
                            "goal_right_back_line": goal_right_back_line} 
          
        keeper_box_lines  = {"keeper_box_left_line": keeper_box_left_line, 
                            "keeper_box_right_line": keeper_box_right_line, 
                            "keeper_box_front_line": keeper_box_front_line}

        penalty_box_lines = {"penalty_box_left_line": penalty_box_left_line, 
                            "penalty_box_right_line": penalty_box_right_line, 
                            "penalty_box_front_line": penalty_box_front_line}
        
        self.lines = {**goal_lines, **keeper_box_lines, **penalty_box_lines}
    
    
        return self.lines


