
import numpy as np    

from interface.camera_calibration import Model


class SquashCourtModel(Model):
    """
        Squash court model based on official world squash court dimensions
        Origin of court is at centre of intersection of floor and front wall, positive X to the right, positive Y
        to the sky and positive Z to the back wall
        Dimensions in mm
        Lines are the average of the red court lines
        
        New coordinates: x left, z towards front, y up, origin centred 10 m from wall negative z direction
    """

    def set_lines(self):
        line_width = 0.050

        left = 3.200
        right = -3.200
        back = 10.000 - 9.750
        front = 10.000

        tint_height = 0.430 + line_width/2
        tint_left  = np.array([left,  tint_height, front, 1])
        tint_right = np.array([right, tint_height, front, 1])

        service_height = 1.780 + line_width/2
        service_left =  np.array([left,  service_height, front, 1])
        service_right = np.array([right, service_height, front, 1])

        front_wall_height = 4.570 + line_width/2
        front_wall_left  = np.array([left,  front_wall_height, front, 1])
        front_wall_right = np.array([right, front_wall_height, front, 1])

        back_height = 2.130 + line_width/2
        side_wall_left_back =  np.array([left,  back_height, back, 1])
        side_wall_right_back = np.array([right, back_height, back, 1])

        floor_left_front  = np.array([left,    0, front, 1])
        floor_right_front = np.array([right,   0, front, 1])
        floor_left_back   = np.array([left,    0, back,  1])
        floor_right_back  = np.array([right,   0, back,  1])

        # Define lines
        tint_line           = (tint_left,        tint_right)
        service_line        = (service_left,     service_right)
        front_out_line      = (front_wall_left,  front_wall_right)
        side_out_line_left  = (front_wall_left,  side_wall_left_back)
        side_out_line_right = (front_wall_right, side_wall_right_back)

        floor_line_front = (floor_left_front, floor_right_front)
        
        floor_line_left  = (floor_left_front, floor_left_back)
        floor_line_right = (floor_right_front, floor_right_back)
        
        vertical_line_left  = (floor_left_front, front_wall_left)
        vertical_line_right = (floor_right_front, front_wall_right)

        
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


