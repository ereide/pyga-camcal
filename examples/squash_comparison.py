
from calibration_common import * 

import numpy as np

from pygacal.geometry.transformations import *

#from resources.squash_court.court_image import *
#from resources.squash_court.court_image_left import *
#from resources.squash_court.court_image_right import *
#from resources.squash_court.court_image_left_2 import *
from resources.squash_court.court_image_right_2 import *

from resources.squash_court.court_model import *



if __name__ == "__main__":  
    img_model = RobertSquashCourtImage()
    model     = SquashCourtModel()

    P_coordtransform = full_projection_matrix([0, np.pi, 0], [0, 0, 10])

    P_robert = np.eye(4)
    P_robert[:3, :3] = img_model.R
    P_robert[:3, 3]  = img_model.T

    P = np.dot(P_coordtransform, np.linalg.inv(P_robert))
    print(P)
    V = projection_to_versor(P)
    

    #project_lines_matrix(model, img_model, P)

    R_min = calibrate(model, img_model)
    project_lines(model, img_model, R_min)
    project_lines(model, img_model, V)
    #theta, t = versor_to_param(R_min)
    #theta, t = versor_to_param(V)
    
    print(versor_to_param(R_min))
    print(versor_to_param(V))





