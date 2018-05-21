
from examples_common import * 

from pygacal.common.plotting import Plot3D

from pygacal.geometry.transformations import *

from resources.squash_court.court_image import RobertSquashCourtImage
from resources.squash_court.court_image_left import RobertSquashCourtImage as RobertSquashCourtImage_Left
from resources.squash_court.court_image_right import RobertSquashCourtImage as RobertSquashCourtImage_Right

from resources.squash_court.court_model import *



if __name__ == "__main__":  
    img_model = RobertSquashCourtImage()
    img_model_left = RobertSquashCourtImage_Left()
    img_model_right = RobertSquashCourtImage_Right()

    model       = SquashCourtModel()

    R_min       = calibrate(model, img_model)
    R_min_left  = calibrate(model, img_model_left)
    R_min_right = calibrate(model, img_model_right)
    
    theta, t = versor_to_param(R_min)
    print(theta, t)

    theta, t = versor_to_param(R_min_left)
    print(theta, t)

    theta, t = versor_to_param(R_min_right)
    print(theta, t)


    new_model = create_model(img_model_left, R_min_left, img_model_right, R_min_right)

    compare_model_lines(model, new_model, img_model, R_min)


    #plot = Plot3D()
    #plot.addLines(new_model.get_lines_ga().values())
    #plot.show()

    #project_lines(new_model, img_model, R_min)
    #project_lines(new_model, img_model_left, R_min_left)
    #project_lines(new_model, img_model_right, R_min_right)



    """
    new_model = align_objects(model, new_model)

    
    R_min       = calibrate(new_model, img_model)
    R_min_left  = calibrate(new_model, img_model_left)
    R_min_right = calibrate(new_model, img_model_right)

    theta, t = versor_to_param(R_min)
    print(theta, t)

    theta, t = versor_to_param(R_min_left)
    print(theta, t)

    theta, t = versor_to_param(R_min_right)
    print(theta, t)

    project_lines(new_model, img_model, R_min)
    project_lines(new_model, img_model_left, R_min_left)
    project_lines(new_model, img_model_right, R_min_right)


    #theta, t = versor_to_param(R_min)
    
    #print(versor_to_param(R_min))

    """




