
from examples_common import * 

from pygacal.geometry.transformations import *

from resources.football_court.court_image import *
from resources.football_court.court_model import *



if __name__ == "__main__":  
    img_model = FootballManCityCourtImage2()
    model     = PenaltyAreaFootballModel()

    R_min = calibrate(model, img_model)
    project_lines(model, img_model, R_min)
    theta, t = versor_to_param(R_min)
    
    print(versor_to_param(R_min))





