
from pygacal.common.cgatools import *


def projectPointToPlane(point, R,  C_orig = up(0) , cP_orig = (ninf + e3)*I5):
    """
    Projects a real point in the world to the cameraplane.
    """
    C = R * C_orig * ~R
    cP = R * cP_orig * ~R   
    PP = C ^ point 
    
    A_img = Meet(cP, PP)
    A_img_d = ~R * A_img * R
    return A_img_d  

def projectLineToPlane(line, R,  C_orig = up(0) , cP_orig = (ninf + e3)*I5):
    """
    Projects a real line in the world to the cameraplane
    """
    C = R * C_orig * ~R
    cP = R * cP_orig * ~R
    P = C ^ line
    
    L_img = Meet(cP, P)
    L_img_d = ~R * L_img * R

    return L_img_d


