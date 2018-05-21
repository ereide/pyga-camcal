

from pygacal.common.cgatools import *

from pygacal.geometry.lines import *
from pygacal.geometry.planes import *


#from clifford_tools.common.g3c.core import *
#from clifford_tools.common.g3c.core import RotorLine2Line, RotorPlane2Plane, ga_exp

import numpy as np


def rotorAbsCostFunction(R, weight = 1):
    rotation = (R - 1)
    translation = R | ep
    #return abs(rotation)**2 + abs(translation)**2
    return float((rotation*~rotation)[0] + weight*(translation*~translation)[0])

#Returns a function pointer to the cost function with a desired weight
def weightedAbsCostFunction(weight):
    return lambda R: rotorAbsCostFunction(R, weight)

def logWeightedRotorAbsCostFunction(weight):
    return lambda R: np.log(rotorAbsCostFunction(R, weight))

def bivectorCostFunction(R):
    return np.linalg.norm(ga_log(R).value)**2
 
"""
def rotorSquareCostFunction(R):
    return float((rotation*~rotation)[0] + (translation*~translation)[0])

"""

def line2lineCostFunction(L1, L2):
    return 1 - float((L1*L2)[0])

def plane2planeCostFunction(P1, P2):
    return 1 + float((P1*P2)[0])


def restrictedCostFunction(costfunction, conversion):
    def decoratedcostfunction(x, object_data):
        R = conversion(x)
        return costfunction(R, object_data)
        
    return decoratedcostfunction

def rotorCostGenerator(rotorgenerator, rotorcost):
    def cost(obj_actual, obj_real):
        rotor_diff = rotorgenerator(obj_actual, obj_real)    #Finds the rotation error 
        return rotorcost(rotor_diff)                    #Maps the error to R
    return cost

def object2objectErrorCostGenerator(costgenerator):
    def object2objectErrorCost(R, objsets):
        totalcost = 0
        N = len(objsets)
        for objset in objsets:
            obj_base    = objset[0]                         #Extracts the starting line
            obj_actual  = objset[1]                         #Finds the measured line

            obj_real = R * obj_base * ~R                    #Uses R to determine its rotation

            cost = costgenerator(obj_actual, obj_real)      #Finds the rotation error 

            totalcost += cost                               #Adds it to the running sum

        #Returns the total cost 
        return totalcost/float(N)

    return object2objectErrorCost

def logObject2objectErrorCostGenerator(costgenerator):
    def object2objectErrorCost(R, objsets):
        totalcost = 0
        N = len(objsets)
        for objset in objsets:
            obj_base    = objset[0]                         #Extracts the starting line
            obj_actual  = objset[1]                         #Finds the measured line

            obj_real = R * obj_base * ~R                    #Uses R to determine its rotation

            cost = costgenerator(obj_actual, obj_real)      #Finds the rotation error 

            totalcost += cost                               #Adds it to the running sum

        #Returns the total cost 
        return np.log(totalcost/float(N))

    return object2objectErrorCost


def singleobject2objectErrorCostGenerator(costgenerator):
    def object2objectErrorCost(R, objdata):
        object, objects = objdata
        totalcost = 0
        N = len(objects)
        obj_base    = object                                #Extracts the starting line

        for obj_data_point in objects:
            obj_actual  = obj_data_point                    #Finds the measured line

            obj_real = R * obj_base * ~R                    #Uses R to determine its rotation

            cost = costgenerator(obj_actual, obj_real)      #Finds the rotation error 

            totalcost += cost                               #Adds it to the running sum

        #Returns the total cost 
        return totalcost/float(N)

    return object2objectErrorCost



#Decorates the functions with the appropiate rotor generator
#Lines
sumLineSquaredErrorCost     =   object2objectErrorCostGenerator(rotorCostGenerator(RotorLine2Line, rotorAbsCostFunction))
#TODO: sumLineMultSquaredErrorCost =   object2objectErrorCostGenerator(rotorCostGenerator(RotorLine2Line, rotorAbsMultCostFunction))
line2lineErrorCost          =   object2objectErrorCostGenerator(line2lineCostFunction) 

logSumSquaredErrorCost      =   logObject2objectErrorCostGenerator(rotorCostGenerator(RotorLine2Line, rotorAbsCostFunction))
 


#Line estimation
lineEstimationErrorCost     =   singleobject2objectErrorCostGenerator(rotorCostGenerator(RotorLine2Line, rotorAbsCostFunction))

#Planes
sumPlaneSquaredErrorCost    =   object2objectErrorCostGenerator(rotorCostGenerator(RotorPlane2Plane, rotorAbsCostFunction))
plane2planeErrorCost        =   object2objectErrorCostGenerator(plane2planeCostFunction) #Currently the same as the line function

#logcostfunction
sumLineLogErrorCost         =   object2objectErrorCostGenerator(rotorCostGenerator(RotorLine2Line, bivectorCostFunction))



def sumWeightedLineSquaredErrorCost(weight):
    return object2objectErrorCostGenerator(rotorCostGenerator(RotorLine2Line, weightedAbsCostFunction(weight)))

def sumLogWeightedLineSquaredErrorCost(weight):
    return object2objectErrorCostGenerator(rotorCostGenerator(RotorLine2Line, logWeightedRotorAbsCostFunction(weight)))

def logSumWeightedLineSquaredErrorCost(weight):
    return logObject2objectErrorCostGenerator(rotorCostGenerator(RotorLine2Line, weightedAbsCostFunction(weight)))

def meet_planes(plane_a,plane_b):
    return ((plane_a*plane_b)(2)).dual()

#Multiview specific 
def costAddition(L_1, L_2, L_3, R_A, R_B, O1 = up(0)):
    P_base  = O1 ^ L_1
    P_A     = R_A * (O1 ^ L_2) * ~R_A
    P_B     = R_B * (O1 ^ L_3) * ~R_B

    #P_test  = (O1 ^ Meet(P_A, P_B)).normal()

    #Lines
    L_A     = Meet(P_base, P_A)
    L_B     = Meet(P_base, P_B)

    #print("")
    #print(L_A)
    #print(L_B)
    #print("")
    #TODO: WHY IS THIS HAPPENING???? NEED A SIGN OPERATOR
    return rotorAbsCostFunction(RotorLine2Line(L_A, L_B))


#Multiview specific 
def costSymmetric(L_A, L_B, L_C, R_A, R_B, R_C, O1 = up(0)):
    P_A     = R_A * (O1 ^ L_A) * ~R_A
    P_B     = R_B * (O1 ^ L_B) * ~R_B
    P_C     = R_C * (O1 ^ L_C) * ~R_C
    #P_test  = (O1 ^ Meet(P_A, P_B)).normal()

    #Lines
    L_A      = Meet(P_A, P_B)
    L_B      = Meet(P_B, P_C)

    #L_A     = Meet(P_B, P_A)
    #L_B     = Meet(P_B, P_C)

    print("")
    print(L_A)
    print(L_B)
    print("")
    #TODO: WHY IS THIS HAPPENING???? NEED A SIGN OPERATOR
    return rotorAbsCostFunction(RotorLine2LineSafe(L_A, L_B))



def sumImageMultiViewBaseImageCostFunction(R_test_list, lines_base, L_list, O1 = up(0), verbose = True):
    if (len(L_list) != len(R_test_list)):
        raise ValueError("Number of images and number of rotors must match")
    if (len(L_list) < 2):
        raise ValueError("Not enough images")
    
    cost = 0
    N = len(lines_base)
    K = len(R_test_list)
    for i in range(N):
        for j in range(-1, K - 1):
            line_base   = lines_base[i]

            line_A      = L_list[j][i]
            line_B      = L_list[j + 1][i]

            R_A_test    = R_test_list[j]
            R_B_test    = R_test_list[j + 1]

            cost += costAddition(line_base, line_A, line_B,  R_A_test, R_B_test)

    cost = cost/(N * K)
    if verbose:
        print("Cost: ", cost/(N * K))
    return cost



def restrictedMultiViewBaseImageCostFunction(costfunction, conversion):
    def decoratedcostfunction(x, lines_base, lines_imgs):
        R_list = conversion(x)
        return costfunction(R_list, lines_base, lines_imgs)
    return decoratedcostfunction 

def sumImageMultiViewCostFunction(R_test_list, L_list, O1 = up(0), verbose = True):
    if (len(L_list) != len(R_test_list)):
        raise ValueError("Number of images and number of rotors must match")
    if (len(L_list) < 2):
        raise ValueError("Not enough images")
    
    cost = 0
    N = len(L_list)
    K = len(R_test_list)
    for i in range(N):
        for j in range(1, K - 1):

            R_A_test    = R_test_list[j - 1]
            R_B_test    = R_test_list[j]
            R_C_test    = R_test_list[j + 1]

            line_A      = L_list[j-1][i]
            line_B      = L_list[ j ][i]
            line_C      = L_list[j+1][i]
            
            cost += costSymmetric(line_A, line_B, line_C, R_A_test, R_B_test, R_C_test, O1)

    cost = cost/(N * K)
    if verbose:
        print("Cost: ", cost)
    return cost


def restrictedMultiViewImageCostFunction(costfunction, conversion):
    def decoratedcostfunction(x, lines_imgs):
        R_list = conversion(x)
        return costfunction(R_list, lines_imgs)
    return decoratedcostfunction 


def sumImageThreeViewAllPairsCostFunction(R_list, lines_base, lines_imgs_d , O1 = up(0), verbose = True):
    #if len(linesets) < 3:
    #    raise ValueError("Need at least 3 lines to determine location")
    
    R_A_test, R_B_test = R_list
    lines_A, lines_B   = lines_imgs_d

    #TODO: For now choose first line set as basestring
    cost = 0
    N = len(lines_base)

    for i in range(N):
        cost += costAddition(lines_base[i], lines_A[i], lines_B[i],  R_A_test, R_B_test)
        cost += costAddition(lines_B[i], lines_base[i], lines_A[i],  ~R_B_test, (~R_B_test*R_A_test))
        cost += costAddition(lines_A[i], lines_B[i], lines_base[i],  (~R_A_test*R_B_test), ~R_A_test)

    if verbose:
        print("Cost: ", cost/(N * 3))

    return cost/(N * 3)



#Image specific

def sumImageGenerator(costfunction, rotation):
    def generalSumImageFunction(R_test, lines, lines_image, O1 = up(0)):
        cost = 0
        N = len(lines)
        for i in range(N):
            P_img_observed =  (O1 ^ lines_image[i]).normal()
            P_img_derived  =  (O1 ^ (~R_test * lines[i] * R_test)).normal()

            cost += costfunction(rotation(P_img_observed, P_img_derived))        

        return cost/N

    return generalSumImageFunction

sumImageFunction         = sumImageGenerator(rotorAbsCostFunction, RotorPlane2Plane)
sumWeightedImageFunction = sumImageGenerator(rotorAbsCostFunction, RotorPlane2Plane) #TODO


def restrictedImageCostFunction(costfunction, conversion):
    def decoratedcostfunction(x, lines, lines_image):
        R = conversion(x)
        return costfunction(R, lines, lines_image)
        
    return decoratedcostfunction


#Comparison metric
def linePointCostMetric(R_test, R_real, N_val):
    validationlines = createRandomLines(N_val)
    N_points = 5

    costs = np.zeros(N_points)

    for line in validationlines:
        linepoints = createPointsOnLine(line, N_points)
        for i in range(N_points):
            startpoint = up(linepoints[i]) #Need to work in CGA space since translations don't make any sense in G3 space'
            newpoint = R_real * startpoint * ~R_real
            testpoint = R_test* startpoint *~R_test
            cost = Distance(newpoint(1), testpoint(1))
            costs[i] +=  cost

    costs /= N_val

    return costs


#Comparison metric
def planePointCostMetric(R_test, R_real, N_val):
    validationplanes = createRandomPlanes(N_val)
    N_points = 5

    costs = np.zeros(N_points)

    for plane in validationplanes:
        planepoints = createPointsOnPlane(plane)
        for i in range(N_points):
            startpoint = up(planepoints[i]) #Need to work in CGA space since translations don't make any sense in G3 space'
            newpoint  = R_real * startpoint * ~R_real
            testpoint = R_test * startpoint * ~R_test
            cost      = Distance(newpoint(1), testpoint(1))
            costs[i] +=  cost

    costs /= float(N_val)

    return costs


