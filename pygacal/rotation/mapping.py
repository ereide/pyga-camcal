import numpy as np

from pygacal.common.cgatools import *

#TODO: from clifford_tools.common.g3c.core import *

from .costfunction import ( sumLineSquaredErrorCost, line2lineErrorCost, 
                            sumPlaneSquaredErrorCost, plane2planeErrorCost, 
                            sumLineLogErrorCost, sumImageFunction,
                            sumImageThreeViewAllPairsCostFunction, sumImageMultiViewCostFunction, 
                            restrictedCostFunction, restrictedImageCostFunction, restrictedMultiViewImageCostFunction, 
                            lineEstimationErrorCost, restrictedMultiViewBaseImageCostFunction, sumImageMultiViewBaseImageCostFunction, 
                            sumWeightedImageFunction, logSumWeightedLineSquaredErrorCost, sumLogWeightedLineSquaredErrorCost, 
                            sumWeightedLineSquaredErrorCost)

def bivectorToVecRepr(B):
    x = np.zeros(6)
    t = B | ep
    x[0:3] = MVto3DVec(t)
    P = (B - t * ninf)
    x[3:6] = -float(P|e12), -float(P|e13), -float(P|e23)   
    return x

def vecReprToBivector(x):
    """
    I will be using the conversion x \in R^(3 + 3)

    B = alpha * P + t * ninf requiring 6 parameters

    Constraint: abs(alpha) < pi

    R = exp(B)
    """       
    alphaP = x[3]*e12 + x[4]*e13 + x[5]*e23
    t = x[0]*e1 + x[1] * e2 + x[2] *e3
    B = alphaP + t * ninf
    return B


def extendedVecReprToBivector(x):
    """
    I will be using the conversion x \in R^(3 + 3 + 1)

    B = alpha * P + t * ninf + omega * E0 requiring 6 parameters

    Constraint: abs(alpha) < pi

    R = exp(B)
    """      
    B = vecReprToBivector(x[:6]) + x[6] * E0

    return B



class Mapping(object):
    name = "Mapping"
    constraints = None
    bounds = None
    opt_method = 'L-BFGS-B'
    costfunction = None
    color = 'b'
    costfunctiondecorator = None
    callback = None


    @staticmethod
    def rotorconversion(x):
        raise NotImplementedError


        
class BivectorMapping(Mapping):
    name = "BivectorMapping"
    constraints = None
    bounds = None
    opt_method = 'L-BFGS-B'
    costfunction = sumLineSquaredErrorCost
    color = 'b'
    costfunctiondecorator = restrictedCostFunction


    @staticmethod
    def rotorconversion(x):
        #return rotorconversion_fast(x) #BROKEN
        return ga_exp(vecReprToBivector(x))

    @staticmethod
    def inverserotorconversion(R):
        B = ga_log(R)
        return bivectorToVecRepr(B)

    @staticmethod
    def startValue():
        return np.random.rand(6)
        #return np.zeros(6) #Equivalent to no rotation

class BivectorLineMapping(BivectorMapping):
    name = "BivectorLineMapping"
    costfunction = sumLineSquaredErrorCost

class BivectorWeightedLineMapping(BivectorMapping):
    name = "BivectorWeightedLineMapping"
    costfunction = sumWeightedLineSquaredErrorCost(1)

    ##Needs a cost function to be implemented

class BivectorLogSumLineMapping(BivectorMapping):
    name = "BivectorLogSumLineMapping"
    costfunction  =logSumWeightedLineSquaredErrorCost(weight = 1)

class BivectorSumLogLineMapping(BivectorMapping):
    name = "BivectorSumLogLineMapping"
    costfunction = sumLogWeightedLineSquaredErrorCost(weight = 1)



class BivectorLineMultMapping(BivectorMapping):
    name = "BivectorMultLineMapping"

    #TODO: costfunction = sumLineMultSquaredErrorCost

class BivectorLogCostLineMapping(BivectorMapping):
    name = "BivectorLogCostLineMapping"
    color = 'b'
    costfunction = sumLineLogErrorCost

class BivectorPlaneMapping(BivectorMapping):
    name = "BivectorPlaneMapping"
    costfunction = sumPlaneSquaredErrorCost

class ExtendedBivectorMapping(Mapping):
    name = "ExtendedBivectorMapping"
    costfunction = sumLineSquaredErrorCost
    color = 'b'
    costfunctiondecorator = restrictedCostFunction

    @staticmethod
    def rotorconversion(x):
        return ga_exp(extendedVecReprToBivector(x))

    @staticmethod
    def inverserotorconversion(R):
        B = ga_log(R)
        return bivectorToVecRepr(B)

    @staticmethod
    def startValue():
        return np.random.rand(7)

class LinePropertyBivectorMapping(BivectorMapping):
    name = "LinePropertyBivectorMapping"
    color = 'g'
    costfunction = line2lineErrorCost
    

class PlanePropertyBivectorMapping(BivectorMapping):
    name = "PlanePropertyBivectorMapping"
    color = 'g'
    costfunction = plane2planeErrorCost


class RotorMapping(Mapping):
    name = "RotorMapping"
    color = 'y'
    constraints = None
    opt_method = 'L-BFGS-B'
    bounds = None
    costfunction = None
    costfunctiondecorator = restrictedCostFunction

    @staticmethod
    def startValue():
        x0 = np.zeros(8)
        x0[0] = 1
        return x0

    @staticmethod
    def rotorconversion(x):
        """
        I will be using the conversion x \in R^(1 + 3 + 3 + 1)

        R = alpha + B + c * ninf + gamma * I3 * ninf 

        """

        alpha = x[0]
        B = x[1] * e12 + x[2] * e13 + x[3]*e23
        c = x[4] * e1 + x[5]*e2 + x[6] * e3
        gamma = x[7]

        R = alpha + B + c * ninf + gamma * I3 * ninf 

        return R.normal()


class RotorLineMapping(RotorMapping):
    name = 'RotorLineMapping'
    costfunction = sumLineSquaredErrorCost


#For line estimation:
class BivectorLineEstimationMapping(BivectorLineMapping):
    name = "BivectorLineEstimationMapping"
    costfunction = lineEstimationErrorCost
    costfunctiondecorator = restrictedCostFunction

#For images:
class BivectorLineImageMapping(BivectorMapping):
    costfunction = sumImageFunction
    costfunctiondecorator = restrictedImageCostFunction

#For images:
class ExtendedBivectorLineImageMapping(ExtendedBivectorMapping):
    costfunction = sumImageFunction
    costfunctiondecorator = restrictedImageCostFunction

class BivectorWeightedLineImageMapping(BivectorMapping):
    costfunction = sumWeightedImageFunction
    costfunctiondecorator = restrictedImageCostFunction

class MultiViewLineImageMapping(Mapping):
    name = "MultiViewMapping"
    constraints = None
    bounds = None
    opt_method =  'L-BFGS-B'
    costfunction = sumImageMultiViewBaseImageCostFunction
    color = 'b'
    costfunctiondecorator = restrictedMultiViewBaseImageCostFunction


    @staticmethod
    def rotorconversion(x):
        """
        See above
        """     
        K = x.size//6
        #R_list = [rotorconversion_fast(x[6*i: 6*i + 6]) for i in range(K)] #BROKEN
        R_list = [ga_exp(vecReprToBivector(x[6*i: 6*i + 6])) for i in range(K)]

        return R_list

    @staticmethod
    def inverserotorconversion(R_list):
        K = len(R_list)
        x = np.zeros(6*K)
        for i in range(K):
            B  = ga_log(R_list)
            x[i*6:(i+1)*6] = bivectorToVecRepr(B)
        return x

    @staticmethod
    def startValue():
        return np.random.rand(12)

class ThreeViewLineImageMapping(MultiViewLineImageMapping):
    costfunction = sumImageThreeViewAllPairsCostFunction
    costfunctiondecorator = restrictedMultiViewImageCostFunction
