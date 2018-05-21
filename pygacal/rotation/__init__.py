from clifford import g3c
from numpy import pi, e
import numpy as np
import time

import scipy.optimize as opt

import warnings

from pygacal.common.cgatools import Sandwich, Dilator, Translator, Reflector, inversion, Rotor, Transversor, I3, I5, VectorEquality, anticommuter, ga_exp, ga_log
from .costfunction import sumLineSquaredErrorCost, linePointCostMetric, restrictedCostFunction

from pygacal.common.plotting import Plot3D

#TODO: Explicit imports
from pygacal.geometry import perturbeObject
from pygacal.geometry.lines import *
from pygacal.geometry.planes import *



layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])



def minimizeError(args, mapping, x0 = None, **kwargs):
    if x0 is None:
        x0 = mapping.startValue()

    method = mapping.opt_method
    rotorconversion = mapping.rotorconversion
    bounds = mapping.bounds
    constraints = mapping.constraints
    costfunction = mapping.costfunction
    decorator = mapping.costfunctiondecorator
    
    res = opt.minimize( decorator(costfunction, rotorconversion), 
                        x0=x0, 
                        args=args, 
                        method=method, 
                        bounds=bounds, 
                        constraints=constraints, 
                        **kwargs)
    R_min = res.x
    success = res.success
    Niterations = res.nit
    if success:
        return mapping.rotorconversion(res.x), Niterations
    else:
        return None
        


