
from numpy import pi, e
import numpy as np

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from clifford import g3c

from pygacal.common.cgatools import MVto3DVec
from pygacal.geometry.lines import findLineParams, createRandomLines, createLine, extract2Dparams
from pygacal.geometry.planes import *

layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])

I3 = e123
I5 = e12345

class Plot2D():
    """
    Used for creating images
    """
    def __init__(self):
        self.fig = plt.figure()
        self.size = 5
    
    def configure(self, val):
        self.size = val
    
    def addPoint(self, A, **kwargs):
        a = MVto3DVec(down(A))
        x, y, z = a
        assert(z == 0)

        self.fig.plot(x, y, **kwargs)

    def addPoints(self, points, **kwargs):
        for point in points:
            self.addPoint(point, **kwargs)

    def plotLine2D(self, Limg, **kwargs):        
        a, ma = extract2Dparams(Limg)
        
        A = a + self.size*ma
        B = a - self.size*ma
        
        plt.plot([A[0], B[0]], [A[1], B[1]], **kwargs)

    def addLines(self, lines, **kwargs):
        for line in lines:
            self.addLine2D(line, **kwargs)  

    def show(self, block = True):
        plt.show(block)

    def save(self, name):
        plt.savefig(name)


    def set_axes(string_list):
        ax.set_xlabel(string_list[0])
        ax.set_ylabel(string_list[1])


class Plot3D():
    def __init__(self):
        self.fig = plt.figure()
        self.axes = Axes3D(self.fig)
        self.axes.axis('equal')
        self.size = 10

    def configure(self, val):
        self.size = val

    def addPoint(self, A, **kwargs):
        a = MVto3DVec(down(A))
        x, y, z = a
        self.axes.scatter(x, y, z, **kwargs)

    def addPoints(self, points, **kwargs):
        for point in points:
            self.addPoint(point, **kwargs)

    def addLine(self, line, **kwargs):
        a, ma = findLineParams(line)
        
        a = MVto3DVec(a) 
        ma = MVto3DVec(ma) 
        
        pointA = a - ma * self.size * 2
        pointB = a + ma * self.size * 2
        vals = np.vstack((pointA, pointB))

        self.axes.plot(vals[:, 0], vals[:, 1], vals[:, 2], **kwargs)

    def addPlane(self, plane, center = None, **kwargs):
        
        a = MVto3DVec(findPlaneParams(plane))
        if center:
            b = MVto3DVec(down(center))
        else:
            b = np.array([0, 0, 0])

        #m1 = np.array([a[1], -a[0], 0])
        #m2 = np.cross(a, m1)
        #m1, m2 = m1/abs(m1), m2/abs(m2)

        d = -np.sqrt(np.dot(a, a))
        normal = -a/d

        # create x,y
        size = self.size

        #TODO: Make more robust and equisized
        xx, yy = np.meshgrid(range(int(b[0])-size, int(b[0]) + size), range(int(b[1])-size, int(b[1])+size))

        # calculate corresponding z
        z = (-normal[0] * xx - normal[1] * yy - d)/normal[2]

        # plot the surface
        self.axes.plot_surface(xx, yy, z, alpha=0.2, **kwargs)

    def addLines(self, lines, **kwargs):
        for line in lines:
            self.addLine(line, **kwargs)       

    def show(self, block = True):
        plt.show(block)

    def save(self, name):
        plt.savefig(name)

    def set_axes(string_list):
        ax.set_xlabel(string_list[0])
        ax.set_ylabel(string_list[1])
        ax.set_zlabel(string_list[2])
    
    



if __name__ == '__main__':
    plot = Plot3D()

    a = e1 + 2*e2 + 3*e3 
    b = -e1
    A, B = up(a), up(b)
    line = createLine(A, B)

    lines = createRandomLines(4)
    plot.addLine(line)
    plot.addPoints((A, B))
    plot.addLines(lines)


    plot.show()
    