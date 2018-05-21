from pygacal.geometry import createRandomVector, createRandomPoint

#TODO: Make useful

class InternalCameraParameters(object):
    def __init__(self, xdir = e1, ydir = e2):
        self.setInternals(xdir, ydir)
        #3 degrees of freedom 2D direction and relative magnitude -> rotation + skew + pixel difference

    def setInternals(self, xdir, ydir):
        self._xdir = xdir
        self._ydir = ydir

        #util parameters
        #raise NotImplementedError()
        self.translator = 0



class Camera(object):
    def __init__(self, pos = up(0), planeVec = e3, internalParameters = InternalCameraParameters()):
        setPosition(pos) 
        setPlaneVec(planeVec)
        self._internal = internalParameters

    def setInternals(self, xdir, ydir):
        self._internal.setInternals(xdir, ydir)

    def setPosition(self, pos):
        self._position = pos
    
    def setPlaneVec(self, vec):
        self._planeVec = vec
        self._plane = definePlane(self._position, self._planeVec)

    def rotatePosition(self, R):
        self._position  = Sandwich(self._position, R)
        self._plane     = Sandwich(self._plane, R)

    def project(self, obj):
        great_obj = (obj ^ self._position)
        projected = great_obj.meet(self._plane)

    def definePlane(self):
        #TODO: Do something to the plane
        raise NotImplementedError()
        self._plane = plane

def createRandomCamera(meanDistance = 1, meanFocalLength = 1):
    camCenter = createRandomPoint(scale)
    planeVec = createRandomVector(focalLength)

    return Camera(camCenter, planeVec)

