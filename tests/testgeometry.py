import unittest

import clifford as cl
from clifford import g3c
from numpy import pi, e
import numpy as np

from pygacal.common.cgatools import (Sandwich, Dilator, Translator, Reflector, inversion, Rotor, Transversor, 
                             I3, I5, VectorEquality, 
                             MVEqual, MVto3DVec, VectoMV, ga_exp)

from pygacal.common.plotting import Plot3D

#TODO: Explicit imports
from pygacal.geometry import *
from pygacal.geometry.transformations import * 
from pygacal.geometry.lines import *
from pygacal.geometry.planes import *
from pygacal.rotation.costfunction import *


#from clifford_tools.common.g3c.core     import RotorLine2Line, ga_exp

#TODO: breaks when adding this line
#from clifford_tools.numerical.g3c.core  import RotorLine2Line, ga_exp, RotorPlane2Plane





layout = g3c.layout
locals().update(g3c.blades)


ep, en, up, down, homo, E0, ninf, no = (g3c.stuff["ep"], g3c.stuff["en"], 
                                        g3c.stuff["up"], g3c.stuff["down"], g3c.stuff["homo"], 
                                        g3c.stuff["E0"], g3c.stuff["einf"], -g3c.stuff["eo"])


def quaternion_mult(q,r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                         x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                         x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)

def point_rotation_by_quaternion(point, quat):
    r = np.array([0, point[0], point[1], point[2]]) 
    q = quat
    q_conj = np.array([q[0],-1*q[1],-1*q[2],-1*q[3]])
    return quaternion_mult(quaternion_mult(q,r),q_conj)[1:]



class TestGeometryTransformations(unittest.TestCase):
    def getParam(self):
        alpha = -0.65
        beta  = 0.43
        gamma = -0.21

        x, y, z = 1.23, 4.56, 78.9

        theta = np.array([alpha, beta, gamma])
        translation = np.array([x, y, z])

        return theta, translation


    def testTransformation(self):
        #TODO: check that the translation is what we think it is
        theta, translation = self.getParam()

        R_rot = rotation_to_rotor(theta)
        R_mat = rotation_to_matrix(theta)
        R_quat  = rotation_to_quaternion(theta)

        a = np.array([1.2, -0.5, 0.3])

        A     = up(VectoMV(a))
        B_rot = R_rot*A*~R_rot

        b_mat  = np.dot(R_mat, a)
        b_rot  = MVto3DVec(down(B_rot))
        b_quat = point_rotation_by_quaternion(a, R_quat)

        assert(np.allclose(b_mat, b_rot))
        assert(np.allclose(b_mat, b_quat))

        a = np.array([1.2, -0.5, 0.3, 1]) 

        P_mat = full_projection_matrix(theta, translation)
        V     = parameters_to_versor(theta, translation)

        A     = up(VectoMV(a))  

        B_rot = V * A * ~V

        b_mat = np.dot(P_mat, a)
        b_rot = MVto3DVec(down(B_rot))

        assert(np.allclose(homogenous_to_vec(b_mat), b_rot))

        a = np.array([1.2, -0.5, 0.3, 1])


        T               = Translator(VectoMV(translation))

        R_translation   = parameters_to_versor(np.zeros(3), translation)
        R_mat           = full_projection_matrix(np.zeros(3), translation)


        B_T             = T * A * ~T
        B_rot           = R_translation * A * R_translation

        b_T     = MVto3DVec(down(B_T))
        b_rot   = MVto3DVec(down(B_rot))
        b_mat  = np.dot(R_mat, a)[:3]
        b_real  = a[:3] + translation

        assert(np.allclose(b_mat, b_rot))
        assert(np.allclose(b_mat, b_real))
        assert(np.allclose(b_mat, b_T))

    def testMatrixTransformation(self):
        theta, translation = self.getParam()
        
         #Testing angles <-> matrix
        R_mat = rotation_to_matrix(theta)
        angles = matrix_to_rotation(R_mat)
        assert(np.allclose(angles, theta))

        projection = full_projection_matrix(theta, translation)
        angles = matrix_to_rotation(projection)
        t_mat = matrix_to_translation(projection)

        assert(np.allclose(angles, theta))
        assert(np.allclose(t_mat , translation))

    def testQuaternionTransformation(self):
        #Testing angles <-> quaternion
        theta, translation = self.getParam()


        rotor = rotation_to_rotor(theta)
        quat_rotor = rotor_to_quaternion(rotor)
        quat_rotation = rotation_to_quaternion(theta)
        assert(np.allclose(quat_rotation, quat_rotor))

        angles = quaternion_to_rotation(quat_rotor)
        assert(np.allclose(angles, theta))


    def testQuaternionRotation(self):
        #Testing angles <-> quaternion
        theta, translation = self.getParam()


        rotor   = rotation_to_rotor(theta)
        quat    = rotor_to_quaternion(rotor)
        mat_rot = rotation_to_matrix(theta)

        a = translation
        A = up(VectoMV(a))
        B = rotor * A * ~rotor

        b_mat  = np.matmul(mat_rot, a)
        b_rot  = MVto3DVec(down(B))
        b_quat = point_rotation_by_quaternion(a, quat)

        assert(np.allclose(b_quat, b_rot))
        assert(np.allclose(b_mat, b_rot))




    def testVersorTransformation(self):
        #Testing versor decomposition
        theta, translation = self.getParam()
        V = parameters_to_versor(theta, translation)
        T, R = versor_decomposition(V)

        assert(MVEqual(R, rotation_to_rotor(theta)))
        assert(MVEqual(T, translation_to_rotor(translation)))

        theta_ver, t_ver = versor_to_param(V) 

        assert(np.allclose(translation, t_ver))
        assert(np.allclose(theta_ver, theta))


    def testProjectionVersorTransformation(self):
        #Testing all together
        theta, translation = self.getParam()

        V = parameters_to_versor(theta, translation)
        P = full_projection_matrix(theta, translation)

        V_proj = projection_to_versor(P)
        P_test = versors_to_projection(V)

        assert(MVEqual(V, V_proj))
        assert(np.allclose(P, P_test))



class TestGeometryUtils(unittest.TestCase):
    def testRandomNoiseGeneration(self):
        """
        Tests the functionality of the random noise generation
        """

        e_0 = createRandomNoiseVector(0)
        assert(e_0.grades() == [])

        e_t = createRandomNoiseVector(0.01)
        assert(e_t.grades() == [1])

        R_e = createRandomRotationNoise(0.05)
        assert(R_e.grades() == [0, 2]) 



class TestLines(unittest.TestCase):
    def testCreateLine(self):
        """
        Test that the create line function indeed produces the desired line, 
        and that it equals the representation using GA(3) vectors

        $ L \propto ((a ^ b ^ n_{inf})  + ((a - b) ^ n_{\inf} ^ n_0)) $
        """

        #Deterministic point
        a = e1 + e2* 3.451
        b = 2 * e3 - e2 * 0.4 + e1 * 0.1

        actual = createLine(up(a), up(b))
        expected = ((a ^ b ^ ninf)  + ((a - b) ^ ninf ^ no)).normal()
        expected2 = (up(a) ^up(b)^ninf).normal()
        assert(actual == expected == expected2)

        #Random point
        a, b = createRandomVectors(2)
        actual = createLine(up(a), up(b))
        expected = ((a ^ b ^ ninf)  + ((a - b) ^ ninf ^ no)).normal()
        expected2 = (up(a) ^up(b)^ninf).normal()
        assert(actual == expected == expected2)

    

    def testLineParameterExtraction(self):
        """
        Tests that we can extract the perpendicular line 

        $ L = ((a ^ ma ^ n_{inf})  + (-ma ^ n_{\inf} ^ n_0)) $

        """

        _a = e1 + 2.7 * e2 - e3
        _ma = (2.7* e1 - e2).normal()
        _b = _a + _ma

        assert(abs(_a | _ma) < 1E-10)

        A, B, = up(_a), up(_b)

        line = A ^ B ^ ninf

        a, ma = findLineParams(line)

        assert(a ==  _a)
        assert(ma == _ma)
    
    def testDeterministicLineRotation(self):
        _a = e1 + e2* 3.451
        _ma = (e1 * 3.451 - e2).normal()
        _b = _a + _ma
        _c = 4.371*e2 + e1
        _mc = (5 * e3 + 4.371 * e1 - e2).normal()
        _d = _c + _mc

        assert(_a | _ma == 0)
        assert(_c | _mc == 0)

        A, B, C, D = up(_a), up(_b), up(_c), up(_d)
        line1 = createLine(A, B)
        line2 = createLine(C, D)
        R_T = RotorLine2Line(line1, line2)

        linecalc = (R_T * line1.normal() * ~R_T)

        assert(linecalc == line2)


    def testRandomLineRotations(self):
        """
        Creates 200 random lines and tests that the rotor converts between them correctly
        """
        print("\nTestRandomLineRotations")

        N = 30
        lines = createRandomLines(2* N)
        for i in range(N):
            line1, line2 = lines[i],  lines[i + N]
            R_T = RotorLine2Line(line1, line2)
            linecalc = (R_T * line1.normal() * ~R_T)
            #print("linecalc= ", linecalc)
            #print("lineactual= ", line2)
            assert(linecalc == line2)
            
    def testRandomNoiseGeneration(self):
        """
        Tests the functionality of the random noise generation
        """

        e_0 = createRandomNoiseVector(0)
        assert(e_0.grades() == [])

        e_t = createRandomNoiseVector(0.01)
        assert(e_t.grades() == [1])

        R_e = createRandomRotationNoise(0.05)
        assert(R_e.grades() == [0, 2]) 

    def testCloseConversion(self):
        """
        Test the difference between a noisy conversion and a regular conversion
        """
        verbose = True 
        if verbose:
            print("\nTestCloseConversion")
        #Variables that should not be touched are marked with an underscore
        _a = e1 + e2* 3.451
        _ma = (e1 * 3.451 - e2).normal()
        _b = _a + _ma
        _c = 4.371*e2 + e1
        _mc = (5 * e3 + 4.371 * e1 - e2).normal()
        _d = _c + _mc

        assert(_c | _mc == 0)
        assert(_a | _ma == 0)


        A, B, C, D  = up(_a), up(_b), up(_c), up(_d)
        L_base      = createLine(A, B)
        L_expected  = createLine(C, D)

        sigma_R = 0.5
        sigma_T = 0.5

        R_T = noisyRotor(_a, _c, _ma, _mc, sigma_R, sigma_T)

        assert(R_T*~R_T == ~R_T * R_T == 1)

        L_actual = (R_T * L_base * ~R_T)
        rotor_diff = RotorLine2Line(L_actual, L_expected)

        c_real, m_c_real = findLineParams(L_actual)
        c_calc, m_c_calc = findLineParams(L_expected)

        L_final = Sandwich(L_actual, rotor_diff)

        if verbose:
            print("diff c:    ", c_calc - c_real)
            print("diff m_c:  ", m_c_calc - m_c_real)

            print("L_base:    ", L_base)
            print("L_expected:", L_expected)
            print("L_actual   ", L_actual)
            print("rotor_diff:", rotor_diff)
            print("L_final:   ", L_final)

        assert(MVEqual(L_final, L_expected))
    
    def testManyCloseConversion(self):

        print("\nTestManyCloseConversion")
        print("Comparing rotors for small perturbations from the expected")
         #Variables that should not be touched are marked with an underscore
        
        '''_a = e1 + e2* 3.451
        _ma = (e1 * 3.451 - e2).normal()
        _c = 4.371*e2 + e1
        _mc = (5 * e3 + 4.371 * e1 - e2).normal()

        assert(_c | _mc == 0)
        assert(_a | _ma == 0)'''

        sigma_R = 0.023
        sigma_T = 0.05
        N = 50
        
        line1, line2 = createRandomLines(2)
        R_real = RotorLine2Line(line1, line2)
        lineSets = createNoisyLineSet(R_real, sigma_R, sigma_T, N)

        totalcost = 0

        for lineset in lineSets:
            L_base = lineset[0]
            L_real = R_real * L_base * ~R_real
            L_actual = lineset[1]
            rotor_diff = RotorLine2Line(L_actual, L_real)
            cost = rotorAbsCostFunction(rotor_diff)

            totalcost += cost

            print("rotor_diff:", rotor_diff)
            print("absolute cost:", cost, "\n")
            L_final = (rotor_diff * L_actual * ~rotor_diff)
            
            #This should have corrected the rotor to the correct one
            #assert(L_final == L_real)

            #Cost should always be postive
            #assert(cost >= 0)

        print("\nTotal cost =", totalcost, "\n")

    #TODO: Doesn't work as expected 
    @unittest.skip
    def testLinePointCost(self):
        np.random.seed(1)

        R_real = ga_exp(createRandomBivector())
        R_other = ga_exp(createRandomBivector())
        N_points = 3
        N_val = 10

        assert(all(linePointCostMetric(R_real, R_real, N_val) < 1E-4)) #TODO: Not as close as I would like it to be
        assert(all(linePointCostMetric(R_real, R_real, N_val) >= 0))

        assert(all(linePointCostMetric(R_real, R_other, N_val) > 0))


    def testCostFunction(self):
        print("\nTesting cost function")
        line = createRandomLines(1)[0]

        t = 1E-1 * (e1 + 2* e2 + 3*e4)
        T = Translator(t)

        sigma_R = 3
        R = createRandomRotationNoise(sigma_R)

        line_T = T * line * ~T
        R_t = RotorLine2Line(line, line_T)

        R_t_2 = (R_t - 1) | ep

        t_cost = rotorAbsCostFunction(R_t)
        
        line_R = R * line * ~R
        R_r = RotorLine2Line(line, line_R)
        r_cost = rotorAbsCostFunction(R_r)

        print("R_t", R_t, R_t_2)
        print("R_error_t", (R_t - 1)*~(R_t - 1), R_t_2*~R_t_2 )
        print("R_r", R_r)
        print("R_error_r", (R_r - 1)*~(R_r - 1))

        print("t_cost", t_cost)
        print("r_cost", r_cost)

        assert(t_cost > 1E-8)   #Needs to incorporate translation error
        assert(r_cost > 1E-8)   #Needs to incorporate rotation error

    '''
    def testInvarianceOfRotorGeneration(self):
        """
        Test that the rotor generation is invariant to location of the line 
        """
        print("\n\nTestInvarianceOfRotorGeneration:")

        #Create two random lines
        lineA, lineB = createRandomLines(2)

        #Define R as the mapping from line A -> B
        R_orignal = RotorLine2Line(lineA, lineB)

        #Find a new random line
        line = createRandomLines(1)[0]

        #Map this to a new location using the true mapping
        new_line = R_orignal * line * ~R_orignal

        #Check that the extracted rotor is the same
        R_extracted = RotorLine2Line(line, new_line)

        #TODO: This currently fails
        print("R_orignal", R_orignal)
        print("R_extracted", R_extracted)

        print("\n")
        assert(R_extracted == R_orignal)
    '''



class TestPlanes(unittest.TestCase):
    def testCreatePlane(self):
        """
        Test that the create line function indeed produces the desired line, 
        and that it equals the representation using GA(3) vectors

        $ \Pi \propto ((a ^ m1 ^ m2 ^ n_{inf})  + (n_0^a ^ (m1 + m2) ^ n_{\inf})) $
        where a, m1, m2 are perpendicular. 
        """

        #Deterministic point
        a = 3*e1 + e2 + 2*e3
        m1 = (e1 -23 * e2 + 10*e3).normal()
        m2 = (-4 *e1 + 2* e2 + 5*e3).normal()

        b = a + m1
        c = a + m2

        assert(a | m1 == 0)
        assert(m2 | m1 == 0)
        assert(m2 | a == 0)


        actual = createPlane(up(a), up(b), up(c))
        expected = (up(a)^up(b)^up(c)^ninf).normal()
        expected_expanded = ((a ^ b ^ c ^ ninf) - (no ^ a ^ (m1 + m2) ^ ninf)) #Something is wrong with this formula
        print(expected)
        print(expected_expanded) #TODO: Doesn't work yet
        assert(actual == expected)

        #Random point
        A, B, C = createRandomPoints(3)
        actual = createPlane(A, B, C)
        expected = (A^B^C^ninf).normal()
        assert(actual == expected)

    
    def testDeterministicPlaneRotation(self):
        _a = e1 + e2* 3.451
        _b = e1 * 3.451 - e2
        _c = 4.371*e2 + e1
        _d = _c + 5 * e3 + 4.371 * e1 - e2
        _e = 3.21 * e1 + 0.421*e2
        _f = 2.21 * e3

        A, B, C, D, E, F = up(_a), up(_b), up(_c), up(_d), up(_e), up(_f)
        P1 = createPlane(A, B, C)
        P2 = createPlane(D, E, F)
        R_T = RotorPlane2Plane(P1, P2)

        planecalc = (R_T * P1 * ~R_T) #Todo: Why negative

        assert(planecalc == P2)



    def testCostFunction(self):
        print("\nTesting cost function")
        plane = createRandomLines(1)[0]

        t = 1E-1 * (e1 + 2* e2 + 3*e4)
        T = Translator(t)

        sigma_R = 3
        R = createRandomRotationNoise(sigma_R)

        plane_T = T * plane * ~T
        R_t = RotorPlane2Plane(plane, plane_T)

        R_t_2 = (R_t - 1) | ep

        t_cost = rotorAbsCostFunction(R_t)
        
        plane_R = R * plane * ~R
        R_r = RotorPlane2Plane(plane, plane_R)
        r_cost = rotorAbsCostFunction(R_r)

        print("R_t", R_t, R_t_2)
        print("R_error_t", (R_t - 1)*~(R_t - 1), R_t_2*~R_t_2 )
        print("R_r", R_r)
        print("R_error_r", (R_r - 1)*~(R_r - 1))

        print("t_cost", t_cost)
        print("r_cost", r_cost)

        assert(t_cost > 1E-8)   #Needs to incorporate translation error
        assert(r_cost > 1E-8)   #Needs to incorporate rotation error

    """
    def testInvarianceOfRotorGeneration(self):
        "
        #Test that the rotor generation is invariant to location of the line 
        "
        print("\n\nTestInvarianceOfRotorGeneration:")

        #Create two random planes
        PA, PB = createRandomPlanes(2)

        #Define R as the mapping from plane A -> B
        R_orignal = RotorPlane2Plane(PA, PB)

        #Find a new random plane
        P3 = createRandomPlanes(1)[0]

        #Map this to a new location using the true mapping
        new_plane = R_orignal * P3 * ~R_orignal

        #Check that the extracted rotor is the same
        R_extracted = RotorPlane2Plane(P3, new_plane)
        PB_extracted = (R_extracted*PA*~R_extracted)

        #TODO: This currently fails
        print("R_orignal    ", R_orignal)
        print("R_extracted  ", R_extracted)

        print("\n")
        print("PB           ", PB)
        print("PB_extracted ", PB_extracted)
        assert(PB == PB_extracted)

        assert(R_extracted == R_orignal)
    """


if __name__ == '__main__':
    unittest.main()