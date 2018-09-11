import numpy as np
import math

#Moves v1 and u1 onto v2 and u2
#Note: v1 and u1 must be perpendicular, as must v2 and u2.
#v is the vector, u is the edge, and c is the calculated cross product
#M is the matrix of v,u,c
###REMINDER: V1 AND U1 MUST ALREADY BE PERPENDICULAR!###
def shiftPerpVectors(v1, u1, v2, u2):
    #print(abs(np.dot(v1,u1)), abs(np.dot(v2,u2)))
    if abs(np.dot(v1,u1))>.01 or abs(np.dot(v2,u2))>.01:
        print("Your input vectors are NOT perpendicular!")
        dot1 = np.dot(v1,u1)
        dot2 = np.dot(v2,u2)
        raise ValueError("The direction vector is not perpendicular to your edge! Dot products: ", dot1, dot2)
    (v1,u1,v2,u2) = [normify(i) for i in (v1,u1,v2,u2)]
    #print(v1, u1, v2, u2)
    c1 = crossProd(v1,u1)
    c2 = crossProd(v2,u2)
    #print(v1,u1,c1,v2,u2,c2)
    M1 = np.transpose((v1,u1,c1)) #Transposed so that the they are column vectors
    M2 = np.transpose((v2,u2,c2))
    #RotationMatrix*M1 = M2, so RotationMatrix = M2*(M1^-1)
    M1inv = np.linalg.inv(M1)
    rotMat = np.matmul(M2,M1inv)
    #return M1,M2,M1inv,rotMat
    return rotMat

def normify(v):
    vSum = 0
    for i in v:
        vSum += i**2
    vLen = vSum**(1/2)
    vNorm = [i/vLen for i in v] 
    return(vNorm)

#Vectors must be the same length (should be length 3 probably)
def dotProd(vec1,vec2):
    dot = 0
    for i in range(0,len(vec1)):
        dot+=vec1[i]*vec2[i]
    return(dot)

#Calculates the cross product of two vectors
def crossProd(vec1,vec2):
    cross0 =   vec1[1]*vec2[2]-vec1[2]*vec2[1]
    cross1 = 0-vec1[0]*vec2[2]+vec1[2]*vec2[0]
    cross2 =   vec1[0]*vec2[1]-vec1[1]*vec2[0]
    return((cross0,cross1,cross2))

#Uses the Euler-Rodriguez formula to find the rotation matrix associated
#with the given axis and twist angle, in radians
def rotateVector(axis, twist):
    axis = normify(axis)
    a = math.cos(twist/2)    
    b = math.sin(twist/2)*axis[0]
    c = math.sin(twist/2)*axis[1]
    d = math.sin(twist/2)*axis[2]
    aa, bb, cc, dd = a*a, b*b, c*c, d*d


    ab, ac, ad, bc, bd, cd = a*b, a*c, a*d, b*c, b*d, c*d
    
    M = ((aa+bb-cc-dd, 2*(bc-ad), 2*(bd+ac)),
         (2*(bc+ad), aa+cc-bb-dd, 2*(cd-ab)),
         (2*(bd-ac), 2*(cd+ab), aa+dd-bb-cc))
    return M


def rotateVectorDegrees(axis, twistDegrees):
    return rotateVector(axis, (twistDegrees/360)*2*math.pi)
    

#M1,M2,M1inv,rotMat = shiftPerpVectors((10,5,0),(0,0,4),(4,0,4),(4,0,-4))

