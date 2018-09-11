import pickle
import shiftPerpVectors
import numpy as np
import copy

#This file has the DNAPolyhedron class and some related methods. 
#It also has the ProteinConnector class.
#
#Generally, this file should be used by creating one or more DNAPolyhedron instances and
#one or more ProteinConnector instances. These instances can also be saved and loaded to text files,
#so that once a polyhedron or connector is designed and created once it can quickly be re-loaded. 
#(If desired, any polyhedron can be sent to a .pdb file so it can be viewed in pymol).
#Then, using iteratePoly, these connectors and polyhedra can be simulated as creating a network of
#connected polyhedra, the polyhedra and connectors for which will be saved to a folder.
#To view the network in pymol, open pymol and type: '{filepath}/{foldername}/*.pdb'
#
#Note: As far as I know, the displayed structure should never be deformed unless a coordinate reaches
#a coordinate location >= 10000 (due to the .pdb file structure).
#
### IMPORTANT NOTE: ###
#This file is written in Python 3. At least sometimes, PyMol seems to run Python 2.
#The biggest difference I've noticed is that in Python 2, static methods (class methods
#that don't require an instance of the class) must be declared as static. Therefore,
#static methods in this file won't work in the PyMol interpreter (unless it can be made
#to run Python 3). Therefore, these methods should be run separately, then the files
#created can be opened in PyMol.


#This is the DNAPolyhedron class. It stores information about a polyhedron with vertices, edges, binding sites,
#and direction vectors associated with the binding sites. Instantiation instructions are below.
#
#When instantiating a DNAPolyhedron, it takes in:
# - newVertices: A list of 3-dimensional points that are added to an array. Indexed in order.
#    - Example: ((1,2,3), (2,3,1), (0,0,0)) creates 3 vertices.
#
# - newEdges: A list of 2-tuples. The entries of each tuple are integers, each integer
#             representing a vertex (by its index) and each pair representing an edge.
#             Stored internally as an adjacency matrix of the vertices.
#    - Example: ((0,3),) links only the first and fourth vertices.
#               ((0,1), (1,2), (2,0)) links the first 3 vertices to make a triangle.
#
#    - Note: Even if you're linking only one pair of vertices, the program expects a list of tuples.
#            So input it in the form ((a,b),). Without the additional comma, it doesn't work.
#
# - newBindingSites: The third entry is a list of 4-tuples, each of which represents a DNA
#                    binding site. The first entry per 4-tuple is an edge (by index),
#                    the second is a number between 0 and 1 that represents the % of the
#                    way along the bond the binding site is located. Goes from the first
#                    point referenced to the second, so if the edge whose index is added
#                    is (2,1) and the % is .6, then the binding site will be 60% of the
#                    way from point 2 to point 1. The third entry is a string identifier
#                    which represents the type of site, which is used to match to a
#                    protein connector.
#                    The fourth entry is a directional vector which represents the
#                    site's direction. This repreesents the direction of a connector
#                    relative to the site, and MUST be perpendicular to the edge.
#                    Stored internally as a 4xn matrix, where each column is the site's
#                    coordinates and then its string ID, followed by its associated vertices
#                    and finally its direction vector.
#    - Example: ((0, .8, "A-site", (1,1,1)), (1, .5, "B-site", (0,0,1))) would create an A-type binding
#               site 80% of the way along the first edge entered and a B-type site halfway along the 
#               second edge entered, in the directions (1,1,1) and (0,0,1).

#Example polyhedron (a triangle):
#trianglePoly = DNAPolyhedron(((10,2,3), (0,0,0), (4,3,12)), ((0,1),(1,2)), ((0, .7, "A", (1,1,1)), (1, .5, "B", (1,1,0))))
class DNAPolyhedron():
    #Initialization function. How to use this is described above.
    def __init__(self, newVertices, newEdges, newBindingSites):
        self.vertices = newVertices #Vertices should already be an array of points

        adjSize = len(self.vertices) #The adjacency matrix must be nxn, where n = #vertices
        #Initializes the nxn adjacency matrix with 0's:
        newAdj = [[0 for i in range(0,adjSize)] for j in range(0,adjSize)]
        for edge in newEdges: #Fills in the adjacency matrix
            newAdj[edge[1]][edge[0]] = 1
            newAdj[edge[0]][edge[1]] = 1
        self.edges = newAdj

        allSites = [] #Creates the 4xn matrix of binding sites
        for site in newBindingSites:
            point1 = self.vertices[newEdges[site[0]][0]] #First vertex
            point2 = self.vertices[newEdges[site[0]][1]] #Second vertex
            newPoint = []
            for i in (0,1,2): #Finds the weighted average of the vertices
                newPoint.append(site[1]*point1[i]+(1-site[1])*point2[i])
            newPoint = (tuple(newPoint), site[2], newEdges[site[0]], site[3]) #Appends the string ID and associated vertices
            allSites.append(newPoint) #Adds it to the final array
        self.bindingSites = tuple(allSites) #Converts to tuple (better data structure)

    #Given a filename (eg "MyPoly.pdb"), makes a file that can be opened in PyMOL (if given the .pdb suffix) and viewed.
    #If you name this the same as an existing file, it WILL overwrite it. Be careful!
    #Don't forget to give it the ".pdb" suffx, and don't forget to "show sticks"!
    #Does not work for coordinates over 10000; uses atomTemplate2 to work for 1000<=x<=9999
    atomTemplate1 = "ATOM  {num:>5} {name:>4} {resName:>3} {chainID:1}{resSeq:>4}    {x:> 8.3f}{y:> 8.3f}{z:> 8.3f}\n"
    atomTemplate2 = "ATOM  {num:>5} {name:>4} {resName:>3} {chainID:1}{resSeq:>4}    {x:> 8.2f}{y:> 8.2f}{z:> 8.2f}\n"
    connectTemplate = "CONECT {vert1:>4} {vert2:>4}\n"
    def createPDB(self, filename):
        pdbFile = open(filename, "w")
        
        curName = "VT"
        resName = "TH1"
        chainID = "A"
        for i in range(0,len(self.vertices)):
            atom = self.vertices[i]
            if atom[0]<1000 and atom[1]<1000 and atom[2]<1000:
                nextLine = DNAPolyhedron.atomTemplate1.format(num=i, name=curName, resName=resName, chainID=chainID, resSeq=i, x=atom[0], y=atom[1], z=atom[2])
            else:
                nextLine = DNAPolyhedron.atomTemplate2.format(num=i, name=curName, resName=resName, chainID=chainID, resSeq=i, x=atom[0], y=atom[1], z=atom[2])
            pdbFile.write(nextLine)

        for i in range(0,len(self.bindingSites)):
            atom = self.bindingSites[i][0]
            if atom[0]<1000 and atom[1]<1000 and atom[2]<1000:
                nextLine = DNAPolyhedron.atomTemplate1.format(num=i+len(self.vertices), name=curName, resName=resName, chainID=chainID, resSeq=i+len(self.vertices), x=atom[0], y=atom[1], z=atom[2])
            else:
                nextLine = DNAPolyhedron.atomTemplate2.format(num=i+len(self.vertices), name=curName, resName=resName, chainID=chainID, resSeq=i+len(self.vertices), x=atom[0], y=atom[1], z=atom[2])
            pdbFile.write(nextLine)
                    
        pdbFile.write("\nTER\n")
        
        for j in range(0,len(self.edges)):
            for i in range(0,len(self.edges[j])):
                if self.edges[j][i] == 1 and i<j:
                    nextLine = DNAPolyhedron.connectTemplate.format(vert1=i, vert2=j)
                    pdbFile.write(nextLine)

        pdbFile.close()

    #The polyhedron instance saves itself for future use in [filename]
    #Filename should be in the form "name.txt"
    def savePoly(self, filename):
        polyFile = open(filename, "wb")
        pickle.dump(self, polyFile)
        polyFile.close()

    #Static method that loads a previously saved polyhedron file.
    #Call as follows: newPoly = DNAPolyhedron.loadPoly("name.txt")
    def loadPoly(filename):
        polyFile = open(filename, "rb")
        returnPoly = pickle.load(polyFile)
        polyFile.close()
        return returnPoly

    #Deprecated method
    #Given a rotational matrix and a translation vector, moves a polyhedron.
    #Meeant to be used by oter methods, and has been replaced by rotateSelf and translateSelf. 
    def moveSelf(self, rotationMatrix, translationVector):
        newVertLoc = np.transpose(np.matmul(rotationMatrix, np.transpose(self.vertices)))#double transpose, column vectors
        #[print(point, translationVector, point+translationVector) for point in newVertLoc]
        newVertLoc = tuple([tuple(point + translationVector) for point in newVertLoc])
        self.vertices = newVertLoc
        #print(np.transpose([point[0] for point in self.bindingSites]))
        newSites = []
        for point in self.bindingSites:
            pointRotated = tuple(np.matmul(rotationMatrix, point[0]))
            pointMoved = [a+b for a,b in zip(pointRotated, translationVector)]
            directionRotated = tuple(np.matmul(rotationMatrix, point[3]))
            newSites.append(((tuple(pointMoved),) + point[1:3] + (directionRotated,)),)  
        self.bindingSites = tuple(newSites)

    #Given a rotational matrix, rotates the coordinates of a polyhedron, its binding sites and
    #the direction vectors for its binding sites.
    #rotateSelf is used because moveSelf couldn't be used to translate a new polyhedron relative 
    #to the binding site of the one it was connecting to.
    def rotateSelf(self, rotationMatrix):
        newVertLoc = np.transpose(np.matmul(rotationMatrix, np.transpose(self.vertices)))#double transpose, column vectors
        newVertLoc = tuple([tuple(point) for point in newVertLoc])
        self.vertices = newVertLoc
        #print(np.transpose([point[0] for point in self.bindingSites]))
        newSites = []
        for point in self.bindingSites:
            pointRotated = tuple(np.matmul(rotationMatrix, point[0]))
            directionRotated = tuple(np.matmul(rotationMatrix, point[3]))
            newSites.append((tuple(pointRotated), point[1], point[2], directionRotated),)
            self.bindingSites = tuple(newSites)
    
    #Given a translation vector, translates the vertices and binding sites of a polyhedron.
    #Replaced moveSelf (see rotateSelf documentation).
    def translateSelf(self, translationVector):
        newVerts = []
        for vert in self.vertices:
            vertMoved = [a+b for a,b in zip(vert, translationVector)]
            newVerts.append(tuple(vertMoved))
        self.vertices = tuple(newVerts)
        
        newSites = []
        for point in self.bindingSites:
            pointMoved = [a+b for a,b in zip(point[0], translationVector)]
            newSites.append(((tuple(pointMoved),) + point[1:]),)
        self.bindingSites = tuple(newSites)
    
    ###STATIC METHODS###
    #IMPORTANT NOTE: This program is written in python 3.

    #Returns the matrix to align poly2 such that it connects to poly1 by connector
    def placePolyhedronMatrix(poly1, site1index, poly2, site2index, twist):
        #twist = 0

        site1 = poly1.bindingSites[site1index]
        vert1a = poly1.vertices[site1[2][1]]
        vert1b = poly1.vertices[site1[2][0]]
        edge1 = [i-j for i,j in zip(vert1a, vert1b)]
        targetVector = [-i for i in site1[3]]
        
        site2 = poly2.bindingSites[site2index]
        vert2a = poly2.vertices[site2[2][1]]
        vert2b = poly2.vertices[site2[2][0]]
        edge2 = [i-j for i,j in zip(vert2a, vert2b)]
        dir2 = site2[3]

        if twist != 0:
            edge2 = np.matmul(shiftPerpVectors.rotateVectorDegrees(dir2,twist),edge2)
        
        return shiftPerpVectors.shiftPerpVectors(dir2, edge2, targetVector, edge1)

    #Uses PlacePolyhedronMatrix to actually place poly2 on the end of the connector to poly1
    def placePoly(poly1, site1index, poly2, site2index, connector):
        matrix = DNAPolyhedron.placePolyhedronMatrix(poly1, site1index, poly2, site2index, connector.twist)
        direction = shiftPerpVectors.normify(poly1.bindingSites[site1index][3])
        poly2.rotateSelf(matrix)
        connectorVector = [i*connector.length for i in direction] 
        site1loc = poly1.bindingSites[site1index][0]
        site2loc = poly2.bindingSites[site2index][0]
        translateVector = [i-j+k for i,j,k in zip(site1loc, site2loc, connectorVector)]
        poly2.translateSelf(translateVector)


#The following 3 lines of code can be used to iterate a simple set of 6-binding-site tetrahedra.
'''
A = DNAPolyhedron.loadPoly("fullTetra.txt")
connector = ProteinConnector("A","A",20,0)
P,D = iteratePoly(5, (A,), (connector,))
'''
#This is the iteratePoly method. It is used to procedurally generate a network of polyhedra connected by the
#given connectors. The basic loop is as follows:
# 0. From among the given types of polyhedra, choose a starter polyhedron.
# 1. Choose a binding site on that random polyhedron, and a connector, and see if either end of the connector
#    matches the site. If it does, continue; if it doesn't, repeat.
# 2. Choose a random polyhedron type. If any of its binding sites match the open end of the new connector,
#    move on. Otherwise, repeat until a poly that matches is found.
# 3. Check if attaching the new polyhedron would cause a collision with an existing polyhedron or connector.
#    If it would, delete it. Otherwise, attach it and add it to a list.
# 4. Return to step 1 and repeat this process for the provided number of iterations.
#
#The method returns 2 lists, one of polyhedra and one of paired points. These can be messed with in python,
#but are mostly meant to be passed as the first two arguments of the massPDB method, which allows them to be
#opened all at once in PyMol using loadall (see top of this file).
import random
def iteratePoly(iterations, polyTypes, connectorTypes):
    openPolies = [] # NOTE: Currently, openPolies is actually ALL polies, not just the ones that have open sites
    firstPoly = random.choice(polyTypes)
    openPolies.append([firstPoly, list(range(len(firstPoly.bindingSites)))])
    allPolies = []
    allPolies.append(firstPoly)
    allConnectors = [] #keeps track of protein bonds between polies
    allCentroids = [] #keeps track of the centroids of each created polyhedron
    allMaxDists = [] #keeps track of the max distance from each poly's centroid to vertex

    newCent,newMaxDist = findCentroidAndMaxDist(firstPoly)
    allCentroids.append(newCent)
    allMaxDists.append(newMaxDist)
    for it in range(iterations):
        flag1=True
        stopInfinite=0
        newConnector=None
        whichSite=None
        nextPolySite=None #Used in the second while loop
        nextPoly=None #Defined here to be used to orient the new poly
        nextSiteIndex=None #Defined here to be used to orient the new poly
        nextPolyIndex=None 
        while(flag1):
            stopInfinite+=1
            if stopInfinite>100:
                raise ValueError("Can't find a connector that fits to any open site! Open polies:", openPolies)
            
            nextPolyIndex = random.randrange(len(openPolies))
            nextPoly = openPolies[nextPolyIndex][0]
            nextSiteIndex = random.choice(openPolies[nextPolyIndex][1])
            #print(openPolies, nextPoly, nextSiteIndex)
            newConnector = random.choice(connectorTypes)
            if newConnector.site1 == nextPoly.bindingSites[nextSiteIndex][1]:
                flag1=False
                nextPolySite = newConnector.site2
            elif newConnector.site2 == nextPoly.bindingSites[nextSiteIndex][1]:
                flag1=False
                nextPolySite = newConnector.site1
        
        flag2=True
        stopInfinite=0
        while(flag2):
            stopInfinite+=1
            if stopInfinite>100:
                raise ValueError("Can't find a poly that fits to open connector! Open connector site:", nextPolySite)

            newPoly = random.choice(polyTypes)

            newPolySiteIndex = 0
            foundMatch = False
            while newPolySiteIndex<len(newPoly.bindingSites) and not foundMatch:
                if (newPoly.bindingSites[newPolySiteIndex][1] == nextPolySite):
                    foundMatch = True
                    #print(newPolySiteIndex, nextPolySite, "success")
                    #print(newPoly.bindingSites[newPolySiteIndex][1],nextPolySite)
                    #print(newPoly.bindingSites[newPolySiteIndex][1]==nextPolySite)
                else:
                    #print(newPolySiteIndex, nextPolySite, "fail")
                    #print(newPoly.bindingSites[newPolySiteIndex][1],nextPolySite)
                    #print(newPoly.bindingSites[newPolySiteIndex][1]==nextPolySite)
                    newPolySiteIndex += 1

            if foundMatch:
                flag2 = False
                newPoly = copy.deepcopy(newPoly)
                DNAPolyhedron.placePoly(nextPoly, nextSiteIndex, newPoly, newPolySiteIndex, newConnector)

                #Collision detection code block:
                polyFail = False
                newCent,newMaxDist = findCentroidAndMaxDist(newPoly)
                connectorsHit = []
                for i in range(len(allConnectors)):
                    if distPointToSegment(newCent, allConnectors[i][0], allConnectors[i][1])<newMaxDist:
                        connectorsHit.append(i)
                poliesHit = []                        
                for i in range(len(allPolies)):
                    if distBetween(newCent, allCentroids[i])<newMaxDist+allMaxDists[i]:
                        #This collision checking is fairly sensitive. If the connector
                        #is too short, a possible collision will always be detected and no
                        #network will form.
                        #print(it, allCentroids[i])
                        poliesHit.append(i)

                if len(connectorsHit) > 0 or len(poliesHit) > 0:
                    polyFail = True
                #End collision detection (Doesn't yet merge)

                if not polyFail:
                    openPolies.append([newPoly,list(range(len(firstPoly.bindingSites)))])
                    #Remove the site that was just used from the new poly:
                    #print(openPolies[len(openPolies)-1][1], newPolySiteIndex)
                    openPolies[len(openPolies)-1][1].remove(newPolySiteIndex)
                    allPolies.append(newPoly)

                    #print(nextPoly.bindingSites[nextSiteIndex], newPoly.bindingSites[newPolySiteIndex])
                    
                    openPolies[nextPolyIndex][1].remove(nextSiteIndex) #The site used is no longer an open site
                    if len(openPolies[nextPolyIndex][1]) == 0:
                        openPolies.pop(nextPolyIndex) #If the poly has no more open sites, remove it from openPolies
                    #print(openPolies)

                    allConnectors.append( (nextPoly.bindingSites[nextSiteIndex][0], newPoly.bindingSites[newPolySiteIndex][0]) )
                    allCentroids.append(newCent)
                    allMaxDists.append(newMaxDist)
             
    return allPolies, allConnectors

#function that should be called from pymol and will turn a list of polies into pymol objects
#polyList is list of polies
#folderName is name of folder to put new poly objects
import os
def massPDB(polyList, connectorsList, folderName):
    os.makedirs(folderName)
    for i in range(len(polyList)):
        polyList[i].createPDB(folderName+"/poly"+str(i)+".pdb")
    for j in range(len(connectorsList)):
        createLinePDB(connectorsList[j], folderName+"/prot"+str(j)+".pdb")

    #if(inPymol):
    #    cmd.loadAll(foldername+"/*.pdb")
                    
#Creates a pdb file for a line between a pair of points. Used by massPDB to create connectors. 
def createLinePDB(pair, filename):
    pdbFile = open(filename, "w")
    
    curName = "VT"
    resName = "TH1"
    chainID = "A"
    for i in range(0,2):
        atom = pair[i]
        if atom[0]<1000 and atom[1]<1000 and atom[2]<1000:
            nextLine = DNAPolyhedron.atomTemplate1.format(num=i, name=curName, resName=resName, chainID=chainID, resSeq=i, x=atom[0], y=atom[1], z=atom[2])
        else:
            nextLine = DNAPolyhedron.atomTemplate2.format(num=i, name=curName, resName=resName, chainID=chainID, resSeq=i, x=atom[0], y=atom[1], z=atom[2])
        pdbFile.write(nextLine)
    pdbFile.write("\nTER\n")
    pdbFile.write("CONECT    0    1")
    pdbFile.close()

#Used for collision detection purposes, finds the centroid of a polyhedron and the maximum distance between centroid
#and vertex.
def findCentroidAndMaxDist(poly):
    xTotal = 0
    yTotal = 0
    zTotal = 0
    for vertex in poly.vertices:
        xTotal += vertex[0]
        yTotal += vertex[1]
        zTotal += vertex[2]
    
    vertNum = len(poly.vertices)
    polyCentroid = (xTotal/vertNum, yTotal/vertNum, zTotal/vertNum)
    dist = 0
    minDist = None
    for vertex in poly.vertices:
        dist = distBetween(vertex, polyCentroid)

    if minDist == None or dist < minDist:
        minDist = dist
    
    return polyCentroid, minDist

#Returns the distance between two points.
def distBetween(point1, point2):
    return ((point1[0]-point2[0])**2
            +(point1[1]-point2[1])**2
            +(point1[2]-point2[2])**2)**(1/2)

#Finds the distance from a point to a line segment defined by two points
#P0 and P1 define the segment, P is the point.
#http://geomalgorithms.com/a02-_lines.html describes why this algorithm works
# (See section "Distance of a Point to a Ray or Segment")
#Basically, it uses dot product of vectors from P0 to P and from P0 to P1
# to determine if the closest point on the segment's associated line to the point
# is on the segment or past one of the endpoints.
def distPointToSegment(P, P0, P1):
    P0_P1 = (P1[0]-P0[0], P1[1]-P0[1], P1[2]-P0[2])
    P0_P = (P[0]-P0[0], P[1]-P0[1], P[2]-P0[2]) 

    #The following statement says that if the dot product of the line segment and endpoint 0
    #to to the point is less than 0, then the distance is simply the distance to endpoint 0.
    #This is because the dot product is |A||B|cos(x), which is only negative if cos(x) < 0,
    #which means abs(x)>90 degrees. If this is the case, then geometrically it is simple to
    #see that the point P does not lie perpendicularly above the line segment.
    c1 = shiftPerpVectors.dotProd(P0_P, P0_P1)
    if c1 <= 0:
        return distBetween(P, P0)
    #This next statement says that if the dot product of the vector from P0 to P1 (aka the line
    #segment) with itself is smaller than the dot product from the previous statement, then
    #the point is not perpendicularly above the line segment. This is because the dot product of
    #the segment with itself is its length squared, while the other dot product is the length of
    #the line segment, times the length from P0 to P, times cosine x. If we divide both sides by
    #the length of the segment, we see that we are comparing the length of the segment and the
    #length of the vector from P0 to P in the direction of the segment (due to the cosine term). 
    c2 = shiftPerpVectors.dotProd(P0_P1, P0_P1)
    if c1 >= c2:
        return distBetween(P, P1)
    #Using the same logic as the previous statement, this finds the distance when the point is
    #perpendicularly above the line segment. c1/c2 is |P0_P|*cos(x)/|P0_P1|, which hen multiplied
    #by vector P0_P1 becomes the point in the P0_P1 direction that is the cos(x) component of the
    #vector P0_P. Aka, the closest point from P to the line segment.
    b = c1/c2
    Pb = [P0[val]+b*P0_P1[val] for val in range(len(P0_P1))]
    return distBetween(P, Pb)

    
#The ProteinConnector class represents a connector between two DNA polyhedra and holds 4 values:
# - site1: The string identifier ID of site 1.
# - site2: The string identifier ID of site 2.
# - length: The length of the protein connector.
# - twist: the twist angle of the connector
class ProteinConnector():
    def __init__(self, newSite1, newSite2, newLength, newTwistAngle):
        self.site1 = newSite1
        self.site2 = newSite2
        self.length = newLength
        self.twist = newTwistAngle

    #The polyhedron instance saves itself for future use in [filename]
    #Filename should be in the form "name.txt"
    def saveConnector(self, filename):
        connectorFile = open(filename, "wb")
        pickle.dump(self, connectorFile)
        connectorFile.close()

    #Static method that loads a previously saved polyhedron file.
    #Call as follows: newPoly = DNAPolyhedron.loadPoly("name.txt")
    def loadConnector(filename):
        connectorFile = open(filename, "rb")
        returnConnector = pickle.load(connectorFile)
        connectorFile.close()
        return returnConnector

