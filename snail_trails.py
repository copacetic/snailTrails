from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

import numpy as np

from time import sleep
import math
import random
import sys

"""A Field is composed of Places which are occupied by Animals"""

MAX_ADJACENT_POSITIONS = 8
SIZE = 500
#types = [Turtle, Rabbit, Monkey]
possible_vectors = [(1,1), (1,0), (0,1), (-1,0), (0, -1), (-1,1), (1,-1), (-1,-1)]
DIM_X, DIM_Y = 600.0, 600.0
X_STEP_SIZE = DIM_X / SIZE
Y_STEP_SIZE = DIM_Y / SIZE

# same sized tuples
def add_tuple_items(a, b):
    if (len(a) != len(b)) or (type(a) != tuple) or (type(b) != tuple):
        raise TypeError
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i]) 
    return tuple(c)

def in_bounds(place):
    return True if (in_range(place[0]) and in_range(place[1])) else False

def in_range(val):
    return True if val >= 0 and val < SIZE else False

class Place:
    def __init__(self):
        self.occupied = False
        self.scent = None
        self.occupant = None
        
    def set_scent(self, smellValue):
        self.scent = smellValue

    def occupy(self, animal):
        self.occupant = animal
        self.occupied = True

    def clear_occupant(self):
        self.occupant = None
        self.occupied = False       
        
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class Animal:
    id_num = 1
    def __init__(self, _field, _pos = (0,0)):
        self.field = _field
        self.id = Animal.id_num
        Animal.id_num += 1
        self.pos = _pos
        self.field[self.pos[0]][self.pos[1]].occupy(self)
        self.color = (random.random(),random.random(),random.random())
        Animal.running = False
        Animal.notMoved = 0

    # gets a random direction and moves there
    def move(self, vectorField):
        old_pos = self.pos
        oldX, oldY = old_pos
        all_places = [add_tuple_items(self.pos, possible_vectors[i]) for i in range(MAX_ADJACENT_POSITIONS)]
        possible = [place for place in all_places if (in_bounds(place) and not self.field[place[0]][place[1]].occupied)]
        #if possible == []:
        #    Animal.notMoved += 1
        #    return
        #print "Vector field: ", vectorField[oldY][oldX]
        #print "Old pos: ", old_pos
        #print "New pos: ", add_tuple_items(self.pos, vectorField[oldY][oldX])
        #print
        if add_tuple_items(self.pos, vectorField[oldY][oldX]) in possible:
            self.pos = add_tuple_items(self.pos, vectorField[oldY][oldX])
            #possible[random.randint(0, len(possible) - 1)]
            self.field[self.pos[0]][self.pos[1]].occupy(self)
            self.field[old_pos[0]][old_pos[1]].clear_occupant()
        else:
            Animal.notMoved += 1
#    def __repr__(self):
#        return str(self.id)
    
def initGL():
    print("Initializing opengl...")
    glClearColor(1.0,1.0,1.0,0.0)
    glColor3f(0.0,0.0, 0.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluOrtho2D(0.0,int(DIM_X),0.0,(DIM_Y))

def draw_field(field):
    to_draw = []

    glClear(GL_COLOR_BUFFER_BIT)
    for row in range(len(field)):
        for column in range(len(field[row])):
            currPlace = field[row][column]
            if currPlace.occupied == True:
                # glColor3f(*currPlace.occupant.color)
                low_x = column*X_STEP_SIZE
                low_y = row*Y_STEP_SIZE
                high_x = low_x + X_STEP_SIZE
                high_y = low_y + Y_STEP_SIZE
                a_square = [
                  [low_x, low_y],
                  [high_x, low_y],
                  [high_x, high_y],
                  [low_x, low_y],
                  [high_x, high_y],
                  [low_x, high_y]
                ]
                to_draw.extend(a_square)
    print to_draw
    the_vbo = vbo.VBO(np.array(to_draw, 'i'))
    the_vbo.bind();

    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(2, GL_INT, 0, the_vbo)
    glDrawArrays(GL_TRIANGLES, 0, len(to_draw))
    glDisableClientState(GL_VERTEX_ARRAY)
    glutSwapBuffers();
#(0, DIM_Y)  (DIM_X, DIM_Y)
#|---|
#|   |
#|___|
#(0,0)  (DIM_X,0)

def gen_vector_field(seed):
    randomVec = possible_vectors[random.randint(0, len(possible_vectors)-1)]
    vectorField = [[randomVec for i in range(SIZE)] for j in range(SIZE)]
    a = 10
    b = .1
    
    getNorm = lambda x,y: math.sqrt(x**2 + y**2)
    for t in range(0,5000, 1):
        t = float(t)/2
        rad = math.radians(t)
        coeff = a*math.exp(b*rad)
        #coeff = 299
        x = 200*math.cos(3*rad) + DIM_X/2 # bottom left corner is 0, 0
        y = 300*math.sin(5*rad) + DIM_Y/2 # so we center it
        nextRad = math.radians(t+.5)
        nextCoeff = a*math.exp(b*(nextRad))
        #nextCoeff = 299
        nextX = 200*math.cos(3*nextRad) + DIM_X/2 
        nextY = 300*math.sin(5*nextRad) + DIM_Y/2 
        
        dirX, dirY = nextX - x, nextY - y

        bestVec = None
        bestCosSim = -1
        for vec in possible_vectors:
            vecX, vecY =  vec
            # get cosine similarity
            numerator = dirX * vecX + dirY * vecY 
            denominator = getNorm(dirX, dirY) * getNorm(vecX, vecY)
            denominator = 1 if denominator == 0 else denominator
            cosSim = 1 - (math.acos(numerator / denominator)/math.pi)
            if cosSim >= bestCosSim:
                bestVec = vec
                bestCosSim = cosSim
        columnOffset = int(x / float(X_STEP_SIZE))
        rowOffset = int(y / float(Y_STEP_SIZE))
        if in_bounds((rowOffset, columnOffset)):
            bestVec = (0,0)
            vectorField[rowOffset][columnOffset] = bestVec
    return vectorField

    #vectorField[rowOffset-1][columnOffset] = bestVec
            #vectorField[rowOffset][columnOffset-1] = bestVec
        #else:
            #print "Out of bounds!"
    #print vectorField
    #[[vec for vec in vectorRow if vec != (0,0)] for vectorRow in vectorField]
    #import sys
    #sys.exit(0)
    #vec2 = possible_vectors[random.randint(0, len(possible_vectors) - 1)]
    #vec = possible_vectors[random.randint(0, len(possible_vectors) - 1)]

def display ():
    critters, field = init_simulation()
    numRounds = 0 
    numVFsGenerated = 0
    vF = gen_vector_field(numRounds)
    toRemove = list()
    delUnmoved = False
    while True:
        critterCount = 0
        for critter in critters:
            moved = critter.move(vF)
            if delUnmoved:
                if not moved and vF[critter.pos[1]][critter.pos[0]] != (0,0):
                    toRemove.append(critterCount)
            critterCount += 1
        print list(reversed(toRemove))
        for toRem in reversed(toRemove):
            field[critters[toRem].pos[0]][critters[toRem].pos[1]].clear_occupant()
            critters.pop(toRem)
        toRemove = []
        print "Animals not moved: ", Animal.notMoved
        print "Moving critters: ", abs(Animal.notMoved - len(critters))
        print "Boundary: ", len(critters)/20.0
        print
        if abs(Animal.notMoved - len(critters)) < len(critters)/100.0:
            numVFsGenerated += 1     
            vF = gen_vector_field(numRounds)
            if numVFsGenerated > 7:
                delUnmoved = True
        Animal.notMoved = 0
        draw_field(field)
        numRounds += 1
        sleep(.01)

def init_simulation():
    critters = []
    field = [[Place() for _ in range(SIZE)] for _ in range(SIZE)]
    for _ in range(20*SIZE):
        newAutomata = Animal(field, (random.randint(0,SIZE - 1), random.randint(0, SIZE - 1)))#types[random.randint(0,len(types) - 1)]()
        critters.append(newAutomata)
    return (critters, field)
    
if __name__ == '__main__':
    glutInit()
    glutInitWindowSize(int(DIM_X),int(DIM_Y))
    glutCreateWindow("Grid")
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutDisplayFunc(display)
    initGL()
    glutMainLoop()


