#!/usr/bin/env python

import pygame
from pygame.locals import *
import numpy as np

# These are the status a person can take
HEALTHY, SICK, CURED, DEAD = range(4)

# Background of the playground
BACKGROUND_COLOR = 200,200,200
# Background of the graph
GRAPH_BACKGROUND_COLOR = 255,255,255
# Colors for the graph
GRAPH_SICK_COLOR = 255,0,0
GRAPH_CURED_COLOR = 0,255,0
GRAPH_DEAD_COLOR = 0,0,0
GRAPH_HEALTHY_COLOR = 255,204,0
GRAPH_UTI_CAPACITY = 0, 0, 255

# Chance of dying from the disease
DIE_CHANCE = 0.05

# This adjusts the width of the graph
# (the smaller, the wider)
TIME_SCALE = 0.5

# Modulus of the velocity of each moving person
VELOCITY = 2.0

# Modify here to simulate social distancing
# (this tells how many people will be stationary,
#  for instance, 0.7 means 70% will not move)
#SOCIAL_DISTANCING = 0.0
SOCIAL_DISTANCING = 0.8
SOCIAL_DISTANCING_MASS = 1000.0

# Size of the population
POPULATION = 100

# Size of population Brasil and number of UTI
POPULATION_BRAZIL = 209e6
LEITOS_UTI_BRAZIL = 25e3

# Percentage of people who have to go to UTI (data of China cases)
PERCENTAGE_GO_UTI = 0.05

UTI_POPULATION_PERCENTAGE = POPULATION*(LEITOS_UTI_BRAZIL/(POPULATION_BRAZIL*PERCENTAGE_GO_UTI))

class Person():
    ''' Class to hold a person status, position and velocity
    '''
    status = HEALTHY
    def __init__(self, pos, vel, sd):
        self.pos = pos
        self.vel = vel
        self.social_distancing = sd
        self.counter = 0
        if self.social_distancing:
            self.mass = SOCIAL_DISTANCING_MASS
        else:
            self.mass = 1.0

class Sim():
    ''' This is the simulation class. Most of the code
        is in here
    '''
    def __init__(self,pop=POPULATION,w=1000,h=400,g=300,r=13):
        # Simulation Parameters
        self.width = w
        self.height = h
        self.radius = r
        self.graph = g
        # Generate the initial population
        self.createPopulation(pop)
        # Init PyGame stuff
        pygame.init()
        # Create the window
        self.screen = pygame.display.set_mode((self.width,self.height+self.graph))
        # Load the images
        self.healthy_img = pygame.image.load('healthy.png')
        self.sick_img = pygame.image.load('sick.png')
        self.cured_img = pygame.image.load('cured.png')
        self.dead_img = pygame.image.load('dead.png')
        # Initialize counters
        self.sick = self.cured = self.dead = 0
        self.counter = 0
        # Simulation starts paused. Press space to start.
        self.start = False
    
    def checkPopulationCollision(self, person):
        # Check for colisions on the initial population
        for other in self.people:
            if other != person:
                if self.checkCollision(person, other):
                    return True
        return False

    def checkPopulationBoundaries(self):
        # Check the "walls" of the playgraound
        for person in self.people:
            if (person.pos[0] < self.radius and person.vel[0] < 0) \
              or (person.pos[0] > self.width - self.radius and person.vel[0] > 0):
                person.vel[0] = -person.vel[0]
                person.pos[0] += person.vel[0]
            if (person.pos[1] < self.radius and person.vel[1] < 0) \
              or (person.pos[1] > self.height - self.radius and person.vel[1]) > 0:
                person.vel[1] = -person.vel[1]
                person.pos[1] += person.vel[1]

    def checkCollision(self, person1, person2):
        # Verify if one person touches another
        if person1 != person2:
            dist = np.linalg.norm(person1.pos - person2.pos)
            if dist <= 2.0*self.radius:
                return True
        return False

    def randomVelocity(self):
        # Generates an initial random velocity
        n = 0
        while n == 0:
            v = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
            n = np.linalg.norm(v)
        return VELOCITY * v / n

    def createPopulation(self, pop):
        # Generates the initial population
        self.people = []
        while len(self.people) < pop:
            # Uniform distribution of positions
            pos = np.array([np.random.uniform(self.radius, self.width-self.radius), \
                            np.random.uniform(self.radius, self.height-self.radius)])
            # Uniform distribution of velocities
            vel = self.randomVelocity()
            # Make sure some people have the necessary social distancing
            social_distancing = False
            if np.random.rand() < SOCIAL_DISTANCING:
                vel = np.array([0.0,0.0])
                social_distancing = True
            # Creates a person
            p = Person(pos,vel,social_distancing)
            # Avoid creating people on top of each other
            if not self.checkPopulationCollision(p):
                self.people.append(p)
        # Randomly chooses someone to be sick
        np.random.choice(self.people).status = SICK

    def drawPopulation(self):
        # Draws the population on the screen
        pygame.draw.rect(self.screen,BACKGROUND_COLOR,(0,0,self.width,self.height))
        for person in self.people:
            pos = person.pos - np.array([self.radius, self.radius])
            x = int(pos[0])
            y = int(pos[1])
            if person.status == HEALTHY:
                self.screen.blit(self.healthy_img, (x,y))
            elif person.status == SICK:
                self.screen.blit(self.sick_img, (x,y))
            elif person.status == CURED:
                self.screen.blit(self.cured_img, (x,y))
            elif person.status == DEAD:
                self.screen.blit(self.dead_img, (x,y))
            # Draw eyes (why not!?)
            if person.status != DEAD:
                leyex = int(person.pos[0]-1-4+0.5*person.vel[0])
                reyex = int(person.pos[0]-1+4+0.5*person.vel[0])
                eyey = int(person.pos[1]-3+0.5*person.vel[1])
                pygame.draw.circle(self.screen, (0,0,0), (leyex,eyey), 1)
                pygame.draw.circle(self.screen, (0,0,0), (reyex,eyey), 1)

    def drawGraph(self):
        # This will plot the cumulative cases in a graph at the bottom of the sim window
        counter = int(self.counter*TIME_SCALE)
        pygame.draw.rect(self.screen,GRAPH_BACKGROUND_COLOR,\
                         (counter,self.height,self.width-counter,self.graph))
        sick = (self.sick / float(len(self.people))) * self.graph
        pygame.draw.line(self.screen,GRAPH_SICK_COLOR,(counter,self.height + self.graph),\
                         (counter, self.height + self.graph - sick))
        dead = (self.dead / float(len(self.people))) * self.graph
        pygame.draw.line(self.screen,GRAPH_DEAD_COLOR,(counter,self.height),\
                         (counter, self.height + dead))
        cured = (self.cured / float(len(self.people))) * self.graph
        pygame.draw.line(self.screen,GRAPH_CURED_COLOR,(counter,self.height + dead + cured),\
                         (counter, self.height + dead))
        pygame.draw.line(self.screen,GRAPH_HEALTHY_COLOR,(counter,self.height + self.graph - sick),\
                         (counter, self.height + dead + cured))
        pygame.draw.line(self.screen, GRAPH_UTI_CAPACITY,\
                         (counter, self.height + self.graph*(1-UTI_POPULATION_PERCENTAGE)),\
                         (counter, self.height + self.graph*(1-UTI_POPULATION_PERCENTAGE)))

    def update(self):
        # Main function to update all the positions and "infect" people
        for i in range(len(self.people)):
            # If someone is not dead, it will move
            if self.people[i].status != DEAD:
                # Update position
                if not self.people[i].social_distancing:
                    self.people[i].pos += self.people[i].vel
                # If the person is sick, the counter will increment
                if self.people[i].status == SICK:
                    self.people[i].counter += 1
                    # When it reaches 300, the person heals or dies
                    if self.people[i].counter == 300:
                        self.sick -= 1
                        if np.random.rand() > DIE_CHANCE:
                            self.people[i].status = CURED
                            self.cured += 1
                        else:
                            self.people[i].status = DEAD
                            self.dead += 1
                # Check collisions (only for moving people)
                if not self.people[i].social_distancing:
                    for j in range(0,len(self.people)):
                        # Can only collide with someone not dead
                        if self.people[j].status != DEAD and i != j:
                            # If there is a colision
                            if self.checkCollision(self.people[i], self.people[j]):
                                # Check if colition is with a socially distanced person
                                if self.people[j].social_distancing:
                                    # If the person is socially distanced, only the "colider" changes direction
                                    v1 = self.people[i].vel
                                    v2 = self.people[j].vel
                                    x1 = self.people[i].pos
                                    x2 = self.people[j].pos
                                    v1_ = v1 - 2.0 * (x1-x2) * np.dot(v1-v2,x1-x2) / 2.0*self.radius
                                    if np.linalg.norm(v1_) > 0.0:
                                        v1_ = 2.0 * v1_ / np.linalg.norm(v1_)
                                    self.people[i].vel = v1_
                                    self.people[i].pos += self.people[i].vel
                                else:
                                    # Elastic colision, agents exchange their velocities (if not stationary)
                                    self.people[i].vel, self.people[j].vel = self.people[j].vel, self.people[i].vel
                                # If one of the two was sick, the other will be sick as well
                                if self.people[i].status == SICK:
                                    if self.people[j].status == HEALTHY:
                                        self.people[j].status = SICK
                                        self.people[j].counter = 0
                                        self.sick += 1
                                elif self.people[i].status == HEALTHY:
                                    if self.people[j].status == SICK:
                                        self.people[i].status = SICK
                                        self.people[i].counter = 0
                                        self.sick += 1
        # Do not let people wander outside the bounding box of the window
        self.checkPopulationBoundaries()
        # Simulation counter
        self.counter += 1

    def run(self):
        # This functin runs the simulation
        quit = False
        while not quit:
            # Checks if the spacebar was pressed already or not
            # (after pressing spacebar, self.start will be True)
            if self.start:
                # Update simulation
                self.update()
                # Draws agents
                self.drawPopulation()
                # Draw graph
                self.drawGraph()
                # Refreshes the actual image on the screen
                pygame.display.flip()
            # Check for quit or spacebar presses
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
                elif event.type == pygame.KEYUP:
                    if event.key == K_SPACE:
                        self.start = True

# Main function
if __name__ == "__main__":
    s = Sim()
    s.run()

