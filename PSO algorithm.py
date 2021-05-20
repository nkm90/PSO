# -------------------------------------------------------------
#      Visualization - PSO animated example
#      Implementation from Ignacio Cabrera based on the code from Jorge Jesus Pomares
# --------------------------------------------------------------
import numpy as np
import random as rand
import matplotlib.pyplot as plt

# 1ST PART: Initialise the Variables for the different parameters and setting up functions
n = 50  # Number of particles
dimensions = 2  # Number of different dimensions for positioning particles (x & y)
# Create the attributes that each particle will have
a = np.empty((dimensions, n))  # actual particle on calculation
v = np.empty((dimensions, n))  # Velocity
Pbest = np.empty((dimensions, n))  # Particle or personal best
Gbest = np.empty((1, 2))  # Global best
r = np.empty(n)  # Result after optimization


def pbest_update(Pbest, a, i):
    # Updating the Particle best from the actual best
    Pbest[0][i] = a[0][i]
    Pbest[1][i] = a[1][i]


def gbest_update(Gbest, Pbest, i):
    # Updating the Global best from the Personal best
    Gbest[0][0] = Pbest[0][i]
    Gbest[0][1] = Pbest[1][i]


def topIsBest(Gbest, Pbest):
    # Takes the value on top of the Personal best as the Global best
    Gbest[0][0] = Pbest[0][0]
    Gbest[0][1] = Pbest[1][0]


def sorting(Pbest, r, n):
    # Bubble sort method to sort Particle best results in position and optimization result, to keep the best result
    # (the lower obtained) first
    for i in range(1, n):
        for j in range(0, n - 1):
            if r[j] > r[j + 1]:
                # sorting the results obtained after optimization
                tempRes = r[j]
                r[j] = r[j + 1]
                r[j + 1] = tempRes

                # sorting the particle best position for x axis
                tempX = Pbest[0][j]
                Pbest[0][j] = Pbest[0][j + 1]
                Pbest[0][j + 1] = tempX
                # sorting the particle best position for y axis
                tempY = Pbest[1][j]
                Pbest[1][j] = Pbest[1][j + 1]
                Pbest[1][j + 1] = tempY


def velocity(n, a, Pbest, Gbest, v):
    # This method calculates the velocity for each particle. To represent a single particle, we generate a loop with
    # the number of particles as the limit(n) and a single particle is i on each iteration. Then the velocity is
    # calculated on each dimension, being v[0] the x axis and v[1] the y axis.
    # From the general formula of the velocity, I extracted the following parameters:
    # v(t+1) = wv + c1r1(Personal(t) - x(t)) + c2r2(Global(t) - x(t)
    # w constant = 0.5 (for balancing testing)
    # v = v[0][i], this is the actual value of the velocity before recalculation
    # c1 and c2 = 1.47 the constant number for incrementation
    # r1 and r2 = random numbers
    # Personal(t) = Pbest position
    # Global(t) = Gbest position
    # x(t) = the actual position of each particle on calculation
    for i in range(n):
        # update velocity on the X axis
        v[0][i] = 0.5 * v[0][i] + (Pbest[0][i] - a[0][i]) * rand.random() * 1.47 + (
                Gbest[0][0] - a[0][i]) * rand.random() * 1.47
        # update the actual particle x position by adding the results obtained from the velocity calculation
        a[0][i] = a[0][i] + v[0][i]
        # update velocity on the Y axis
        v[1][i] = 0.5 * v[1][i] + (Pbest[1][i] - a[1][i]) * rand.random() * 1.47 + (
                Gbest[0][1] - a[1][i]) * rand.random() * 1.47
        a[1][i] = a[1][i] + v[1][i]


def optimizer_function(x, y):
    # Rosenbrock function Optimization
    # function based on Rosenbrock with arguments from pySwarms. The parameters that has been passed on are a=y,
    # b=1 and c=0, so c is not added at the end.
    # f = 100 * ((y - (x ** 2)) ** 2) + ((1 - (x ** 2)) ** 2)

    # Ackley fucntion
    # Based on the general formula with the parameters a = 20, b = 0.2, c = 2pi
    TWO_PI = np.pi * 2.0  # Variable to represent double pi
    EXP = np.exp(1)  # variable to represent the exponent of 1 = 2.71828182846
    # f = (-20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) -
    #    np.exp(0.5 * (np.cos(TWO_PI * x) + np.cos(TWO_PI * y))) +
    #    EXP + 20)

    # Rastrigin function
    # From the basic function I have simplified it by introducing the elements (x, y, and -10) into their corresponding
    # places in the parenthesis. As a result, f= 20 + (x^2 - 10cos(double pi * x) + (y^2 - 10cos(double pi * x))
    f = 20 + (x ** 2 - 10 * np.cos(TWO_PI * x)) + (y ** 2 - 10 * np.cos(TWO_PI * y))
    return f


def plotting(ax, a, Gbest):
    # Plot lines, 1 with the actual particles position (line1) and with the global best (line2)
    line1 = ax.plot(a[0], a[1], 'r+')
    line2 = ax.plot(Gbest[0][0], Gbest[0][1], 'g*')
    # Set the limits on x and y axis of the plot space created
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    # Draw the results into a canvas to display the results obtained
    fig.canvas.draw()
    ax.clear()
    ax.grid(True)


# 2ND PART: Initialising all particles, sorting results and setting the ground for plotting them
# Initialising the attributes for each particle (i = X axis and j = Y axis)
for i in range(0, dimensions):
    for j in range(0, n):
        Pbest[i][j] = rand.randint(-20, 20)
        a[i][j] = Pbest[i][j]
        v[i][j] = 0

# filling up r with the results after evaluation with the optimization algorithm each particle results
for i in range(0, n):
    r[i] = optimizer_function(a[0][i], a[1][i])

# using the sorting method to sort each particle best results for Pbest and r
sorting(Pbest, r, n)
# The Particle situated on the top result is the best Global.
topIsBest(Gbest, Pbest)
# Initialising the plot figure to display the results
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)

# 3RD PART: Optimization loop
# Initial generation parameter to control the number of iterations during the optimization loop
generation = 0
while generation < 100:
    for i in range(n):
        # STEP 1: UPDATING PARTICLE BEST
        # Getting each particle best (Pbest) after optimization algorithm. In case, the new result is smaller than
        # the previous particle best, the new result is updated as the best result.
        if optimizer_function(a[0][i], a[1][i]) < optimizer_function(Pbest[0][i], Pbest[1][i]):
            pbest_update(Pbest, a, i)

        # STEP 2: UPDATING GLOBAL BEST
        # Checks that the result obtained after optimization is smaller than Global best (Gbest).
        # In case, the new Pbest will be stored as the Global best
        if optimizer_function(Pbest[0][i], Pbest[1][i]) < optimizer_function(Gbest[0][0], Gbest[0][1]):
            gbest_update(Gbest, Pbest, i)

        # STEP 3: UPDATING VELOCITY
        # Calculates velocity using the function
        velocity(n, a, Pbest, Gbest, v)
    # Increase the generation after calculations and display the results
    generation = generation + 1
    # DISPLAYS THE MESSAGE WITH RESULTS AND UPDATES THE GRAPHICS WITH NEW POSITIONS
    # Display the global best for each iterations on the log and plots it on the figure
    print('Generation: ' + str(generation) + ' - - - Gbest: ' + str(Gbest))
    plotting(ax, a, Gbest)
