#!/usr/bin/env python3

import copy
import time
import math
import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt



# inv-pendulum constants
l = 3.0             # length of rod
m = 0.4             # bob weight
M = 1.0             # cart weight   
g = 9.8             # gravity constant
# initial state 
x0 = np.array([ [0.0],
                [0.0],
                [1.0],
                [0.0]  ])
# model param
Q = np.diag([0.0, 1.0, 1.0, 0.0])
R = np.diag([0.01])
nx = 4              # number of state [x, xdot, theta, thetadot]
nu = 1              # number of input [u] :: force
T = 50              # Time Horizon
dt = 0.1            # time discretization

animate = True




def state_space_matrix():
    '''
    input : dt : fraction of time to discretize the time horizon

    output : matrix A and B (for the state space model)
    '''

    global dt
    
    # parameters
    A = np.array([  [1.0, dt, 0.0, 0.0],
                    [0.0, 1.0, -(m*g)*dt/M, 0.0],
                    [0.0, 0.0, 1.0, dt],
                    [0.0, 0.0, g*dt*(M+m)/(l*M), 1.0]  ])

    B = np.array([  [0.0],
                    [dt / M],
                    [0.0],
                    [-dt / (l*M)]  ])


    print(f' {A} \n {B} ')
    return A, B


def step(x, u):

    A, B = state_space_matrix()
    # update 
    x = np.dot(A, x) + np.dot(B, u)
    return x


    print(x)


def controller(x0):
    
    x = cp.Variable((nx, T+1))
    u = cp.Variable((nu, T))

    A, B = state_space_matrix()

    cost_fn = 0.0
    constr = []
    constr += [x[:, 0] == x0[:,0]]

    # mpc loop
    for t in range(T):
        cost_fn += cp.quad_form(u[:, t], R)
        cost_fn += cp.quad_form(x[:, t+1], Q)

        constr += [x[:, t+1] == A @ x[:, t] + B @ u[:,t]]

    print(constr)

    prob = cp.Problem(cp.Minimize(cost_fn), constr)

    start = time.time()
    prob.solve()
    stop = time.time()
    print(f' Calc Time : {stop-start:.2f} sec ')


    if prob.status == cp.OPTIMAL:
        ox = x.value[0][0]
        dx = x.value[1][0]
        theta = x.value[2][0]
        dtheta = x.value[3][0]

        ou = u.value[0][0]

    return ox, dx, theta, dtheta, ou



def draw_cart(xt, theta):
    cart_w = 2.0
    cart_h = 1.0
    radius = 0.25

    cx = np.matrix([-cart_w / 2.0, cart_w / 2.0, cart_w /
                    2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.matrix([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.matrix([0.0, l * math.sin(theta)])
    bx += xt
    by = np.matrix([cart_h, l * math.cos(theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = [radius * math.cos(a) for a in angles]
    oy = [radius * math.sin(a) for a in angles]

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + float(bx[0, -1])
    wy = np.copy(oy) + float(by[0, -1])

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title("x:" + str(round(xt, 2)) + ",theta:" +
              str(round(math.degrees(theta), 2)))

    plt.axis("equal")



def flatten(a):
    return np.array(a).flatten()






print(f' Initial State : {x0} ')
x = copy.deepcopy(x0)

for i in range(50):

    # simulation loop
    ox, dx, theta, dtheta, u = controller(x)
    x = step(x, u)
        
    if animate:
        plt.clf()
        px = float(x[0])
        theta = float(x[2])
        
        draw_cart(px, theta)
        plt.ylim([0.0, 5.0])
        plt.pause(1) if i == 0 else plt.pause(1/30)



print('\n\n\n')
print(f' Final state : {x} ')
print(f' Final input : {u} ')


