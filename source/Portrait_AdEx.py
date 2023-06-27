# Dynamical Systems
# examples from:
# http://systems-sciences.uni-graz.at/etextbook/content.html

import numpy as np
import scipy
import scipy.linalg

import matplotlib
import matplotlib as ml
import matplotlib.pyplot as plt



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict,self).__init__(*args,**kwargs)
        self.__dict__ = self


def quiverplot( exe, f1, f2, p, limits, grid_step, ax ):
    # define a grid and compute direction at each point
    x = np.linspace( limits[0][0], limits[0][1], grid_step)
    y = np.linspace( limits[1][0], limits[1][1], grid_step)
    X1 , Y1  = np.meshgrid(x, y) # create grid
    DX1 = X1.copy()
    DY1 = Y1.copy()
    for j in range(len(y)):
        for i in range(len(x)):
            x1, y1 = exe( f1, f2, p, (x[i],y[j]), 0.01, 2) # compute 1 step growth
            # print(x1, y1)
            DX1[j][i] = x1[1]-x1[0]
            DY1[j][i] = y1[1]-y1[0]
    M = (np.hypot(DX1, DY1))
    plt.rcParams['image.cmap'] = 'RdPu' # rose
    # plt.rcParams['image.cmap'] = 'PuRd' # pink
    # plt.rcParams['image.cmap'] = 'Purples' # violet
    ax.quiver(X1, Y1, DX1, DY1, color='red', pivot='mid', units='xy')
    # ax.quiver(X1, Y1, DX1, DY1, M, pivot='mid', units='xy')



###################################################
# DYNAMICS


def nullcline(f, params, limits, steps):
    fn = []
    c = np.linspace( limits[0], limits[1], steps ) # linearly spaced numbers
    for i in c:
        fn.append( f(i,params) )
    return c, fn


# FIXED POINTS
# brute force: iterate through possibility space(r)
def find_fixed_points(f1, f2, p, r):
    # EIGEN values and vectors
    A = np.array([[-.5,1],[-1, .5]]) # oscillator
    eigvals,eigvecs = scipy.linalg.eig( A )
    print("eigenvals",eigvals)
    print("eigenvecs",eigvecs)
    fp = []
    for x in range(r):
        for y in range(r):
            if ( (f1(x,y,p,0)<0.1) and (f2(x,y,p)<0.1) ):
                fp.append((x,y))
                print('The system has a fixed point in %s,%s' % (x,y))
    return fp


def f(x,y,p,I):
    return ( -(x-p.v_rest) + p.delta_T*np.exp((x-p.v_thresh)/p.delta_T) -y +I ) / p.tau_m # AdEx
def f_nullcline(x,p):
    return -(x-p.v_rest) + p.delta_T*np.exp((x-p.v_thresh)/p.delta_T) + p.I  # AdEx

def g(x,y,p):
    return ( p.a * (x-p.v_rest) -y ) / p.tau_w
def g_nullcline(x,p):
    return p.a * (x-p.v_rest)


# f1, f2 = function for each variable, iv = initial vector, dt = timestep, time = range
def Euler( f1, f2, params, iv, dt, time):
    x = np.zeros(time)
    y = np.zeros(time)
    # initial values:
    x[0] = iv[0]
    y[0] = iv[1]
    # compute and fill lists
    i=1
    while i < time:
        I = 0 # init
        if i > time/3 and i < 2*(time/3):
            I = params.I #

        # integrating
        x[i] = x[i-1] + ( f1(x[i-1],y[i-1],params,I) )*dt
        y[i] = y[i-1] + ( f2(x[i-1],y[i-1],params) )*dt

        # discontinuity
        if x[i] >= params.v_spike:
            x[i-1] = params.v_spike
            x[i] = params.v_reset
            y[i] = y[i] + params.b
            # refractory period
            for j in range(int(params.tau_refrac / dt)): # refrac is in ms already
                if i+j < time:
                    x[i+j] = params.v_reset
                    y[i+j] = y[i] + params.b
            i = i + j-1
        i = i+1 # iterate

    return x, y





###################################################
# PARAMETERS

# regular spiking params
params = AttrDict({
    'v_spike'    : 0.0,   # mV, spike detection
    'I'          : 0.,    # pA, Input current
    'tau_refrac' : 2.5,   # ms, refractory period
    'v_reset'    : -60.0, # mV, reset after spike
    'v_thresh'   : -50.0, # mV, spike threshold
    'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold
    'cm'         : 0.150, # nF, tot membrane capacitance
    'a'          : 0.0,   # nS, conductance of adaptation variable
    'b'          : 0.05,  # nA, increment to adaptation variable
    'tau_m'      : 15.0,  # ms, time constant of leak conductance (cm/gl)
    'v_rest'     : -65.0, # mV, resting potential E_leak
    'tau_w'      : 200.0, # ms, time constant of adaptation variable
})

# # bursting params
# params = AttrDict({
#     'v_spike'    : 0.0,   # mV, spike detection
#     'I'          : 0.,    # pA, Input current
#     'tau_refrac' : 2.5,   # ms, refractory period
#     'v_reset'    : -50.0, # mV, reset after spike
#     'v_thresh'   : -45.0, # mV, spike threshold (modified by adaptation)
#     'delta_T'    : 2.5,   # mV, steepness of exponential approach to threshold
#     'cm'         : 0.20,  # nF, tot membrane capacitance
#     'a'          : 3.0,   # nS, conductance of adaptation variable
#     'b'          : 0.01,  # nA, increment to adaptation variable
#     'tau_m'      : 20.0,  # ms, time constant of leak conductance (cm/gl)
#     'v_rest'     : -70.0, # mV, resting potential E_leak
#     'tau_w'      : 200.0, # ms, time constant of adaptation variable
# })


###################################################
# COMPUTING

# dynamic
Time = 60000
dt = 0.01
init = (-75, 5)
params.I = 0 # pA
x, y = Euler( f, g, params, init, dt, Time )

# to draw the field
fp = find_fixed_points(f, g, params, 100) # iterations
xn1Ip, xn2Ip = nullcline( f_nullcline, params, (-100,0), 100 )
xn1, xn2 = nullcline( f_nullcline, params, (-100,0), 100 )
yn1, yn2 = nullcline( g_nullcline, params, (-100,0), 100 )



###################################################
# PLOTTING

fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(wspace = 0.5, hspace = 0.3)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(x, 'r-', label='Vm')
ax1.plot(y, 'b-', label='w')
ax1.set_title("Dynamics in time")
ax1.set_ylabel("Potential V (mV), adaptation w")
ax1.set_xlabel("time (ms)")
ax1.grid()
axi = plt.gca()
plt.draw()
labels = ax1.get_xticklabels()
for i,label in enumerate(labels):
    labels[i] = "{:d}".format(int(i*dt*1000))
ax1.set_xticklabels(labels)
ax1.legend()

ax2.axis([-90,-35,-50,80])
# ax2.axis([-100,20,-500,500])
ax2.plot(x, y, color="red")
ax2.plot(xn1, xn2, '-', color="black")
ax2.plot(xn1Ip, xn2Ip, '--',color="black")
ax2.plot(yn1, yn2, color="blue")
for p in fp:
    ax2.plot(fp, 'bo')
# ax2.grid()
quiverplot( Euler, f, g, params, [(-90,-35),(-50,80)], 10, ax2 ) #
ax2.set_xlabel("V (mV)")
ax2.set_ylabel("w (pA)")
ax2.set_title("Phase space")

plt.savefig( "DynamicAnalysis_AdEx.png" )
