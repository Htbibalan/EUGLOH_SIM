#!/usr/bin/env python

#import a library for graphical representation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# time step for integration
DT=0.01
# number of iterations
Max_It=10000

#Parameters
spikeThreshold=0.0  # mV, spike detection
Cm=0.15 # nF, tot membrane capacitance
I=0.   # pA, Input current
gL=0.01 # nS, leak conductance
El=-65. # mV, resting potential E_leak
deltaT=2. # mV, steepness of exponential approach to threshold
Vt=-50. # mV, spike threshold
a=0.  # nS, conductance of adaptation variable
b=0.05  # nA, increment to adaptation variable
tauw = 200 # ms, time constant of adaptation variable
refractorytime=2.5  # ms, refractory period

Refrac_It = refractorytime/DT

#initial condition
V=-65.
w= 0
#Create a list to save the value of the variable at each time step
Vm=[]
W=[]
def f(V):
    return -gL*(V-El)+gL*deltaT*np.exp((V-Vt)/deltaT)

# iterate
for i in range(Max_It):
    if 5000<i<6000:
        I = 1.
    else:
        I =0.
    # calculate the variation of V at each time step
    dVdt = (f(V)+I-w)/Cm
    dwdt = (a*(V-El)-w)/tauw
    # calculate the new value of V at each time step
    V = V+dVdt*DT
    w = w+dwdt*DT
    if V>spikeThreshold:
        V=-55.
        w+=b
    # Save the value of V into a list
    Vm.append(V)
    W.append(w)
# create the time vector
TimeVec=[i*DT for i in range(Max_It)]

# Plot the Vm over time
plt.plot(TimeVec, Vm)
plt.savefig( "./AdExVm.png" )
plt.plot(TimeVec, W)
plt.savefig( "./AdExW.png" )
