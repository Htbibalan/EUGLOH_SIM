#import a library for graphical representation
import matplotlib
import matplotlib.pyplot as plt


################################
#Leaky Integrate and Fire model#
################################

# time step for integration
DT=0.01
# number of iterations
MaxIt=60000

# Parameters
Cm = 0.150 # nF, membrane capacitance
I = 0 # pA
El = -65.0 # mV,
reset = -65.0 # mV,
gl = 0.01 # nS
tau_refrac = 5 # ms
# initial condition
V = -65.0 # mV
# Threshold for spike detection
Threshold = -50.0 # mV

# Create a list to save the value of the variable at each time step
Vm=[]
# iterate
for i in range(MaxIt):
    # calculate the variation of V at each time step
    dVdt = ( -(V-El) +I) / (Cm/gl)
    # calculate the new value of V at each time step
    V = V+dVdt *DT
    if V > Threshold:
        V = reset
    # Save the value of V into a list
    Vm.append(V)

# create the time vector
TimeVec=[i*DT for i in range(MaxIt)]
# Plot the Vm over time
plt.plot(TimeVec, Vm)
plt.savefig( "./LIF.png" )
