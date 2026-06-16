
import numpy as np

#Get v_rms from peak to peak voltage reading on signal generator
def v_rms(v):
    return (v * 0.707)

#Get current from voltage of wire in kicker coil and v_rms voltage
def current(v_rms, R=6):
    return v_rms / R

#Calculate B field from current, radius of kicker coil, turns in coil and constants
def B(I, n=20, r=0.07):
    #Define permeability of free space
    mu_0 = (4 * np.pi ) * 10 ** (-7)

    return (((4 / 5) ** (3/2)) * mu_0 * n * I) / r

#Calculate time needed to tip magnetic dipoles by a certain theta in degrees
def time(theta, v, source="xe"): #v is signal generator reading
        
    theta0 = theta * (2 * np.pi) / 360

    v0 = v_rms(v)
    I0 = current(v0)
    B1 = B(I0)

    if (source == "xe"):
        gamma = 73.197 * (10 ** 6)

    else:
        gamma = 203.789 * (10 ** 6)
    
    t = theta0 / (B1 * gamma)

    return t


time = time(5,10,source = 'he')
print(time * 38100)

