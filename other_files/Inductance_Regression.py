#%
import numpy as np
import scipy
import scipy.odr as odr
import math as m

#Calculate resonance frequency of coil from capacitance and indctance
def f(C,L=0.000144):
    
    freq = 1 / (2*np.pi * np.sqrt(L*C))

    return freq

#Calculate Inductance from resonant freq. and capacitance
def L(f,C):
    return 1 / (((2*np.pi * f) **2) * C)

def L_N(N, r, l):
    return ((4*np.pi * 10 ** -7) * (N **2) * np.pi * (r**2)) / l

def C_f(f,L):
    return 1 / (((2*np.pi * f) **2) * L) 

def regression(freq, cap):

    def LC_func(L,C):
        return (1 / (2 * m.pi * np.sqrt(L * C)))
    
    model = odr.Model(LC_func)

    mydata = odr.Data(cap, freq)
    myodr = odr.ODR(mydata, model, beta0=[.1])

    my_out = myodr.run()

    L = float(my_out.beta[0])


    return L

freq = np.array([43000,26000,20000,20000])
cap = np.array([0.1, .2, 0.44, 0.47]) * (10 ** -6)

#L = regression(freq, cap)


Ind = L(33000, 0.147E-6)

print(Ind)

C  = L(38100, Ind)
print(C)

print('new')
print(f(870E-12, 1.81E-4))
r = 0.075 / 2
print(L_N(590, r, 0.3))
print(C_f(401000, 5.44E-3))


