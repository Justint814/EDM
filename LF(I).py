import numpy as np
import edmpackage as e

#Calculate B field from current, radius of Helmhholtz coil, turns in coil and constants
def B_field(I, n=222, r=0.2):
    #Define permeability of free space
    mu_0 = (4 * np.pi ) * 10 ** (-7)

    B = (((4 / 5) ** (3/2)) * mu_0 * n * I) / r

    return B 

#Calculate Larmour precession frequency based on source
def L_freq(I, source="xe"):

    B = B_field(I)

    if (source=="xe"):
        gamma = 73.197 * (10 ** 6)

    else:
        gamma = 203.789 * (10 ** 6)

    freq = (B * gamma) / (2 * np.pi)

    return freq

#Plot Larmor precession frequency w.r.t. current to H coil
import plotly.graph_objects as go

x = np.linspace(0, 5, 1000)
y = L_freq(x, source="he")
y2 = L_freq(x)

mu_0 = (4 * np.pi) * 10E-7
gamma = 203.789E6
sig_f = e.sig_freq(222, 1.16, 0.2, 2, 0.0006, 0.005, mu_0, gamma)
print("sigma f =", sig_f)


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='He'))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Xe'))

fig.update_layout(
    title='Larmor Precession Frequency as a function of Current',
    xaxis_title='I(A)',
    yaxis_title='f(Hz)'
)

fig.show()

print(B_field(1.15))



