import numpy as np
import ROOT
import edmpackage as e

mp = e.munpy


#Load in file
workdir = "/Users/justintraywick/research/NOPTREX/EDM/"
filename = workdir + "good_data/data/3/minus3.csv"
save_file = workdir + "data/nois_fits/minus1.root"

data = e.csv2npy(filename)

#e.plot(data, plotstyle="matplotlib")
data = mp(data).range(6, 6.4).astype(np.double)
print(data)

y_errors = np.full_like(data[1,:], 0)
guess = np.array([4, 0.0247, -37206, .05, -4.4, -0.25])
params = e.root_fit(data[0,:], data[1,:], guess, y_errors, save_file, root_func=e.exp_sin_call)[0]

#print(params)
