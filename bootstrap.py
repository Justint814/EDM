#Python script for applying bootstrapping to the residuals of fit parameters in a fit

#IMPORTS
import numpy as np
import edmpackage as e

#ALIASES
munpy = e.munpy

#Load in file
workdir = "/Users/justintraywick/research/NOPTREX/EDM/"
filename = workdir + "good_data/data/2/minus2.csv"

save_file = workdir + "data/boot_hists/minus2freqs.root"


data = munpy(e.csv2npy(filename)).range(2,11)
freqs = e.bootstrap(data, 0.15, e.exp_sin_call, 4)[2]

np.save(workdir + "param_boot_hist", freqs)


#e.root_hist(freqs, 100, save_file, title="Bootstrapped values of frequency", root_func=e.gaus_call)

print(freqs)


