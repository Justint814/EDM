##IMPORTS
import edmpackage as e
import numpy as np

##Define aliases
munpy = e.munpy


#Define working directory and filename
workdir = "/users/justintraywick/research/NOPTREX/EDM/data/"
filename = workdir + "/good_data/data/1/plus1.csv"

#Convert base file to numpy array
data_arr = e.csv2npy(filename)

#Plot FID signal and isolate FID range
#e.plot(data_arr)
freq_dist = []
time=[]
for i in np.arange(0,8,.05):
    data = munpy(data_arr).range(1 + i, 6 + i)
    freq_dist.append(e.freq_approx(data))
    time.append(i + 3)
freq_dist = np.array([time, freq_dist])
e.plot(freq_dist)

#e.plot(data)
#Approximate average frequency
#frequency, freq_data = e.freq_approx(data, return_arr=True)
#e.plot(freq_data)
#print(frequency)

#Make histogram of average frequencies
#e.root_hist(freq_dist[1,:],10, workdir + "frequency_hists/avg_freq_dist.root", title = "-E Config. Frequency Distrubution", start=0, stop=-1,root_func=e.gaus_call)

d1 = e.freq_dif(37396.8, 37345.4, 4, 4)
d2 = e.freq_dif(37410.3, 37439.2, 4, 4)
d3 = e.freq_dif(37531.5, 37494.6, 4, 4)

print(d1[0], d2[0], d3[0])

