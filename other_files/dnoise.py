import numpy as np
import math as m
import matplotlib.pyplot as plt
import os


#Function to return array of file names in a specified path
def file_list(path,reorder=False):
    entries = []
    obj = os.scandir(path)
    for entry in obj:
        if entry.is_file():
            if not entry.name.startswith('.'):
                entries.append(entry.name)

    files = np.array(entries)
         
    return files

#Function to delete text from .csv files and save as .npy
def csv2npy(file):
    time, p2p = np.loadtxt(file, skiprows = 18, usecols = [0,1], delimiter = ',', unpack = True) # time (s), p2p (v)
    array = np.array([time, p2p])

    return array

def scope_ft(signal):
    tot_time = signal[0][-1] - signal[0][0] #Get total time
    sample_rate = (len(signal[1])) / tot_time #Get number of samples per second
	
    amplitudes = signal[1]
    n = len(signal[1])
	
    freq = np.fft.fftfreq(n, 1/sample_rate) / 1000 #Get frequencies from FFT. Divide by 1000 to get in kHZ
    amps = np.fft.fft(amplitudes) #Get amplitudes from FFT.

    fft_data = np.array([freq[:n//2], amps[:n//2]]) #Join arrays and consider only positive values (Symmetrical over y axis)
    #fft_data = np.array([freq,amps])
    return fft_data

signal = csv2npy("./data/6-17/good/FID.9.csv")
beg_cut = m.floor((0.0150 + 0.0175) / (10 ** -6))
rate = np.shape(signal)[1] / (signal[0,-1] - signal[0,0])
beg_cut = m.floor((0.0150 + 0.0175) * rate)
signal = signal[:,700:3200]

fft_data = scope_ft(signal)
max_freq = fft_data[0][np.argmax(fft_data[1,:])]
print(max_freq)


count=0
'''for i in range(np.shape(fft_data)[1]):
    if abs(fft_data[0,i]) > 50 or abs(fft_data[0,i]) < 25:
        fft_data[1,i] = 0 + 0j
        count +=1

clean_sig = np.fft.ifft(fft_data[1,:])

print(count, "frequencies removed.")
'''
#Plot data of choosing
plt.plot(fft_data[0],fft_data[1])
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude')
plt.title('FFT Amplitude Spectrum')
plt.grid(True)
plt.show()

#Plot data of choosing
plt.plot(signal[0],signal[1])
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude')
plt.title('FFT Amplitude Spectrum')
plt.grid(True)
plt.show()