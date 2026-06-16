import numpy as np
import os
import scipy
import scipy.odr as odr
import math as m
import matplotlib.pyplot as plt
import ROOT
import csv

#File Locations
save_dir = './data/NMR-peaks' #Place to save Lorentzian peaks
target_dir = './data/6-17/good' #Data containing csv files from scope
load_dir = './data/npy/6-13'

#Functions:

#Function to delete text from .csv files and save as .npy
def csv2npysave(fetch_path, save_path):
    #Get filenames from o-scope directory
    files = file_list(fetch_path)
    print(files)

    #Loop through files and convert .csv to .npy and save
    for i in files:
        csv_filename = fetch_path + '/' + i
        time, p2p = np.loadtxt(csv_filename, skiprows = 18, usecols = [0,1], delimiter = ',', unpack = True) # time (s), p2p (v)
        np_filename = '.' + (save_path + '/' + i).split('.')[1]
        np.save(np_filename, [time, p2p])

#Function to delete text from .csv files and save as .npy
def csv2npy(file):
    x_arr = [] #list of 1st column values
    y_arr = [] #list of 2nd column values
    trial = [] #Dummy list for checking data type validity

    with open(file, 'r') as csv_file:
        csv_reader  = csv.reader(csv_file)

        for row in csv_reader:
            try:
                trial.append(float(row[0]))
                trial.append(float(row[1]))
            except:
                pass
            else:
                x_arr.append(float(row[0]))
                y_arr.append(float(row[1])) 
        
        array = np.array([x_arr,y_arr])

        return array

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

#Function to get the index of the second maximum value in an array
def sec_max_index(array):
  if len(np.unique(array)) < 2:
    return None
  sorted_indices = np.argsort(array)
  return sorted_indices[-2]

#Function to perform FFT on signal (in kHz) from specified numpy array
def scope_ft(signal,abs=False):
    tot_time = signal[0][-1] - signal[0][0] #Get total time
    sample_rate = (len(signal[1])) / tot_time #Get number of samples per second
	
    amplitudes = signal[1]
    n = len(signal[1])

    freq = np.fft.fftfreq(n, 1/sample_rate) #/ 1000 #Get frequencies from FFT. Divide by 1000 to get in kHZ
    amps = np.abs(np.fft.fft(amplitudes)) #Get amplitudes from FFT. Divide by 1000 to get in kHz. Magnitude of amplitudes.
    
    if abs==True:
           fft_data = np.array([freq[:n//2], np.abs(amps[:n//2])]) #Join arrays and consider only positive values (Symmetrical over y axis)
 
    else:
        fft_data = np.array([freq,amps])

    return fft_data

#Function to fit FID decay signal to FID decay function
def FID_fit(time, voltage):

    def FID_func(x,A,tau,omega): #Function representation of FID decay
        return np.exp(-x / tau) * A * np.sin(omega * 2 * np.pi * x)

    popt, pcov = scipy.optimize.curve_fit(FID_func, time, voltage)
    y = FID_func(time, *popt)
    print(popt)

    return y

#Get the maximum frequency index of the spectrum not including zero
def max_freq(fft_data, b_range):   
    max_freq_ind = np.argmax(np.abs(fft_data[1,3:]))
    max_freq = np.abs(fft_data[0][max_freq_ind])
    bounds = np.array([max_freq_ind - b_range, max_freq_ind + b_range])
    print(bounds)

    if bounds[0] < 0 or bounds[1] < 0:
        raise ValueError("Bounds too large for array around maximum frequency, reduce bound range.")
    #Define new array of spectrum between bounds around peak. 
    peak_arr = fft_data[:,bounds[0]:bounds[1]]

    return peak_arr, max_freq

#Plot data of choosing
def plot(*tuple, x_axis='x',y_axis='y',title='plot'):
    #If the input array is one diminesional, use the length of the array as the x axis
    if len(tuple) == 1:
        pts = np.array([range(len(tuple[0])),[*tuple[0]]])
    else:
        pts = np.array([[*tuple[0]],[*tuple[1]]])

    plt.plot(pts[0],pts[1])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.grid(True)
    plt.show()

#Function to cut out signals in a certain range around a specified frequency on fft data
def sig_cut(fft_data,freq,width): 
    for i in range(np.shape(fft_data)[1]):
        if abs(fft_data[0,i]) > freq + width or abs(fft_data[0,i]) < freq - width:
            fft_data[1,i] = 0
    return(fft_data)

#Function to smooth spikes in fft_data when they are not the maximum frequency
def smooth(fft_data):
    max_freq = np.max(fft_data[1,:])
    for j in range(4):
        for i in range(len(fft_data[0,:])-1):
            if (fft_data[1,i] - fft_data[1,i+1]) > 3 and fft_data[1,i] != max_freq:
                fft_data[1,i] = fft_data[1,i+1]
   
    return fft_data

#Function that subtracts a fraction of the frequencies in signal_a from signal_b
def f_sub(noise,signal_b,fraction,thresh=1): #Thresh decide the range in which 2 freqs. are "the same"

    fft_b = scope_ft(signal_b, abs=True)[:,100:]
    noise_arr = []
    for h in noise:
        noise_arr.append(scope_ft(h, abs=True)[:,100:])

    #Loop over frequencies and subtract frequencies in common
    for fft_noise in noise_arr:
        for i in range(len(fft_noise[0,:])):
            for j in range(len(fft_b[0,:])):
                if fft_b[0,j] > fft_noise[0,i] - thresh and fft_b[0,j] < fft_noise[0,i] + thresh:
                    fft_b[1,j] = fft_b[1,j] - (fraction * fft_noise[1,i])
    return fft_b

#Lorentzian functional form used for fitting
def l_form(x,A,x_0,G):
   return (A * (G/2) ** 2) / ((x - x_0) ** 2 + (G / 2) ** 2) #A, x_0, Gamma

#Function to get peaks of a signal and store into array for fitting
def pk_sig(signal, t=3):
    x = []
    y = []
    for i in np.arange(len(signal[0,:]) - t):
        if signal[1,i] > signal[1,i+t] and signal[1,i] > signal[1,i-t]:
            x.append(signal[0,i])
            y.append(signal[1,i])

    data = np.array([x,y])
    
    return data

#Perform a fit with root
def root_fit(x_arr, y_arr, func_str, guess_arr, y_errors, filename, plot="on"):
    
    out_file = ROOT.TFile(filename,"RECREATE")
    graph = ROOT.TGraphErrors(len(x_arr), x_arr.astype(np.double), y_arr.astype(np.double), np.zeros_like(x_arr).astype(np.double), y_errors.astype(np.double))
   
    peaks = pk_sig(np.array([x_arr,y_arr]))
    fitdat = ROOT.TGraphErrors(len(peaks[1,:]), peaks[0,:] ,peaks[1,:] , np.zeros_like(peaks[0,:]), np.zeros_like(peaks[1,:]))

    fit_func = ROOT.TF1("fit", func_str, min(x_arr), max(x_arr))
    fit_func.SetParameters(guess_arr.astype(np.double))
    
    fitdat.Fit(fit_func, "S")

    params = []
    errors = []
    for i in range(len(guess_arr)):
        params.append(fit_func.GetParameter(i))
        errors.append(fit_func.GetParError(i))
    
    errors = np.array(errors)
    params = np.array(params)
    chi2 = fit_func.GetChisquare()
    
    print("\n")
    print("root_fit return form: (params, errors, chi2)")


    if plot=="on":
        canvas = ROOT.TCanvas("myCanvas", "My Plot", 800, 600)
        graph.SetLineStyle(1)
        graph.SetLineColor(38)
        graph.Draw("AL")
        fitdat.Draw("P SAMES")
        fitdat.SetMarkerStyle(20)
        fitdat.SetMarkerSize(0.5)

        canvas.Update()
        canvas.Write()

        out_file.Close()

    #input("\n") #Uncomment the input to display the graph 

    return(params, errors, chi2)
    
#Function to get peaks of a signal and store into array for fitting
def pk_sig(signal, t=5):
    x = []
    y = []
    for i in np.arange(len(signal[0,:]) - t):
        if signal[1,i] > signal[1,i+t] and signal[1,i] > signal[1,i-t]:
            x.append(signal[0,i])
            y.append(signal[1,i])

    data = np.array([x,y])
    
    return data

#Function to sort array by max values and select number of points from
def max_sig(signal, pts=2000):
    x_arr = []
    y_arr = []
    j = 0
    sort_arr = np.argsort(signal[1,:])[::-1]
    for i in sort_arr[:pts]:
        if i > j:
            x_arr.append(signal[0,i])
            y_arr.append(signal[1,i])
            j = i

    array = np.array([x_arr,y_arr])

    return array

signal = csv2npy('./data/3/m/minus1.csv')
#signal = signal[:,1000:3600]

#Perform fft
fft_data = scope_ft([signal[0,980:],signal[1,980:]])
#plot(fft_data[0,:],fft_data[1,:])
noise = csv2npy("./data/noise/noise1.csv")
fft_noise = scope_ft(noise)
#plot(fft_noise[0,2:],fft_noise[1,2:])

clean_fft = smooth(fft_data[:,50:])

#Get data around maximum frequency
max_frequency = (max_freq(clean_fft, 500)[0]).astype(np.double)

peak_sig = pk_sig(signal)
#plot(signal[0,980:],signal[1,980:])



#guess = np.array([1,.006,5])
guess = np.array([1,60,38000]).astype(np.double)
#print(root_fit(signal[0,980:], signal[1,980:], "[0]*exp(-x / [1]) - [2]", guess, np.zeros_like(signal[1,980:]), "./data/root/EDM_test2.root",plot='on'))
#print(root_fit(max_frequency[0,:], max_frequency[1,:], "([0] * ([1]/2) ** 2) / ((x - [2]) ** 2 + ([1] / 2) ** 2)", guess, np.zeros_like(max_frequency[1,:]), "./data/root/lorentzian.root",plot='on'))





