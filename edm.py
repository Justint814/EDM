#edm module
#Updated: 7/16/25
#Author: Justin Traywick
#Email: jtraywick8@gmail.com
#GitHub User: Justint814

# import numpy as np
import scipy
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import csv
import os
from PyEMD import EMD
from scipy.signal import hilbert
import ROOT
import math as m
import numpy as np

###################~FUNCTIONS~#####################
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

#Plot data of choosing
def plot(*tuple, x_axis='x',y_axis='y',title='plot', plotstyle='plotly', markers='markers'): #Use markers='lines' for line plot

    #Define dataset for each axis

    #If the input array is one diminesional, use the length of the array as the x axis. If it is 2 dim., use the columns as the axes.
    if len(tuple) == 1:
        #Get shape of *tuple
        shape = np.shape(tuple[0])

        if len(shape) == 1:
            pts = np.array([[range(len(tuple[0]))],[*tuple[0]]])
        elif len(shape) == 2:
            pts = tuple[0]
    #If the input tuple has two elements, treat each as an axis.
    else:
        pts = np.array([[*tuple[0]],[*tuple[1]]])


    #Plot data in one of two configurations
    if plotstyle == 'plotly':
        #Plot pts with plotly interactive plot
        fig = go.Figure() #Make the figure
        fig.add_trace(go.Scatter(x=pts[0], y=pts[1], mode=markers, name=title)) #Add data to the figure

        fig.update_traces(line=dict(width=.6)) #Update line width

        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis
        ) #Add titles

        fig.show() #Show the figure

    else:
        #Plot pts with matplotlib
        plt.plot(pts[0],pts[1]) #Plot data
        plt.xlabel(x_axis) #Define x axis label
        plt.ylabel(y_axis) #Define y axis label
        plt.title(title) #Title the plot
        plt.grid(True) #Overlay grid on plot
        plt.show() #Show the plot

#Lorentzian fit
def lor_root(x, par, guess_arr=np.array([1, 60, -38000,0])): #guess_arr is of size: 4
    #Define Lorentzian fit ROOT object
    return par[3] + (par[0] * (par[1]/2) ** 2) / ((x[0] - par[2]) ** 2 + (par[1] / 2) **2)

#Function for calling lor_root() in root_hist()
def lor_call():
    return "Lorentzian", lor_root

#Function to fit a data set to a specified root function and apply bootstrapping to the parameters using the residuals
def bootstrap(data, y_err, root_func, N): #data: 2D numpy array, y_err: scalar error, root_func: Calling function for desired root fit, N: Number of bootstrap iterations
    #Define name of fit and fitting function
    name, func, n_par, guess = root_func()
    xmin = np.min(data[0,:])
    xmax = np.max(data[0,:])

    #Define array of error in y 
    y_err = np.full_like(data[1,:], y_err)

    def fit_dat(data, name, func, n_par, guess):
        #Define TGraph for data
        gr = ROOT.TGraphErrors(len(data[0,:]), data[0,:].astype(np.double), data[1,:].astype(np.double), np.zeros_like(data[0,:]).astype(np.double), y_err.astype(np.double))

        #Create ROOT fit function object and set initial parameters
        fit_func = ROOT.TF1(name, func, xmin, xmax, n_par)
        fit_func.SetParameters(guess.astype(np.double))

        #Fit function to data
        gr.Fit(fit_func)

        return fit_func
    
    in_fit = fit_dat(data, name, func, n_par, guess)
    #Get initial fit parameters
    params = []
    for i in range(len(guess)):
        params.append(in_fit.GetParameter(i))
    
    params = np.array(params)

    #Get fit data
    x = data[0,:]
    y_fit = func(x, params)

    #Get residuals
    resids = data[1,:] - y_fit

    #Define Numpy array for storing bootstrapped fit parameters
    params_boot = np.zeros((len(guess), N))

    #Randomly sample residuals, apply fitting to data based on them and then return parameters
    for i in range(N):
        #Randomly select residuals with replacement
        rand_resids = np.random.choice(resids, size=len(resids), replace=True)

        #Define new data to fit to
        y_prime = y_fit + rand_resids

        #Define new data array based on y_prime
        data_it = np.array([x, y_prime])

        #Fit function to new data and add parameters to array
        func_prime = fit_dat(data_it, name, func, n_par, guess)
        for j in range(len(guess)):
            params_boot[j,i] = func_prime.GetParameter(j)
    
    return params_boot #Return 2D array with all bootstrapped parameter distribution
 
#Perform FID fit with root
def root_fit(x_arr, y_arr, guess_arr, y_errors, filename, root_func=lor_call):
    out_file = ROOT.TFile(filename,"RECREATE")
    name, func = root_func()
    graph = ROOT.TGraphErrors(len(x_arr), x_arr.astype(np.double), y_arr.astype(np.double), np.zeros_like(x_arr).astype(np.double), y_errors.astype(np.double))
   
    #Define fit function for FID
    def exp_sin(x, par):
        return par[0] * np.exp(-(x[0] + par[3]) / par[1]) *  ROOT.TMath.Sin(par[2] * 2 * ROOT.TMath.Pi() * x[0] + par[4]) + par[5]

    fit_func = ROOT.TF1(name, func, x_arr[0], x_arr[-1], 6) #(name, func, lower bound, upper bound, num params)
    fit_func.SetParameters(guess_arr.astype(np.double))
    
    graph.Fit(fit_func, "S")

    params = []
    errors = []
    for i in range(len(guess_arr)):
        params.append(fit_func.GetParameter(i))
        errors.append(fit_func.GetParError(i))
    
    errors = np.array(errors)
    params = np.array(params)
    chi2 = fit_func.GetChisquare()

    canvas = ROOT.TCanvas("myCanvas", "My Plot", 800, 600)

    fit_func.SetLineColor(ROOT.kRed)
    fit_func.SetLineWidth(2)
    fit_func.SetLineStyle(1)

    #Draw histogram and fit
    graph.Draw("graph  E1")
    fit_func.Draw("SAME")

    canvas.Update()
    canvas.Write()

    out_file.Close() 

    return (params, errors, chi2)

#Gaussian fit
def gaus_root(x, par, guess_arr=np.array([1, 60, -38000,0])):
    return par[0] + par[1] * ROOT.TMath.Gaus(x[0], par[2], par[3])

#Function for calling gaus_call in hist_root()
def gaus_call():
    return "gaussian", gaus_root
    
#Function that root fits to in ~root_fit~. Using this for plotting the fit. Takes parameter array returned by root fit.
def exp_sin_fit(x,par):
    return par[0] * np.exp(-(x[0] + par[3]) / par[1]) *  np.sin(par[2] * 2 * np.pi * x[0] + par[4]) + par[5]

#Function for calling exp_sin_fit() in other root functions
def exp_sin_call(guess=np.array([1, 0.06, 37000, .5, .5, 2])): #Returns name, function, number of parameters and parameter guess
    return "exp_sin_fit", exp_sin_fit, 6, guess

#Function to generate a 1D histogram of a dataset using ROOT and fit a Lorentzian function to it.
def root_hist(data, bins, filename, title='Histogram', plot='on', root_func=lor_call,  start=0, stop=-1): #Data array, bin count, filename, plot title, plot setting, histogram starting entry, histogram ending entry

    #Define outfile to write data to
    out_file = ROOT.TFile(filename,"RECREATE")

    name, func = root_func()

    xmin = np.sort(data)[start] #Get min value of data
    xmax = np.sort(data)[stop] #Get max value of data

    #Define histogram with specified number of bins, minimum and maximum
    hist = ROOT.TH1D("hist", f"{title};Bins(Hz);Frequency", bins, xmin, xmax)

    #Fill histogram with data
    for entry in data:
        hist.Fill(entry.astype(np.double))

    #Configure fit
    fit_func = ROOT.TF1(name, func, xmin, xmax, 4) #Define fit function
    guess = np.array([1, 1, -37000,20])

    fit_func.SetParameters(guess.astype(np.double))

    #Add legend
    legend = ROOT.TLegend(0.75, 0.55, 0.95, 0.75)

    #Add legend entries
    legend.AddEntry(hist, "Histogram", "l")
    legend.AddEntry(fit_func, f"{name} Fit", "l")

    #Perform Gaussian fit to histogram

    hist.Fit(fit_func, "Q")

    # Get fit parameters
    param0 = fit_func.GetParameter(2)
    param0_err = fit_func.GetParError(2)
    chi2 = fit_func.GetChisquare()
    ndf = fit_func.GetNDF()
    reduced_chi2 = chi2 / ndf if ndf > 0 else 0
    entries = int(hist.GetEntries())

    # Create custom stats box (TPaveText)
    stats_box = ROOT.TPaveText(0.7, 0.25, 1, 0.55, "NDC")
    stats_box.SetBorderSize(1)
    stats_box.SetFillColor(0)
    stats_box.SetTextAlign(12)
    stats_box.SetTextFont(42)
    stats_box.AddText(f"Freq. (Hz) = {np.abs(param0):.4f} #pm {param0_err:.4f}")
    stats_box.AddText(f"#chi^{{2}}/ndf = {reduced_chi2:.2f}")
    stats_box.AddText(f"Entries = {entries}")
    stats_box.SetTextSize(100)

    if plot =='on':
        canvas = ROOT.TCanvas("myCanvas", "My Plot", 800, 600)

        fit_func.SetLineColor(ROOT.kRed)
        fit_func.SetLineWidth(2)
        fit_func.SetLineStyle(1)

        #Draw histogram and fit
        hist.Draw("HIST  E1")
        fit_func.Draw("SAME")

        #Draw legend
        legend.Draw()
        stats_box.Draw()


        canvas.Update()
        canvas.Write()

        out_file.Close()
        #canvas.Print("/Users/justintraywick/research/NOPTREX/EDM/data/root/histogram_omega.pdf")
    return fit_func.GetParameter(2), reduced_chi2

#Function to get differences in frequencies across E field configuration
def freq_dif(f_p, f_m, sig_p, sig_m):
    dif = np.abs(f_p - f_m)
    sigma = np.sqrt(sig_p**2 + sig_m**2)

    return dif, sigma

#Perform ~root_fit()~ over a chunks of a dataset with a specified sweep rate. "order=" kwarg specifies the order of magnitude of your input values in seconds.
def it_fit(t_min, t_max, set_size, data, sweep=.05, numpy=False): #Sweep kwarg and set size must be given on the order that is chosen for input. Milisecond is standard
    arr = munpy(data)
    t_max = np.max(data[0,:]) * 1000

    guess = np.array([-100,.002,37000,.001,20,0])

    i = 0
    omega = []
    tau = []
    chi2_arr = []
    lim = t_min + set_size
    while lim < t_max:
        l_ind = t_min + i * sweep #Define upper bound
        u_ind = l_ind + set_size #Define lower bound
        set = arr.range(l_ind, u_ind) #Define set to perform fit on

        chi2_lim = len(set[0,:]) * 1

        params, error, chi2 = root_fit(set[0,:], set[1,:], guess, np.full_like(set[0,:], 0.15))

        if chi2 < chi2_lim:
            tau.append(params[1])
            omega.append(params[2])
            chi2_arr.append(chi2)

        lim = u_ind + sweep
        i+=1
        
    #Convert lists to numpy arrays
    if numpy == True:
        tau = np.array(tau)
        omega = np.array(omega)
        chi2_arr = np.array(chi2_arr)

    print(f"{i} fits completed.")
        
    return tau, omega, chi2_arr

#Function to calculate EDM
def edm_calc(f_p, f_m, V, a, sig_p, sig_m, sig_V, sig_a):
    ###~FUNCTION VARIABLES~###:
    #f_p: Positive E-field Larmor frequency (Hz)
    #m_p: Negative E-field Larmor frequency (Hz)
    #V: E-field Supply voltage (Volts)
    #a: Plate separation distance (m)
    #sig_p: Positive E-field Larmor frequency uncertainty (Hz)
    #sig_m: Negative E-field Larmor frequency uncertainty (Hz)
    #sig_V: E-field Supply voltage uncertainty (Volts)
    #sig_a: Plate separation distance uncertainty (m)
    
    #Constants
    h = 6.62607015E-34 #Planck's Constant h in joule seconds

    #Calculate EDM
    d = h * a * (f_p - f_m) / (4*V)

    #Propogate uncertainty
    sig_1 = ((h * a) / (4 * V)) * sig_p #Error contribution from omega p 
    sig_2 = ((-h * a) / (4 * V)) * sig_m #Error contribution from omega m
    sig_3 = (-(h * a * (f_p - f_m)) / (16 * V ** 2)) * sig_V #Error contribution from V
    sig_4 = ((h) * (f_p - f_m) / (4 * V)) * sig_a #Error contribution from d

    sigma = np.sqrt(sig_1 ** 2 + sig_2 ** 2 + sig_3 ** 2 + sig_4 ** 2) #Compile uncertainties

    #Convert EDM and uncertainty to e*cm
    d = d * 6.2E18 * 100
    sigma = sigma * 6.2E18 * 100

    return d, sigma

#Function to perform EMD and hilbert huang transform on first IMF
def h_huang(data, return_signal=False):
    emd = EMD()
    IMFs = emd.emd(data[1,:])

    #Apply Hilbert Transform and identify noise IMFs (example: based on visual inspection or frequency content)
    keep_list = np.array([0]) #Define list of IMF elements to keep

    analytic_signal = hilbert(IMFs[0,:])
    sampling_rate = len(data[0,:]) / (data[0,0] - data[0,-1])
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi)) * sampling_rate

    time = data[0,:-1]
    inst_freq = np.array([time, instantaneous_frequency])

    if return_signal==True:
        #Extra code to isolate IMFs
        denoised_IMFs = []
        for i in keep_list:
            denoised_IMFs.append(IMFs[i,:])

        denoised_IMFs = np.array(denoised_IMFs)

        #Reconstruct the denoised signal
        denoised_signal = np.sum(denoised_IMFs, axis=0)
        denoised = np.array([data[0,:],denoised_signal])

        return inst_freq, denoised

    else:
        return inst_freq

#Function to approximate the frequency of signal data. Takes 2D numpy array.
def freq_approx(data, return_arr=False): 
    #Define time and amplitude from data as x and y 
    x = data[0,:]
    y = data[1,:]

    #Define lists for storing time and amplitude data to count cycles
    time = []
    amp = []

    #Define rough estimate of center estimate of data to count cycles
    mean = np.average(y)

    #Define variables to count crosses to upper side of mean
    up_count = 0

    #Iterate over data and store time stamps each time the signal crosses the mean line.
    for i in range(len(x)):
        if (y[i] > mean and up_count % 2 == 0): #If y > mean and up_count is even
            time.append(x[i]) #Append time element to time list
            amp.append(y[i]) #Append amplitude element to amp list

            #Add 1 to up count such that it is odd
            up_count += 1

        if (y[i] < mean and up_count % 2 != 0):
            #Add 1 to up count such that it is even
            up_count += 1

    #Convert time and amp lists to numpy arrays
    time = np.array(time)
    amp = np.array(amp)

    #Define upper and lower time indexes to compute frequency
    t1 = time[0]
    t2 = time[-1]

    #Calculate average frequency in Hz with dcycles / dt
    dt = t2 - t1 #Define change in time
    dcycles = len(amp) #Define number of cycles

    freq = dcycles / dt #Calculate frequency in HZ
    
    if return_arr == True:
        return freq, np.array([time, amp])
    else:
        return freq
    
#Make dummy data for testing
def exp_sin(x, par):
    return par[0] * np.exp(-x / par[1]) * np.sin(par[2] * 2 * np.pi * x)

#Function to perform FFT on signal (in kHz) from specified numpy array
def scope_ft(signal,abs=False):
    tot_time = signal[0][-1] - signal[0][0] #Get total time
    sample_rate = (len(signal[1])) / tot_time #Get number of samples per second
	
    amplitudes = signal[1]
    n = len(signal[1])

    freq = np.fft.fftfreq(n, 1/sample_rate) #/ 1000 #Get frequencies from FFT. Divide by 1000 to get in kHZ
    amps = np.fft.fft(amplitudes) #Get amplitudes from FFT. Divide by 1000 to get in kHz. Magnitude of amplitudes.
    
    if abs==True:
           fft_data = np.array([np.abs(freq[:n//2]), np.abs(amps[:n//2])]) #Join arrays and consider only positive values (Symmetrical over y axis)
 
    else:
        fft_data = np.array([freq,amps])

    return fft_data
####################~CLASSES~######################
#Class to handle indexing an array using a column value instead of index number
class munpy:
    def __init__(self, array, order=-3): #'order=' kwarg defines the order of magnitude that inputs given in. Ex: for 'order=-3', give time in miliseconds. For seconds, use 'order=0'
        self.array = array
        self.order = order
        self.shape = np.shape(array) #Shape of the array
        self.period = array[0,-1] - array[0,-2] #Amount of time between indices

    def find(self, t_val): #Return the index of the element closest to the time value given to the function.
        diffs = np.abs(self.array[0,:] - (t_val * 10 ** self.order))

        index = np.argmin(diffs)

        return index
    
    def range(self, t1, t2): #Return a sliced array within the range t1-t2
        cut_arr = self.array[:,self.find(t1) - 1 : self.find(t2)]

        return(cut_arr)