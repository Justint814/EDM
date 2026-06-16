############  ~IMPORTS~  ########################################################

import numpy as np
import ROOT
import math
import matplotlib.pyplot as plt
import os
import csv
import plotly.graph_objects as go
##################################################################################




############  ~FUNCTIONS~  #######################################################

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
def plot(*tuple, x_axis='x',y_axis='y',title='plot', plotstyle='plotly'):

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
        fig.add_trace(go.Scatter(x=pts[0], y=pts[1], mode='lines', name=title)) #Add data to the figure

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

#Perform a fit with root
def root_fit(x_arr, y_arr , guess_arr, y_errors):
    
    graph = ROOT.TGraphErrors(len(x_arr), x_arr.astype(np.double), y_arr.astype(np.double), np.zeros_like(x_arr).astype(np.double), y_errors.astype(np.double))
   
    #Define fit function for FID
    def exp_sin(x, par):
        return par[0] * np.exp(-(x[0] + par[3]) / par[1]) *  ROOT.TMath.Sin(par[2] * 2 * ROOT.TMath.Pi() * x[0] + par[4]) + par[5]

    fit_func = ROOT.TF1("decay_sin", exp_sin, x_arr[0], x_arr[-1], 6) #(name, func, upper bound, lower bound, num params)
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

    return (params, errors, chi2)

#Generate a TGraph based on a root fit. Returns ROOT TGraph object, or a numpy array of fit data
def fit_graph(param_arr, x_vals, fit_func, arr_return=False):
    y_vals = fit_func(x_vals, param_arr)

    gr = ROOT.TGraph(len(y_vals), x_vals.astype(np.double), y_vals.astype(np.double))

    fit_dat = np.array([x_vals, y_vals])

    if arr_return == True:
        return fit_dat
    
    else:
        return gr

#Function to plot multiple ROOT objects and save 
def rplot_seq(*g_objs, filename): 
    #Define save file
    out_file = ROOT.TFile(filename, "RECREATE")

    #Create canvas for plotting
    c1 = ROOT.TCanvas("mycanvas", "Combined Plot, 800 600")

    #Plot each object                  
    for i in g_objs:
        i.Draw("SAME") #Draw object on c1 canvas
        c1.Update()
    
    #Write canvas to outfile
    c1.Write()
    out_file.Close()

    print(f"File saved at {filename}")

#Function to calculate residuals from two datasets
def resid(data_arr, fit_arr): #Both arrays should be one dimensional (outputs)

    data_arr = data_arr.astype(np.float64)
    fit_arr = fit_arr.astype(np.float64)

    #Raise value error if arrays are different lengths
    if len(data_arr) != len(fit_arr):
        raise ValueError("Arrays given are different sizes")
    
    residuals = (data_arr - fit_arr) ** 2


    return residuals

#Function to generate a 1D histogram of a dataset using ROOT and fit a gaussian function to it.
def root_hist(data, bins, filename, title='Histogram', plot='on'): #Data array, bin count

    #Define outfile to write data to
    out_file = ROOT.TFile(filename,"RECREATE")

    xmin = np.sort(data)[0] #Get min value of data
    xmax = np.sort(data)[-1] #Get max value of data

    #Define histogram with specified number of bins, minimum and maximum
    hist = ROOT.TH1D("hist", f"{title};Bins;Frequency", bins, xmin, xmax)

    #Fill histogram with data
    for entry in data:
        hist.Fill(entry.astype(np.double))

    #Perform Gaussian fit to histogram
    fit_result = hist.Fit("gaus", "S", "SAME")

    fit_func = hist.GetFunction("gaus")

    if plot =='on':
        canvas = ROOT.TCanvas("myCanvas", "My Plot", 800, 600)

        fit_func.SetLineColor(ROOT.kRed)
        fit_func.SetLineWidth(2)
        fit_func.SetLineStyle(1)

        hist.Draw()
        fit_func.Draw("SAME")

        canvas.Update()
        canvas.Write()

        out_file.Close()
        #canvas.Print("/Users/justintraywick/research/NOPTREX/EDM/data/root/histogram_omega.pdf")

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

#Function to cut out signals in a certain range around a specified frequency on fft data
def sig_cut(fft_data,freq,width): 
    for i in range(np.shape(fft_data)[1]):
        if abs(fft_data[0,i]) > freq + width or abs(fft_data[0,i]) < freq - width:
            fft_data[1,i] = 0
    return(fft_data)

#Function to perform FFT on signal (in kHz) from specified numpy array
def scope_ft(signal,abs=False):
    tot_time = signal[0][-1] - signal[0][0] #Get total time
    sample_rate = (len(signal[1])) / tot_time #Get number of samples per second
	
    amplitudes = signal[1]
    n = len(signal[1])

    freq = np.fft.fftfreq(n, 1/sample_rate) #/ 1000 #Get frequencies from FFT. Divide by 1000 to get in kHZ
    amps = np.fft.fft(amplitudes) #Get amplitudes from FFT. Divide by 1000 to get in kHz. Magnitude of amplitudes.
    
    if abs==True:
           fft_data = np.array([freq[:n//2], np.abs(amps[:n//2])]) #Join arrays and consider only positive values (Symmetrical over y axis)
 
    else:
        fft_data = np.array([freq,amps])

    return fft_data

#Function that root fits to in ~root_fit~. Using this for plotting the fit. Takes parameter array returned by root fit.
def exp_sin_fit(x,par):
    return par[0] * np.exp(-(x + par[3]) / par[1]) *  np.sin(par[2] * 2 * np.pi * x + par[4]) + par[5]

#Function that fits only exponential of FID root fit function. Takes parameter array returned by root_fit.
def exp_fit(x, par):
    return par[0] * np.exp(-(x + par[3]) / par[1]) + par[5]

#Function that fits only sin function part of FID root fit function. Takes parameter array returned by root_fit.
def sin_fit(x, par):
   return np.sin(par[2] * 2 * np.pi * x) #+ par[5] #+ par[4]) + par[5] 

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

#Function to iterate through FID files and calculate average frequency values
def omeg_search():
    #Scan directories and get list of full path names for each file:
    plus_files = plus_dir + file_list(plus_dir)
    minus_files = minus_dir + file_list(minus_dir)

    #Define arrays for parameters and error from each piece of data
    omeg_p = []
    omeg_p_err = []
    omeg_m = []
    omeg_m_err = []

    #Iterate through plus files and process data:
    count = 0
    for i in plus_files:
        plus_file = i #Define file for processing

        plus_arr = csv2npy(plus_file) #Convert to numpy array

        plot(plus_arr, x_axis="Time(s)", y_axis="Voltage(V)", title=f"Positive E Field FID - file-{count}") #Plot positive E field FID to check range

        tmin = float(input(f"Enter the time value that the fit should begin at in miliseconds for Plot{count}: ")) #Get user input regarding range the fit should be performed over
        tmax = float(input(f"Enter the time value that the fit should end at in miliseconds for Plot{count}: ")) #Get user input regarding range the fit should be performed over
        
        plus_arr = munpy(plus_arr).range(tmin, tmax) #Slice array between tmin and tmax
        guess = np.array([-100,.002,37000,.001,20,0]) #Define guess array for fit parameters
        y_error = .128 #Define uncertainty in voltage

        fit = root_fit(plus_arr[0,:], plus_arr[1,:], guess, np.full_like(plus_arr[0,:], y_error)) #Perform fit and return parameters/ errors
        params = fit[0] #Get parameters
        errors = fit[1] #Get error on each parameter

        omeg_p.append(params[2]) #Append omega to list of omegas
        omeg_p_err.append(errors[2]) #Append omega's error to list of omega errors

        #Perform iterative fitting within the range specified and append omegas to omega lists
        tau, omega, chi2 = it_fit(tmin, tmax, 2, plus_arr) #Perform it_fit on plus E FID and return omega and tau distributions

        omeg_p = omeg_p + omega


        count+=1 #Iterate the count for indexing purposes

    #Iterate through minus files and process data:
    count = 0
    for i in minus_files:
        minus_file = i #Define file for processing

        minus_arr = csv2npy(minus_file) #Convert to numpy array

        plot(minus_arr, x_axis="Time(s)", y_axis="Voltage(V)", title=f"Negative E Field FID - file-{count}") #Plot Negative E field FID to check range

        tmin = float(input(f"Enter the time value that the fit should begin at in miliseconds for Plot{count}: ")) #Get user input regarding range the fit should be performed over
        tmax = float(input(f"Enter the time value that the fit should end at in miliseconds for Plot{count}: ")) #Get user input regarding range the fit should be performed over
        
        minus_arr = munpy(minus_arr).range(tmin, tmax) #Slice array between tmin and tmax
        guess = np.array([-100,.002,37000,.001,20,0]) #Define guess array for fit parameters
        y_error = .128 #Define uncertainty in voltage

        fit = root_fit(minus_arr[0,:], minus_arr[1,:], guess, np.full_like(minus_arr[0,:], y_error)) #Perform fit and return parameters/ errors
        params = fit[0] #Get parameters
        errors = fit[1] #Get error on each parameter

        omeg_m.append(params[2]) #Append omega to list of omegas
        omeg_m_err.append(errors[2]) #Append omega's error to list of omega errors

        #Perform iterative fitting on minus arrays and append to omeg_m
        tau, omega, chi2 = it_fit(tmin, tmax, 2, minus_arr)

        omeg_m = omeg_m + omega

        count+=1 #Iterate the count for indexing purposes

    #Convert all lists to numpy arrays
    omeg_p = np.array(omeg_p)
    omeg_p_err = np.array(omeg_p_err)
    omeg_m = np.array(omeg_m)
    omeg_m_err = np.array(omeg_m_err)

    p_savename = workdir + "npy/omegaplus_arr.npy" 
    m_savename = workdir + "npy/omegaplus_arr.npy"

    #Save values of omega into arrays for further processing
    np.save(p_savename, omeg_p)
    np.save(m_savename, omeg_m)

    #Generate histograms from plus and minus E field omega arrays
    root_hist(omeg_p, 100, workdir + "frequency_hists/plus/omeg_phist.root", title="Larmar Frequency(Hz), +E") #Create root histogram for omega plus values
    root_hist(omeg_m, 100, workdir + "frequency_hists/minus/omeg_m_hist.root", title="Larmar Frequency(Hz), -E") #Create root histogram for omega minus values

#Function to calculate the EDM value and its uncertainty
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

edm, sigma = edm_calc(37217.1, 37160, 2000, .0855, 0.5, 0.5, 5, 0.001)
print(edm, sigma)

##################################################################################


############  ~CLASSES~ ##########################################################

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
##################################################################################


############  ~MAIN CODE~  #######################################################

#Define Working directory:
workdir = "/Users/justintraywick/research/NOPTREX/EDM/data/"

#Define directories for FID from both E field configurations:
plus_dir = workdir + "plus_dir/"
minus_dir = workdir + "minus_dir/"

############ ~ITERATIVE APPROACH~ ################################################

#omeg_search()

#Calculate EDM value with uncertainty from each set of omegas
#edm, sigma = edm_calc(37139, 37308, 2000, .0855, 22, 24, 20, 0.001)
#print(edm, sigma)



############ ~INDIVIDUAL APPROACH~ ######################################################

#Define files for processing
plus_file = workdir + "plus_dir/plus7-92.csv" #Plus E
minus_file = workdir + "minus_dir/minus1.csv" #Minus E

#Convert files to numpy arrays
plus_arr = csv2npy(plus_file) #Plus E
minus_arr = csv2npy(minus_file) #Minus E

#Plot data to find fit range:
#plot(plus_arr, x_axis="Time(s)", y_axis="Voltage(V)", title="Positive E Field FID") #Plot positive E field FID
#plot(minus_arr, x_axis="Time(s)", y_axis="Voltage(V)", title="Negative E Field FID") #Plot negative E field FID

#Designate range to fit data on
data_arr = munpy(plus_arr).range(2, 20).astype(np.double)
#full_data = munpy(minus_arr).range(1.5,11)

#Get params from fit
guess = np.array([-100, .002,37000,.001,20,0])
y_error  = 0.15 #Define 150mV error
params = root_fit(data_arr[0,:], data_arr[1,:], guess, np.full_like(data_arr[0,:], y_error))[0]

fit_dat = np.array([data_arr[0,:], exp_sin_fit(data_arr[0,:], params)])
plot(fit_dat)

#Get array of fit values
#fit_arr = fit_graph(params, full_data[0,:], exp_sin_fit, arr_return=True)

#Get residuals against time value
#residuals = np.array([full_data[0,:], resid(full_data[1,:], fit_arr[1,:])])

#plot(residuals)

#chi2 = it_fit(1.6, 13, 1.5, minus_arr, numpy=True)[2] #Perform it_fit on plus E FID and return omega and tau distributions

#plot(np.arange(len(chi2)),chi2)







'''
#Fit data iteratively with root:
#tau, omega = it_fit(1.6, 13, 1.5, plus_arr) #Perform it_fit on plus E FID and return omega and tau distributions
#tau, omega = it_fit(1.5, 13.8, 5, minus_arr) #Perform it_fit on minus E FID and return omega and tau distributions 


#Save omega and tau as .npy files for loading:
omega_savename = workdir + "npy/omega_arr.npy" #Define omega .npy file name
tau_savename = workdir + "npy/tau_arr.npy" #Define tau .npy file name
#np.save(omega_savename, omega) #Save tau as .npy
#np.save(tau_savename, tau) #Save tau as .npy

#Load saved .npy files:
#omega = np.load(npy_savename)

#Fit a single section:
guess = np.array([-100, .002,37000,.001,20,0])
plus_arr = munpy(plus_arr).range(2, 17)
params = root_fit(plus_arr[0,:], plus_arr[1,:], guess, np.zeros_like(plus_arr[0,:]))[0]

#Generate histogram of frequency and time decay with root:
#root_hist(tau, 5, workdir + "frequency_hists/plus/omega_hist.root", title="Larmar Frequency(Hz), +E")
'''