#Script to perform discrete schroeder integration on a signal
import numpy as np
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
import seaborn as sns

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

############# ~ROOT FIT FUNCTIONS~ ###############################################
#Functions which return root function objects that are callable in root_hist() function. Initial parameters are defaulted, but can be reguessed using kwargs.

#Lorentzian fit
def lor_root(x, par, guess_arr=np.array([1, 60, -38000,0])): #guess_arr is of size: 4
    #Define Lorentzian fit ROOT object
    return par[3] + (par[0] * (par[1]/2) ** 2) / ((x[0] - par[2]) ** 2 + (par[1] / 2) **2)

def lor_call():
    return "Lorentzian", lor_root
#Perform FID fit with root
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

#Gaussian fit
def gaus_root(x, par, guess_arr=np.array([1, 60, -38000,0])):
    return par[0] + par[1] * ROOT.TMath.Gaus(x[0], par[2], par[3])

def gaus_call():
    return "gaussian", gaus_root
    
##################################################################################

#Function that root fits to in ~root_fit~. Using this for plotting the fit. Takes parameter array returned by root fit.
def exp_sin_fit(x,par):
    return par[0] * np.exp(-(x + par[3]) / par[1]) *  np.sin(par[2] * 2 * np.pi * x + par[4]) + par[5]

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

#Perform EMD
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

##########~MAIN CODE~#####################################################################

#Load in file
workdir = "/Users/justintraywick/research/NOPTREX/EDM/"
filename = workdir + "good_data/data/2/minus2.csv"


data = csv2npy(filename)
#plot(data, plotstyle="mlp")
data = munpy(data).range(float(2),float(9))
inst_freq = h_huang(data)
#plot(inst_freq)
freqs = munpy(inst_freq).range(2,6)

test = 0

if test == 1:

    reduced_chi2 = []
    for i in range(300):
        #Generate root histogram of frequency vs. time data and fit to Lorentzian
        reduced_chi2.append(root_hist(freqs[1,:], i+10, workdir + "data/frequency_hists/test.root", title = "-E Config. Frequency Distrubution", start=10, stop=-15,root_func=lor_call)[1])

    diff_chi2 = np.abs(np.array(reduced_chi2) - 1)

    bin_count = np.argmin(diff_chi2) + 10

    print(bin_count)

else:
    pass
    #root_hist(freqs[1,:], 100, workdir + "/good_data/data/2/minus_lor.root", title = "-E Config. Frequency Distrubution", start=3, stop=-6,root_func=gaus_call)


d1 = freq_dif(37169.57, 37217, 14.68, 11.41)
d2 = freq_dif(37117.724, 37141.996, 29.98, 14.337)
d3 = freq_dif(37200.99, 37222.78, 17.13, 33.06)
#d4 = freq_dif(37153.4843, 7.8266) Not viable
d5 = freq_dif(37338.5316, 37358.3275, 38.98, 18.64)

diff_gr = np.array([d1[0],d2[1],d3[1],d5[1]])
print(diff_gr)

#d4 = freq_dif(37240, 37347, 14, 13.6)
