import numpy as np
import pyvisa as pv
import time

GPIBname = pv.ResourceManager().list_resources() #Get list of GPIB connections

print(GPIBname)

#SG = pv.ResourceManager().open_resource(GPIBname[0]) #Define signal generator connection

#SG = pv.ResourceManager().open_resource('USB0::0x3923::0x709B::016AE8AD::RAW')

#SG.write("FREQ 5.0E+3")

