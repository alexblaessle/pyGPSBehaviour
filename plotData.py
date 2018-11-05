"""Script to have a better look at Grey seal GPS data."""

# Import modules
import os
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

import plot_module as pm
from GPSData import GPSDataset
import GPS_module as gps


# Define location of sable island
sable=[43.9499962, -59.909496362]
sableCart=gps.toCartesian(sable[0],sable[1])

# Pick some ID
ID=98439

# Load data
g=GPSDataset()
#g.loadFile("../GreySealData/greyseal2009_short.csv")
g.loadFile("../GreySealData/greyseal2009.csv")

# Center 
g.centerCartesian(sableCart)

# Plot all data
g.showData()
#g.showStats()
raw_input()




