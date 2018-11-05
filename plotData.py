"""Script to have a better look at Grey seal GPS data."""

# Import modules
import os
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

import plot_module as pm
import GPS_module as gps
import basic_module as bam
from GPSData import GPSDataset


# Define location of sable island
sable=[43.9499962, -59.909496362]
sableCart=gps.toCartesian(sable[0],sable[1])

# Pick some ID
ID=98439

# Load data
g=GPSDataset()
g.loadFile("GreySealData/greyseal2009_short.csv")

# Center 
g.centerCartesian(sableCart)

g.showStats()
raw_input()
##g.data[ID].swoopWin()
#for ID in g.IDs:
	##g.data[ID].computeWindowClusters(20,2000,show=False)
	
	#g.data[ID].computeRangedWindowClusters(15,25,10,show=False)
	
	#w=g.data[ID].findBestWindowSet()
	
	#w.plotClusters()
	#ax=None
	##colors=['b','g','r','m','y','orange','k']
	##for i,win in enumerate(w.windows):
		##ax=win.plotCartTraj(ax=ax,color=colors[i])
	
	
	
		
	#w.showClustersOnTracks()
	
	#raw_input()	
	
	



##for ID in g.IDs:
	

	
	
	##g.data[ID].cleanUpData()
	
	##g.data[ID].showAllPlots()

	##stats_module.
#raw_input()

## Create figure
#fig,axes=pm.makeSubplot([2,2])

## Draw map
#m=pm.drawMap(axes[0])




## Plot location of sable island
##pm.plotSquareAroundCoordinate(m,sable,1)



## Plot trajectories into basemap
#for ID in g.IDs:
	#g.data[ID].plotLonLatTraj(m)
	
	
## Plot trajectories in cartesian coordinates
#for ID in g.IDs:
        #g.data[ID].plotCartTraj(axes[1])
        

## Histogram over all distances
#g.plotDHist(axes[2])

## Histogram over all velocities
#g.plotVHist(axes[3])

#raw_input()



