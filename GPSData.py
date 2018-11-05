"""Base class to save a GPS dataset."""

import GPS_module as gps
import basic_module as bam
import plot_module as pm
import stats_module 
from window import window,windowSet

import numpy as np
from datetime import datetime
import matplotlib.cm as cmap
import matplotlib.pyplot as plt


class GPSData:
        
        def __init__(self,ID,color='r'):
                
                self.ID=ID
                self.date=[]
                self.lon=[]
                self.lat=[]
                self.color=color
                self.Windows=[]
                 
        def readData(self,header,data,fixLon=True):
                
                """Reads data from array.
                
                Reads in data and subsequently standardizes data and finally computes movement stats.
                
                Args:
			header (list): Header list.
			data (nunmpy.ndarray): Array with GPS data.
			
                Keyword Args:
			fixLon (bool): Fixes sign error in longitude of GPS data.
                
                """
                
                idx=np.where(data[:,header.index("id")].astype(int)==self.ID)[0]     
                
                self.date=data[:,header.index("date")][idx] # Still need to convert this to datetime
                self.lat=data[:,header.index("lat")].astype(float)[idx]         
                self.lon=data[:,header.index("lon")].astype(float)[idx]
                
                if fixLon and max(self.lon)>180:
                        self.lon=self.lon-360.
		
		
		self.standardizeData()
		self.computeMoveStats()
		
		#self.findRestPeriods(meth='kMeans')
		#self.findRestPeriods(meth='thresh')
		
	def standardizeData(self):
		
		"""Standardizes data by converting data to cartesian coordinates, distances between coordinates and timestamps to datetime objects.
		"""
		
		self.toCartesian()
                self.computeDistances()
		self.toDateTime()
		
	
	def computeMoveStats(self):
		
		"""Computes basic stats of GPS data.
		
		Including:
			* Duration of timesteps.
			* Velocities.
			* Movement angles.
		
		"""
		
		self.computeDeltaT()
		self.computeVelocity()
		self.computeAngle()

	def findPartitions(self,n):
		
		"""Partitions GPS data into n+1 subsets.
		
		Args:
			n (int): Number of subsets.
		
		Returns:
			tuple: Tuple containing:
			
				* list: List of subsets (windows).
				* lust: Indices that define windows.
		
		"""
		
		
		# Create random numbers
		N=len(self.dt)
		I=bam.getIncreasingRandInt(n,N)
		
		# Close intervals
		I=np.array(list(I)+[N])
		
		# Array of windows
		windows=[]
		Idxs=[]
		
		# Create indices interval
		k=0
		for i in I:
	
			idxs=np.arange(k,i)
			k=i
			
			windows.append(window(idxs,self))	
			
			Idxs.append(idxs)
			
		return windows,idxs
	
	def findNPartitions(self,n,N):
		
		"""Creates N partitions of GPS data into n+1 subsets.
		
		Args:
			n (int): Number of subsets.
			N (int): Number of partitions.
			
		Returns:
			list: List of Windowsets.
		
		"""
		
		Windows=[]
		Idxs=[]
		
		for i in range(N):
			w,I=self.findPartitions(n)
			Windows.append(windowSet(w,I,self))
			
		return Windows
		
	def computeWindowClusters(self,n,N,show=False,add=True,nbouts=3):
		
		"""Bins movement data in time windows and performs clustering by movement stats.
		
		Args:
			n (int): Number of windows of the dataset.
			N (int): Number of window sets to be created.
			
		Keyword Args:
			show (bool): Show cluster plots.
			add (bool): Add new window sets.
			nbouts (int): Number of clusters/bouts/
			
		Returns:
			list: List of window sets.
		"""
		
		if add:
			self.Windows=self.Windows+self.findNPartitions(n,N)
		else:
			self.Windows=self.findNPartitions(n,N)
		
		for w in self.Windows:
			w.performKMeans(nbouts=nbouts)
		
		if show:
			self.showClusters()
		
		return self.Windows
	
	def computeRangedWindowClusters(self,nMin,nMax,N,show=False,add=True,nbouts=3,printStatus=True):	
		
		"""Bins movement data in time windows and performs clustering by movement stats for a range of different window sizes.
		
		Args:
			nMin (int): Minimal number of windows of the dataset.
			nMax (int): Maximal number of windows of the dataset.
			N (int): Number of window sets to be created.
			
		Keyword Args:
			show (bool): Show cluster plots.
			add (bool): Add new window sets.
			nbouts (int): Number of clusters/bouts/
			
		Returns:
			list: List of window sets.
		"""
		
		ns=np.arange(nMin,nMax)
		
		for n in ns:
			print "Computing window set with n = ", n ," windows."
			self.Windows=self.computeWindowClusters(n,N,show=show,add=add,nbouts=nbouts)
		
		return self.Windows
	
	def findBestWindowSet(self):
		
		"""Returns the optimal partition of all window sets by minimal kmean scores.
		
		Returns:
			window.windowSet: Optimal window set.
		
		"""
		
		scores=[]
		for w in self.Windows:	
			scores.append(w.kMeansScore)
				
		return self.Windows[scores.index(min(scores))]
		
	def showClusters(self,axes=None):
		
		# Get number of window sets
		N=len(self.Windows)
			
		# Compute number of plots displayed
		Nx=int(np.ceil(N/2.))
		Ny=int(np.floor(N/2.))
		
		# Show scatter plot for each window set
		proj=N*['3d']
		fig,axes=pm.makeSubplot([Nx,Ny],proj=proj)
		for i,w in enumerate(self.Windows):
			w.plotClusters(ax=axes[i],var='dta')
		
		# Show labeled trajectories for each set
		fig,axes=pm.makeSubplot([Nx,Ny])	
		for i,w in enumerate(self.Windows):
			w.showClustersOnTracks(ax=axes[i])
			

        def plotLonLatTraj(self,ax=None,onMap=True,lw=2):
                
		"""Plots trajectory in lon/lat of this specific dataset.
		
		.. note:: Creates new figure if axes is not specified.
		
		Keyword Args:
			ax (matplotlib.axes): Axes or basemap to plot in.
			onMap (bool): Plot on map.
			lw (str): Linewidth of plot.
			color (str): Color of trajectory.
			
		Returns:
			matplotlib.axes: Modified axes.
			
		"""	
        
                ax=pm.plotLonLatTraj(self.lon,self.lat,ax=ax,onMap=onMap,color=self.color,lw=lw)
                
                return ax
        
        def plotCartTraj(self,ax=None,vel=[],showVel=True,centered=True,showCenter=True):
		
		"""Plots cartesian trajectory of specific dataset.
		
		.. note:: Creates new figure if axes is not specified.
	
		Args:
			x (numpy.ndarray): Array with x-coordinates.
			y (numpy.ndarray): Array with y-coordinates.
			
		Keyword Args:
			ax (matplotlib.axes): Axes to plot in.
			vel (list): List of velocities or other values.
			showVel (bool): Show velocities.
			centered (bool): Show centered coordinates.
			showCenter (bool): Display center.
		
		Returns:
			matplotlib.axes: Modified axes.
		
		"""
	
		if centered:
			x=np.array(self.xC)
			y=np.array(self.yC)
		else:
			x=np.array(self.x)
			y=np.array(self.y)
		
		if showVel and len(vel)==0:
			vel=self.v
			
		if centered:	
			pm.plotCartTraj(x,y,ax=ax,vel=vel,showVel=showVel,centered=centered,showCenter=showCenter,center=[0,0,0],color=self.color)
		else:
			pm.plotCartTraj(x,y,ax=ax,vel=vel,showVel=showVel,centered=centered,showCenter=showCenter,center=self.center,color=self.color)
			
		return ax
	
	def toCartesian(self,r=6378137,e=8.1819190842622e-2,h=0):

		"""Converts the GPS data to cartesian coordinates."""
		
		self.x,self.y,self.z=gps.toCartesian(self.lat,self.lon,r=r,e=e,h=h)
	
	def centerCartesian(self,c):
		
		"""Centers cartesian coordinates around center.
		
		Args:
			c (list): Coordinates of center.
		
		"""
		
		self.xC=self.x-c[0]
		self.yC=self.y-c[1]
		
		self.flipCoords()
		#self.transform2KM(self.xC,self.yC)
		

		self.center=-1*np.array(c)/1000.
		
	def flipCoords(self):
		
		"""Multiplies all coordinates by -1 such that lon/lat plots look the same as cartesian."""
		
		self.xC=-1*self.xC
		self.yC=-1*self.yC
	
	def transform2KM(self,x,y):
		
		"""Transforms distances to km.
		
		Args:
			x (numpy.ndarray): Cartesian x-coordinates in m.
			y (numpy.ndarray): Cartesian y-coordinates in m.
		
		Returns:
			tuple: Tuple containing:
			
				* numpy.ndarray: x-coordinates in km.
				* numpy.ndarray: y-coordinates in km.
		
		"""
		
		return x/1000.,y/1000.
		
		
	def computeDistances(self):
		
		"""Computes distances of all steps."""
		
		self.d=gps.computeDistances(self.x,self.y)
		
	def toDateTime(self):
		
		"""Converts dates to datetime objects."""
		
		self.date=list(self.date)
		self.date=[datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in self.date]
	
	def computeAngle(self):
		
		"""Computes all turning angles."""
		
		angles=[]
		
		# Get directions
		for i in range(len(self.x)-2):
			
			# Define movement vectors
			vec1=np.array([self.x[i+1]-self.x[i],self.y[i+1]-self.y[i]])
			vec2=np.array([self.x[i+2]-self.x[i+1],self.y[i+2]-self.y[i+1]])
			
			# Compute angles
			angle = bam.direcAngle(vec1,vec2)
			angles.append(angle)
		
		self.angles=angles
		
		return self.angles
		
	def computeDeltaT(self):
		
		"""Computes deltaT between timepoints in seconds."""
		
		self.dt=[]
		
		for i in range(len(self.date)-1):
			t=self.date[i+1]-self.date[i]
			self.dt.append(t.seconds)
		
		self.t=[self.dt[0]]
		
		for d in self.dt[1:]:
			self.t.append(self.t[-1]+d)
		
		self.dt=np.array(self.dt)
		self.t=np.array(self.t)
		
	def computeVelocity(self):
		
		"""Computes velocity for each step."""
		
		self.v=self.d/np.array(self.dt)
		
	
		
	#def showBouts(self,ax):
		
		#self.findBouts()
		
		#ax.scatter(self.dt,self.d,c=self.bouts.labels_)
		
		#plt.draw()
		
		
	#def findBouts(self,meth='kMeans',nBouts=3):
		
		#self.bouts=stats_module.performKMeans(self.dt,self.d,nBouts)
		
		#print len(np.where(self.bouts.labels_>1.5)[0])
		#print len(np.where(self.bouts.labels_>0.5)[0])
	
	#def findRestPeriods(self,meth='thresh',thresh=3600.):
		
		#if meth=='thresh':
		
			#restIdx=np.where(self.dt>thresh)[0]
		
			
		#if meth=='PCA':
			
			#stats_module.performPCA(self.dt,self.d,2)
	
		#if meth=='kMeans':
			
			#x=stats_module.performKMeans(self.dt,self.d,2)
			#restIdx=np.where(x.labels_>0.1)[0]
		
		#self.restIdx=restIdx
	
	#def showAllPlots(self):
		
		#"""Shows all plots for one individual."""
		
		## Create figure
		#fig,axes=pm.makeSubplot([3,5],sup="Individual "+str(self.ID))
		
		## Draw map
		#m=pm.drawMap(axes[0])

		## Plot trajectories into basemap
		#self.plotLonLatTraj(m)
			
		## Plot trajectories in cartesian coordinates
		#self.plotCartTraj(axes[1])
		
		## Histogram over distances
		#self.plotDHist(axes[2])

		## Histogram over velocities
		#self.plotVHist(axes[3])
		
		#self.plotDvsV(axes[4])
	
		#self.plotDOverT(axes[5])
		
		#self.plotVOverT(axes[6])
		
		#self.plotPhiOverT(axes[7])
		
		#self.plotPhivsV(axes[8])
		
		#self.plotPhivsD(axes[9])
		
		#self.plotDT(axes[10])
		
		#self.plotDTvsD(axes[11])
		
	
	
	
	#def plotDHist(self,ax,bins=100):
		
		#"""Creates histogram of steplength distribution of individual."""
		
		#ax=self.plotHist(ax,self.d,bins=bins,xlabel="Stepsize (m)",ylabel="Frequency",title="Overall stepsizes")
		#plt.draw()
		
		#return ax
	
	#def plotVHist(self,ax,bins=100):
		
		#"""Creates histogram of speed distribution of individual."""
		
		#self.plotHist(ax,self.v,bins=bins,xlabel="Velocity (m/s)",ylabel="Frequency",title="Overall velocities")	
		#plt.draw()
		
		#return ax

	#def plotHist(self,ax,x,bins=100,xlabel="",ylabel="",title=""):
		
		#"""Histogram plotting function."""
		
		#pm.setLabels(ax,xlabel=xlabel,ylabel=ylabel,title=title)
		#ax.hist(x,bins=bins)
		
		#return ax
	
	#def makeVSPlot(self,ax,vX,vY,scaleX=1,scaleY=1,xlabel="",ylabel="",title="",color=None):
		
		#"""Creates scatter plot of two variables against each other."""
		
		#if color==None:
			#color=self.color
		
		#ax.scatter(np.array(vX)/scaleX,np.array(vY)/scaleY,color=color)	
		#pm.setLabels(ax,xlabel=xlabel,ylabel=ylabel,title=title)
		
		#return ax
	
	#def makeOverPlot(self,ax,vX,vY,scaleX=1,scaleY=1,xlabel="",ylabel="",title="",color=None):
		
		#"""Creates line plot of two variables against each other."""
		
		#if color==None:
			#color=self.color
		
		#ax.plot(np.array(vX)/scaleX,np.array(vY)/scaleY,color=color)	
		#pm.setLabels(ax,xlabel=xlabel,ylabel=ylabel,title=title)
		
		#return ax
	
	#def plotDvsV(self,ax):
		
		#"""Creates scatter plot showing steplength vs speed of individual."""
		
		#self.makeVSPlot(ax,self.d,self.v,xlabel="Stepsize (m)",ylabel="Velocity (m/s)",title="D vs V")
	
	#def plotDOverT(self,ax):
		
		#"""Creates line plot showing time vs distance of individual."""
		
		#self.makeOverPlot(ax,self.t,self.d,scaleX=3600.,xlabel="Time (h)",ylabel="Stepsize (m)",title="Stepsize over Time")
		
	#def plotVOverT(self,ax):
		
		#"""Creates line plot showing time vs velocity of individual."""
		
		#self.makeOverPlot(ax,self.t,self.v,scaleX=3600.,xlabel="Time (h)",ylabel="Velocity (m/s)",title="Velocity over Time")
	
	#def plotPhiOverT(self,ax):
		
		#"""Creates line plot showing time vs angles of individual."""
	
		#self.makeOverPlot(ax,self.t[1:],self.angles,scaleX=3600.,xlabel="Time (h)",ylabel="Phi (radians)",title="Turning angles over Time")
		
	#def plotDTvsD(self,ax):
		
		#"""Creates scatter plot showing dt vs steplength of individual."""
		
		#self.makeVSPlot(ax,self.dt,self.d,scaleX=3600.,xlabel="Duration (h)",ylabel="Stepsize (m)",title="DT vs D")
		
	#def plotPhivsV(self,ax):
		
		#"""Creates scatter plot showing angles vs velocity of individual."""
	
		#self.makeVSPlot(ax,self.angles,self.v[:-1],xlabel="Turning angle (radians)",ylabel="Velocity (m/s)",title="Phi vs V")
		
	#def plotPhivsD(self,ax):
		
		#"""Creates scatter plot showing angles vs steplengths of individual."""
		
		#self.makeVSPlot(ax,self.angles,self.d[:-1],xlabel="Turning angle (radians)",ylabel="Distance (m)",title="Phi vs D")	
	
	#def plotDT(self,ax):
		
		#"""Creates scatter plot showing angles vs steplengths of individual."""
		
		#self.makeOverPlot(ax,np.arange(len(self.dt)),self.dt,scaleX=100.,scaleY=3600.,xlabel="Timepoint (100s)",ylabel="Time inbetween (h)",title="Datapoint vs. DeltaT")
	
	
	#def cleanUpData(self):
		
		## Create figure
		#fig,axes=pm.makeSubplot([2,2],sup="Individual "+str(self.ID),proj=[None,None,None,'3d'])
		
		## Scatter of kmeans
		#axes[0].scatter(self.dt[self.restIdx],self.d[self.restIdx],color='r')
		#axes[0].scatter(np.delete(self.dt, self.restIdx),np.delete(self.d, self.restIdx),color='b')
		
		##
		#vel=np.zeros(np.shape(self.dt))
		#vel[self.restIdx]=1
		#self.plotCartTraj(axes[1],vel=vel)
		
		#plt.draw()
		
		
		
		
		##self.showBouts(axes[2])
		
		#print len(self.angles)
		#print self.dt.shape
		
		##raw_input()
		
		#axes[3].scatter(self.dt,self.d,[0]+list(self.angles))
		#plt.draw()
		
		#raw_input()
		
	#def swoopWin(self,lenWin=60.*60.):
		
		#wins=[]
		#win=[]
		#sumT=0
		
		#for i,t in enumerate(self.dt):
			
			
			
			
			#sumT=sumT+t
			
			
			
			#if sumT>lenWin:
				
				
				
				#win=[i]
				
				
				#wins.append(win)
				
				#sumT=t
					
			#else:
				#win.append(i)
			
			
			
		#return wins
	
	
	
class GPSDataset:
	
	
	def __init__(self):
		
		self.data={}
		self.IDs=[]
			
	def loadFile(self,fn):
		
		"""Loads GPS data from file and creates a GPSData object for each individual.
		
		Args:
			fn (str): Path to gps data text file.
		
		"""
		
		# Load data
		self.rawData=np.loadtxt(fn,dtype=str,delimiter=',')
		self.header=bam.removeQuotationMarksFromArr(list(self.rawData[0]))
		
		# Sort by individual
		self.data=self.getIndividuals()
		
	def getIndividuals(self):
		
		"""Finds individuals in data and assigns data into dict and GPS data."""
		
		# Create dict
		newdata={}
	
		# Get list of IDs
		self.IDs=np.unique(self.rawData[1:,self.header.index("id")]).astype(int)
		
		
		# Create list of colors
		colors=self.getColors()
		
		for i,ID in enumerate(self.IDs):
			g=GPSData(ID,color=colors[i])
			g.readData(self.header,self.rawData[1:])
			newdata[ID]=g
		
		self.data=newdata
			
		return self.data
	
	def centerCartesian(self,c):
		
		"""Centers all GPS datasets around center.
		
		Args:
			c (list): Coordinates of center.
		"""
		
		for ID in self.IDs:
			
			self.data[ID].centerCartesian(c)
	
	def getColors(self,cpick='jet'):
		
		"""Creates a list of distinct colors for each individual.
		
		Keyword Args:
			cpick (str): Colormap to pick colors from.
			
		Returns:
			list: List of colors.
		"""	
		
		# Get number of individuals
		n=len(self.IDs)
		
		return pm.getColors(n,cpick=cpick)
	
	def collectProp(self,prop):
		
		"""Collects defined property from all individuals.
		
		Args:
			prop (str): Name of property.
			
		Returns:
			list: Collected properties.
		"""
		
		p=[]
		
		for ID in self.IDs:
			p=p+list(getattr(self.data[ID],prop))
			
		return p
		
	def collectDistances(self):
		
		"""Collects all distances of all individuals.
		
		Returns:
			list: Collected distances.
		"""
		
		return self.collectProp('d')
	
	def collectVelocities(self):
		
		"""Collects all velocities of all individuals.
		
		Returns:
			list: Collected distances.
		"""
		
		return self.collectProp('v')
	
	def collectAngles(self):
		
		"""Collects all angles of all individuals.
		
		Returns:
			list: Collected angles.
		"""
		
		return self.collectProp('angles')
	
	def collectDeltaTs(self):
		
		"""Collects all angles of all individuals.
		
		Returns:
			list: Collected angles.
		"""
		
		return self.collectProp('dt')
	
	def propHist(self,prop,ax=None,bins=100,xlabel="",ylabel="",title="",xlim=[],ylim=[]):
		
		"""Creates a histogram of a specific property.
	
		.. note:: Creates new figure if axes is not specified.
		
		Args:
			p (list): List of values to make hist of.
			
		Keyword Args:
			ax (matplotlib.axes): Axes.
			bins (int): Number of bins.
			xlabel (str): Label of x-axis.
			ylabel (str): Label of y-axis.
			title (str): Plot title.
			xlim (list): Limits of x-axis.
			ylim (list): Limits of y-axis.
		
		Returns:
			matplotlib.axes: Modified axes.
		"""
		
		# Collect property
		p=self.collectProp(prop)
		
		# Make hist
		ax=pm.plotHist(p,ax=ax,bins=bins,xlabel=xlabel,ylabel=ylabel,title=title,xlim=xlim,ylim=ylim)
		
		return ax
		
	def plotDHist(self,ax=None,bins=100):
		
		"""Creates a histogram of all distances of all individuals.
	
		.. note:: Creates new figure if axes is not specified.
		
		Keyword Args:
			ax (matplotlib.axes): Axes.
			bins (int): Number of bins.
		
		Returns:
			matplotlib.axes: Modified axes.
		"""
		
		ax=self.propHist('d',ax=ax,bins=bins,xlabel="Stepsize (m)",ylabel="Frequency",title="Overall stepsizes")
		
		return ax
		
	def plotVHist(self,ax=None,bins=100):
		
		"""Creates a histogram of all velocities of all individuals.
	
		.. note:: Creates new figure if axes is not specified.
		
		Keyword Args:
			ax (matplotlib.axes): Axes.
			bins (int): Number of bins.
		
		Returns:
			matplotlib.axes: Modified axes.
		"""
		
		ax=self.propHist('v',ax=ax,bins=bins,xlabel="Velocity (m/s)",ylabel="Frequency",title="Overall velocities")
		
		return ax
	
	def plotAnglesHist(self,ax=None,bins=100):
		
		"""Creates a histogram of all angles of all individuals.
	
		.. note:: Creates new figure if axes is not specified.
		
		Keyword Args:
			ax (matplotlib.axes): Axes.
			bins (int): Number of bins.
		
		Returns:
			matplotlib.axes: Modified axes.
		"""
		
		ax=self.propHist('angles',ax=ax,bins=bins,xlabel="Angle (radians)",ylabel="Frequency",title="Overall angles")
		
		return ax
	
	def plotDeltaTHist(self,ax=None,bins=100):
		
		"""Creates a histogram of all delta ts of all individuals.
	
		.. note:: Creates new figure if axes is not specified.
		
		Keyword Args:
			ax (matplotlib.axes): Axes.
			bins (int): Number of bins.
		
		Returns:
			matplotlib.axes: Modified axes.
		"""
		
		ax=self.propHist('dt',ax=ax,bins=bins,xlabel="dt (s)",ylabel="Frequency",title="Overall dt")
		
		return ax
	
	def showStats(self,axes=[],bins=100):
		
		"""Creates four histograms showing overall movement stats of dataset of all delta.
	
		.. note:: Creates new figure if axes is not specified.
		
		Keyword Args:
			ax (matplotlib.axes): Axes.
			bins (int): Number of bins.
		
		Returns:
			list: List of modified axes.
		"""
		
		# Create axes if not specified
		if len(axes)!=4:
			fig,axes=pm.makeSubplot([2,2])
		
		self.plotDHist(ax=axes[0],bins=bins)
		self.plotVHist(ax=axes[1],bins=bins)
		self.plotAnglesHist(ax=axes[2],bins=bins)
		self.plotDeltaTHist(ax=axes[3],bins=bins)
		
		return axes
	
	