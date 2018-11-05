import numpy as np
import stats_module
import plot_module as pm

class window:
        
        def __init__(self,idx,data):
                
		self.idx=idx
		self.x=data.x[idx]
		self.y=data.y[idx]
		self.d=data.d[idx]
		self.dt=data.dt[idx]
		self.v=data.v[idx]
		self.angles=np.array(data.angles)[idx[:-1]]
		
	def computeAllStats(self):
		
		"""Computes all key stats.
		
		Computes mean and std for distances, velocities, time duration, and movement angles.
		
		"""
		
		self.meand,self.stdd=self.computeStats(self.d)
		self.meandt,self.stddt=self.computeStats(self.dt)
		self.meanv,self.stdv=self.computeStats(self.v)
		self.meanangles,self.stdangles=self.computeStats(self.angles)
		
		if np.isnan(self.getKeyStats()).any():
		
			print self.idx
			raw_input()
	
		
	def computeStats(self,x):
		
		"""Computes basic stats, that is mean and std, of variable.
		
		Args:
			x (numpy.ndarray): Array of variable.
			
		Returns:
			tuple: Tuple containing:
			
				* float: Mean of variable.
				* float: std of variable.
		"""
		
		return np.mean(x), np.std(x)
	
	def getKeyStats(self):
		
		"""Returns key stats of window."""
		
		return self.meand,self.meandt,self.meanv,self.stdangles
	
	def plotCartTraj(self,ax=None,vel=[],showVel=False,centered=False,color='b',center=[0,0,0],showCenter=False):
		
		"""Plots trajectory of gps data in window.
		
		Keyword Args:
			ax (matplotlib.axes): Axes to plot in.
			vel (list): List of velocities or other values.
			showVel (bool): Show velocities.
			centered (bool): Show centered coordinates.
			color (str): Color of plot.
			center (list): Center of plot.
			showCenter (bool): Display center.
		
		Returns:
			matplotlib.axes: Modified axes.
		
		"""
		
		ax=pm.plotCartTraj(self.x,self.y,ax=ax,vel=vel,showVel=showVel,centered=centered,showCenter=showCenter,center=center,color=color)
		return ax
		
class windowSet:

	def __init__(self,windows,idxs,data):
		
		self.windows=windows
		self.idxs=idxs
		self.data=data
		
		self.computeAllStats()
		self.collectKeyStats()
		
		
	def computeAllStats(self):
		
		"""Computes all key stats for each window."""
		
		for w in self.windows:
			w.computeAllStats()
		
	def collectKeyStats(self):
		
		"""Collects all key stats from each window and stores them in nunmpy array."""
		
		stats=[]
		
		for w in self.windows:
			stats.append(w.getKeyStats())
			
		self.stats=np.array(stats)

	def performKMeans(self,nbouts=3):
		
		"""Performs kmeans clustering into nbouts bouts of the data.
		
		Keyword Args:
			nbouts (int): Number of clusters.
		
		"""
		
		x=stats_module.performKMeans(self.stats,nbouts)
		
		self.kMeans=x
		self.kMeansScore=self.kMeans.inertia_
		self.kMeansLabels=self.kMeans.labels_
		
		
	def plotClusters(self,ax=None,var='dta',showLabels=False,cpick='jet',alg='kMeans',nbouts=3):
		
		"""Shows scatter plot of gps data binned in windows.
		
		If algorithm is 'kMeans', performs additional kMeans and labels windows by bout.
		
		``var`` defines which variables are shown on x/y/z-axis in the order they are specified. 
		
			* 'd': distances
			* 't': deltaT
			* 'a': angles
			* 'v': velocities
			
		Keyword Args:
			ax (matplotlib.axes): Axes to plot in.
			var (str): Define which variables are plotted.
			showLabels (bool): Show labels.
			cpick (str): Colormap used for coloring.
			alg (str): Algorithm used for clustering.
			nbouts (int): Number of bouts to cluster data in.
			
		Returns:
			matplotlib.axes: Modified axes.
			
		"""
		
		idxs=[]
		
		# Figure out what to plot
		if 'd' in var:
			idxs.append(0)
		if 't' in var:
			idxs.append(1)
		if 'v' in var:
			idxs.append(2)
		if 'a' in var:
			idxs.append(3)
		
		# Cluster
		if alg=='none':
			labels=np.zeros(np.shape(self.stats)[0])
		if alg=='kMeans':
			self.performKMeans(nbouts=nbouts)
			labels=self.kMeansLabels
		
		# Create axes if necessary
		if ax==None:
			fig,axes=pm.makeSubplot([1,1],proj=['3d'])
			ax=axes[0]
		
		# Plot
		pm.labeledScatter(self.stats[:,idxs[0]],self.stats[:,idxs[1]],self.stats[:,idxs[2]],labels,ax=ax,cmap=cpick)
		
		# Labels
		ax.set_xlabel(var[0])
		ax.set_ylabel(var[1])
		ax.set_zlabel(var[2])
		ax.get_figure().canvas.draw()
		
		return ax
	
	def showClustersOnTracks(self,ax=None,centered=True,nbouts=3,showCenter=False):
		
		"""Plots all trajectory of gps data colorized by kmeans-clustered groups.
		
		Keyword Args:
			ax (matplotlib.axes): Axes to plot in.
			nbouts (int): Number of bouts to cluster data in.
			centered (bool): Show centered coordinates.
			center (list): Center of plot.
			showCenter (bool): Display center.
		
		Returns:
			matplotlib.axes: Modified axes.
		
		"""
		
		# Create axes if necessary
		if ax==None:
			fig,axes=pm.makeSubplot([1,1])
			ax=axes[0]
		
		#self.performKMeans(nbouts=nbouts)
		labels=self.kMeansLabels
		print labels
		colors=pm.getColors(len(np.unique(labels)))
		
		for i,w in enumerate(self.windows):
			w.plotCartTraj(ax=ax,vel=[],showVel=False,centered=centered,color=colors[labels[i]],center=[0,0,0],showCenter=showCenter)
		
		
		