"""Plot module for grey seal project."""

# Import modules
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as bm
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmap

def makeSubplot(size,titles=None,tight=False,sup=None,proj=None,fig=None,show=True):
	
	"""Generates matplotlib figure with (x,y) subplots.
	
	.. note:: List of ``titles`` needs to be same size as desired number of axes.
	   Otherwise will turn off titles.
	
	.. note:: List of ``proj`` needs to be same size as desired number of axes.
	   Otherwise will turn off projections.
	   
	Example:
	
	>>> makeSubplot([2,2],titles=["Axes 1", "Axes 2", "Axes 3", "Axes 4"],proj=[None,None,'3d',None])
	
	Args:
		size (list): Size of subplot arrangement.
	
	Keyword Args:
		titles (list): List of axes titles.
		tight (bool): Use tight layout.
		sup (str): Figure title.
		proj (list): List of projections.
		fig (matplotlib.figure): Figure used for axes.
		show (bool): Show figure right away.
	
	Returns:
		tuple: Tuple containing:
			
			* fig (matplotlib.figure): Figure.
			* axes (list): List of Matplotlib axes.
	
	"""
	
	#How many axes need to be created
	n_ax=size[0]*size[1]
	
	if proj==None:
		proj=n_ax*[None]
	
	#Creating figure
	if fig==None:
		fig=plt.figure()
		if show:
			fig.show()
	fig.set_tight_layout(tight)
	
	#Suptitle
	if sup!=None:
		fig.suptitle(sup)
	
	#Add axes
	axes=[]
	for i in range(n_ax):
		
		try:
			if proj[i]!=None:
				ax=fig.add_subplot(size[0],size[1],i+1,projection=proj[i])	
			else:
				ax=fig.add_subplot(size[0],size[1],i+1)
		except IndexError:
			print "Axes " + str(i) + "does not have a projection specified. Will create normal 2D plot."
			ax=fig.add_subplot(size[0],size[1],i+1)
				
		axes.append(ax)
		
		#Print titles 
		if titles!=None:
			try:
				ax.set_title(titles[i])
			except IndexError:
				print "Axes " + str(i) + " does not have a title specified."
	#Draw
	plt.draw()
	
	#Return axes handle
	return fig,axes


def drawSquareAroundCoordinate(m,c,s):
	
	"""Draws square around coordinate on map.
	
	Args:
		m (matplotlib.axes): Basemap to be drawn on.
		c (numpy.ndarray): Center coordinate of square.
		s (float): Half sidelength of square
	
	Returns:
		matplotlib.axes: Modified axes.
	
	"""
	
	m.plot([c[0]],[c[1]],color='k',lw=20,latlon=True)
	
	m.plot([c[0]-s,c[0]+s,c[0]+s,c[0]-s],[c[1]-s,c[1]-s,c[1]+s,c[1]+s],color='k',lw=10,latlon=True)
	plt.draw()
	
	return m

def drawMap(ax):
	
	"""Creates basemap with standard properties.
	
	Args:
		ax (matplotlib.axes): Axes to display plots in.
		
	Returns:
		matplotlib.basemap.basemap: Basemap.
	
	"""
	
	# Create basemap
	m = bm.Basemap(#llcrnrlon=-lonMin,llcrnrlat=latMin,urcrnrlon=lonMax,urcrnrlat=latMax,
		projection='lcc',lat_0=45.,lon_0=-63.,
		resolution ='l',area_thresh=10000.,ax=ax,suppress_ticks=True,
		width=1000*1E3, height=700*1E3)
	
	# Some settings
	m.drawcoastlines()
	m.fillcontinents(color='coral',lake_color='aqua')
	m.drawmapboundary(fill_color='aqua')
	m.drawparallels(np.arange(-80,80,10.))
	m.drawmeridians(np.arange(-180,180.,10))
	
	return m

def setLabels(ax,xlabel="",ylabel="",title=""):
	
	"""Sets labels of axis.
	
	Args:
		ax (matplotlib.axes): Axes.
		
	Keyword Args:	
		xlabel (str): String for x-label.
		ylabel (str): String for y-label.
		title (str): String for title.
		
		
	Returns:
		matplotlib.axes: Modified axes.

	"""
	
	# Label
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	
	# Title
	ax.set_title(title)
	
	return ax

def zoomToLimits(ax,x,y):
	
	"""Adjusts xlim and ylim to actual minimum and maximum of data.
	
	Args:
		ax (matplotlib.axes): Axes.
		x (numpy.ndarray): List of x-values.
		y (numpy.ndarray): List of y-values.
	
	Returns:
		matplotlib.axes: Modified axes.
	
	"""
	
	ax.set_xlim([min(x),max(x)])
	ax.set_ylim([min(y),max(y)])
	
	return ax
	
def coordsToLC(x,y,vals,cmap='jet'):
	
	"""Converts coordinates to LineCollection and labels them with colormap.
	
	Args:
		x (numpy.ndarray): List of x-values.
		y (numpy.ndarray): List of y-values.
		vals (numpy.ndarray): Segment values.
	
	Keyword Args:	
		cmap (str): Name of colormap.
	
	Returns:
		matplotlib.collections.LineCollection: LineCollection of the data.
	
	"""
	
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)

	lc = LineCollection(segments, cmap=plt.get_cmap(cmap),norm=plt.Normalize(0, max(vals)))
	
	lc.set_array(vals)
	lc.set_linewidth(3)
	
	return lc
	
def labeledScatter(x,y,z,labels,ax=None,cmap='jet'):
	
	"""Creates labeled 3D scatter plot.
	
	.. note:: Creates new figure if axes is not specified.
	
	Args:
		x (numpy.ndarray): Values on x-axis.
		y (numpy.ndarray): Values on y-axis.
		z (numpy.ndarray): Values on z-axis.
		labels (numpy.ndarray): Values to labels.
		
	Keyword Args:	
		cmap (str): Name of colormap.
		ax (matplotlib.axes): Axes to plot in.
	
	Returns:
		matplotlib.axes: Modified axes.
		
	"""
	
	# Get number of labels
	if len(np.unique(labels))>1:
		labelVals=np.sort(np.unique(labels))	
	else:	
		labelVals=np.unique(labels)	
	nLabels=len(labelVals)
	
	# Get some colors
	colors=getColors(nLabels,cpick=cmap)
	if len(colors)==1:
		colors=['b']
	
	# Create figure if necessary
	if ax==None:
		fig,axes=makeSubplot([1,1],proj=['3d'])
		ax=axes[0]
	
	# Plot each label
	for i in range(nLabels):
		
		# Get indices of this particular label
		idxs=np.where(labels==labelVals[i])[0]
		ax.scatter(x[idxs],y[idxs],z[idxs],color=colors[i])
		
	return ax

def getColors(n,cpick='jet'):
	
	"""Returns n different colors from a colormap.
	
	Args:
		n (int): Number of colors
		cpick (str): Name of cmap.
		
	Returns:
		list: List of colors.
	"""
	
	# Create colormap
	c = cmap.get_cmap(cpick)
	
	# Create list of colors
	colors=[]
	for i in range(n):
		colors.append(c(float(i)/n))
	
	return colors

def plotLonLatTraj(lon,lat,ax=None,lw=2,color='k'):
	
	"""Plots trajectory in lon/lat.
        
        .. note:: Creates new figure if axes is not specified.
        
        Args:
		lon (numpy.ndarray): Array with longitude coordinates.
		lat (numpy.ndarray): Array with latitude coordinates.
        
        Keyword Args:
		ax (matplotlib.axes): Axes or basemap to plot in.
		onMap (bool): Plot on map.
		lw (str): Linewidth of plot.
		color (str): Color of trajectory.
		
	Returns:
		matplotlib.axes: Modified axes.
		
	"""	
        
        # Create axes if necessary
	if ax==None:
		fig,axes=makeSubplot([1,1])
		ax=axes[0]
        
        # Plot
	if onMap: 
		ax.plot(lon,lat,latlon=True,lw=lw,color=color)
	else: 
		ax.plot(lon,lat,lw=lw,color=color)
	
	# Redraw
	ax.get_figure().canvas.draw()
	
	return ax

def plotCartTraj(x,y,ax=None,vel=[],showVel=True,centered=True,showCenter=False,center=[],color='b'):
		
	"""Plots cartesian trajectory.
	
	.. note:: Creates new figure if axes is not specified.
	
	Args:
		x (numpy.ndarray): Array with x-coordinates.
		y (numpy.ndarray): Array with y-coordinates.
		
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
	
	# Create axes if necessary
	if ax==None:
		fig,axes=makeSubplot([1,1])
		ax=axes[0]
		
	# Show velocity values if necessary
	if showVel and len(vel)>0:
		lc=coordsToLC(x,y,vel)
		ax.add_collection(lc)
		zoomToLimits(ax,x,y)		
	else:
		ax.plot(x,y,color=color)
		plt.draw()
	
	if centered:
		if showCenter and len(center)>0:
			drawCenter(ax)
	
	return ax

def drawCenter(x,ax,s=3):
	
	"""Show rectangular box around center.
	
	Args:
		x (numpy.ndarray): Array with center coordinates.
		ax (matplotlib.axes): Axes to plot in.
		s (float): Sidelength of center.
	
	"""
			
	ax.plot([x[0]-s,x[0]+s,x[0]+s,x[0]-s,x[0]-s],[x[1]-s,x[1]-s,x[1]+s,x[1]+s,x[1]-s],color='k',lw=2)
	
def plotHist(p,ax=None,bins=100,xlabel="",ylabel="",title="",xlim=[],ylim=[]):
	
	"""Creates a histogram.
	
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
	
	# Create figure
	if ax==None:
		fig,axes=makeSubplot([1,1])
		ax=axes[0]
		
	# Make hist
	ax.hist(p,bins=bins)
	
	# Set labels/limits
	setLabels(ax,xlabel=xlabel,ylabel=ylabel,title=title)
	setxlim(ax,xlim)
	setylim(ax,ylim)
	
	# Redraw
	ax.get_figure().canvas.draw()
	
	return ax

def setxlim(ax,xlim):
	
	"""Sets xlim for specified axes.
	
	Only sets limits if xlim has length 2.
	
	Args:
		ax (matplotlib.axes): Axes.
		xlim (list): Limits.
		
	Returns:
		matplotlib.axes: Modified axes.
	"""
	
	if len(xlim)==2:
		ax.set_xlim(xlim)
	
	return ax
	
def setylim(ax,ylim):
	
	"""Sets ylim for specified axes.
	
	Only sets limits if xlim has length 2.
	
	Args:
		ax (matplotlib.axes): Axes.
		ylim (list): Limits.
		
	Returns:
		matplotlib.axes: Modified axes.
	"""

	if len(ylim)==2:
		ax.set_xlim(ylim)
	
	return ax
		