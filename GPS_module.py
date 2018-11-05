"""GPS module of seal project. Contains all functions necessary to read and convert GPS coordinates."""

# Import modules 
import numpy as np

def toCartesian(lat,lon,r=6378137,e=8.1819190842622e-2,h=0):
	
	"""Converts the GPS data to cartesian coordinates.
	
	Args:
		lat (numpy.ndarray): Latitude values.
		lon (numpy.ndarray): Longitude values.
		
	Keyword Args:
		r (float): Radius.
		e (float): Excentriciity.
		h (float): Height of location. Normally assumed to be sea-level.
		
	Returns:
		
		tuple: Tuple containing:
		
			* x (numpy.ndarray): x coordinates.
			* y (numpy.ndarray): y coordinates.
			* z (numpy.ndarray): z coordinates.
			
	"""
	
	# To radians
	latRad=lat/180.*np.pi
	lonRad=lon/180.*np.pi	

	N = r/np.sqrt(1 - e**2 * np.sin(latRad)**2)

	x = -1*(N+h)*np.cos(latRad)*np.cos(lonRad)
	y = -1*(N+h)*np.cos(latRad)*np.sin(lonRad)
	z = ((1-e**2)*N+h)*np.sin(latRad)
	
	return x,y,z

def computeDistances(x,y):
		
	"""Computes distances of all steps.
	
	Args:
		x (numpy.ndarray): x coordinates.
		y (numpy.ndarray): y coordinates.
		
	Returns:
		numpy.ndarray: Distances.
	
	"""
	
	d=np.sqrt(abs(np.diff(x))**2+abs(np.diff(y))**2)	
	
	return d

