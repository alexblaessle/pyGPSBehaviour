"""Basic module for grey seal project."""

# Import module

from GPSData import GPSData
import numpy as np

def removeQuotationMarksFromArr(arr):
        
        """Removes unnecessary quotation marks from entries in array.
        
        Args:
		arr (list): List of strings.
		
	Returns:
		list: List of strings without quotation marks.
        """
        
        arr2=[]
        for s in arr:
                arr2.append(removeQuotationMarks(s))
        return arr2
        

def removeQuotationMarks(s):
        
        """Removes unnecessary quotation marks from string.
        
        Args:
		s (str): Some string.
		
	Returns:
		str: String without quotation marks.
        """
        
        return s.replace('"','')

def computeAngle(vec1,vec2):
	
	"""Computes angle between two vectors in 2D.
	
	Args:
		vec1 (numpy.ndarray): Vector 1.
		vec2 (numpy.ndarray): Vector 2.
		
	
	Returns:
		float: Angle.
	
	"""
	
	#Determine sign through atan2
	dv=vec1-vec2
	sign = np.arctan2(-dv[1],dv[0])

	if sign<0:
		sign=-1
	else:
		sign=1
	
	#Compute vector length
	l_vec1=norm(vec1)
	l_vec2=norm(vec2)
	
	#Compute dot product
	prod=np.dot(vec1,vec2)
	
	#Compute angle
	if l_vec1*l_vec2>0:
		if sign==1:
			angle=sign*np.arccos(prod/(l_vec1*l_vec2))
		elif sign==-1:	
			angle=2*np.pi+sign*np.arccos(prod/(l_vec1*l_vec2))
	else:
		#Make sure not to return NaN
		angle=0.
		
	return angle

def direcAngle(v1,v2):
	
	"""Computes directional angle between two vectors in 2D.
	
	Args:
		v1 (numpy.ndarray): Vector 1.
		v2 (numpy.ndarray): Vector 2.
		
	
	Returns:
		float: Angle.
	
	"""
	
	v1_u = unitVector(v1)
	v2_u = unitVector(v2)
	angle = np.arccos(np.dot(v1_u, v2_u))
	if np.isnan(angle):
		if (v1_u == v2_u).all():
			return 0.0
		else:
			return np.pi
		
	sign=getAngleSign(v1,v2)
	
	angle=sign*angle
	
	return angle

def getAngleSign(vec1,vec2):
	
	"""Determine sign of angle through atan2.
	
	Args:
		vec1 (numpy.ndarray): Vector 1.
		vec2 (numpy.ndarray): Vector 2.
		
	Returns:
		int: Sign of angle.
	
	"""
	
	dv=vec1-vec2
	sign = np.arctan2(-dv[1],dv[0])

	if sign<0:
		sign=-1
	else:
		sign=1
	
	return sign

def unitVector(vector):
	
	"""Compute unit vector.
	
	Args:
		vector (numpy.ndarray): Some vector.
		
	Returns:
		numpy.ndarray: Unit vector with same direction.
	"""
	
	return vector / np.linalg.norm(vector)

def norm(vec):
	
	"""Compute L2 norm of vector.
	
	Args:
		vec (numpy.ndarray): Some vector.
		
	Returns:
		float: L2 norm of vector.
	"""
	
	return np.linalg.norm(vec)

def getIncreasingRandInt(n,N,minDist=3,offset=3):
	
	"""Creates an array of increasing random integers.
	
	Args:
		n (int): Number of integers to be created.
		N (int): Maximum allowed integer.
		
	Keyword Args: 
		minDist (int): Minimum difference between integers.
	
	"""
	
	
	while True:
		
		x=np.random.random_integers(offset,high=N-offset,size=n)
		
		x=np.sort(x)
		diffs=np.diff(x)
		
		if len(np.unique(x))==len(x) and (diffs>minDist).all():
			
			return x
		
		
			