import numpy as np

def xdiff(x):
	
	return np.diff(x[::-1])

def constrainFunc(x):
	
	return xdiff(x).any()>0

