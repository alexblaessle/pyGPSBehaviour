"""Statistics module for grey seal project."""

# Import modules
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def performPCA(x,y,nComponents):
	
	"""Experimental."""

	X = np.stack((x,y), axis=0)

	pca = PCA(n_components=nComponents)
	pca.fit(X)
	

def performKMeans(*args):
	
	"""Performs kMeans as implemented by scikit learn.
	
	Args:
		X (np.ndarray): Array.
		nComponents (int): Number of clusters.
	
	Returns:
		sklearn.KMeans: kMeans object.
	
	"""
	
	nComponents=args[-1]
	
	if len(args)>2:
		X = np.stack(args[:-1], axis=1)
	else:
		X=args[0]
	
	kmeans = KMeans(n_clusters=nComponents, random_state=0).fit(X)
		
	return kmeans
	