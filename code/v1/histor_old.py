
def histor(bin,cap,mid= None,Weigh = None):
	import numpy as np
	import math


	#print('**************************')
	import numpy as np

	c = np.asarray(cap)
	#max = 0
	#print(c.shape)
	l = c.shape[0]
	m = c.shape[1]
	mid = [l,m]
	soeb = np.zeros((l,m))
	weight_factor = math.sqrt(mid[0]**2 + mid[1]**2)
	hist = np.zeros((bin**3))
	hist_un = np.zeros((bin**3))
	bin_square = bin**2
	factor = 1.0/32
	for i in range(l):
		for j in range(m):
			soeb[i][j] = 1 #weight_factor -  math.sqrt((mid[0]-i)**2 + (mid[0]-j)**2)
			for k in range(2):
				b = int(c[i][j][0]*factor)
				g = int(c[i][j][1]*factor)
				r = int(c[i][j][2]*factor)
				temp = b*bin_square + g*bin + r
				hist[temp] = hist[temp] + soeb[i][j]
				hist_un[temp] = hist_un[temp] + 1 

	#histr, bin_edges = np.histogram(cap[:,:], bins=bin, range=None, normed=True, weights = soeb)
	#histg, bin_edges = np.histogram(cap[:,:,1], bins=bin, range=None, normed=True)
	#histb, bin_edges = np.histogram(cap[:,:,2], bins=bin, range=None, normed=True)
	hist = hist/sum(hist)
	hist_un = hist_un/sum(hist_un)
	#[a,] =histr.shape
	#print(a) 
	#his = np.zeros((a))
	#for i in range(a):
	#	his[i] = histr[i,]
	#his[:,1] = histg[:,]
	#his[:,2] = histb[:,]
	#print(hist)


	#numpy.histogram(a, bins=10, 
		#range=None, normed=False, weights=None, 
		#density=None)
	return hist , hist_un
