
def histo(bin,cap,mid= None,Weigh = None):
	import numpy as np
	import math


	#print('**************************')
	import numpy as np

	#c = np.asarray(cap)
	#max = 0
	#print(c.shape)
	#l# = c.shape[0]
	#m = c.shape[1]
	#soeb = np.zeros((l,m))
	#weight_factor = math.sqrt(mid[0]**2 + mid[0]**2)
	#for i in range(l):
		#for j in range(m):
			#soeb[i][j] = weight_factor -  math.sqrt((mid[0]-i)**2 + (mid[0]-j)**2)
			#if max<soeb[i][j]:
				#max = soeb[i][j]
				#print(i*100000 + j)
			#if  i==j and i==mid[0]:
				#print(soeb[i][j])





	histr, bin_edges = np.histogram(cap[:,:,0], bins=bin, range=None, normed=True)
	histg, bin_edges = np.histogram(cap[:,:,1], bins=bin, range=None, normed=True)
	histb, bin_edges = np.histogram(cap[:,:,2], bins=bin, range=None, normed=True)
	
	[a,] =histr.shape
	#print(a) 
	his = np.zeros((a,3))
	his[:,0] = histr[:,]
	his[:,1] = histg[:,]
	his[:,2] = histb[:,]
	#print(his.shape)


	#numpy.histogram(a, bins=10, 
		#range=None, normed=False, weights=None, 
		#density=None)
	return his
