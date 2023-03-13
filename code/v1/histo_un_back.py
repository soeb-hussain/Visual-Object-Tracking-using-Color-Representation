
def histor_unweighted_background(bin,cap):
	import numpy as np
	import math


	#print('**************************')
	import numpy as np

	c = np.asarray(cap)
	#max = 0
	soeb = cap
	#print(c.shape)
	l = c.shape[0]
	m = c.shape[1]
	mid = [l,m]
	#soebi = np.zeros((l,m))
	#weight_factor = math.sqrt(mid[0]**2 + mid[1]**2)
	Red = np.asarray(soeb[:,:,0])
	Blue = np.asarray(soeb[:,:,1])
	green = np.asarray(soeb[:,:,2])
	r = (Red/32) * 64
	b = (Blue/32)* 8
	g = green/32
	sum = r + g+ b
	hist_un = np.zeros((512,1))
	#hist = np.zeros((512,1))

	for i in range(Red.shape[0]):
		for j in range(Red.shape[1]):
			#soebi[i][j] = weight_factor -  math.sqrt((mid[0]-i)**2 + (mid[0]-j)**2)
			hist_un[sum[i][j]] = hist_un[sum[i][j]] + 1
			#hist[sum[i][j]] = hist[sum[i][j]] + soebi[i][j]

	hist_un = hist_un/np.sum(hist_un)
	#hist = hist/np.sum(hist)


	#hist_un = (np.array(hist_un)).flatten()
	#hist = (np.array(hist)).flatten()

	#hist_un = np.asarray(hist_un)
	#hist_un = hist_un.flatten()

	#print(hist_un.shape)
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
	return hist_un
