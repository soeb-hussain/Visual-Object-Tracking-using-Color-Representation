# last function 

# updates the target distribution
#input 
#target histo 
# new estimated state 
#alpha - covariance 
#   output 
#target - updated target model 


def update(target,s,obs,bin,alpha):
	import math
	from histor import histor
	import numpy as np
	from numpy import linalg
	import cv2

	tl = [max(1,abs(s[0] - s[4])), max(1,abs(s[1] - s[5] ) )]

	br = [min(360,s[0] + s[4]), min(240,s[1] + s[5])]
	#print(tl)
	#print(br)

	window = obs[int(tl[0]) : int(br[0]) , int(tl[1]) : int(br[1]),0]
	#print('window window')
	#print(tl)
	#print(br)

	mid = []
	mid.append(s[4]/2)
	mid.append(s[5]/2)
	#print(s)
	soeb = np.zeros((int(mid[0]*2),int(mid[1]*2)))
	weight_factor = math.sqrt(mid[0]**2 + mid[0]**2)

	#for i in range(int(mid[0]*2)):
	#	for j in range(int(mid[1]*2)):
	#		soeb[i][j] = 1 # math.sqrt((mid[0]-i)**2 + (mid[0]-j)**2)

	#print(soeb)		
	cv2.imshow('update',window)
	cv2.waitKey(1)
	

	posteriordist = histor(bin,window,mid,soeb)
	#histo(window,mid,,bin)
	for i in range(7):
		target[i] = (1-alpha)*target[i]  + alpha*posteriordist[i]

	#for i in range(4,7):
	#	target[i] = target[i]	

	return target






