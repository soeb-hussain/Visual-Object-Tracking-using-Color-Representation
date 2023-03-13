#bhattacharye distance

#input p = proposedhistograms
# q = target histrogram

def bhatta_dist(p,q):
	import math 
	import numpy as np
	sum =0
	#print('tttttttttt')
	#print(q.shape)
	#print(p.shape)
	temp = np.zeros((len(p),))
	temp1 = np.zeros((len(p),))
	sum = 0 
	for i in range(len(p)):
		
		temp[i] = math.sqrt(max(0.0,q[i,]))
		temp1[i] = math.sqrt(max(0.0,p[i,]))
		sum = sum + (temp[i])*(temp1[i])


	#sum = np.dot(temp,temp1)	

		
	#print(sum)

	d = sum

	#print(d)

	
	d = math.sqrt(1 - d)

	#print(sum(p))
	#print('333######################################################')
	#print(d)
	return d 
