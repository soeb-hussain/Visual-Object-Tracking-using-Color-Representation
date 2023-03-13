#intiate particle 
def intiate_particle(target,N_sample,param):

	#input - histo , N_sample in set , param
	#output - S - set of particles 
	# weight - set of weight 
	S = []
	W = []
	delta = []
	import numpy as np 
	#print('ssssss')
	#print(N_sample)

	delta_max = param
	for i in range(N_sample):
		so = len(delta_max)
		ty = np.random.rand(so)
		delta = []
		for j in range(len(param)):
			delta.append(-delta_max[j] + 2*delta_max[j]*ty[j])

		dx = target[0] + delta[0]
		dy = target[1] + delta[1]
		dvx = delta[0]
		dvy = delta[1]
		dHx = int(target[4] *(1 + delta[2]))
		dHy =  int(target[5]) * (1 + delta[2])
		dsc = delta[2]
		s = [int(dx),int(dy),int(dvx),int(dvy),int(dHx),int(dHy),dsc]

		S.append(s)
		#print('parparpaprapraprar')
		#print(ty)
		#print('parparpaprapraprar')
		W.append(1.0/N_sample)
		#print((1.0/N_sample))

	return S,W	


