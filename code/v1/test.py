import numpy as np 
S = []
W = []
delta = []


delta_max = param
for i in range(N_sample):
	so = len(delta_max)
	ty = np.random.rand(so)
	for j in range(len(param)):
		delta.append(-delta_max[j] + 2*delta_max[j]*ty[j])

	delta[0] = int(delta[0])
	delta[1] = int(delta[1])
	dx = target[0] + delta[0]
	dy = target[1] + delta[1]
	dvx = delta[0]
	dvy = delta[1]
	dHx = int(target[4] *(1 + delta[2]))	
	dHy =  int(target[5]) * (1 + delta[2])
	dsc = delta[2]
	s = [dx,dy,dvx,dvy,dHx,dHy,dsc]

	S.append(s)
	W.append(1/N_sample)
#print(W)
#print(S)	
