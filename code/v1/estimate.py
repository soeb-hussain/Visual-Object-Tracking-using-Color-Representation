def estimate(S,W):
	import numpy as np

	S_array =  np.asarray(S)
	N1_S_array = S_array.shape[0]
	N2_S_array = 7

	#print(len(S))#S_array.shape[1]



	#print(len(S))

	#print(meanstate)
	#print(N1_S_array)
	#print(N2_S_array)
	meanstate = np.zeros((N2_S_array))
	for i in range(N1_S_array):
		s = S[i]

		for j in range(N2_S_array):
			#print(W[i])
			meanstate[j] = meanstate[j] + (W[i]* s[j])

		#print(meanstate)			

	#print meanstate			
	return meanstate	
