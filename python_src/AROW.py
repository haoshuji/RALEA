import numpy as np
import time
import math
import sys
def AROW(X,Y):
	err = 0
	gamma = 1
	n,d = X.shape
	w = np.zeros((1,d))
	Sigma = np.identity(d)
	start = time.time()
	for t in range(n):
		yt = Y[t]
		xt = X[t,:]		
		ft = np.dot(xt,w.transpose())[0,0]
		hat_yt=np.sign(ft)
		if hat_yt==0:
			hat_yt=1
		if hat_yt != yt:			
			err +=1
		
		if ft*yt<1:
			v_t = np.sum(xt*Sigma*xt.transpose())
			beta_t = 1.0/(v_t+gamma)
			alpha_t = max(0,1-yt*ft)*beta_t
			w = w + alpha_t*yt*xt*Sigma;
			Sigma = Sigma-beta_t*Sigma*xt.transpose()*xt*Sigma
	end = time.time()
	# print sys._getframe().f_code.co_name, 1-float(err)/n
	return 1-float(err)/n, w
	