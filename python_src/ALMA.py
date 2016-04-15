import numpy as np
import time
import math
import sys
from numpy import linalg as LA
def ALMA(X,Y):
	n,d = X.shape
	w = np.zeros(d)
	alpha = 0.9
	B = 1/alpha
	C = math.sqrt(2)
	p = 2
	k = 1
	err=0
	start = time.time()
	for t in range(n):
		yt = Y[t]
		xt = X[t,:]
		# xt = xt/LA.norm(xt)
		ft = np.dot(xt,w)
		hat_yt=np.sign(ft)
		if hat_yt==0:
			hat_yt=1

		if hat_yt != yt:			
			err +=1

		gamma_t = B*math.sqrt((p-1)/k)
		if yt*ft <=(1-alpha)*gamma_t:
			eta_t = C/math.sqrt((p-1)*k)
			w = w + eta_t*yt*xt;
			w = w/max(1,LA.norm(w))
			k = k+1
	end = time.time()
	# print sys._getframe().f_code.co_name, 1-float(err)/n
	return 1-float(err)/n, w