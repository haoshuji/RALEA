import numpy as np
import time
import math
import sys
from numpy import linalg as LA
def PA1(X,Y):
	err = 0	
	n,d = X.shape
	w = np.zeros((1,d))
	C = 5;
	for t in range(n):
		yt = Y[t]
		xt = X[t,:]		
		ft = np.dot(xt,w.transpose())[0,0]
		hat_yt=np.sign(ft)
		if hat_yt==0:
			hat_yt=1
		if hat_yt != yt:			
			err +=1
		lt = max(0,1-yt*ft)
		if lt > 0:
			xtNorm = LA.norm(xt)
			if xtNorm == 0.0:
				continue
			tau_t = min(C,lt/(math.pow(xtNorm,2)))
			w=w+tau_t*yt*xt
	end = time.time()
	# print sys._getframe().f_code.co_name, 1-float(err)/n
	return 1-float(err)/n, w
	