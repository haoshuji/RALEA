import numpy as np
import time
import sys

def PERCEPTRON(X,Y):
	n,d = X.shape
	w = np.zeros(d)
	err=0
	start = time.time()
	for t in range(n):
		yt = Y[t]
		xt = X[t]
		# print xt, yt
		# exit(1)
		# print xt.shape, w.shape
		hat_yt=np.sign(np.dot(xt,w))
		if hat_yt==0:
			hat_yt=1
		if hat_yt != yt:
			w = w + yt*xt
			err +=1
	end = time.time()
	# print sys._getframe().f_code.co_name, 1-float(err)/n
	return 1-float(err)/n, w