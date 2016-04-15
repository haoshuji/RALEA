import numpy as np
import time
import math
import sys
from numpy import linalg as LA
def ROMMA(X,Y):
	err = 0	
	n,d = X.shape
	w = np.zeros(d)
	
	for t in range(n):
		yt = Y[t]
		xt = X[t,:]		
		ft = np.dot(xt,w)
		hat_yt=np.sign(ft)
		if hat_yt==0:
			hat_yt=1
		if hat_yt != yt:			
			err +=1
		if LA.norm(w) == 0:
			if hat_yt!=yt:
				w=w+yt*xt
		else:
			if LA.norm(xt) != 0:
				if hat_yt != yt:
					xtNormpow = math.pow(LA.norm(xt),2); 
					wNormpow = math.pow(LA.norm(w),2)
					ct = (xtNormpow*wNormpow-yt*ft)/(xtNormpow*wNormpow-ft*ft)
					dt = (wNormpow*(yt-ft))/(xtNormpow*wNormpow-ft*ft)
					w = ct*w+dt*xt
	end = time.time()
	# print sys._getframe().f_code.co_name, 1-float(err)/n
	return 1-float(err)/n, w
	