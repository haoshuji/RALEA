import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pylab import figure
def EWAF(X,Y,options):
	########################################################
	########################################################
	# Change label space from {-1, +1} to {0, 1}
	Y=(Y+1)/2
	########################################################
	########################################################

	n,d = X.shape
	eta = options['eta']	
	W = options['W']
	N = options['num_all_experts']

	w = np.ones(N)/N
	delta = options['delta']
	loss_experts = np.zeros(N)
	loss_forecaster = 0
	loss_experts_queried = np.zeros((1,N))
	loss_experts_unqueried = np.zeros((1,N))
	num_queried = 1
	num_non_queried = 0	
	num_accurate_preditcted = 0

	wij_list=[]

	random.seed(time.time())
	start = time.time()
	for t in range(n):
		xt = X[t]
		yt = Y[t]

		# receive annotations and transfer the annotation from [-1,+1] to [0,+1]
		import pdb
		# pdb.set_trace()
		f=np.maximum(np.zeros(N),np.minimum(np.ones(N), np.dot(W,xt.transpose()).flatten()+0.5))	
		# add noisy annotations
		for i in range(options['num_noisy_experts']):
			f[5+i]=random.random()
		
		_p_t = np.dot(f,w.transpose())[0,0]		
		
		_hat_y_t = 1 if _p_t >= 0.5 else 0

		_query_or_not=False
		
		if options['alg_name']=='EWAF':			
			_query_or_not = True
		
		elif options['alg_name']=='AEWAF':		
			# if (np.amax(f)-np.amin(f)) > 0.99:
			# 	wij_list.append((np.amax(f)-np.amin(f)))	
			
			if (np.amax(f)-np.amin(f)) > delta:				
				_query_or_not = True
		
		elif options['alg_name'] == 'RAEWAF':			
			_mu = float(num_non_queried)/(num_queried+1)
			_rel = np.zeros((N,N))

			for i in range(N):
				for j in range(N):					
					_rel[i,j] = math.exp( -eta*((1+_mu)*loss_experts_queried[0,i] + loss_experts_queried[0,j]) )
			# _rel = np.power(_rel,-eta)
			# # print sumRel,f
			# if  _rel.sum()== 0.0:
			# 	# print "_sum_rel == 0.0 in RAEWAF"
			# 	# print num_non_queried,num_queried,sumRel, _mu, loss_experts_queried
			# 	continue
			
			_rel_fij = 0.0
			for i in range(N):
				for j in range(i,N):
					import pdb
					# pdb.set_trace()
					abs_fij = abs(f[0,i]-f[0,j])
					_rel_fij = _rel_fij + _rel[i,j]*abs_fij
					if i != j:
						_rel_fij = _rel_fij + _rel[j,i]*abs_fij		
			_wij = _rel_fij/_rel.sum();
			
			# wij_list.append(abs(_wij))
			if abs(_wij) > delta:			
				_query_or_not = True
		else: # REWAF
			if random.random() <= delta:
				_query_or_not = True
		
		_ell_experts = np.absolute(f-yt)
		loss_experts = loss_experts + _ell_experts
		loss_forecaster = loss_forecaster + abs(_p_t-yt)
		
		if _hat_y_t == yt:
			num_accurate_preditcted += 1

		if _query_or_not:
			num_queried += 1			
			loss_experts_queried = loss_experts_queried+_ell_experts
			# w = w*np.exp(-eta*_ell_experts)
			w = np.exp(-eta*loss_experts_queried)
			sum_w = np.sum(w)
			w = w/sum_w
		else: 
			num_non_queried += 1		
		
		# loss_experts_unqueried = (float(num_non_queried)/float(num_queried))*loss_experts_queried

	end = time.time()
	reg = loss_forecaster - np.amin(loss_experts)	
	# if len(wij_list):
	# 	if options['alg_name'] in ['RAEWAF','AEWAF'] and options['que_ind'] == 0  and options['fold_ind']==0:
	# 		wij_list.sort()
	# 		# print wij_list[20000],wij_list[10000]
	# 		if options['alg_name'] == 'RAEWAF':
	# 			wij_list = [x for x in wij_list if x <= 0.00005]

	# 		fig = plt.figure()
	# 		ax = fig.add_subplot(1,1,1)  
	# 		plt.hist(np.asarray(wij_list))
	# 		plt.savefig(options['output_file_name']+'_his'+'.pdf')
	# 		plt.close(fig) 
	# 	# print min(wij_list),max(wij_list)
	return float(num_queried)/n, float(reg)/n, (end-start), float(num_accurate_preditcted)/n	