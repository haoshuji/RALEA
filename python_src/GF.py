import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pylab import figure
def GF(X,Y,options):

	########################################################
	########################################################
	# Change label space from {-1, +1} to {0, 1}
	Y=(Y+1)/2
	########################################################
	########################################################
	# print min(Y),max(Y)
	n,d = X.shape
	eta = options['eta']	
	W = options['W']
	N = options['num_all_experts']

	wij_list=[]

	w = np.ones(N)/N
	delta = options['delta']
	
	loss_experts = np.zeros((1,N))
	loss_forecaster = 0
	loss_experts_queried = np.zeros((1,N))
	loss_forecaster_queried = 0
	num_queried = 0
	num_non_queried = 0	
	
	num_accurate_preditcted = 0

	random.seed(time.time())
	start = time.time()
	for t in range(n):
		xt = X[t]
		yt = Y[t]
		
		# transfer the prediction in [-1,+1] to [0,+1]
		_f=np.maximum(np.zeros(N), np.minimum(np.ones(N), np.dot(W,xt.transpose()).flatten()+0.5))			
		for i in range(options['num_noisy_experts']):
			_f[5+i]=random.random()				
		
		_ell_1 = np.absolute(_f-1)
		_ell_0 = np.absolute(_f-0)
		_temp_alpha = np.exp(eta*(loss_forecaster_queried-loss_experts_queried-_ell_1))
		_temp_beta = np.exp(eta*(loss_forecaster_queried-loss_experts_queried-_ell_0))
		_sum_temp_alpha = _temp_alpha.sum()
		_sum_temp_beta = _temp_beta.sum()
		
		if _sum_temp_beta <=0 or _sum_temp_alpha <=0:
			print "_sum_temp_beta <=0 or _sum_temp_alpha <=0 in GF"
		
		_bar_p_t = 0.5+math.log(_sum_temp_alpha/_sum_temp_beta)*1/(2*eta)
		_hat_p_t = max(0,min(1,_bar_p_t))

		_hat_y_t = 1 if _hat_p_t >= 0.5 else 0

		_query_or_not=False
		
		if options['alg_name']=='GF':			
			_query_or_not = True
		
		elif options['alg_name']=='AGF':
			# if abs(np.amax(_f-_bar_p_t)) > 0.99:
			# 	wij_list.append(abs(np.amax(_f-_bar_p_t)))
			if np.amax(_f-_bar_p_t) > delta or np.amin(_f-_bar_p_t)< -delta:				
				_query_or_not = True
				
		elif options['alg_name'] == 'RAGF':
			ratio = float(num_non_queried)/(num_queried+1)
			mu = np.zeros(N)
			for i in range(N):
				mu[i]=math.exp(-eta*((1+ratio)*loss_experts_queried[0,i]+_ell_0[0,i]))		
			
			_numerator=0
			for i in range(N):
				_numerator = _numerator +  mu[i]*math.exp(2*eta*abs(_f[0,i]-_bar_p_t))
			
			max_ft = 1/(2*eta)*math.log(_numerator/mu.sum())
			
			# wij_list.append(max_ft)			
			if max_ft > delta:
				_query_or_not = True
		else: #RGF
			if random.random() <= delta:
				_query_or_not = True
		
		_ell_experts = np.absolute(_f-yt)
		loss_experts = loss_experts + _ell_experts

		_ell_forecaster = abs(_hat_p_t-yt)
		loss_forecaster = loss_forecaster + _ell_forecaster

		if _hat_y_t == yt:
			num_accurate_preditcted += 1

		if _query_or_not:
			num_queried += 1			
			loss_experts_queried = loss_experts_queried+_ell_experts
			loss_forecaster_queried = loss_forecaster_queried + _ell_forecaster
		else: 
			num_non_queried += 1		
		
	# if len(wij_list):
	# 	if options['alg_name'] in ['RAGF','AGF'] and options['que_ind'] == 0 and options['fold_ind']==0:
	# 		wij_list.sort()
	# 		# print wij_list[20000],wij_list[10000]
	# 		if options['alg_name'] == 'RAGF':
	# 			wij_list = [x for x in wij_list if x <= 0.005]
	# 		fig = plt.figure()
	# 		ax = fig.add_subplot(1,1,1)  
	# 		plt.hist(np.asarray(wij_list))
	# 		plt.savefig(options['output_file_name']+'_his'+'.pdf')
	# 		plt.close(fig) 
		# print min(wij_list),max(wij_list)

	end = time.time()
	reg = loss_forecaster - np.amin(loss_experts)	
	# print num_queried
	return float(num_queried)/n, float(reg)/n, (end-start), float(num_accurate_preditcted)/n