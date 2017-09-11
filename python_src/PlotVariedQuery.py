import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import figure

def PlotVariedQuery(results,options):
    data_name = options['data_name']
    num_queries = options['num_queries']

    selected_indexes = options['selected_indexes']

    algorithm_names=['EWAF','AEWAF','REWAF','RAEWAF','GF','AGF','RGF','RAGF']
    colors = ['black','blue', 'green', 'red', 'black','blue', 'green', 'red']    
    markers = ['x','o', 's', 'd', 'x','o', 's', 'd']
    filltypes = ['none','none','none','none','none','none','none','none']
    ls = ['-','-','-','-','-','-','-','-']
    line_width = 2;    num_alg = 7
    marker_edge_width=[2,2,2,2,2,2,2,2]
    label_size = 20;    tick_size = 18
    legend_size = 18;    legend_font_size = 18
    marker_size = 10
    _query_ratio_passive_algorithm=np.linspace(0,1,num_queries)  

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)        
    for i in range(4):
        if algorithm_names[i]=='EWAF':
            plt.plot(_query_ratio_passive_algorithm, results[algorithm_names[i]]['reg'], \
                lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
        else:         
            # print algorithm_names[i]
            # print selected_indexes[algorithm_names[i]]   
            # print results[algorithm_names[i]]['que']
            _query_ratio = [results[algorithm_names[i]]['que'][j] for j in selected_indexes[algorithm_names[i]]]
            _regret = [results[algorithm_names[i]]['reg'][j] for j in selected_indexes[algorithm_names[i]]]
            
            # print _query_ratio
            plt.plot(_query_ratio, _regret, lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
    plt.xlabel('que ratios',fontsize=label_size)
    plt.ylabel('Per-round reg',fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    plt.legend(loc='best', ncol=1, shadow=True, fancybox=True,prop={'size':legend_size},fontsize=legend_font_size)
    plt.grid(True,which="both",ls="--", color='0.4')
    plt.savefig(options['output_file_name']+"_regret"+"_EWAF"+options['output_file_extension'])
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)        
    for i in range(4):
        if algorithm_names[i]=='EWAF':
            plt.plot(_query_ratio_passive_algorithm, results[algorithm_names[i]]['acc'], \
                lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
        else:         
            _query_ratio = [results[algorithm_names[i]]['que'][j] for j in selected_indexes[algorithm_names[i]]]
            _acc = [results[algorithm_names[i]]['acc'][j] for j in selected_indexes[algorithm_names[i]]]
            
            if algorithm_names[i] == 'RAEWAF':
                plt.plot(_query_ratio, _regret, lw=line_width,label = 'RAWA', ls=ls[i], color=colors[i], marker = markers[i],\
                    fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
            else:
                plt.plot(_query_ratio, _regret, lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                    fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
    plt.xlabel('Number of Control Questions (Percentage) ',fontsize=label_size)
    plt.ylabel('Aggregation Accuracy',fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    plt.legend(loc='best', ncol=1, shadow=True, fancybox=True,prop={'size':legend_size},fontsize=legend_font_size)
    plt.grid(True,which="both",ls="--", color='0.4')
    plt.savefig(options['output_file_name']+"_acc"+"_EWAF"+options['output_file_extension'])
    plt.close(fig) 

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)        
    for i in range(4):
        if algorithm_names[i]=='EWAF':
            plt.plot(_query_ratio_passive_algorithm, results[algorithm_names[i]]['time'], \
                lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
        else:            
            _query_ratio = [results[algorithm_names[i]]['que'][j] for j in selected_indexes[algorithm_names[i]]]
            _regret = [results[algorithm_names[i]]['time'][j] for j in selected_indexes[algorithm_names[i]]]
            # print algorithm_names[i]
            # print selected_indexes[algorithm_names[i]]
            # print _query_ratio
            plt.plot(_query_ratio, _regret, lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
    plt.xlabel('que ratios',fontsize=label_size)
    plt.ylabel('Per-round reg',fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    plt.legend(loc='best', ncol=1, shadow=True, fancybox=True,prop={'size':legend_size},fontsize=legend_font_size)
    plt.grid(True,which="both",ls="--", color='0.4')
    plt.savefig(options['output_file_name']+'_time'+"_EWAF"+options['output_file_extension'])
    plt.close(fig) 


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)           
    for i in range(4,8):
        if algorithm_names[i]=='GF':
            plt.plot(_query_ratio_passive_algorithm, results[algorithm_names[i]]['reg'], \
                lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
        else:
            _query_ratio = [results[algorithm_names[i]]['que'][j] for j in selected_indexes[algorithm_names[i]]]
            _regret = [results[algorithm_names[i]]['reg'][j] for j in selected_indexes[algorithm_names[i]]]
            # print algorithm_names[i]
            # print selected_indexes[algorithm_names[i]]
            # print _query_ratio
            plt.plot(_query_ratio,_regret,lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
    plt.xlabel('que ratios',fontsize=label_size)
    plt.ylabel('Per-round reg',fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    plt.legend(loc='best', ncol=1, shadow=True, fancybox=True,prop={'size':legend_size},fontsize=legend_font_size)
    plt.grid(True,which="both",ls="--", color='0.4')
    plt.savefig(options['output_file_name']+"_regret"+"_GF"+options['output_file_extension'])
    plt.close(fig) 

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)           
    for i in range(4,8):
        if algorithm_names[i]=='GF':
            plt.plot(_query_ratio_passive_algorithm, results[algorithm_names[i]]['reg'], \
                lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
        else:
            _query_ratio = [results[algorithm_names[i]]['que'][j] for j in selected_indexes[algorithm_names[i]]]
            _regret = [results[algorithm_names[i]]['time'][j] for j in selected_indexes[algorithm_names[i]]]
            # print algorithm_names[i]
            # print selected_indexes[algorithm_names[i]]
            # print _query_ratio
            plt.plot(_query_ratio,_regret,lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
    plt.xlabel('que ratios',fontsize=label_size)
    plt.ylabel('Per-round reg',fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    plt.legend(loc='best', ncol=1, shadow=True, fancybox=True,prop={'size':legend_size},fontsize=legend_font_size)
    plt.grid(True,which="both",ls="--", color='0.4')
    plt.savefig(options['output_file_name']+'_time'+"_GF"+options['output_file_extension'])
    plt.close(fig) 

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)        
    for i in range(4,8):
        if algorithm_names[i]=='GF':
            plt.plot(_query_ratio_passive_algorithm, results[algorithm_names[i]]['acc'], \
                lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
        else:         
            _query_ratio = [results[algorithm_names[i]]['que'][j] for j in selected_indexes[algorithm_names[i]]]
            _acc = [results[algorithm_names[i]]['acc'][j] for j in selected_indexes[algorithm_names[i]]]
            
            if algorithm_names[i] == 'RAGF':
                plt.plot(_query_ratio, _regret, lw=line_width,label = 'RAGA', ls=ls[i], color=colors[i], marker = markers[i],\
                    fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
            else:
                plt.plot(_query_ratio, _regret, lw=line_width,label = algorithm_names[i], ls=ls[i], color=colors[i], marker = markers[i],\
                    fillstyle=filltypes[i],markersize=marker_size,mew=marker_edge_width[i])
    plt.xlabel('Number of Control Questions (Percentage) ',fontsize=label_size)
    plt.ylabel('Aggregation Accuracy',fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    plt.legend(loc='best', ncol=1, shadow=True, fancybox=True,prop={'size':legend_size},fontsize=legend_font_size)
    plt.grid(True,which="both",ls="--", color='0.4')
    plt.savefig(options['output_file_name']+"_acc"+"_GF"+options['output_file_extension'])
    plt.close(fig) 

    pass