# Import example:
    # from FBSNNs_plots import *
    # from FBSNNs_plots import plot_results
    
import matplotlib.pyplot as plt
import numpy as np

def plot_results(model, res, r=range(0,5), flag_err_info=0, u_name='u', Y_name='Y', ax = None):
    '''
    plot_results(model, res, r=range(0,5), flag_err_info=0, u_name='u', Y_name='Y', ax = None)
    вывод траекторий в заданной области
    '''
    t_test, X_pred, Y_pred, Y_test = res
    # в классе FBSNN нет аттрибутов name, s
    name_ = 'FBSNNs' if not hasattr(model, 'name') else model.name
    s_ = 0 if not hasattr(model, 's') else model.s # s=0
    ## s_ = t_test[0,0,0] if not hasattr(model, 's') else model.s # взять s из t_test 
    # в классе FBSNN_tf1 есть аттрибуты name, s   
    rmin = min(r)
    rmax = max(r)+1
    if ax is None:
        f,ax = plt.subplots(1,1) ## 
    # 0-я траектория 
    ax.plot(t_test[rmin,:,0].T,Y_pred[rmin,:,0].T,'b',label=f'Learned ${u_name}(t,X_t)$')
    ax.plot(t_test[rmin,:,0].T,Y_test[rmin,:,0].T,'r--',label=f'Exact ${u_name}(t,X_t)$')
    ax.plot(t_test[rmin,-1,0],Y_test[rmin,-1,0],'ks',label=f'${Y_name}_T = {u_name}(T,X_T)$')
    # остальные траектории
    ax.plot(t_test[rmin+1:rmax,:,0].T, Y_pred[rmin+1:rmax,:,0].T, 'b')
    ax.plot(t_test[rmin+1:rmax,:,0].T, Y_test[rmin+1:rmax,:,0].T, 'r--')
    ax.plot(t_test[rmin+1:rmax,-1,0],  Y_test[rmin+1:rmax,-1,0],  'ks')

    # plt.plot([s_],Y_test[0,0,0],'ks',label = f'$Y_s = Y_\{{s_}\}=u(\{{s_}\},X_\{{s_}\}=\\xi)$')
    ax.plot([s_],Y_test[0,0,0],'ko',
             label = f'${Y_name}_s = {Y_name}_' + '{' + f'{s_}' + '}' + f'={u_name}(' +'{' + f'{s_}' + '},X_{' + f'{s_}' + '}=\\xi)$')
    
    ax.set_xlabel('$t$') # plt.xlabel('$t$')
    ax.set_ylabel(f'${Y_name}_t = {u_name}(t,X_t)$') # plt.ylabel(f'${Y_name}_t = {u_name}(t,X_t)$')
    
    rel_err_t0 = abs( (Y_pred[0,0,0]-Y_test[0,0,0])/Y_test[0,0,0] ) # ошибка в значении u(0, xi)
    it,loss = np.array(model.loss_history).T[:,-1]
    nn_info1 = f'N={model.N}, M={model.M}, layers={model.layers}, \n act_f={model.act_func_name}, it={it}, loss={loss:.4e}\n'
    nn_info2 = ''
    # if hasattr(model, 'loss_n'): nn_info2 = nn_info2 + f'loss_n = {model.loss_n}   '
    # if hasattr(model, 'flag_v1eqDu'): nn_info2 = nn_info2 + f'v1eqDu={model.flag_v1eqDu}'
    if nn_info2 != '': nn_info2 = nn_info2 +'\n'
    if flag_err_info == 1:
        err_info = ' $rel\_err_{t=' + f'{s_}'+'}$ = '+f'{rel_err_t0:.4e}' 
        # err_info = 'err_info' + ' $rel\_err_{t=' + f'{s_}'+'}$ = '+f'{rel_err_t0:.4f}' 
        ''' err_info = f' rel_err = {calc_error_point(Y_pred, Y_test, r)[3]} \n' +
              f' $||rel\_err||_1$ = {calc_rel_error_traj(Y_pred, Y_test, r, norm_ord=1)} \n' +
              f' $||rel\_err||_2$ = {calc_rel_error_traj(Y_pred, Y_test, r, norm_ord=2)} \n' +
              f' $||rel\_err||_\infty$ = {calc_rel_error_traj(Y_pred, Y_test, r, norm_ord=np.inf)} \n' +
              ' $rel\_err_{t=0}$ = '+f'{rel_err_t0:.4f}' 
        '''
    else: err_info = ''
    ax.set_title( f'{model.D}-dim {name_} \n(trajectories={r}) \n' + nn_info1 + nn_info2 + err_info )
    ax.legend()
    ax.grid()
    return ax
    
    # savefig('./figures/HJB_Apr18_50', crop = False)
