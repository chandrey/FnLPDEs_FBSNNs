 # Import example:
    # from FBSNNs_tf1_FnL_1D import FBSNN_tf1_FnL_1D
    
## TODO переименовать в FBSNNs_tf1_SqL1D (system (2eq=(1+D)eq) quasi linear PDEs in 1D)

## процесс V (1-dim) вычисляется сетью

from FBSNNs_tf1 import FBSNN_tf1

import tensorflow as tf # for tf.sin, tf.add, tf.matmul
#tf.disable_v2_behavior()
import numpy as np # for: np.zeros, ... 
import time

from abc import abstractmethod

## здесь D == 1

class FBSNN_tf1_FnL_1D(FBSNN_tf1): 
    ''' 
        class FBSNN_tf1_FnL_1D(FBSNN_tf1) - new class Forward-Backward Stochastic Neural Network for 
                -- Fully non Linear Parabolic Equation (x in 1D);
                -- system (2eq=(1+D)eq) quasi linear PDE in 1D.
            - Added abstract method: mu_tf, sigma_tf, phi_tf, g_tf, psi_tf, f_tf.       
            - Modified methods (with respect to class FBSNN_tf1):
                __init__, Df_tf, net_u, loss_function.
    '''
    
    def __init__(self, Xi, T,
                       M, N,
                       hlayers, 
                       act_func = tf.tanh, s=0, name='Model_FBSNN_FnL_1D'): # , act_f = 'sin' 
        ## self.K = 1 # D # размерность для V (2го уравнения)            
        D = 1
        layers = [D+1] + hlayers + [1+D]
        self._debug_ = 0 # 1
        self.flag_v1eqDu = False
        super().__init__(Xi, T,   M, N, D,    layers,   act_func, s, name) # FBSNN_tf1.__init__
        
        self.net_out_size = 1+D
        # new parameters
        #
    ###########################################################################
    # X process
    ## V=gradU, GV=gradV
    @abstractmethod  
    def mu_tf(self, t, X, U, V, GV): # M x 1, M x D, M x 1, M x D, M x D x D --> M x D
        M = self.M
        D = self.D
        return np.zeros([M,D]) # M x D    
    @abstractmethod
    def sigma_tf(self, t, X, U, V, GV): # M x 1, M x D, M x 1, M x D, M x D x D --> # M x D x D
        M = self.M
        D = self.D
        return tf.linalg.diag(tf.ones([M,D])) # M x D x D
    
    # equation 1
    @abstractmethod
    def phi_tf(self, t, X, U, V, GV): # M x 1, M x D, M x 1, M x D, M x D x D --> M x 1
        pass # M x 1    
    @abstractmethod
    def g_tf(self, X): # M x D --> M x 1
        pass # M x 1
    
    # equation 2    
    @abstractmethod
    def psi_tf(self, t, X, U, V, GV): # M x 1, M x D, M x 1, M x D, M x D x D --> M x D
        pass # M x D    
    @abstractmethod
    def f_tf(self, X): # M x D --> M x D
        pass # M x D
        
    
    ###########################################################################   
    
    def Df_tf(self, X): # M x D
        return tf.gradients(self.f_tf(X), X)[0] # M x D
    
    def net_u(self, t, X): # M x 1, M x D        
        uv = self.neural_net(tf.concat([t,X], 1), self.weights, self.biases) # M x (1+D)
        ## if self._debug_==1: print(f'from net_u: \t uv.shape={uv.shape}')
        u = uv[0:,0:1]
        v1 = uv[0:,1:2] # 1:]
        if self._debug_==1: print(f'from net_u: \t u.shape={u.shape}, v1.shape={v1.shape}')
        Duv = tf.gradients(uv, X)[0] # M x D почему??? надо     M x (1+D) x D
        Du = tf.gradients(u, X)[0] # M x D почему??? надо     M x 1 x D
        Dv1 = tf.gradients(v1, X)[0] # M x D почему??? надо     M x D x D
        D2u = tf.gradients(Du, X)[0] # M x D почему??? надо     M x D x D
        #print(f'from net_u: \t Duv={Duv}')    
        if self._debug_==1: 
            print(f'from net_u: \t Duv.shape={Duv.shape}, Du.shape={Du.shape}, Duv1.shape={Dv1.shape}, D2u.shape={D2u.shape}') 
        if self.flag_v1eqDu: return u,Du,Du,D2u # uv, Duv0
        else: return u,v1,Du,Dv1 # uv, Duv0
    
    # def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
    #     return super().loss_function(t, W, Xi)
    ### @tf.function # для tf.math.is_nan(...)
    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = 0
        X_list = []
        Y_list = []  
        V_list = []       
        t0 = t[:,0,:] # M x 1
        W0 = W[:,0,:] # M x D
        X0 = tf.tile(Xi, [self.M,1]) # 1 x D, ... --> M x D     # X0 = tf.tile(Xi,[self.M,1]) # M x D
        #Y0, Z0 = self.net_u(t0,X0) # M x 1, M x D --> M x 1, M x D 
        #V0, G0 = Y0, Z0 #  None, None 
        Y0, V0, Z0, G0 = self.net_u(t0,X0)
        #V0 = Z0
        
        ## self.net_out_size = (Y0.shape, Z0.shape)
        
        X_list.append(X0)
        Y_list.append(Y0)   
        V_list.append(V0)      
        for n in range(0,self.N):
            t1 = t[:,n+1,:] # M x 1
            W1 = W[:,n+1,:] # M x D
            dt = t1-t0 # --> M x 1
            dW = W1-W0 # --> M x D
            s_dW = tf.squeeze( tf.matmul( self.sigma_tf(t0,X0,Y0,V0,G0), tf.expand_dims(dW,-1) ), axis=[-1] ) # --> M x D             
            # Y1_tilde = Y0 + self.phi_tf(t0,X0,Y0,Z0)*(t1-t0) + \
            #           tf.reduce_sum(Z0*tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1))), axis=1, keepdims = True)
            Y1_tilde = Y0 + self.phi_tf(t0,X0,Y0,Z0,G0)*dt + tf.reduce_sum( Z0*s_dW, axis=1, keepdims = True)            
            # X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*(t1-t0) + \
            #      tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)), axis=[-1]) 
            V1_tilde = V0 + self.psi_tf(t0,X0,Y0,Z0,G0)*dt + tf.reduce_sum( G0*s_dW, axis=1, keepdims = True)   
            X1 = X0 + self.mu_tf(t0,X0,Y0,Z0,G0)*dt + s_dW # --> M x D      
            
            # Y1, Z1 = self.net_u(t1, X1)    
            # V1, G1 = Y1, Z1
            Y1, V1, Z1, G1 = self.net_u(t1,X1)
            #V1 = Z1
            sY = tf.reduce_sum(tf.square(Y1 - Y1_tilde))
            # if tf.math.is_nan(sY): raise ValueError(f'sY value == NaN!\n ')
            loss += sY   
            sV = tf.reduce_sum(tf.square(V1 - V1_tilde))
            # if tf.math.is_nan(sV): raise ValueError(f'sV value == NaN!\n ')
            loss += sV        
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1 
            V0 = V1
            G0 = G1           
            X_list.append(X0)
            Y_list.append(Y0)  
            V_list.append(V0) 
        # if tf.math.is_nan(loss): raise ValueError(f'loss value == NaN!\n ')
        sum0N = loss            
        ## loss2 = 1/self.N * loss # MY        
        loss_uvT  = tf.reduce_sum(tf.square(Y1 - self.g_tf(X1))) + tf.reduce_sum(tf.square(V1 - self.f_tf(X1))) # MY
        loss_DuvT = tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1))) + tf.reduce_sum(tf.square(G1 - self.Df_tf(X1)))# MY         
        loss =  sum0N + loss_uvT + loss_DuvT # расшир Raissi loss       
        loss0 = 1/self.M * ( dt[0][0]*sum0N + loss_uvT )  # MY       
        loss1 = 1/self.M * ( dt[0][0]*sum0N + loss_uvT + loss_DuvT ) # MY
        # loss3 = 1/self.M * ( sum0N + loss_uT + 1/self.D * loss_DuT ) # MY
        
        # Y_list: (M x 1)*(N+1)
        # Y: M x (N+1) x 1
        X = tf.stack(X_list,axis=1)
        Y = tf.stack(Y_list,axis=1)
        V = tf.stack(V_list,axis=1)
        
        if self.loss_n == 0: loss_ = loss0
        elif self.loss_n == 1: loss_ = loss1
        # elif self.loss_n == 3: loss_ = loss3
        else: loss_ = loss
            
        # YV = Y
        YV = tf.stack([Y,V],axis=0)
        # YV: 2 x M x (N+1) x 1
        
        # OperatorNotAllowedInGraphError: Using a symbolic `tf.Tensor` as a Python `bool` is not allowed in Graph execution. 
        # Use Eager execution or decorate this function with @tf.function.
        ### if tf.math.is_nan(loss_): raise ValueError(f'NaN loss value! \n X={X0}')
            
        return loss_, X, YV, [Y[0,0,0],V[0,0,0]]   
    
    
       