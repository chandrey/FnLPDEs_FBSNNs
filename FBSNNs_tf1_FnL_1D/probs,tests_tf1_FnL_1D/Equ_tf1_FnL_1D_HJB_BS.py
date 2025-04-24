# Import example:
    # from Equ_tf1_FnL_1D_HJB_BS import Equ_tf1_FnL_1D_HJB_BS

from FBSNNs_tf1_FnL_1D import FBSNN_tf1_FnL_1D 

import numpy as np # for: np.exp, np.sum
import tensorflow as tf # for: tf.sin, tf.reduce_sum, tf.linalg.diag

from abc import abstractmethod

# Xi = np.zeros([1,D])
# TODO подумать каа задать ТУТ Xi



class Equ_tf1_FnL_1D_HJB_BS(FBSNN_tf1_FnL_1D):  # FnL_1D_HJB_BS_ic_exp
    def __init__(self, Xi, T,
                       M, N,
                       hlayers, 
                       act_func = tf.tanh, s=0, name='model_FnL_1D_HJB_BS',
                       mu=0.02, sigma=0.1 ):
        ### DEBUG MODE
        self._debug_ = 0
        
        self.mu = mu # np.sqrt(0.1)
        self.sigma = sigma # np.sqrt(0.1)
        self.mu__tf = self.mu # 0.1 # tf.sqrt(2.0) 
        self.sigma__tf = self.sigma
        
        self.eps = 1e-8
        super().__init__(Xi, T,
                         M, N,
                         hlayers, 
                         act_func=act_func, s=s, name=name) # FBSNN_tf1_FnL_1D.__init__(...)
        
        ### DEGUB PRINT sigma_tf
        # X = tf.tile(Xi, [self.M,1])
        # print( f'sigma_tf = {self.sigma_tf(t=0, X=X)} = {self.sigma_tf(t=0, X=X).eval(session=self.sess.as_default())}' )
        ## print( f'sigma_tf = {self.sigma_tf(t=0, X=X)} ' )
        
    def __str__(self):
        return super().__str__() + f'\n\t HJB_BS: mu={self.mu}, sigma={self.sigma}'
        
    ## init abstract method     
    # X process 
    def mu_tf(self, t, X, U, V, GV): # M x 1, M x D, M x 1, M x D, M x D x D --> M x D
        # return super().mu_tf(t, X, U, V, GV) # M x D
        return np.zeros([self.M,self.D]) ## 
    def sigma_tf(self, t, X, U=None, V=None, G=None): # M x 1, M x D, M x 1, M x D, M x D x D --> # M x D x D
        # TODO dependence sigma from V,G
        # return self.sigma__tf*tf.linalg.diag(tf.ones([M,D])) # super().sigma_tf(t, X, U, V, GV) # M x D x D        
        # return 2 * super().sigma_tf(t, X, U, V, G) # 
        # a_ = self.mu__tf/self.sigma__tf * V/(G+self.eps) ## ?? abs( self.mu__tf/self.sigma__tf * V/G )
        a_ = self.mu__tf/self.sigma__tf * abs(V) /(abs(G)+self.eps)  ## ?? abs( self.mu__tf/self.sigma__tf * V/G )
        if self._debug_==1: print( f'a_.shape = {a_.shape}' )
        a = tf.squeeze(a_)
        # print( f'a.shape = {a.shape}, a = {a}' )
        D = self.D
        dd = tf.stack([a]*D,axis=1)
        # print( f'dd = \n{dd}' )
        sigma = tf.linalg.diag(dd)
        # print( f'tf.rank(sigma) = {tf.rank(sigma)}' )
        if self._debug_==1: print( f'sigma.shape = {sigma.shape}' )
        # print( f'sigma = \n{sigma}' )
        return sigma 
    
    # equation 1           
    def phi_tf(self, t, X, U, V, G): # M x 1, M x D, M x 1, M x D, M x D x D --> M x 1
        res = + (self.mu__tf/self.sigma__tf)**2 * V**2/(abs(G)+self.eps)
        if self._debug_==1: print(f'from phi_tf: res.shape = {res.shape}')
        return res 
        # return tf.reduce_sum(V**2, 1, keepdims = True) # M x 1    
   
    # equation 2     
    def psi_tf(self, t, X, U, V, G): # M x 1, M x D, M x 1, M x D, M x D x D --> M x D
        ### FOR D==1
        return + (self.mu__tf/self.sigma__tf)**2 * V
        # return tf.reduce_sum(V**2, 1, keepdims = True) # M x 1  
    
    ## exact_solution
    @abstractmethod
    def u_exact(self, t, X): # NC x 1, NC x D --> NC x 1   
        pass
    
    @abstractmethod
    def Du_exact(self, t, X): # NC x 1, NC x D --> NC x 1
        pass
    