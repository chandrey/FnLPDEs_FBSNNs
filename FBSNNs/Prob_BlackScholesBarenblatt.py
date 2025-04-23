# Import example:
    # from Prob_BlackScholesBarenblatt import BlackScholesBarenblatt
    
from FBSNNs import FBSNN

import numpy as np # for np.exp, np.sum
import tensorflow as tf # for tf.sin, tf.reduce_sum, tf.linalg.diag

# r = 0.0
# sigma = 0.1
class BlackScholesBarenblatt(FBSNN):
    # r = 0.0
    # sigma = 0.1
    def __init__(self, Xi, T,
                       M, N, D,
                       layers):        
        self.r = 0.05
        self.sigma = 0.40
        super().__init__(Xi, T,
                         M, N, D,
                         layers)        
        # self.r = 0.05
        # self.sigma = 0.40
        
    def __str__(self):
        return super().__str__() + f'; r={self.r}, sigma={self.sigma}'
        
    # equation           
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return self.r*(Y - tf.reduce_sum(X*Z, 1, keepdims = True)) # M x 1    
    # terminal condition
    def g_tf(self, X): # M x D
        return tf.reduce_sum(X**2, 1, keepdims = True) # M x 1
    
    # X process 
    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D        
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return self.sigma*tf.linalg.diag(X) # M x D x D
        # return sigma*tf.linalg.diag(X) # M x D x D
    
    # exact_solution
    def u_exact(self, t, X): # (N+1) x 1, (N+1) x D
        return np.exp( (self.r + self.sigma**2)*(self.T - t) ) * np.sum(X**2, 1, keepdims = True) # (N+1) x 1