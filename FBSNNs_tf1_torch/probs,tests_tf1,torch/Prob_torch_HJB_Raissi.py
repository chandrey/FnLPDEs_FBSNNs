# Import example:
    # from Prob_torch_HJB_Raissi import HJB_Raissi
    
from FBSNNs_torch import FBSNN_torch

import torch # for: torch.sin,
import numpy as np # for: np.exp, np.sum

class HJB_Raissi(FBSNN_torch):
    '''
    class HJB_Raissi(FBSNN_torch)
    '''
    def __init__(self, Xi, T,
                       M, N, D,
                       layers, 
                       act_func = torch.sin, s=0, name='model_HJB_Raissi'):
        super().__init__(Xi, T,
                         M, N, D,
                         layers, 
                         act_func=act_func, s=s, name=name) # FBSNN_torch.__init__(...)
        self.sigma = np.sqrt(2.0)
        self._debug_ = 0
        
    def __str__(self):
        return super().__str__() + f'; sigma={self.sigma}'
        
    ## init abstract method     
    # equation 
    def phi_torch(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        # return self.r*(Y - tf.reduce_sum(X*Z, 1, keepdims = True)) # M x 1  
        # return self.r*(Y - torch.sum(X*Z, dim=1).unsqueeze(-1)) # M x 1   
        # return tf.reduce_sum(Z**2, 1, keepdims = True) # M x 1   
        return torch.sum(Z**2, dim=1).unsqueeze(-1) # M x 1
    # terminal condition
    def g_torch(self, X): # M x D       
        # return tf.reduce_sum(X**2, 1, keepdims = True) # M x 1  
        # return torch.sum(X**2, dim=1).unsqueeze(1) # M x 1  
        # return tf.math.log( 0.5 + 0.5*tf.reduce_sum(X**2, 1, keepdims = True) ) # M x 1
        return torch.log( 0.5 + 0.5*torch.sum(X**2, dim=1).unsqueeze(1) ) # M x 1  
    # X process 
    def mu_torch(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D        
        # return super().mu_tf(t, X, Y, Z) # M x D        
        return torch.zeros([self.M, self.D])  # M x D        
    def sigma_torch(self, t, X, Y): # M x 1, M x D, M x 1
        # return self.sigma*tf.linalg.diag(X) # M x D x D
        # return self.sigma*torch.diag_embed(X) # M x D x D
        # return self.sigma__tf*super().sigma_tf(t, X, Y) # M x D x D
        return self.sigma*torch.diag_embed(X) # M x D x D 
    
    ## exact_solution
    def g(self, X): # MC x NC x D
        return np.log(0.5 + 0.5*np.sum(X**2, axis=-1, keepdims=True)) # MC x N x 1  ## axis=2
    def u_exact(self, t, X, MC=10**4): # NC x 1, NC x D
        # NB!! global var: sigma
        # MC = 10**4 # 10**5
        NC = t.shape[0]
        D = X.shape[1] # self.D
        T = self.T      
        W = np.random.normal(size=(MC,NC,D)) # MC x NC x D        
        # return -np.log( np.mean( np.exp( -g(X + np.sqrt(2.0*np.abs(T-t))*W) ), axis=0 ) )
        return -np.log( np.mean( np.exp( -self.g(X + self.sigma*np.sqrt(np.abs(T-t))*W) ), axis=0 ) )# r = 0 