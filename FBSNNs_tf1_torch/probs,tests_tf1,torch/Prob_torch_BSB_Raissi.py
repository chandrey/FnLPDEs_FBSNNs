# Import example:
    # from Prob_torch_BSB_Raissi import BSB_Raissi
    
from FBSNNs_torch import FBSNN_torch

import torch # for: torch.sin,
import numpy as np # for: np.exp, np.sum

class BSB_Raissi(FBSNN_torch):
    '''
    class BSB_Raissi(FBSNN_torch)
    '''
    def __init__(self, Xi, T,
                       M, N, D,
                       layers, 
                       act_func = torch.sin, s=0, name='model_BSB_Raissi'):
        super().__init__(Xi, T,
                         M, N, D,
                         layers, 
                         act_func=act_func, s=s, name=name) # FBSNN_torch.__init__(...)
        self.r = 0.05
        self.sigma = 0.40
        self._debug_ = 0
        
    def __str__(self):
        return super().__str__() + f'; r={self.r}, sigma={self.sigma}'
        
    ## init abstract method     
    # equation 
    def phi_torch(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return self.r*(Y - torch.sum(X*Z, dim=1).unsqueeze(-1)) # M x 1    
    # terminal condition
    def g_torch(self, X): # M x D
        return torch.sum(X**2, dim=1).unsqueeze(1) # M x 1   
    # X process 
    def mu_torch(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return torch.zeros([self.M, self.D])  # M x D        
    def sigma_torch(self, t, X, Y): # M x 1, M x D, M x 1
        return self.sigma*torch.diag_embed(X) # M x D x D
    
     ## exact_solution
    def u_exact(self, t, X): # (N+1) x 1, (N+1) x D
        return np.exp( (self.r + self.sigma**2)*(self.T - t) ) * np.sum(X**2, 1, keepdims = True) # (N+1) x 1