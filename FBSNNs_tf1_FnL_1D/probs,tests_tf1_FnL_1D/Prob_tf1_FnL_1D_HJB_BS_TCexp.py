# Import example:
    # from Prob_tf1_FnL_1D_HJB_BS_ICexp import Prob_tf1_FnL_1D_HJB_BS_ICexp

from Equ_tf1_FnL_1D_HJB_BS import Equ_tf1_FnL_1D_HJB_BS

import numpy as np # for: np.exp, np.sum
import tensorflow as tf # for: tf.sin, tf.reduce_sum, tf.linalg.diag

# Xi = np.zeros([1,D])
# TODO подумать каа задать ТУТ Xi

## здесь D == 1

class Prob_tf1_FnL_1D_HJB_BS_TCexp(Equ_tf1_FnL_1D_HJB_BS): 
    def __init__(self, Xi, T,
                       M, N,
                       hlayers, 
                       act_func = tf.tanh, s=0, name='model_FnL_1D_HJB_BS__ICexp',
                       mu=0.02, sigma=0.1, 
                       gamma=0.5): 
        ## TODO gamma >0 
        self._gamma = gamma
        super().__init__(Xi, T,
                         M, N,
                         hlayers, 
                         act_func=act_func, s=s, name=name,
                         mu=mu, sigma=sigma) # Equ_tf1_FnL_1D_HJB_BS.__init__(...)      
        
    def __str__(self):
        return super().__str__() + f'; IC: u(T,x)=-exp(-gamma*x), gamma={self._gamma}'
    
    # equation 1           
    # terminal condition g_tf(x) = u(T, x)
    def g_tf(self, X): # M x D --> M x 1
        # FOR g(x)=u(T,x)=-exp(-gamma*x)
        return -tf.math.exp(-self._gamma*X) # M x 1 
    
    # equation 2  
    # terminal condition f_tf(x) = v(T, x)
    def f_tf(self, X): # M x D --> M x D
        # FOR f(x)=v(T,x)=gamma*exp(-gamma*x)
        return self._gamma * tf.math.exp(-self._gamma*X) # M x 1
    
    ## exact_solution
    def u_exact(self, t, X): # NC x 1, NC x D --> NC x 1
        # FOR g(x)=u(T,x)=-exp(-gamma*x)
        return -np.exp(-self._gamma*X) * np.exp( -1/2*(self.mu/self.sigma)**2 *(self.T-t) )
    
    def Du_exact(self, t, X): # NC x 1, NC x D --> NC x 1
        # FOR g(x)=u(T,x)=-exp(-gamma*x)
        return self._gamma * np.exp(-self._gamma*X) * np.exp( -1/2*(self.mu/self.sigma)**2 *(self.T-t) )
    