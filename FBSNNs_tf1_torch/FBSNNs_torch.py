"""
  Documentation of FBSNNs_torch
  модификация (by Chubatov Andrey) для кода https://github.com/Shine119/FBSNNs_pytorch 
      - class neural_net(nn.Module)
      - class FBSNN_torch(nn.Module)  
"""

# Import example:
    # from FBSNNs_torch import FBSNN_torch

import torch    
import numpy as np
import time
# import torch.nn as nn
# import torch.optim as optim
#import matplotlib.pyplot as plt
# from plotting import newfig, savefig

from abc import abstractmethod

class neural_net(torch.nn.Module):
    '''
    class neural_net(torch.nn.Module) - модификация (by Chubatov Andrey) для кода https://github.com/Shine119/FBSNNs_pytorch
    ''' 
    def __init__(self, act_func = torch.sin): #,     pathbatch=100, n_dim=100+1, n_output=1):
        super(neural_net, self).__init__()
        #self.pathbatch = pathbatch
        #self.n_dim = n_dim
        #self.n_output = n_output

        #self.relu = nn.ReLU()
        #self.prelu = nn.PReLU()
        #self.tanh = nn.Tanh()
        
        self.act_func = act_func
        
        # self.xavier_init()        
            
    def initialize_NN(self, layers):
        # TODO добавить учет layers        
        '''
        self.fc_1 = nn.Linear(self.n_dim, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 256)
        self.fc_4 = nn.Linear(256, 256)
        self.out = nn.Linear(256, self.n_output)
        '''
        self.layers = layers
        num_layers = len(layers)
        self.fc = []
        for l in range(0,num_layers-2):
            # self.fc[l] = nn.Linear(layers[l], layers[l+1])
            self.fc.append( torch.nn.Linear(layers[l], layers[l+1]) )
        self.out = torch.nn.Linear(layers[-2], layers[-1])
        with torch.no_grad():
            # <ipython-input-2-ee67bbafe65f>:16: UserWarning: 
            # nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.
            '''
            torch.nn.init.xavier_uniform_(self.fc_1.weight) # torch.nn.init.xavier_uniform(self.fc_1.weight)
            torch.nn.init.xavier_uniform_(self.fc_2.weight) # torch.nn.init.xavier_uniform(self.fc_2.weight)
            torch.nn.init.xavier_uniform_(self.fc_3.weight) # torch.nn.init.xavier_uniform(self.fc_3.weight)
            torch.nn.init.xavier_uniform_(self.fc_4.weight) # torch.nn.init.xavier_uniform(self.fc_4.weight)
            '''
            for l in range(len(layers)-2):
                torch.nn.init.xavier_uniform_(self.fc[l].weight) 
            
    # act_func == torch.sin
    def forward(self, state, train=False):
        '''state = self.act_func(self.fc_1(state)) # torch.sin(self.fc_1(state))   
        state = self.act_func(self.fc_2(state)) # torch.sin(self.fc_2(state))  
        state = self.act_func(self.fc_3(state)) # torch.sin(self.fc_3(state))  
        state = self.act_func(self.fc_4(state)) # torch.sin(self.fc_4(state))
        fn_u = self.out(state) 
        '''
        for l in range(len(self.layers)-2):
            state = self.act_func(self.fc[l](state))
        fn_u = self.out(state)
        return fn_u

    
class FBSNN_torch(torch.nn.Module): # Forward-Backward Stochastic Neural Network
    '''
    class FBSNN_torch(torch.nn.Module) - модификация (by Chubatov Andrey) для кода https://github.com/Shine119/FBSNNs_pytorch
    ''' 
    # def __init__(self, Xi, T, M, N, D, learning_rate):
    def __init__(self, Xi, T,
                       M, N, D,
                       layers, 
                       act_func = torch.sin, s=0, name='Model_FBSNN'): # , act_f = 'sin' 
        '''
        FBSNN_torch.__init__(Xi, T,
                       M, N, D,
                       layers, 
                       act_func = tf.sin, s=0, name='ModelFBSNN'):
        '''
        super().__init__()
        self._debug_ = 0 
        self.tool = 'pyTorch'
        self.Xi = Xi # initial point
        self.T = T # terminal time
        
        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions    
        # layers
        self.layers = layers # (D+1) --> 1 # layers = [D+1] + hlayers + [1]
        self.fn_u = neural_net(act_func=act_func) #,    pathbatch=M, n_dim=D+1, n_output=1) # вместо neural_net
        self.initialize_NN(layers)
        
        # self.learning_rate = 1e-2 # learning_rate
        # перенесено train self.optimizer = optim.Adam(self.fn_u.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam # optim.Adam
        
        # new attributes
        self.act_func = act_func
        self.iterations = 0 # iterations
        self.loss_n = 1 # use loss1
        self.loss_history = [] 
        self.it_loss_history = 10
        # new parameters
        self.s = s 
        self.name = name        

    # добавлен     
    def __str__(self):
        _d = min(4,self.Xi.shape[1])
        return f'{self.name}: s={self.s}, T={self.T}, D={self.D}, Xi[0:{_d}]={self.Xi[0:_d]}; \t N={self.N}; '+\
               f'\n\t NNparams: M={self.M}, layers={self.layers}, act_func={self.act_func}, optimizer={self.optimizer}; '+\
               f'\n\t Learning: it={self.iterations}; '
    
    def initialize_NN(self, layers): # def initialize_NN(self, layers):
        self.iterations = 0 # iterations
        self.fn_u.initialize_NN(layers)
        
    # вместо net_u 
    def net_u_Du(self, t, X): # M x 1, M x D
        tX = torch.cat([t, X], dim=1)
        u = self.fn_u(tX) # вместо neural_net
        Du = torch.autograd.grad(torch.sum(u), X, retain_graph=True)[0]  
        return u, Du

    def fetch_minibatch(self):
        # код как для TF
        T = self.T
        s = self.s
        M = self.M
        N = self.N
        D = self.D
        
        Dt = np.zeros((M,N+1,1)) # M x (N+1) x 1
        DW = np.zeros((M,N+1,D)) # M x (N+1) x D
        
        dt = (T-s)/N
        
        Dt[:,0,:] = s
        Dt[:,1:,:] = dt
        DW[:,1:,:] = np.sqrt(dt)*np.random.normal(size=(M,N,D))
        
        t = np.cumsum(Dt,axis=1) # M x (N+1) x 1
        W = np.cumsum(DW,axis=1) # M x (N+1) x D
        
        return torch.from_numpy(t).float(), torch.from_numpy(W).float() # в TF тут return t, W

    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        '''
        FBSNN_torch.loss_function(t, W, Xi):
            - кроме loss еще вычисляется loss0, loss1
                loss0 = 1/self.M * ( sum0N + loss_uT )  # (self.loss_n == 0)    
                loss1 = 1/self.M * ( sum0N + loss_uT + loss_DuT )  # (self.loss_n == 1)      
                loss =  sum0N + loss_uT + loss_DuT # original Raissi loss (self.loss_n == else == OTHERWISE)
            - чуть преписаны вычисления (например, s_dW вычисляется 1 раз, а не 2)
                s_dW = torch.matmul(self.sigma_torch(t0,X0,Y0), (W1-W0).unsqueeze(-1)).squeeze(2)                
        '''
        loss = torch.zeros(1)
        X_list = []
        Y_list = []
        
        t0 = t[:,0,:] # M x 1   
        W0 = W[:,0,:] # M x D
        X0 = torch.cat([Xi]*self.M) # M x D 
        X0.requires_grad = True
        Y0, Z0 = self.net_u_Du(t0,X0) # M x 1, M x D     
        X_list.append(X0)
        Y_list.append(Y0)
        
        for n in range(0,self.N):
            t1 = t[:,n+1,:]
            W1 = W[:,n+1,:]            
            dt = t1-t0            
            dW = W1-W0            
            s_dW = torch.matmul(self.sigma_torch(t0,X0,Y0), dW.unsqueeze(-1)).squeeze(2)
            _flag_print_ = self._debug_ and n==0 and self.iterations == 0 
            if _flag_print_: 
                print(f'_flag_print_={_flag_print_}, it={self.iterations}')
                print(f'X0.shape = {X0.shape}, Y0.shape = {Y0.shape}') 
                print(f'Z0.shape = {Z0.shape}, s_dW.shape = {s_dW.shape}')
            Y1_tilde = Y0 + self.phi_torch(t0,X0,Y0,Z0)*dt + \
                torch.sum(Z0*s_dW, dim=1).unsqueeze(1)
            mu = self.mu_torch(t0,X0,Y0,Z0)
            if _flag_print_: 
                print(f'mu.shape = {mu.shape}, dt.shape = {dt.shape}')
            mu_dt = mu * dt # mu_dt = torch.matmul(mu, dt)
            if _flag_print_: 
                print(f'mu_dt.shape = {mu_dt.shape}')
            X1 = X0 + mu_dt + s_dW # M x D 
            Y1, Z1 = self.net_u_Du(t1,X1)            
            if _flag_print_: 
                print(f'Y1_tilde.shape = {Y1_tilde.shape}, Y1.shape = {Y1.shape}, Z1.shape = {Z1.shape}')            
            loss = loss + torch.sum( (Y1 - Y1_tilde)**2 )            
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1        
            X_list.append(X0)
            Y_list.append(Y0)           
        sum0N = loss           
        loss_uT  = torch.sum((Y1 - self.g_torch(X1))**2) # MY
        loss_DuT = torch.sum((Z1 - self.Dg_torch(X1))**2)
        loss =  sum0N + loss_uT + loss_DuT # original Raissi loss       
        loss0 = 1/self.M * ( sum0N + loss_uT )  # MY       
        loss1 = 1/self.M * loss        
        
        X = torch.stack(X_list,dim=1) # M x N x D 
        Y = torch.stack(Y_list,dim=1) # M x N x 1
        
        if self.loss_n == 0: loss_ = loss0
        elif self.loss_n == 1: loss_ = loss1
        # elif self.loss_n == 3: loss_ = loss3
        else: loss_ = loss
            
        return loss_, X, Y, Y[0,0,0]  

    def train(self, N_Iter, learning_rate, it_print=100):        
        # self.learning_rate = learning_rate
        previous_it = self.iterations

        start_time = time.time()
        for it in range(previous_it+1, previous_it + N_Iter+1):
            
            t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            
            # self.optimizer = self.opt(self.fn_u.parameters(), lr=self.learning_rate)
            self.train_op = self.optimizer(self.fn_u.parameters(), lr=learning_rate)
            
            self.train_op.zero_grad()
            loss.backward()
            self.train_op.step()
            
            # Print
            if it % it_print == 0:
                elapsed = time.time() - start_time
                # print( f'It: {it}, Loss: {loss:.3e}, Y0: {Y0_pred}, Time: {elapsed:.2f}, Learning Rate: {learning_rate:.3e}' )
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' \
                      % (it, loss, Y0_pred, elapsed,learning_rate))
                start_time = time.time()
            # loss history
            if it % self.it_loss_history == 0:
                if np.isnan(loss.detach().numpy()): raise ValueError(f'NaN loss value! it={it}\n ')
                self.loss_history.append( (it,loss.detach().numpy()) )# my
            self.iterations += 1
        # self.iterations += N_Iter

    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)
        
        return X_star, Y_star
    
    @abstractmethod 
    def phi_torch(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass
    
    @abstractmethod 
    def g_torch(self, X): # M x D
        pass
    
    def Dg_torch(self, X): # M x D
        return torch.autograd.grad(torch.sum(self.g_torch(X)), X, retain_graph=True)[0] # M x D
    
    @abstractmethod
    def mu_torch(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass
    
    @abstractmethod
    def sigma_torch(self, t, X, Y): # M x 1, M x D, M x 1
        pass
    