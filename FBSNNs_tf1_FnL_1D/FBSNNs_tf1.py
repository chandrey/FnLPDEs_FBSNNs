"""
  Documentation of FBSNNs_tf1
		- class FBSNN_tf1(FBSNN) - modification (by A. Chubatov) for the original class FBSNN (by M. Raissi)
"""

# Import example:
    # from FBSNNs_tf1 import FBSNN_tf1

from FBSNNs import FBSNN

import tensorflow as tf # for tf.sin, tf.add, tf.matmul
# tf.disable_v2_behavior()
import numpy as np # for: np.zeros, ... 
import time

class FBSNN_tf1(FBSNN): # Forward-Backward Stochastic Neural Network
    '''
        class FBSNN_tf1(FBSNN) - modification (by A. Chubatov) for the original class FBSNN (by M. Raissi)
            - Added new parameters: s, name
                s -- initial time in the interval [s,T] (modified fetch_minibatch method)
                name -- name of the model (task) for output in print and plot
            - Added attributes: act_func, iterations, loss_n, loss_history, it_loss_history
                act_func -- activation function (modified neural_net method)
                iterations = 0 -- iteration counter (modified methods: initialize_NN, train)
                loss_n = 1 -- different loss-function (more details in FBSNN_tf1.loss_function.__doc__) (modified loss_function method)
                loss_history = [] -- loss history list (modified train method)
                it_loss_history = 10 -- how often it is saved loss history (modified train method)
            - Added __str__() method
            - Modified methods (compared to FBSNN): __init__, neural_net, initialize_NN, fetch_minibatch, loss_function, train
    '''
    '''
    class FBSNN_tf1(FBSNN) - модификация (by A. Chubatov) для оригинального класса FBSNN (by M. Raissi)
        - Добавлены новые параметры: s, name 
            s -- начальный момент времени в промежутке [s,T] (модифицирован метод fetch_minibatch)
            name -- имя модели (задачи) для вывода в print и plot  
        - Добавлены атрибуты: act_func, iterations, loss_n, loss_history, it_loss_history
            act_func -- активационая функция (модифицирован метод neural_net)
            iterations = 0 -- счетчик итераций (модифицированы методы: initialize_NN, train)
            loss_n = 1 -- разные loss-function (подробнее в FBSNN_tf1.loss_function.__doc__) (модифицирован метод loss_function)
            loss_history = [] -- loss history list (модифицирован метод train)
            it_loss_history = 10 -- как часто сохраняется loss history (модифицирован метод train)
        - Добавлен метод  __str__()
        - Модифицированы методы (по сравнению с FBSNN): __init__, neural_net, initialize_NN, fetch_minibatch, loss_function, train
    '''
    
    # переписан, т.к. добавлены новые аттрибуты и параметры
	# method rewritten because added new attributes and parameters 
    def __init__(self, Xi, T,
                       M, N, D,
                       layers, 
                       act_func = tf.sin, s=0, name='Model_FBSNN'): # , act_f = 'sin' 
        '''
        FBSNN_tf1.__init__(Xi, T,
                       M, N, D,
                       layers, 
                       act_func = tf.sin, s=0, name='ModelFBSNN'):
           
        '''
        # new attributes
        self.act_func = act_func;
        self.iterations = 0 # iterations
        self.loss_n = 1 # use loss1
        self.loss_history = [] 
        self.it_loss_history = 10
        super().__init__(Xi, T,
                         M, N, D,
                         layers) # FBSNN.__init__(...)
        # new parameters
        self.s = s 
        self.name = name
        
    # добавлен 
	# method added    
    def __str__(self):
        _d = min(4,self.Xi.shape[1])
        return f'{self.name}: s={self.s}, T={self.T}, D={self.D}, Xi[0:{_d}]={self.Xi[0:_d]}; \t N={self.N}; '+\
               f'\n\t NNparams: M={self.M}, layers={self.layers}, act_func={self.act_func}, optimizer={self.optimizer}; '+\
               f'\n\t Learning: it={self.iterations}; '
    
	# method rewritten because added new attributes and parameters  
    def neural_net(self, tX, weights, biases):    
        '''
        FBSNN_tf1.neural_net(X, weights, biases):
            - добавлен учет параметра act_func_ 
        '''
        if hasattr(self, 'act_func'): 
            act_func_ = self.act_func
        else: 
            act_func_ = tf.sin # by default
            
        num_layers = len(weights) + 1
        
        H = tX        
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = act_func_( tf.add(tf.matmul(H, W), b) ) # self.act_func( tf.add(tf.matmul(H, W), b) )
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
	# method rewritten because added new attributes and parameters 
    def initialize_NN(self, layers):
        '''
        FBSNN_tf1.initialize_NN(layers):
            - в методе initialize_NN() обнулен аттрубут self.iterations = 0 
        '''
        self.iterations = 0 # iterations
        self.loss_history = [] 
        self.it_loss_history = 10
        
        self.weights, self.biases = super().initialize_NN(layers)
        return self.weights, self.biases
    
	# method rewritten because added new attributes and parameters  
    def fetch_minibatch(self):
        '''
        FBSNN_tf1.fetch_minibatch():
            - учет начального момента s
        '''
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
        
        return t, W
    
	# method rewritten because added new attributes and parameters 
    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        '''
        FBSNN_tf1.loss_function(t, W, Xi):
            - легкая модификация loss'ов - добавление кроме loss еще loss0, loss1, loss_1, loss_2
                loss =  sum0N + loss_uT + loss_DuT # original Raissi loss (self.loss_n == else == OTHERWISE)
                loss0 = 1/self.M * ( sum0N + loss_uT )  # (self.loss_n == 0)                  
                loss0_ = 1/self.M * ( sum0N + loss_uT )  #  
                loss1 = 1/self.M * ( sum0N + loss_uT + loss_DuT )  # (self.loss_n == 1)     
                loss_1 = 1/self.M * loss_uT   # (self.loss_n == -1)    
                loss_2 = 1/self.M * ( loss_uT + loss_DuT )  # (self.loss_n == -2)     
            - чуть преписаны вычисления (например, s_dW вычисляется 1 раз, а не 2)
                s_dW = tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)), axis=[-1])                
        '''
        loss = 0
        X_list= []
        Y_list = []     
        
        t0 = t[:,0,:] # M x 1
        W0 = W[:,0,:] # M x D
        X0 = tf.tile(Xi, [self.M,1]) # 1 x D, ... --> M x D     # X0 = tf.tile(Xi,[self.M,1]) # M x D
        Y0, Z0 = self.net_u(t0,X0) # M x 1, M x D --> M x 1, M x D         
        X_list.append(X0)
        Y_list.append(Y0)        
        for n in range(0,self.N):
            t1 = t[:,n+1,:] # M x 1
            W1 = W[:,n+1,:] # M x D
            dt = t1-t0 # --> M x 1
            dW = W1-W0 # --> M x D
            s_dW = tf.squeeze( tf.matmul( self.sigma_tf(t0,X0,Y0), tf.expand_dims(dW,-1) ), axis=[-1] ) # --> M x D             
            # Y1_tilde = Y0 + self.phi_tf(t0,X0,Y0,Z0)*(t1-t0) + \
            #           tf.reduce_sum(Z0*tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1))), axis=1, keepdims = True)
            Y1_tilde_ = Y0 + self.phi_tf(t0,X0,Y0,Z0)*dt + tf.reduce_sum( Z0*s_dW, axis=1, keepdims = True)            
            # X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*(t1-t0) + \
            #      tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)), axis=[-1]) 
            X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*dt + s_dW # --> M x D              
            Y1, Z1 = self.net_u(t1, X1)            
            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde_))            
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1            
            X_list.append(X0)
            Y_list.append(Y0)            
        sum0N = loss            
        ## loss2 = 1/self.N * loss # MY        
        loss_uT  = tf.reduce_sum(tf.square(Y1 - self.g_tf(X1))) # MY
        loss_DuT = tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1))) # MY         
        loss =  sum0N + loss_uT + loss_DuT # original Raissi loss        
        loss_1 = 1/self.M * ( loss_uT )  # MY                      
        loss_2 = 1/self.M * ( loss_uT + loss_DuT )  # MY                
        loss0_ = 1/self.M * ( sum0N + loss_uT )  # MY             
        loss0 = 1/self.M * ( dt[0] * sum0N + loss_uT )  # MY       
        # loss1 = 1/self.M * loss        # MY  # 
        loss1 = 1/self.M * ( sum0N*dt + loss_uT + 0*loss_DuT )
        # loss3 = 1/self.M * ( sum0N + loss_uT + 1/self.D * loss_DuT ) # MY
        
        X = tf.stack(X_list,axis=1)
        Y = tf.stack(Y_list,axis=1)
        
        if self.loss_n == 0: loss_ = loss0
        elif self.loss_n == 1: loss_ = loss1
        elif self.loss_n == -1: loss_ = loss_1
        elif self.loss_n == -2: loss_ = loss_2
        # elif self.loss_n == 3: loss_ = loss3
        else: loss_ = loss
            
        return loss_, X, Y, Y[0,0,0]     
    
	# method rewritten because added new attributes and parameters     
    def train(self, N_Iter, learning_rate, it_print=100): # def train(self, N_Iter, learning_rate):
        '''
        FBSNN_tf1.train(N_Iter, learning_rate, it_print=100):
            - добавлен аргумент it_print=100
            - учтен iterations, it_loss_history 
            - в процессе обучения заполняется loss_history
            - добавлен учет NaN
                if np.isnan(loss_value[0]): raise ValueError(f'NaN loss value! it={it}')
        '''
        
        # loss_history = [] # my TODO array
        ## loss_temp = np.array([]) ### ?
        
        previous_it = self.iterations

        start_time = time.time()
        for it in range(previous_it+1, previous_it + N_Iter+1):
            
            t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
            
            tf_dict = {self.Xi_tf: self.Xi, self.t_tf: t_batch, self.W_tf: W_batch, self.learning_rate: learning_rate}
            
            self.sess.run(self.train_op, tf_dict)
            
            ## loss_temp = np.append( loss_temp, loss_value )
            
            # Print
            if it % it_print == 0:
                elapsed = time.time() - start_time
                ### возврат loss 
                loss_value, Y0_value, learning_rate_value = self.sess.run([self.loss, self.Y0_pred, self.learning_rate], tf_dict)
                # print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' % 
                #      (it, loss_value, Y0_value, elapsed, learning_rate_value))                    
                print( f'It: {it}, Loss: {loss_value:.3e}, Y0: {Y0_value}, Time: {elapsed:.2f}, Learning Rate: {learning_rate_value:.3e}' )
                                
                ## if np.isnan(Y0_value[0]): raise ValueError(f'NaN value! \n ')
                    
                # print( f'It: {it}, Loss: {loss_value:.3e}, Y0: {Y0_value:.3e}, Time: {elapsed:.2f}, Learning Rate: {learning_rate_value:.3e}' )
                start_time = time.time()
                           
            # loss history
            if it % self.it_loss_history == 0:
                ### возврат loss 
                loss_value = self.sess.run([self.loss], tf_dict)
                if np.isnan(loss_value[0]): raise ValueError(f'NaN loss value! it={it}\n ')
                self.loss_history.append( (it,loss_value[0]) )# my
                
                # loss_history.append( loss_temp.mean() )# my
                # loss_temp = np.array([])
            ## self.iterations += 1
        self.iterations += N_Iter
        # return self.loss_history
        