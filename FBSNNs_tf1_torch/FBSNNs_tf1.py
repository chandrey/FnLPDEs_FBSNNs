"""
  Documentation of FBSNNs_tf1
      - class FBSNN_tf1(FBSNN) - модификация (by Chubatov Andrey) для оригинального класса FBSNN (by Raissi)
"""

# Import example:
    # from FBSNNs_tf1 import FBSNN_tf1

### from FBSNNs import FBSNN

import tensorflow as tf # for tf.sin, tf.add, tf.matmul
#tf.disable_v2_behavior()
import numpy as np # for: np.zeros, ... 
import time

from abc import ABC, abstractmethod

def act_func_str2func_tf1(act_func_name):
    if act_func_name == 'relu': 
        act_func = tf.nn.relu         
    elif act_func_name == 'sin':
        act_func = tf.sin      
    elif act_func_name == 'tanh':
        act_func = tf.tanh
    else:
        act_func = None
    return act_func 



class FBSNN_tf1(ABC): # Forward-Backward Stochastic Neural Network
    '''
    class FBSNN_tf1(FBSNN) - модификация (by Chubatov Andrey) для оригинального класса FBSNN (by Raissi)
        - Добавлены новые параметры: s, name 
            s -- начальный момент времени в промежутке [s,T] (модифицирован метод fetch_minibatch)
            name -- имя модели (задачи) для вывода в print и plot  
        - Добавлены атрибуты: act_func, iterations, loss_n, loss_history, it_loss_history
            act_func -- активационая функция (модифицирован метод neural_net)
            iterations = 0 -- счетчик итераций (модифицированы методы: initialize_NN, train)
            loss_n = 1 -- разные loss-function (подробнее в FBSNN_tf1.loss_function.__doc__) (модифицирован метод loss_function)
            loss_history = [] -- loss history list (модифицирован метод train)
            it_loss_history = 10 -- как часто сохраняется loss history (модифицирован метод train)
            learning_rate_history = []
        - Добавлен метод  __str__()
        - Модифицированы методы (по сравнению с FBSNN): neural_net, initialize_NN, fetch_minibatch, loss_function, train
    '''
    tool = 'TensorFlow_1x'
    
    # переписан, т.к. добавлены новые аттрибуты и параметры 
    def __init__(self, Xi, T,
                       M, N, D,
                       layers, 
                       act_func_name = 'sin', s=0, name='Model_FBSNN'): # , act_f = 'sin' 
        '''
        FBSNN_tf1.__init__(Xi, T,
                       M, N, D,
                       layers, 
                       act_func = tf.sin, s=0, name='ModelFBSNN'):
           
        '''
        self.tool = FBSNN_tf1.tool
        # new attributes
        # self.act_func = act_func;
        self.act_func_name = act_func_name
        self.act_func = act_func_str2func_tf1(act_func_name)
        self.lambda_1 = 1 
        self.lambda_2 = 1 
        self.iterations = -10 # iterations
        self.loss_n = 1 # use loss1
        self.loss_history = [] 
        self.it_loss_history = 10
        self.learning_rate_history = []
        self.prev_learning_rate = None
        self.learning_rate = None
        
        ### super().__init__(Xi, T, M, N, D, layers) # FBSNN.__init__(...)
        self.Xi = Xi # initial point
        self.T = T # terminal time        
        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions        
        # layers
        self.layers = layers # (D+1) --> 1        
        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)    
        # self.initialize_NN(layers)   
#        print( f'self.weights ={self.weights}, self.biases = {self.biases}' )
        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))        
        # tf placeholders and graph (training)
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[M, self.N+1, 1]) # M x (N+1) x 1
        self.W_tf = tf.compat.v1.placeholder(tf.float32, shape=[M, self.N+1, self.D]) # M x (N+1) x D
        self.Xi_tf = tf.compat.v1.placeholder(tf.float32, shape=[1, D]) # 1 x D
        # loss_function
        self.loss, self.X_pred, self.Y_pred, self.Y0_pred = self.loss_function(self.t_tf, self.W_tf, self.Xi_tf)   
        # optimizers
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate = self.learning_rate,
            beta1=0.9,                # Экспоненциальная затухающая скорость среднего градиента
            beta2=0.999,              # Экспоненциальная затухающая скорость среднеквадратичного значения градиента
            epsilon=1e-8 #,              # Малое значение для численной стабильности
            # amsgrad=False              # Включить/Выключить модификацию AMSGrad
        )
        self.train_op = self.optimizer.minimize(self.loss)        
        # initialize session and variables
        init = tf.compat.v1.global_variables_initializer()        
        self.sess.run(init)
            
            
        # new parameters
        self.s = s 
        self.name = name
    
    def initialize_NN(self, layers):
        self.iterations = 0 # -1 # iterations
        self.loss_history = [] 
        self.it_loss_history = 10
        
        #### self.weights, self.biases = super().initialize_NN(layers)
        weights = []
        biases = []
        # print( f'\t running FBSNN.initialize_NN ...' ) ## MY
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        self.weights, self.biases = weights, biases ## MY
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                               stddev=xavier_stddev), dtype=tf.float32)
    
    
    # добавлен     
    def __str__(self):
        _d = min(4,self.Xi.shape[1])
        return f'{self.name}: s={self.s}, T={self.T}, D={self.D}, Xi[0:{_d}]={self.Xi[0:_d]}; \t N={self.N}; '+\
               f'\n\t NNparams: M={self.M}, layers={self.layers}, act_func={self.act_func}, optimizer={self.optimizer}; '+\
               f'\n\t Learning: it={self.iterations}; '
     
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
    
    def net_u(self, t, X): # M x 1, M x D        
        u = self.neural_net(tf.concat([t,X], 1), self.weights, self.biases) # M x 1
        Du = tf.gradients(u, X)[0] # M x D        
        return u, Du  
    
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
#        print('123456')
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
        loss_uT  = 1 * tf.reduce_sum(tf.square(Y1 - self.g_tf(X1))) # MY
        loss_DuT = 1 * tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1))) # MY  
        
        loss =  sum0N + loss_uT + loss_DuT # original Raissi loss        
        loss_1 = 1/self.M * ( loss_uT )  # MY                      
        loss_2 = 1/self.M * ( loss_uT + loss_DuT )  # MY                
        # loss0_ = 1/self.M * ( sum0N + loss_uT )  # MY             
        loss0 = 1/self.M * ( sum0N + loss_uT )  # MY       
        # loss1 = 1/self.M * loss        # MY  # 
        loss1 = 1/self.M * ( dt[0][0] * sum0N + self.lambda_1 * loss_uT )
        loss2 = 1/self.M * ( dt[0][0] * sum0N + self.lambda_1 * loss_uT + self.lambda_2 * loss_DuT )
        loss3 = 1/self.M * ( dt[0][0] * sum0N + self.lambda_1 * loss_uT + 1/self.D * loss_DuT ) # MY 
        
        X = tf.stack(X_list,axis=1)
        Y = tf.stack(Y_list,axis=1)
        
        if self.loss_n == 0: loss_ = loss0
        elif self.loss_n == 1: loss_ = loss1
        elif self.loss_n == 2: loss_ = loss2
        elif self.loss_n == -1: loss_ = loss_1
        elif self.loss_n == -2: loss_ = loss_2
        # elif self.loss_n == 3: loss_ = loss3
        else: loss_ = loss
            
        return loss_, X, Y, Y[0,0,0]     
        
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
            if it==1 or it % it_print == 0:
                elapsed = time.time() - start_time
                ### возврат loss 
                loss_value, Y0_value, learning_rate_value = self.sess.run([self.loss, self.Y0_pred, self.learning_rate], tf_dict)
                # print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' % 
                #      (it, loss_value, Y0_value, elapsed, learning_rate_value))               
#                print( f'It: {it}, Loss: {loss_value[0]}, Y0: {Y0_value}' )
                print( f'It: {it}, Loss: {loss_value:.3e}, Y0: {Y0_value}, Time: {elapsed:.2f}, Learning Rate: {learning_rate_value:.3e}' )
                                
                ## if np.isnan(Y0_value[0]): raise ValueError(f'NaN value! \n ')
                    
                # print( f'It: {it}, Loss: {loss_value:.3e}, Y0: {Y0_value:.3e}, Time: {elapsed:.2f}, Learning Rate: {learning_rate_value:.3e}' )
                start_time = time.time()
                           
            # loss history
            if it==1 or it % self.it_loss_history == 0:
                ### возврат loss 
                loss_value = self.sess.run([self.loss], tf_dict)[0]
#                print( f'loss_value = {loss_value}' )
                if np.isnan(loss_value): raise ValueError(f'NaN loss value! it={it}\n ')
                self.loss_history.append( (it,loss_value) )# my
                
                # loss_history.append( loss_temp.mean() )# my
                # loss_temp = np.array([])
            self.iterations += 1
        # self.iterations += N_Iter        
        if self.prev_learning_rate != learning_rate :
            self.learning_rate_history.append( ((previous_it+1,self.iterations), learning_rate) )
            self.prev_learning_rate = learning_rate
        else:
            _lr_h_end = self.learning_rate_history[-1]
            self.learning_rate_history[-1] = ((_lr_h_end[0][0],self.iterations), _lr_h_end[1])
        # return self.loss_history
        
        
    def predict(self, Xi_star, t_star, W_star):        
        tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}        
        X_star = self.sess.run(self.X_pred, tf_dict)
        Y_star = self.sess.run(self.Y_pred, tf_dict)        
        return X_star, Y_star
    
    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass # M x1
    
    @abstractmethod
    def g_tf(self, X): # M x D
        pass # M x 1
    
    def Dg_tf(self, X): # M x D
        return tf.gradients(self.g_tf(X), X)[0] # M x D
    
    @abstractmethod
    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return np.zeros([M,D]) # M x D
    
    @abstractmethod
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return tf.linalg.diag(tf.ones([M,D])) # M x D x D
    ###########################################################################