# FnLPDEs_FBSNNs framework

## Dependencies-relationships between classes (packages)
* ./FBSNNs/FBSNNs.py - minimally modified of original file (by M. Raissi)
    class FBSNNs - minimally modified of original class (by M. Raissi)
        - so that it runs on tensorFlow2x
            -- added row `tf.compat.v1.disable_eager_execution()`
            -- `tf.placeholder` <-replace-> `tf.compat.v1.placeholder`
        - added attribute self.tool = 'tensorflow 1x' # "code for TF1 running in TF2"
        
### tf1-forks of FBSNN class
    
* ./FBSNNs_tf1_torch/FBSNNs_tf1.py - modification (by A. Chubatov) for original framework FBSNNs (by M. Raissi)
    class FBSNN_tf1(FBSNN) - (fork) modification (by A. Chubatov) for the original FBSNN class (by M. Raissi)
        - Added new parameters: s, name
        - Added attributes: act_func, iterations, loss_n, loss_history, it_loss_history
        - Added method __str__()
        - Modified methods (with respect to class FBSNN):
            neural_net, initialize_NN, loss_function, fetch_minibatch, train   
        
* ./FBSNNs_tf1_FnL_1D/FBSNNs_tf1_FnL_1D.py - extention (by A. Chubatov) for the original FBSNN class (by M. Raissi)
    class FBSNN_tf1_FnL_1D(FBSNN_tf1) - new class Forward-Backward Stochastic Neural Network for 
            -- Fully non Linear Parabolic Equation (x in 1D);
            -- system (2eq=(1+D)eq) quasi linear PDE in 1D.
        - Added abstract method: mu_tf, sigma_tf, phi_tf, g_tf, psi_tf, f_tf.       
        - Modified methods (with respect to class FBSNN_tf1):
            __init__, Df_tf, net_u, loss_function.
            
### torch-forks of FBSNN class
* ./FBSNNs_tf1_torch/FBSNNs_torch.py - modification (by A. Chubatov) for the code https://github.com/Shine119/FBSNNs_pytorch