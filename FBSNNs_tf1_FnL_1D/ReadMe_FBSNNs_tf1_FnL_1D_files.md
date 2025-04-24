# FOLDER `FBSNNs_tf1_FnL_1D`

### FBSNNs_tf1_FnL_1D package - a extention (by A. Chubatov) for original FBSNN framework (by M. Raissi)
        (running on TensorFlow2 (TF2) + pyTorch) 
* FBSNNs.py - minimally modified original file (by M. Raissi)
* FBSNNs_plots.py - package for plot of trajectories
    plot_results(...) - output of trajectories
* FBSNNs_tf1.py - modification (by A. Chubatov) for original framework FBSNNs (by M. Raissi)
    class FBSNN_tf1(FBSNN) - (fork) modification (by A. Chubatov) for the original FBSNN class (by M. Raissi)
        - Added new parameters: s, name.
        - Added attributes: act_func, iterations, loss_n, loss_history, it_loss_history.
        - Added method: __str__.
        - Modified methods (with respect to class FBSNN):
            __init__, neural_net, initialize_NN, loss_function, fetch_minibatch, train
* ./FBSNNs_tf1_FnL_1D/FBSNNs_tf1_FnL_1D.py - extention (by A. Chubatov) for the original FBSNN class (by M. Raissi)
    class FBSNN_tf1_FnL_1D(FBSNN_tf1) - new class Forward-Backward Stochastic Neural Network for 
            -- Fully non Linear Parabolic Equation (x in 1D);
            -- system (2eq=(1+D)eq) quasi linear PDE in 1D.
        - Added abstract method: mu_tf, sigma_tf, phi_tf, g_tf, psi_tf, f_tf.       
        - Modified methods (with respect to class FBSNN_tf1):
            __init__, Df_tf, net_u, loss_function.
 
#### folder `probs,tests_tf1_FnL_1D`  - new classes and test-examples
toolkit == TF2
* Equ_tf1_FnL_1D_HJB_BS.py - class for a HJB (HamiltonJacobiBellman) equation for the Black-Sholes (BS) market
* Prob_tf1_FnL_1D_HJB_BS_TCexp.py - class for the problem with exp-TC (terminal condition) for the HJB equation 
tests for some HJB problem
* tests__FBSNNs_tf1_FnL_1D,Prob_tf1_FnL_1D_HJB_BS_TCexp.ipynb - test for the HJB problem with exp-TC
 