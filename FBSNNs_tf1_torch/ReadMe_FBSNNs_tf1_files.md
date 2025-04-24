# FOLDER ./FBSNNs_tf1_torch

### package FBSNNs_tf1_torch - modification (by A. Chubatov) for original FBSNN framework (by M. Raissi)
        (running on TensorFlow2 (TF2) + pyTorch) 
* FBSNNs.py - minimally modified original file (by M. Raissi)
* FBSNNs_plots.py - package for plot of trajectories
    plot_results(...) - output of trajectories
* FBSNNs_tf1.py - modification (by A. Chubatov) for original framework FBSNNs (by M. Raissi)
    class FBSNN_tf1(FBSNN) - (fork) modification (by A. Chubatov) for the original FBSNN class (by M. Raissi)
        - Added new parameters: s, name
        - Added attributes: act_func, iterations, loss_n, loss_history, it_loss_history
        - Added method __str__()
        - Modified methods (with respect to class FBSNN):
            neural_net, initialize_NN, loss_function, fetch_minibatch, train
* FBSNNs_torch.py - modification (by A. Chubatov) for the code https://github.com/Shine119/FBSNNs_pytorch

#### folder `probs,tests_tf1,torch` - Raissi examples
toolkit == TF2
* Prob_tf1_BSB_Raissi.py - class for BSB equation (BlackScholesBarenblatt) problem
* Prob_tf1_HJB_Raissi.py - class for HJB equation (HamiltonJacobiBellman) problem
toolkit == pyTorch
* Prob_torch_BSB_Raissi.py - class for BSB equation (BlackScholesBarenblatt) problem
* Prob_torch_HJB_Raissi.py - class for HJB equation (HamiltonJacobiBellman) problem
test for BSB equation
* tests__FBSNNs_tf1,Prob_tf1_BSB_Raissi.ipynb   
* tests__FBSNNs_torch,Prob_torch_BSB_Raissi.ipynb
test for HJB equation
* tests__FBSNNs_tf1,Prob_tf1_HJB_Raissi.ipynb
* tests__FBSNNs_torch,Prob_torch_HJB_Raissi.ipynb