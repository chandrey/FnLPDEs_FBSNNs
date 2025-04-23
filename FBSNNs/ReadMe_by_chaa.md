# [Forward-Backward Stochastic Neural Networks...](https://github.com/.../FBSNNs.../) #
This is a fork of the original repository at https://github.com/maziarraissi/FBSNNs, modified by Chubatov Andrey


# FOLDER ./FBSNNs

### 'package' FBSNNs
* FBSNNs_orig.py - original Raissi file
* LICENSE - original Raissi file LICENSE

* FBSNNs.py - minimally modified of original Raissi's file
    class FBSNNs - minimally modified of original Raissi's class
        - so that it runs on tensorFlow2x
            -- added row 
                `tf.compat.v1.disable_eager_execution()`
            -- `tf.placeholder` <-replace-> `tf.compat.v1.placeholder`
        - added attribute self.tool = 'tensorflow 1x' # "code for TF1 running in TF2"
* original Raissi examples: AllenCahn20D.py, BlackScholesBarenblatt100D.py, HamiltonJacobiBellman100D.py
* plotting.py - original Raissi file

Added files
* Prob_BlackScholesBarenblatt.py (the BlackScholesBarenblatt problem described in a separate file)
* tests__FBSNNs,Prob_BlackScholesBarenblatt.ipynb (notebook for BlackScholesBarenblatt)

### module FBSNNs_plots
* FBSNNs_plots.py
    plot_results(...) - output of trajectories 



# [Forward-Backward Stochastic Neural Networks](https://maziarraissi.github.io/FBSNNs/)

Classical numerical methods for solving partial differential equations suffer from the curse of dimensionality mainly due to their reliance on meticulously generated spatio-temporal grids. Inspired by modern deep learning based techniques for solving forward and inverse problems associated with partial differential equations, we circumvent the tyranny of numerical discretization by devising an algorithm that is scalable to high-dimensions. In particular, we approximate the unknown solution by a deep neural network which essentially enables us to benefit from the merits of automatic differentiation. To train the aforementioned neural network we leverage the well-known connection between high-dimensional partial differential equations and forward-backward stochastic differential equations. In fact, independent realizations of a standard Brownian motion will act as training data. We test the effectiveness of our approach for a couple of benchmark problems spanning a number of scientific domains including Black-Scholes-Barenblatt and Hamilton-Jacobi-Bellman equations, both in 100-dimensions. 

For more information, please refer to the following: (https://maziarraissi.github.io/FBSNNs/)

  - Raissi, Maziar. "[Forward-Backward Stochastic Neural Networks: Deep Learning of High-dimensional Partial Differential Equations](https://arxiv.org/abs/1804.07010)." arXiv preprint arXiv:1804.07010 (2018).
  
  - Video: [Forward-Backward Stochastic Neural Networks](https://youtu.be/-Pu_ZTJsMyA)
  
  - Slides: [Forward-Backward Stochastic Neural Networks](https://github.com/maziarraissi/FBSNNs/blob/master/docs/FBSNNs.pdf)

## Citation

    @article{raissi2018forward,
      title={Forward-Backward Stochastic Neural Networks: Deep Learning of High-dimensional Partial Differential Equations},
      author={Raissi, Maziar},
      journal={arXiv preprint arXiv:1804.07010},
      year={2018}
    }