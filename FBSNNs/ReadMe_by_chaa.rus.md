# [Forward-Backward Stochastic Neural Networks...](https://github.com/.../FBSNNs.../) #
This is a fork of the original repository at https://github.com/maziarraissi/FBSNNs, modified by Chubatov Andrey


# ПАПКА ./FBSNNs

### 'пакет' FBSNNs
* FBSNNs_orig.py - оригинальный файл Raissi
* LICENSE - оригинальный файл Raissi LICENSE

* FBSNNs.py - минимально исправленный оригинальный файл Raissi
    class FBSNNs - минимально исправленный оригинальный файл Raissi
        -- чтобы он запускался на tensorFlow2x
        -- добавлен аттрибут self.tool = 'tensorflow 1x' # "code for TF1 running in TF2"
* оригинальные примеры Raissi: AllenCahn20D.py, BlackScholesBarenblatt100D.py, HamiltonJacobiBellman100D.py   
* plotting.py - оригинальный файл Raissi

Добавлены файлы
* Prob_BlackScholesBarenblatt.py (задача описана в отдельном файле)
* tests__FBSNNs,Prob_BlackScholesBarenblatt.ipynb (ноутбук для BlackScholesBarenblatt)

### модуль FBSNNs_plots
* FBSNNs_plots.py
    plot_results(...) - вывод траекторий в заданной области


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