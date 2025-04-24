# ПАПКА ./FBSNNs_tf1
   
### пакет FBSNNs_tf1
* FBSNNs.py - минимально исправленный оригинальный файл Raissi  
* FBSNNs_plots.py - пакет вывода рисунков с траекториями
    plot_results(...) - вывод траекторий  
* FBSNNs_tf1.py
    class FBSNN_tf1(FBSNN) - (fork) модификация (by Chubatov Andrey) для оригинального класса FBSNN (by Raissi) 
        - Добавлены новые параметры: s, name 
        - Добавлены атрибуты: act_func, iterations, loss_n, loss_history, it_loss_history
        - Добавлен метод  __str__()
        - Модифицированы методы (по сравнению с FBSNN): 
            neural_net, initialize_NN, loss_function, fetch_minibatch, train
* FBSNNs_torch.py - (fork) модификация (by Chubatov Andrey) для кода https://github.com/Shine119/FBSNNs_pytorch
    
##### папка probs,tests_tf1 - примеры Raissi
Prob_tf1_BSB_Raissi.py - задача для уравнения BSB (BlackScholesBarenblatt)
tests__FBSNNs_tf1,Prob_tf1_BSB_Raissi.ipynb - ноутбук для уравнения BSB (BlackScholesBarenblatt)
Prob_tf1_HJB_Raissi.py - задача для уравнения HJB (HamiltonJacobiBellman) 
tests__FBSNNs_tf1,Prob_tf1_HJB_Raissi.ipynb - ноутбук для уравнения HJB (HamiltonJacobiBellman)

        
        
        