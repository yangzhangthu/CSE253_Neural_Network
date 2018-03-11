1. Download the data and save them in './data/'

2. open command, run 

   ```shell
   python neural_network.py
   ```

   ```shell
   usage: python neural_network.py 
       				 [--layer {1,2}]
       				 [--nnode NNODE]
   					 [--shuffle {True,False}]
                     [--activation {lr,sigmoid,relu}]
                     [--wnorm {True,False}]
                     [--optim {nesterov,momentum,sgd}]
                     [--vanilla {True,False}]
                     [--lr LR]
                     [--maxiter MAXITER]        
   ```

   --layer: choose 1 layer or 2 layers network, 1 by default.

   --nnode: the number of hidden node, 64 by default.

   --shuffle: shuffle training data after each epoch, False by default.

   --activation: choose activation function, lr by default.

   --wnorm: weight normalization, False by default.

   --optim: choose optimizer for 2 layers network, nesterov by default.

   --vanilla: use sgd for 1 layer network, True by default. If False, use momentum.

   --lr: set learning rate, 0.0001 by default.

   --maxiter: set max iteration number, 5 by default.