1. Download the training data and these three files.

2. Make sure cuda is available.

3. Run train_rnn.py

   ```shell
   $ python train_rnn.py
       				 [--seqlen SEQLEN]
       				 [--bsz BSZ]
   					 [--dropout DROPOUT]
                     	 [--nhidd NHIDD]
                        [--optim {'SGD', 'Adam', 'RMSprop', 'Adagrad'}]
                        [--lr LR]
                        [--maxiter MAXITER] 
                        [--inp INP]
                        [--outp OUTP]
   ```

   --seqlen: length of sequence fed in, which will decide how many rnn is unrolled, 32 by default.

   --bsz: batch size, 32 by default.

   --dropout: dropout rate, 0 by default.

   --nhidd: number of hidden units, 100 by default.

   --optim: optimizer, 'Adam' by default.

   --lr: set learning rate, 0.01 by default.

   --maxiter: set max iteration number, 10 by default.

   --inp: path to the input.txt, './input.txt' by default.

   --outp: save model to this folder, './' by default.

4. Run generate_music_show_heatmap.py

   ```shell
   $ python generate_music_show_heatmap.py
       				                   [--seqlen SEQLEN]
       				                   [--bsz BSZ]
   					                   [--dropout DROPOUT]
                     	                   [--nhidd NHIDD]
                                          [--id ID]
                                          [--inp INP]
                                          [--outp OUTP]
   ```

   --seqlen: length of sequence fed in, which will decide how many rnn is unrolled, 32 by default.

   --bsz: batch size, 32 by default.

   --dropout: dropout rate, 0 by default.

   --nhidd: number of hidden units, 100 by default.

   --id: which hidden node's heatmap will be shown, 0 by default.

   --inp: path to the input.txt, './input.txt' by default.

   --outp: load model from this folder, './' by default.