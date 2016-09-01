# nce-lm
Generative language model in Theano/Lasagne using noise contrastive estimation (NCE)

Trains a word-level language model on a text file using an LSTM or GRU. Once the model is trained a piece of text is generated.

Using NCE greatly improves efficiency for word-level language models where the large vocabulary size makes computing softmax inefficient. 

NCE is only used during the training of the model - a full softmax is used in evaluation.
