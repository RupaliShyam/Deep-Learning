# Deep-Learning

This repository contains Deep Learning models for various applications:
- Basic neural network

- Convolutional Neural Networks for category and domain prediction using AlexNet and its variations such as:
  1. Larger kernel sizes - resulting in smaller output
  2. Smaller number of filters - practical for smaller number of categories or domains.
  3. Pooling stratergies - using AvgPool2d instead of MaxPool2d
  4. Atrous convolutions - the receptive fields are enlarged by inserting holes into the convolutional kernels.
  
- Recurrent Neural Networks for the following applications:
  1. Sentiment Analysis 
  2. Sentence generation
  3. Visualization of LSTM gates
  The RNN modules use GRU, LSTM and variations of LSTM cells such as PeepholedLSTM and CoupledLSTM.
