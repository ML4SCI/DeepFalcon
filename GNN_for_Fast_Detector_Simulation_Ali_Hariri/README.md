# FALCON:Graph Generative models for particle collision events reconstruction

# Introduction 
The aim of the FALCON project is to develop Fast Simulation tools to be used in High Energy Physics applications. Previous simulation techniques include Monte Carlo Simulations 
that use probabilistic models to learn probabiltiy densities of collision data. Future upgrades taking place at CERN's Large Hadron Collider will allow it to operate at higher 
luminosities, meaning that more data will be collected and processed. As a result, more sophisticated algorithms are required to process detector data in a timely efficient manner. 

# Graph Generative Models
Recently, deep generative models have shown notable potential in computer vision applications such as image generation andsegmentation. Their ability to model complex distributions
in addition to dealing with missing data and interpolation makes them an interesting candidate for detector physics applications.The most widely used generative models include 
Generative Adversarial Networks and Variational Auto-Encoders. 

While VAEs have shown significant potential with regards to generative models that operate
on images, their applications on irregular data remain under-explored. Recent work in this
field shed light on graph generative models. In contrast to regular VAEs whose input consists
of a regularly-structured input X, Graph-VAEs take two inputs X and A. The former is a NxC
feature matrix where N is the number of nodes and C is the feature dimensions. On the other
hand, A is an NxN adjacency matrix describing the topology of a graph. To transfer the
convolution operations to graph structures, neural networks assigned to encoders/decoders
are replaced by graph convolutional networks inspired from [9]

# Results

![Sample Reconstruction](https://github.com/ahariri13/FALCON/blob/master/pract_img.png)

We use the k-nearest neighbour algorithm to connect each node representing a reconstructed hit in a detector cell to k neighbouring cells closest in Euclidean distance given by $\sqrt{(x-x_i)^2+(y-y_i)^2}$. Therefore, the reconstructed detector hits are mapped into nodes that contain 3 features: x and y locations and the reconstructed energy of the hit. We next learn the high dimensional representation of the nodes using a Graph Variational Autoencoder (GVAE), a geometric deep learning architecture that learns the graph embeddings of the non-Euclidean data in a latent space

![Model Architecture](https://github.com/ahariri13/FALCON/blob/master/model.PNG)

We train the GVAE model on a cluster using Volta V100 GPUs with 16 GB of RAM. Following profiling and code optimization for enhanced CPU performance, the training is scaled on multiple GPUs using the Horovod library. We compare the results to the training on a single GPU with a batch size of 32 for 100 iterations, i.e a total of 3200 graph samplesWe notice an increase in the performance with an increase in the GPU devices used as shown in the tabe below.

![Scaling performance](https://github.com/ahariri13/FALCON/blob/master/tablescaling.PNG)

# Prerequisites 

-PyTorch 1.6.0 and CUDA 10.1
-Pytorch Geometric 

# References 
CMS Open Data Release: http://opendata.cern.ch

# Contact
- Please contact [Ali Hariri](https://github.com/ahariri13) for questions/comments related to this repository. 
- This project was mentored by [Prof. Sergei V. Gleyzer](http://sergeigleyzer.com/).

# About me

Ali Hariri has recently finished his Master's of Engineering at the American University of Beirut and was part of the 2021 Google Summer of Code. 


