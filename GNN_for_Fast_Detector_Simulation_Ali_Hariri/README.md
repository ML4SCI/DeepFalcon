# FALCON:Graph Generative models for particle collision events reconstruction

# Introduction 
The aim of the FALCON project is to develop Fast Simulation tools to be used in High Energy Physics applications. Previous simulation techniques include Monte Carlo Simulations 
that use probabilistic models to learn probabiltiy densities of collision data. Future upgrades taking place at CERN's Large Hadron Collider will allow it to operate at higher 
luminosities, meaning that more data will be collected and processed. As a result, more sophisticated algorithms are required to process detector data in a timely efficient manner. 

# Graph Generative Models
Recently, deep generative models have shown notable potential in computer vision applications such as image generation andsegmentation. Their ability to model complex distributions
in addition to dealing with missingdata and interpolation makes them an interesting candidate for detector physics applications.The most widely used generative models include 
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

![Sample Reconstruction](https://github.com/ahariri13/FALCON/blob/master/rec1.PNG)

![Sample Reconstruction2](https://github.com/ahariri13/FALCON/blob/master/rec2.PNG)

# Prerequisites 

-PyTorch 1.6.0 and CUDA 10.1
-Pytorch Geometric 

# References 
CMS Open Data Release: http://opendata.cern.ch
