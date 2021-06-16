# Binaural Speech Enhancement Based on Deep Attention Layers

<p align="justify">
Deep learning based speech enhancement methods have been proven to be successful at removing unwanted interfering signals from target ones. Generally, these algorithms focus on removing the noisy elements from single-channel signals. However, humans listen with two ears, allowing the central auditory pathways to merge the auditory messages sent by the two cochleae into auditory objects. This is known as binaural hearing and constitutes a crucial aspect of auditory perception by means of separating target signals from noise and competing sources. 
</p>

<p align="justify">
This repository contains our submission for the 1st Clarity Enhancement Challenge. Here, a TensorFlow [1] implementation of the model can be found. The model is based on two conv-TasNets [2] and combines the information contained on each of the listening sides to provide the model with potential binaural cues. This information is combined through intermediate layers that we refer to as "attention layers", inspired by the classical attention layers used in sequence to sequence modeling [3]. The implemented model is fed with stereo signals and outputs its de-noised version. 
</p>

# Requirements
See [Requirements.txt](https://github.com/APGDHZ/BinAttSE/blob/main/requirements.txt)

# References
<p align="justify">
[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[2] Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation. arXiv preprint arXiv:1809.07454, 2018.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H.Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017.
</p>
