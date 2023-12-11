# IHODroid
This repository is used to store the code of the IHODroid malware detection model and the address of the dataset used

1. DE-GAN: 
We utilize the data augmentation feature of Generative Adversarial Networks (GANs) to complete the imbalanced nodes in the dataset. 
In the data feature processing of the real Android malware dataset, we address the issue of dataset imbalance by employing a novel 
adversarial generative graph network model called DE-GAN to balance and expand the original dataset.

2. Ho2vec:
To address the over-smoothing phenomenon that occurs after stacking multiple layers in the original Graph Convolutional Network (GCN), 
we designed the HO2vec method for high-order hidden embedding representation. This method involves sampling K neighbors for each node, 
and it utilizes residual connections and identity mapping techniques to alleviate the over-smoothing phenomenon.

3. Baseline methodology:
The core aspects of this paper include the construction of imbalanced networks, malicious software detection algorithms, 
and heterogeneous graph embedding algorithms. The comparison and evaluation of baseline methods in this paper are based 
on the same design principles. Therefore, three state-of-the-art baseline methods have been selected for comparison:
(1) Unbalanced network embedding algorithm RECT;
(2) malware detection algorithms Scorpion and Hawk;
(3) Heterogeneous graph embedding algorithms HGT , SimpleHGN and HPN.

4. Drebin Dataset
In our testing, all the data used is sourced from the DREBIN dataset. The DREBIN dataset consists of a total of 123,453
samples of Android applications, including 5,560 malicious samples, and encompasses up to 545,333 behavioral features.
