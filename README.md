# Network-SDE-Inference
Codes and data for paper: Learning interpretable dynamics of stochastic complex systems from experimental data

Authors: Ting-Ting Gao, Baruch Barzel, and Gang Yan*

## GNN-based Stochastic Network Dynamics Inference Framework

This repository contains a GNN-based framework for inferring stochastic network dynamics, as well as several examples showcasing its usage. 

Example: Fig1_Lorenz_unweighted.ipynb


### To get started, we recommend checking out the Fig1_Lorenz_unweighted example. 
This example demonstrates how to use the framework to infer the dynamics of an unweighted network based on the Lorenz system. The data for all examples can be found in the Data directory, and the inference and plotting code can be found in the corresponding files (Figure1).

To use the framework on your own data, simply modify the input data format to match the examples provided. The framework is designed to work with both weighted and unweighted networks, and can handle a variety of different network topologies. The inference and plotting code can be found in the corresponding files.

Please note that the data for Fig3 and Fig4 is too large to upload at Github. Please download it from https://doi.org/10.6084/m9.figshare.24804894.v3 and modify the file path to ensure proper execution.

Please look for all datasets in Figshare: https://doi.org/10.6084/m9.figshare.24804894.v3, or contact Ting-Ting Gao (ttinggao314@gmail.com) and Gang Yan for large datasets.

## Requirements
This framework requires Python 3.8 or higher, as well as several common scientific computing libraries, as shown below. These libraries can be installed using pip or conda:

matplotlib                3.5.1

numpy                     1.22.0

pandas                    1.4.1

scikit-learn              1.1.2

scipy                     1.7.3

scs                       3.2.0

seaborn                   0.11.2 

torchvision               0.11.3

pytorch                   1.10.2 

pytorch-cluster           1.5.9  

pytorch-scatter           2.0.9    

pytorch-sparse            0.6.12 

pytorch-spline-conv       1.2.1  

## Citation
If you use this framework in your research, please cite our paper.


## Troubleshooting
*If you would like to use the code in your research and have any questions, you can reach out to us via email: eegyan@gmail.com.

Thanks to helpful discussions with Dr. Miles Cranmer on GitHub platform and Dr. Felix Dietrich with email.
