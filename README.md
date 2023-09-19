## Multi-frame Full-rank Spatial Covariance Analysis for Underdetermined Blind Source Separation and Dereverberation

This repository contains sample codes for [1]. There are two main programs, main_synthetic.ipynb and main_speech.py.

### main_synthetic.ipynb

In this sample code, synthetic mixtures are made, and then they are separated. The situation is simple, and therefore the code execution is not computationally demanding, especially compared to the following one. 

### main_speech.py

This sample code separates mixtures of 3 speeches mixed in a real room environment with a reverberation time of 450 ms. The code execution is computationally demanding. Therefore, it is recommended to run it with cupy, which accelerates the computation with NVIDIA CUDA.

### Reference

[1]: Hiroshi Sawada, Rintaro Ikeshita, Keisuke Kinoshita, and Tomohiro Nakatani, "Multi-frame Full-rank Spatial Covariance Analysis for Underdetermined Blind Source Separation and Dereverberation," IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.
