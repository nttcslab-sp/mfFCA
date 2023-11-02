## Multi-frame Full-rank Spatial Covariance Analysis for Underdetermined Blind Source Separation and Dereverberation

This repository contains sample codes for [1]. There are two main programs, main_synthetic.ipynb and main_speech.py.

### Main programs

#### main_synthetic.ipynb

In this sample code, synthetic mixtures are made, and then they are separated. The situation is simple, and therefore the code execution is not computationally demanding, especially compared to the following one. 

#### main_speech.py

This sample code separates mixtures of 3 speeches mixed in a real room environment with a reverberation time of 450 ms. The code execution is computationally demanding. Therefore, it is recommended to run it with cupy, which accelerates the computation with NVIDIA CUDA.

### Sound examples

[Sound examples](https://www.kecl.ntt.co.jp/icl/signal/sawada/demo/mffca/index.html)

### Reference

[1]: Hiroshi Sawada, Rintaro Ikeshita, Keisuke Kinoshita, and Tomohiro Nakatani, "Multi-frame Full-rank Spatial Covariance Analysis for Underdetermined Blind Source Separation and Dereverberation," IEEE/ACM Trans. Audio, Speech, and Language Processing, vol. 31, pp. 3589-3602, 2023, [doi: 10.1109/TASLP.2023.3313446](https://ieeexplore.ieee.org/document/10244107). ([PDF](https://www.kecl.ntt.co.jp/icl/signal/sawada/mypaper/IEEEtaslp2023sawada.pdf))

```
@ARTICLE{sawada2023multi,
  author={Sawada, Hiroshi and Ikeshita, Rintaro and Kinoshita, Keisuke and Nakatani, Tomohiro},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Multi-Frame Full-Rank Spatial Covariance Analysis for Underdetermined Blind Source Separation and Dereverberation}, 
  year={2023},
  volume={31},
  number={},
  pages={3589-3602},
  doi={10.1109/TASLP.2023.3313446}
}
```
