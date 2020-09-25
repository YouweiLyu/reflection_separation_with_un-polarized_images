# Reflection Separation using a Pair of Unpolarized and Polarized Images
By Youwei Lyu<sup>\*</sup>, [Zhaopeng Cui<sup>\*</sup>](https://zhpcui.github.io/), Si Li, [Marc Pollefeys](https://people.inf.ethz.ch/pomarc/), [Boxin Shi](http://ci.idm.pku.edu.cn/)
<br>
(<sup>\*</sup>equal contribution)
<br>
<p align="center">
	<img src='/data/image/pipeline.png' height="260">
</p>


Abstract
--------------------------
When we take photos through glass windows or doors, the transmitted background scene is often blended with undesirable reflection. Separating two layers apart to enhance the image quality is of vital importance for both human and machine perception. In this paper, we propose to exploit physical constraints from a pair of unpolarized and polarized images to separate reflection and transmission layers. Due to the simplified capturing setup, the system becomes more underdetermined compared with existing polarization based solutions that take three or more images as input. We propose to solve semireflector orientation estimation first to make the physical image formation well-posed and then learn to reliably separate two layers using a refinement network with gradient loss. Quantitative and qualitative experimental results show our approach performs favorably over existing polarization and single image based solutions.

Setup
--------------------------
This code is based on Pytorch and is tested on Linux Distributions (Ubuntu 16.04).
### Environment
- python 3.6
- pytorch 1.1.0
- cv2
- matplotlib

We recommend Anaconda and provide a conda environment setup file including the above dependencies. Please create the environment ```polar``` by running:
```
	conda env create -f env.yml
```

Pretrained models and dataset
--------------------------

 - Download the pretrained model and dataset for test from [Google Drive](https://drive.google.com/open?id=1CTry_JRlVzxn65EJbAvdkOj5ij3xU-DU)
 - Unpack the ```.zip``` file, and then move the folders containing pretrained models and test images to the ```data``` directory.

Inference
--------------------------

Run
```
	python test_sync.py
```
to get the separation results.

Citation
--------------------------
If you find this work helpful to your research, please cite:
```
@incollection{lyu2019_polarRS,
title = {Reflection Separation using a Pair of Unpolarized and Polarized Images},
author = {Lyu, Youwei and Cui, Zhaopeng and Li, Si and Pollefeys, Marc and Shi, Boxin},
booktitle = {Advances in Neural Information Processing Systems 32 (NeurIPS)},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {14559--14569},
year = {2019},
publisher = {Curran Associates, Inc.},
}
```
 
 
