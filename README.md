# Reflection Separation using a Pair of Unpolarized and Polarized Images
Youwei Lyu<sup>\*</sup>, Zhaopeng Cui<sup>\*</sup>, Si Li, Marc Pollefeys, Boxin Shi
<br>
(<sup>\*</sup>equal contribution)
<br>
<p align="center">
	<img src='/data/image/pipeline.png' height="260">
</p>


Abstract
--------------------------
When we take photos through glass windows or doors, the transmitted background scene is often blended with undesirable reflection. Separating two layers apart to enhance the image quality is of vital importance for both human and machine perception. In this paper, we propose to exploit physical constraints from a pair of unpolarized and polarized images to separate reflection and transmission layers. Due to the simplified capturing setup, the system becomes more underdetermined compared with existing polarization based solutions that take three or more images as input. We propose to solve semireflector orientation estimation first to make the physical image formation well-posed and then learn to reliably separate two layers using a refinement network with gradient loss. Quantitative and qualitative experimental results show our approach performs favorably over existing polarization and single image based solutions.

Dependence
--------------------------
- python 3.6
- pytorch 1.1.0
- cv2
- lmdb
- tensorpack

Contents
--------------------------

 - **Trained model and real data for test**

The pretrained model for test on real data can be download via [Google Drive](https://drive.google.com/open?id=1CTry_JRlVzxn65EJbAvdkOj5ij3xU-DU)
 
 