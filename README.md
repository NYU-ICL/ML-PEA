<div align="center">

<h1>ML-PEA: Machine Learning-Based Perceptual Algorithms for Display Power Optimization</h1>

[**Kenneth Chen**](https://kenchen10.github.io/)<sup>1</sup>,
[**Nathan Matsuda**](https://www.nathanmatsuda.com)<sup>2</sup>,
[**Thomas Wan**](https://www.linkedin.com/in/thomas-cc-wan)<sup>2</sup>,
[**Ajit Ninan**](https://www.linkedin.com/in/ajitninan)<sup>2</sup>,
[**Alexandre Chapiro**](https://achapiro.github.io/)<sup>2</sup>
[**Qi Sun**](https://qisun.me/)<sup>1</sup>

<sup>1</sup>
<a href="https://www.immersivecomputinglab.org/research/"><img width="30%" src="assets/nyulogo.png" style="vertical-align: top;" alt="NYU logo"></a>
&emsp;
<sup>2</sup>
<a href=""><img width="22%" src="assets/meta.png" style="vertical-align: top;" alt="Intel logo"></a>

</div>

<div style="width: 90%; margin: 0 auto;">
<img src="assets/teaser.png" width="100%" />
  <sub>
  Our pipeline generates images which consume less power than the original when shown on a display, while minimizing perceptual impact. Here, we show an example of an image generated with our technique compared to the reference and its uniformly dimmed version. The corresponding dimming maps are shown in the insets, with the multiplicative scaling factor presented in the color bar on the right. Note that both the uniformly dimmed image and the image generated with our technique in this figure consume the same amount of display power: 52.1% of the reference.
  </sub>
</div>

## Abstract 
Image processing techniques can be used to modulate the pixel intensities of an image to reduce the power consumption of the display device. A simple example of this consists of uniformly dimming the entire image. Such algorithms should strive to minimize the impact on image quality while maximizing power savings. Techniques based on heuristics or human perception have been proposed, both for traditional flat panel displays and modern display modalities such as virtual and augmented reality (VR/AR). In this paper, we focus on developing and evaluating display power-saving techniques that use machine learning (ML). This pipeline was validated via quantitative analysis using metrics and through a subjective study. Our results show that participants prefer our technique over a uniform dimming baseline for high target power saving conditions. In the future, this work should serve as a template and baseline for future applications of deep learning for display power optimization.

## Quick Startup

To train a model, run the following command:

`python train.py --w_vgg 0.5 --w_ssim 5 --w_power 50 --method MULT --dataset div2k`

where `w_vgg`, `w_ssim`, and `w_power` are weights on the VGG, SSIM, and power losses, respectively. 
`--method MULT` sets the dimming map modulation to multiplicative, `I_new = I * dimming_map`.
Specify the training dataset (e.g. `div2k`).
Place dataset images in `div2k/train/*.png` and `div2k/test/*.png`.

## Checkpoints

Saved checkpoints are found at our [Google Drive link](https://drive.google.com/drive/folders/1Fpr_0HNpvti6gM-gYtaqBM0uRxJuEi-T?usp=share_link).