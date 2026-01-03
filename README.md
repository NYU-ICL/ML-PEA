<div align="center">

<h1>ML-PEA: Machine Learning-Based Perceptual Algorithms for Display Power Optimization</h1>

</div>

## Quick Startup

To train a model, run the following command:

`python train.py --w_vgg 0.5 --w_ssim 5 --w_power 50 --method MULT --dataset div2k`

where `w_vgg`, `w_ssim`, and `w_power` are weights on the VGG, SSIM, and power losses, respectively. 
`--method MULT` sets the dimming map modulation to multiplicative, `I_new = I * dimming_map`.
Specify the training dataset (e.g. `div2k`).
Place dataset images in `div2k/train/*.png` and `div2k/test/*.png`.

## Checkpoints

Saved checkpoints are found at our [Google Drive link](https://drive.google.com/drive/folders/1Fpr_0HNpvti6gM-gYtaqBM0uRxJuEi-T?usp=share_link).