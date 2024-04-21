# TexTile: A Differentiable Metric for Texture Tileability

Official repository of "TexTile: A Differentiable Metric for Texture Tileability" [CVPR 2024]

[Carlos Rodriguez-Pardo](https://carlosrodriguezpardo.es/), [Dan Casas](https://dancasas.github.io/), [Elena Garces](https://www.elenagarces.es/), [Jorge Lopez-Moreno](http://www.jorg3.com/)

In [CVPR](https://arxiv.org/abs/2403.12961v1), 2024.

![Textile-Teaser](https://raw.githubusercontent.com/crp94/textile/main/textile/data/teaser.png)


We introduce TexTile, a novel differentiable metric to quantify the degree upon which a texture image can be concatenated with itself without introducing repeating artifacts (i.e., the tileability). Existing methods for tileable texture synthesis focus on general texture quality, but lack explicit analysis of the intrinsic repeatability properties of a texture. In contrast, our TexTile metric effectively evaluates the tileable properties of a texture, opening the door to more informed synthesis and analysis of tileable textures. Under the hood, TexTile is formulated as a binary classifier carefully built from a large dataset of textures of different styles, semantics, regularities, and human annotations. Key to our method is a set of architectural modifications to baseline pre-train image classifiers to overcome their shortcomings at measuring tileability, along with a custom data augmentation and training regime aimed at increasing robustness and accuracy. We demonstrate that TexTile can be plugged into different state-of-the-art texture synthesis methods, including diffusion-based strategies, and generate tileable textures while keeping or even improving the overall texture quality. Furthermore, we show that TexTile can objectively evaluate any tileable texture synthesis method, whereas the current mix of existing metrics produces uncorrelated scores which heavily hinders progress in the field.


### Quick start

Run `pip install textile-metric`. The following Python code is all you need. Note that models are downloaded automatically.

```python
import textile
from textile.utils.image_utils import read_and_process_image
loss_textile = textile.Textile() 
image = read_and_process_image(YOUR_PATH)
textile_value = loss_textile(image)
```

### Image preprocessing
Please note that we provide the necessary functionality for image pre-processing on `textile/utils/image_utils.py`

### Installation
- Install PyTorch 1.0+ and torchvision fom http://pytorch.org

```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/crp94/textile
cd textile
```


## Citation

If you find this repository useful for your research, please use the following.

```
@inproceedings{Rodriguez-Pardo_2024_CVPR,
    author = {Rodriguez-Pardo, Carlos and Casas, Dan and Garces, Elena and Lopez-Moreno, Jorge},
    title = {TexTile: A Differentiable Metric for Texture Tileability},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    year = {2024}
}
```


## Acknowledgements

This repository is loosely inspired by the [LPIPS](https://github.com/richzhang/PerceptualSimilarity) repository. 