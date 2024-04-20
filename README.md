# TexTile: A Differentiable Metric for Texture Tileability

Official repository of "TexTile: A Differentiable Metric for Texture Tileability" [CVPR 2024]


### Quick start

Run `pip install textile-metric`. The following Python code is all you need.
```python
import textile
import torch
loss_textile = textile.Textile() # best forward scores
image = torch.zeros(1,3,512,512) 
textile = loss_textile(image)
```

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