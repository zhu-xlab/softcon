# SoftCon
Multi-label Guided Soft Contrastive Learning for Efficient Earth Observation Pretraining

## Pretrained models

| Model | Modality | BigEarthNet-10% linear | EuroSAT linear | download |
| :---: | :---: | :---: | :---: | :---: |
| RN50 | MS | 84.8 | 98.6 | [backbone](https://huggingface.co/wangyi111/softcon/resolve/main/B13_rn50_softcon.pth) |
| ViT-S/14 | MS | 85.0 | 97.1 | [backbone](https://huggingface.co/wangyi111/softcon/resolve/main/B13_vits14_softcon.pth) |
| ViT-B/14 | MS | 86.8 | 98.0 | [backbone](https://huggingface.co/wangyi111/softcon/resolve/main/B13_vitb14_softcon.pth) |
| RN50 | SAR | 78.9 | 87.1 | [backbone](https://huggingface.co/wangyi111/softcon/resolve/main/B2_rn50_softcon.pth) |
| ViT-S/14 | SAR | 80.3 | 87.1 | [backbone](https://huggingface.co/wangyi111/softcon/resolve/main/B2_vits14_softcon.pth) |
| ViT-B/14 | SAR | 81.4 | 89.1 | [backbone](https://huggingface.co/wangyi111/softcon/resolve/main/B2_vitb14_softcon.pth) |


## Usage

Clone this repository:
```bash
git clone https://github.com/zhu-xlab/softcon
cd softcon
```

Download the pretrained weights and put them in the `./pretrained` directory.
```bash
wget https://huggingface.co/wangyi111/softcon/resolve/main/B13_rn50_softcon.pth -P ./pretrained
...
```

Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/locally/), e.g.,
```bash
pip3 install torch torchvision # CUDA 12.1
```

Open a Python interpreter and run:
```python
import torch
from torchvision.models import resnet50

# create RN50 model for multispectral
model_r50 = resnet50(pretrained=False)
model_r50.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model_r50.fc = torch.nn.Identity()

# load pretrained weights
ckpt_r50 = torch.load('./pretrained/B13_rn50_softcon.pth')
model_r50.load_state_dict(ckpt_r50)

# encode one image
model_r50.eval()
img = torch.randn(1, 13, 224, 224)
with torch.no_grad():
    out = model_r50(img)
print(out.shape) # torch.Size([1, 2048])
```

Similarly, for ViT backcbones run:

```python
import torch
from models.dinov2 import vision_transformer as dinov2_vits

# create ViT-S/14 model for SAR
model_vits14 = dinov2_vits.__dict__['vit_small'](
    img_size=224,
    patch_size=14,
    in_chans=2,
    block_chunks=0,
    init_values=1e-5,
    num_register_tokens=0,
)

# load pretrained weights
ckpt_vits14 = torch.load('./pretrained/B2_vits14_softcon.pth')
model_vits14.load_state_dict(ckpt_vits14)

# encode one image
model_vits14.eval()
img = torch.randn(1, 2, 224, 224)
with torch.no_grad():
    out = model_vits14(img)
print(out.shape) # torch.Size([1, 384])
```

## Data normalization
It may depend on the downstream task for the most suitable data preprocessing. As a general case, we recommend using the per-channel mean/std of the [SSL4EO-S12](https://arxiv.org/abs/2211.07044) dataset (our pretraining dataset) or the target dataset for input normalization. Our normalization function is as follows:
```python
def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
```


## Citation