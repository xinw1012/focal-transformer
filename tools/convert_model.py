"""
Convert pretrained models to the defition of FocalNet
"""
import torch

model_paths = {
    'focalnet_tiny_3333': '/datadrive/azureblobs/model_storage2/projects/focalv3/amlt-results/7354148258.48326-4bb8c2e2-6002-4e5c-a1a1-151bd5f1be3b/ckpt_epoch_297.pth'
}
for key, path in model_paths.items():
    print(key)
    ckpt = torch.load(path)
    import pdb; pdb.set_trace()
