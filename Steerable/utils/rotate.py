import torch

from Steerable.nn.Steerable2d.utils import rotate_2D_image
from Steerable.nn.Steerable3d.utils import rotate_3D_image

class RandomRotate:
    def __init__(self, dataset:torch.utils.data.Dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        if targets.ndim == 2:
            degree = torch.randint(0, 360, (1,)).item()
            inputs = rotate_2D_image(inputs, degree, order=1)
            targets = rotate_2D_image(targets, degree, order=0)
        elif targets.ndim == 3:
            degree = torch.randint(0, 360, (3,))
            inputs = rotate_3D_image(inputs, degree, order=1)
            targets = rotate_3D_image(targets, degree, order=0)
        else:
            ValueError("Only 2D or 3D image data are supported.")
        return inputs, targets
            
    def __len__(self):
        return len(self.dataset)
