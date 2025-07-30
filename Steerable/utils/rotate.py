import torch

from Steerable.nn.utils import rotate_image

class RandomRotate(torch.utils.data.Dataset):
    def __init__(self, dataset:torch.utils.data.Dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        if targets.ndim == 2:
            degree = torch.randint(0, 360, (1,)).item()
            inputs = rotate_image(inputs, degree, order=1)
            targets = rotate_image(targets, degree, order=0)
        elif targets.ndim == 3:
            degree = torch.randint(0, 360, (3,))
            inputs = rotate_image(inputs, degree, order=1)
            targets = rotate_image(targets, degree, order=0)
        else:
            ValueError("Only 2D or 3D image data are supported.")
        return inputs, targets
            
    def __len__(self):
        return len(self.dataset)
    
    
    
class AddGaussianNoise(torch.utils.data.Dataset):
    def __init__(self, dataset:torch.utils.data.Dataset, sd:float):
        self.dataset = dataset
        self.sd = sd
        
    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        inputs = inputs + torch.randn_like(inputs) * self.sd
        return inputs, targets
            
    def __len__(self):
        return len(self.dataset)