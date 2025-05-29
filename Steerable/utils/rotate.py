import torch
from scipy.ndimage.interpolation import rotate

class RotateImage:
    def __init__(self, order=1):
        self.order = order
    
    def rotate_image_slice(self, image_slice, degree):
        if image_slice.ndim == 2:
            image_slice = torch.from_numpy(rotate(image_slice, degree, (1,0), reshape=False, order=self.order))
        if image_slice.ndim == 3:
            image_slice = torch.from_numpy(rotate(image_slice, degree[0], (1,0), reshape=False, order=self.order))
            image_slice = torch.from_numpy(rotate(image_slice, degree[1], (0,2), reshape=False, order=self.order))
            image_slice = torch.from_numpy(rotate(image_slice, degree[2], (1,0), reshape=False, order=self.order))
        return image_slice
        
        
    def __call__(self, image, degree=None):
        assert image.ndim in [3,4], f'Only accepted 3D (C,H,W) or 4D (C,H,W,D) tensors but got {image.ndim}D tensor.'
        degree = torch.tensor(degree) if degree else torch.randint(0, 360, (3,))
            
        return torch.vstack([self.rotate_image_slice(image[i], degree) for i in range(image.shape[0])]).view(*image.shape)
            