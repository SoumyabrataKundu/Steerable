import torch
from scipy.ndimage.interpolation import rotate

class RotateImage:
    def __init__(self, dimension, order=1):
        self.order = order
        assert dimension in [2,3], "'dimension' must be either 2 or 3."
        self.dimension = dimension

    def rotate_image_slice(self, image_slice, degree):
        image_slice = torch.from_numpy(rotate(image_slice, degree[0], (1,0), reshape=False, order=self.order))
        if self.dimension == 3:
            image_slice = torch.from_numpy(rotate(image_slice, degree[1], (0,2), reshape=False, order=self.order))
            image_slice = torch.from_numpy(rotate(image_slice, degree[2], (1,0), reshape=False, order=self.order))
        return image_slice

    def __call__(self, image, degree=None):
        assert image.ndim >= self.dimension, f'Input should have at least {self.dimension} dimensions but got only {image.ndim} dimensions.'
        degree = degree if degree else torch.randint(0, 360, (3,)) 
        image_shape = image.shape
        image = image.reshape(-1, *image_shape[-self.dimension:])
        return torch.vstack([self.rotate_image_slice(image[i], degree) for i in range(image.shape[0])]).view(*image_shape)
