import torch
from math import floor, ceil

class PatchifyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, kernel_size, stride = 1):
        self.dataset = dataset
        self.kernel_size = kernel_size
        self.patchify = Patchify(kernel_size, stride)
               
    def __getitem__(self, index):
        image, target = self.dataset[index]
        return self.patchify(image), self.patchify(target.unsqueeze(0)).squeeze(1)
    
    def __len__(self):
        return len(self.dataset)
    
from math import ceil, floor
class Patchify:
    def __init__(self, kernel_size, stride=1, transform=None) -> None:
        self.kernel_size = kernel_size
        self.dimension = len(self.kernel_size)
        self.stride = stride if type(stride) is tuple else tuple([stride]*self.dimension)      
        self.transform = transform
        
        if self.dimension != len(self.stride):
            raise ValueError(f'kernel_size ({self.dimension}) should have same number of dimension as padding ({len(self.stride)}).') 
        
        pass 
    
    def __call__(self, image):
        return _ImagePatches(image, self.kernel_size, self.stride, self.transform)
    
    def get_padding(self, image_shape):
        padding = []
        for i in range(self.dimension):
            p = ((self.kernel_size[i] - image_shape[i])%self.stride[i])/2.0
            padding.append([ceil(p), floor(p)])
        return torch.tensor(padding)
    
    def get_num_patches_per_dim(self, image_shape):
        padding = self.get_padding(image_shape)
        return torch.tensor([(image_shape[i] + torch.sum(padding[i]).item() - self.kernel_size[i])//self.stride[i] + 1 for i in range(self.dimension)])
    
    def get_num_patches(self, image_shape):
        return torch.prod(self.get_num_patches_per_dim(image_shape))
    
class _ImagePatches(Patchify):
    def __init__(self, image, kernel_size, stride=1, transform=None) -> None:
        self.kernel_size = kernel_size
        self.dimension = len(self.kernel_size)
        self.stride = stride
        self.transform = transform
        self.image = torch.nn.functional.pad(image, torch.flip(self.get_padding(image.shape[-self.dimension:]), [0]).flatten().tolist())

        self.num_patches_per_dim = self.get_num_patches_per_dim(self.image.shape[-self.dimension:])
        self.num_patches = self.get_num_patches(self.image.shape[-self.dimension:])
        start = torch.cartesian_prod(*[torch.arange(self.num_patches_per_dim[i]) * self.stride[i]  for i in range(self.dimension)])
        self._patch_positions = [tuple(slice(start[i,d], start[i,d]+self.kernel_size[d]) for d in range(self.dimension)) for i in range(len(start))]

        self._idx = 0            # cursor for iteration
        
    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= self.num_patches:
            raise StopIteration

        index = self._idx
        patch = self.image[(Ellipsis,) + self._patch_positions[index]]
        if self.transform:
            patch = self.transform(patch)
        self._idx += 1

        return patch
    
    def __len__(self):
        return self.num_patches
    
    

class Reconstruct(Patchify):
    def __init__(self, kernel_size, image_shape, stride=1, sigma=1.0):
        self.kernel_size = kernel_size
        self.dimension = len(self.kernel_size)
        self.image_shape = image_shape
        self.sigma = sigma
        self.stride = stride if type(stride) is tuple else tuple([stride]*self.dimension)
        self.num_patches_per_dim = self.get_num_patches_per_dim(self.image_shape)
        self.num_patches = self.get_num_patches(self.image_shape)
        
        if self.dimension != len(self.stride):
            raise ValueError(f'kernel_size ({len(self.dimension)}) should have same number of dimension as padding ({len(self.stride)}).')
        
        if self.dimension != len(self.image_shape):
            raise ValueError(f'shape of kernel ({self.dimension}) and image shape ({self.image_shape}) should match.')
        
        self.padding = self.get_padding(self.image_shape)
        self.pad_image_shape = tuple(torch.sum(self.padding[i]).item() + self.image_shape[i] for i in range(self.dimension))
        start = torch.cartesian_prod(*[torch.arange(self.num_patches_per_dim[i]) * self.stride[i]  for i in range(self.dimension)])
        self.patch_positions = [tuple(slice(start[i,d], start[i,d]+self.kernel_size[d]) for d in range(self.dimension)) for i in range(len(start))]
        self.kernel = self._gaussian_kernel()
        self.weights = self._get_weights(self.kernel)
        
    def __call__(self, patches):
        if len(patches) != self.weights.shape[0]:
            raise ValueError(f'Number of pacthes should be {self.weights.shape[0]}, but {len(patches)} were given.')
        
        patch = next(patches)
        patch_shape = patch.shape 
        self.weights = self.weights.to(patch.device)
        output = torch.zeros(*patch_shape[:-self.dimension], *self.pad_image_shape, device=patch.device)
        
        for i, (position, patch) in enumerate(zip(self.patch_positions, patches)):
            output[(Ellipsis,)+position] += self.weights[i]*patch
        
        slices = tuple(slice(self.padding[i,0], self.padding[i,0]+self.image_shape[i]) for i in range(self.dimension))
        output = output[(Ellipsis, )+slices]

        return output

    def _gaussian_kernel(self):
        coords = [torch.arange(0, k, 1) for k in self.kernel_size]
        mesh = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
        c = torch.tensor([(k-1)/2 for k in self.kernel_size])
        
        return -torch.sum((mesh-c)**2/(2*self.sigma**2), dim=-1)
    
    def _get_weights(self, kernel):
        log_denominator = torch.empty(*self.pad_image_shape)
        log_denominator.fill_(float('-inf'))
        for pos in self.patch_positions:
            log_denominator[pos] = torch.logaddexp(log_denominator[pos], kernel)
        weights  = torch.empty(self.num_patches,*self.kernel_size)
        for i, pos in enumerate(self.patch_positions):
            weights[i] = kernel - log_denominator[pos]
        return torch.exp(weights)
    