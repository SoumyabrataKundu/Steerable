import torch
import torch.nn.functional as F
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
    
class Patchify:
    def __init__(self, kernel_size, stride=1):
        self.kernel_size = kernel_size
        self.dimension = len(self.kernel_size)
        self.stride = stride if type(stride) is tuple else tuple([stride]*self.dimension)      
        
        if self.dimension != len(self.stride):
            raise ValueError(f'kernel_size ({self.dimension}) should have same number of dimension as padding ({len(self.stride)}).') 
        
    def __call__(self, tensor):
        tensor = F.pad(tensor, torch.flip(self.get_padding(tensor.shape[-self.dimension:]), [0]).flatten().tolist())
        batch_dim = tensor.shape[:-self.dimension]
        for i in range(self.dimension):
            tensor = tensor.unfold(len(batch_dim)+i, self.kernel_size[i], self.stride[i])
        tensor = tensor.reshape(*batch_dim, -1, *tensor.shape[-self.dimension:])
        tensor = torch.movedim(tensor, len(batch_dim), 0)
        
        return tensor
    
    def get_padding(self, image_shape):
        padding = []
        for i in range(self.dimension):
            p = ((self.kernel_size[i] - image_shape[i])%self.stride[i])/2.0
            padding.append([ceil(p), floor(p)])
        return torch.tensor(padding)
    
    def num_patches_per_dim(self, image_shape):
        padding = self.get_padding(image_shape)
        return tuple((image_shape[i] + torch.sum(padding[i]).item() - self.kernel_size[i])//self.stride[i] + 1 for i in range(self.dimension))
    
    def num_patches(self, image_shape):
        return torch.prod(torch.tensor(self.num_patches_per_dim(image_shape)))
    
    
class Reconstruct:
    def __init__(self, kernel_size, image_shape, stride=1, sigma=1.0):
        self.kernel_size = kernel_size
        self.dimension = len(self.kernel_size)
        self.image_shape = image_shape
        self.sigma = sigma
        self.stride = stride if type(stride) is tuple else tuple([stride]*self.dimension)
        
        
        if self.dimension != len(self.stride):
            raise ValueError(f'kernel_size ({len(self.dimension)}) should have same number of dimension as padding ({len(self.stride)}).')
        
        if self.dimension != len(self.image_shape):
            raise ValueError(f'shape of kernel ({self.dimension}) and image shape ({self.image_shape}) should match.')
        
        self.padding = self.get_padding()
        self.pad_image_shape = tuple(torch.sum(self.padding[i]).item() + self.image_shape[i] for i in range(self.dimension))
        self.kernel = self._gaussian_kernel()
        self.num_patches_per_dim = self.get_num_patches_per_dim()
        start = torch.cartesian_prod(*[torch.arange(self.num_patches_per_dim[i]) * self.stride[i]  for i in range(self.dimension)])
        self.patch_positions = [tuple(slice(start[i,d], start[i,d]+self.kernel_size[d]) for d in range(self.dimension)) for i in range(len(start))]
        self.weights = self._embed_kernel_into_volume()
        
    def __call__(self, patches):
        if patches.shape[0] != self.weights.shape[0]:
            raise ValueError(f'Number of pacthes should be {self.weights.shape[0]}, but {patches.shape[0]} were given.')
        channel = len(patches.shape) == self.dimension+2
        patches = patches if channel else patches.unsqueeze(1)
        
        self.recon = self.weights.to(patches.device)*patches
        output = self._unfold(self.recon).reshape(-1, patches.shape[1], *self.pad_image_shape)[0]
        slices = tuple(slice(self.padding[i,0], self.padding[i,0]+self.image_shape[i]) for i in range(self.dimension))
        output = output[(Ellipsis, )+slices]

        return output if channel else output[0]

    def _gaussian_kernel(self):
        coords = [torch.arange(0, k, 1) for k in self.kernel_size]
        mesh = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
        c = torch.tensor([(k-1)/2 for k in self.kernel_size])
        
        return -torch.sum((mesh-c)**2/(2*self.sigma**2), dim=-1)
    
    def _form_conv_transpose_input(self):
        N = torch.prod(self.num_patches_per_dim).item()
        self.inp = torch.zeros(1, N, *self.num_patches_per_dim, device=self.kernel.device, dtype=self.kernel.dtype)
        coords  = [torch.arange(patch, device=self.kernel.device) for patch in self.num_patches_per_dim]
        grid = [v.flatten() for v in torch.meshgrid(*coords, indexing='ij')]
        idx_flat = torch.zeros_like(grid[0])
        for i in range(self.dimension):
            idx_flat += grid[i] * torch.prod(self.num_patches_per_dim[i+1:])
        self.inp[tuple([0,idx_flat,*grid])] = 1
    
    def _embed_kernel_into_volume(self, fill_value = -float('inf')):
        self._form_conv_transpose_input()
        N = torch.prod(self.num_patches_per_dim).item()
        
        weight_val  = self.kernel.reshape(1,1,*self.kernel.shape).repeat(N,1,*[1]*self.dimension)
        weight_mask = torch.ones_like(weight_val)
        out_val = self._unfold(weight_val, grouped=True)
        out_mask = self._unfold(weight_mask, grouped=True)
        
        embedded_kernel = torch.nan_to_num(torch.softmax(torch.where(out_mask.bool(), out_val, fill_value), dim=0),0)
        unfolded_kernel = Patchify(kernel_size=self.kernel.shape, stride=self.stride)(embedded_kernel)
        return torch.movedim(torch.diagonal(unfolded_kernel, dim1=0, dim2=1), -1,0)
    
    def get_padding(self):
        padding = []
        for i in range(self.dimension):
            p = ((self.kernel_size[i] - self.image_shape[i])%self.stride[i])/2.0
            padding.append([ceil(p), floor(p)])
        return torch.tensor(padding)
    
    def _unfold(self, patches, grouped=False):
        if grouped:
            outputs = torch.zeros(*patches.shape[0:2], *self.pad_image_shape)
        else:
            outputs = torch.zeros(patches.shape[1], *self.pad_image_shape)
            
        for i, pos in enumerate(self.patch_positions):
            if grouped:
                outputs[(i,Ellipsis,)+pos] += patches[i]
            else:
                outputs[(Ellipsis,)+pos] += patches[i]
        return outputs
    
    def get_num_patches_per_dim(self):
        return torch.tensor([(self.pad_image_shape[i] - self.kernel.shape[i])//self.stride[i] + 1 for i in range(self.dimension)])
    
    def num_patches(self):
        return torch.prod(self.get_num_patches_per_dim())
