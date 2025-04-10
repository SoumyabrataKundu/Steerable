import torch
import torch.nn.functional as F

class PatchifyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, kernel_size, stride = 1):
        self.dataset = dataset
        self.kernel_size = kernel_size
        
        self.patchify = Patchify(kernel_size, stride)
        self.patches_per_image = len(self.patchify(dataset[0][0]))
        
    def __getitem__(self, index):
        image, target = self.dataset[index // self.patches_per_image]
        extracted_patches = self.patchify(image)[index % self.patches_per_image]
        extracted_targets = self.patchify(target.unsqueeze(0))[index % self.patches_per_image,0]
        return extracted_patches, extracted_targets
    
    def __len__(self):
        return self.patches_per_image*len(self.dataset)
    
class Patchify:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.dimension = len(self.kernel_size)
        self.stride = stride if type(stride) is tuple else tuple([stride]*self.dimension)
        self.padding = padding if type(padding) is tuple else tuple([padding]*self.dimension)
        if self.dimension != len(self.stride):
            raise ValueError(f'kernel_size ({self.dimension}) should have same number of dimension as padding ({len(self.stride)}).')

        if self.dimension != len(self.padding):
            raise ValueError(f'kernel_size ({self.dimension}) should have same number of dimension as padding ({len(self.padding)}).')
        self.padding = tuple(v for x in self.padding[::-1] for v in (x, x))
        
    def __call__(self, tensor):
        tensor = F.pad(tensor, self.padding)
        batch_dim = tensor.shape[:-self.dimension]
        for i in range(self.dimension):
            tensor = tensor.unfold(len(batch_dim)+i, self.kernel_size[i], self.stride[i])
        tensor = tensor.reshape(*batch_dim, -1, *tensor.shape[-self.dimension:])
        tensor = torch.movedim(tensor, len(batch_dim), 0)
        
        return tensor
    
    
class Reconstruct:
    def __init__(self, kernel_size, image_shape, stride=1, padding=0, sigma=1.0):
        self.kernel_size = kernel_size
        self.dimension = len(self.kernel_size)
        self.image_shape = image_shape
        self.sigma = sigma
        self.stride = stride if type(stride) is tuple else tuple([stride]*self.dimension)
        self.padding = padding if type(padding) is tuple else tuple([padding]*self.dimension)
        
        if self.dimension != len(self.stride):
            raise ValueError(f'kernel_size ({len(self.dimension)}) should have same number of dimension as padding ({len(self.stride)}).')

        if self.dimension != len(self.padding):
            raise ValueError(f'kernel_size ({len(self.dimension)}) should have same number of dimension as padding ({len(self.padding)}).')
        
        if self.dimension != len(self.image_shape):
            raise ValueError(f'shape of kernel ({self.dimension}) and image shape ({self.image_shape}) should match.')
        
        self.kernel = self._gaussian_kernel()
        self.num_patches_per_dim = self.get_num_patches_per_dim()
        self.weights = self._embed_kernel_into_volume()
        
    
    def __call__(self, patches):
        if patches.shape[0] != self.weights.shape[0]:
            raise ValueError(f'Number of pacthes should be {self.weights.shape[0]}, but {patches.shape[0]} were given.')
        self.recon = self.weights.to(patches.device)*patches
        output = self._unfold(self.recon).reshape(-1, patches.shape[1], *self.image_shape).sum(dim=0)

        return output

    def _gaussian_kernel(self):
        coords = [torch.arange(0, k, 1) for k in self.kernel_size]
        mesh = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
        c = torch.tensor([(k-1)/2 for k in self.kernel_size])
        
        return torch.sum((mesh-c)**2/(2*self.sigma**2), dim=-1)
    
 
    
    def _form_conv_transpose_input(self):
        # number of placements vertically & horizontally
        N = torch.prod(self.num_patches_per_dim).item()

        # Build the one-hot “input” map of shape (1, N, n_y, n_x)
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
        out_val = self._unfold(weight_val).squeeze(0)
        out_mask = self._unfold(weight_mask).squeeze(0)
        
        embedded_kernel = torch.softmax(torch.where(out_mask.bool(), out_val, fill_value), dim=0)
        unfolded_kernel = Patchify(kernel_size=self.kernel.shape, stride=self.stride, padding=self.padding)(embedded_kernel)
        
        return torch.movedim(torch.diagonal(unfolded_kernel, dim1=0, dim2=1), -1,0).unsqueeze(1) 
    
    def _unfold(self, weights):
        # Compute needed output_padding to exactly hit (H,W)
        N = torch.prod(self.num_patches_per_dim).item()
        op = [self.image_shape[i] - (self.num_patches_per_dim[i]-1)*self.stride[i] + 2*self.padding[i] - self.kernel.shape[i] for i in range(self.dimension)]
        
        if self.dimension == 2:
            output = F.conv_transpose2d(self.inp, weights, stride=self.stride, padding=self.padding,output_padding=op,groups=N)
        elif self.dimension==3:
            output = F.conv_transpose3d(self.inp, weights, stride=self.stride, padding=self.padding,output_padding=op,groups=N)
        else:
            raise ValueError(f'Only 2D and 3D kernel shape is supported.')
        return output
    
    def get_num_patches_per_dim(self):
        return torch.tensor([(self.image_shape[i] + 2*self.padding[i] - self.kernel.shape[i])//self.stride[i] + 1 for i in range(self.dimension)])
