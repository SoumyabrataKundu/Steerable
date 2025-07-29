import torch
from scipy.ndimage import map_coordinates


def get_interpolation_matrix(kernel_size, n_radius, n_angular, interpolation_order=1):
    R = torch.tensor([(kernel_size[i] - 1)/2 for i in range(len(kernel_size))])
    A1 = torch.arange(0, torch.pi, torch.pi / n_angular)
    A2 = torch.arange(0, 2*torch.pi, 2*torch.pi / n_angular)
    sphere_coord = torch.ones(1)
    r_values = torch.vstack([torch.arange(r / (n_radius+1), r, r / (n_radius+1))[:n_radius] for r in R])
    for i in range(len(kernel_size)-1):
        A = A1 if i<len(kernel_size)-2 else A2
        sphere_coord = torch.vstack([
                    torch.tensordot(sphere_coord[:-1], torch.ones(n_angular), dims=0),
                    torch.tensordot(sphere_coord[-1:], torch.cos(A), dims=0), 
                    torch.tensordot(sphere_coord[-1:], torch.sin(A), dims=0)])
        
    sphere_coord = torch.einsum('dr, da -> dra', r_values, sphere_coord.flatten(1)) + R.reshape(-1, 1,1)
    
    I = torch.zeros(n_radius * n_angular**(len(kernel_size)-1), *kernel_size, dtype=torch.float)
    kernel = torch.cartesian_prod(*[torch.arange(0, kernel_size[d], 1) for d in range(len(kernel_size))])
    for i in range(len(kernel)):
        f = torch.zeros(*kernel_size)
        f[tuple(kernel[i].tolist())] = 1
        I[(Ellipsis,) + tuple(kernel[i].tolist())] = torch.from_numpy(map_coordinates(f, sphere_coord.flatten(1), order=interpolation_order, mode='nearest'))

    return I.reshape(n_radius, -1, *I.shape[-len(kernel_size):])