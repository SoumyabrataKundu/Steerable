import torch
from scipy.ndimage.interpolation import rotate
from Steerable.nn.utils import get_interpolation_matrix
from math import sqrt

#######################################################################################################################
################################################# Fint Matrix #########################################################
#######################################################################################################################

def get_Fint_matrix(kernel_size, n_radius, n_theta, max_m, interpolation_type=1):
    R1 = (kernel_size[0]-1)/2
    R2 = (kernel_size[1]-1)/2
    r1_values = (torch.arange(R1/(n_radius+1), R1, R1/(n_radius+1))[:n_radius]).type(torch.cfloat)
    r2_values = (torch.arange(R2/(n_radius+1), R2, R2/(n_radius+1))[:n_radius]).type(torch.cfloat)
    
    if interpolation_type == -1:
        x_range = torch.arange(-kernel_size[0]/2, kernel_size[0]/2, 1) + 0.5  # Example range for x-coordinate
        y_range = torch.arange(-kernel_size[1]/2, kernel_size[1]/2, 1) + 0.5  # Example range for y-coordinate
        X, Y = torch.meshgrid(x_range, y_range, indexing='xy')
        
        norm = torch.sqrt(X**2 + Y**2).reshape(kernel_size[0], kernel_size[1])
        theta = torch.arctan2(Y, X)
        tau_r = torch.exp(-(( (torch.arange(n_radius)+1).reshape(-1,1,1) - norm)**2)/0.72).type(torch.cfloat)

        Fint = torch.stack([torch.exp( m * 1j * theta) for m in range(max_m)], dim = 0)
        Fint = torch.einsum('rxy, mxy-> mrxy', tau_r, Fint)
        
    elif 0 <= interpolation_type and interpolation_type<=5 and type(interpolation_type) == int:
        # Interpolation
        I = get_interpolation_matrix(kernel_size, n_radius, n_theta, min(interpolation_type, min(kernel_size)-1)).type(torch.cfloat)
        # Fourier Transform Matrix
        FT = (torch.fft.fft(torch.eye(max_m, n_theta)) / sqrt(n_theta))
        Fint = torch.einsum('r, mt, rtxy -> mrxy', torch.sqrt(r1_values*r2_values), FT, I)
    else:
        raise ValueError("'interpolation_type' integer takes values between -1 and 5.")
    
    return Fint

def get_CG_matrix(max_m):
    CG_Matrix = torch.tensor([[[(m1+m2-m)%max_m == 0 
                                            for m2 in range(max_m)] 
                                        for m1 in range(max_m)] 
                                    for m in range(max_m)]).type(torch.cfloat)
    return CG_Matrix

#######################################################################################################################
################################################# Positional Encoding #################################################
#######################################################################################################################

def get_pos_encod(kernel_size, max_m):
    # Define the range for each dimension
    x_range = torch.arange(0, kernel_size[0], 1)  # Example range for x-coordinate
    y_range = torch.arange(0, kernel_size[1], 1)  # Example range for y-coordinate
    
    # Create the mesh grid
    X, Y = torch.meshgrid(x_range, y_range, indexing='xy')

    # Flatten the mesh grid to get coordinates
    coordinates = torch.stack((X.flatten(), Y.flatten()), dim=1)
    
    # Compute Pairwise differences
    num_points = coordinates.shape[0]
    pairwise_diffs = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
    pairwise_diffs = pairwise_diffs.view(-1, 2)  # Flatten the pairwise differences

    x, y = pairwise_diffs[:, 0], pairwise_diffs[:, 1]
    
    # Convert to polar coordinates
    r_square = (x**2 + y**2).reshape(num_points, num_points)
    #phi_r = torch.sqrt(r_square).unsqueeze(0).repeat(max_m, 1)
    phi_r = torch.stack([torch.exp(-(r_square) / 2).fill_diagonal_(0) for m in range(max_m)], dim=0)
    
    theta = torch.arctan2(y, x)
    result = torch.stack([torch.exp( m *1j * theta) for m in range(max_m)], dim = 0) 
    result = result.reshape(max_m, num_points, num_points) * phi_r
    result = result.reshape(max_m, 1, num_points, 1, num_points)
    
    return result

#######################################################################################################################
################################################### Rotate Image  #####################################################
#######################################################################################################################

def rotate_2D_image(image, degree=None, order=5):
    if degree is None:
        degree = torch.randint(0, 360, (1,)).item()
    
    def rotate_slice_image(image_slice, degree):
        image_slice = torch.from_numpy(rotate(image_slice, degree, (1,0), reshape=False, order=order))
        
        return image_slice
    
    image_shape = image.shape
    image = image.reshape(-1, image_shape[-2], image_shape[-1])
    image = torch.vstack([rotate_slice_image(image[i], degree) for i in range(image.shape[0])]).view(*image_shape)
    
    return image
