import torch
from math import floor, pi
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import RectBivariateSpline
from math import sqrt

#######################################################################################################################
################################################# Interpolation Matrix ################################################
#######################################################################################################################


# def get_interpolation_matrix(kernel_size, n_radius, n_phi, interpolation_type="cubic"):
#     R = (kernel_size[0]-1)/2
#     r_values = torch.arange(R/(n_radius+1), R, R/(n_radius+1))[:n_radius]
#     phi_values = torch.arange(0, 2 * pi, 2 * pi / n_phi)
#     I = torch.zeros(n_radius, n_phi, kernel_size[0], kernel_size[1], dtype=torch.cfloat)
#     x_new = torch.tensordot(r_values, torch.cos(phi_values), dims=0) + R
#     y_new = torch.tensordot(r_values, torch.sin(phi_values), dims=0) + R

#     def w(x,y):
#         if interpolation_type == "nearest":
#             return float(x>= 0.5) * float(y>=0.5)

#         if interpolation_type == "linear":
#             return x*y

#         if interpolation_type == "cubic":
#             return x*x*y*y*(9 - 6*x - 6*y + 4*x*y)

#     for r in range(n_radius):
#         for phi in range(n_phi):
#             integer_x, fraction_x = floor(x_new[r, phi]), x_new[r, phi] - floor(x_new[r, phi])
#             integer_y, fraction_y = floor(y_new[r, phi]), y_new[r, phi] - floor(y_new[r, phi])

#             I[r, phi, integer_x, integer_y] = w(1-fraction_x, 1-fraction_y)
#             I[r, phi, integer_x + 1, integer_y] = w(fraction_x, 1-fraction_y)
#             I[r, phi, integer_x, integer_y + 1] = w(1 - fraction_x, fraction_y)
#             I[r, phi, integer_x + 1, integer_y + 1] = w(fraction_x, fraction_y)

# #    I = I/torch.sum(I, dims = (2,3), keepdim = True)
#     return(I)


def get_interpolation_matrix(kernel_size, n_radius, n_theta, k=1):
    R1 = (kernel_size[0]-1)/2
    R2 = (kernel_size[1]-1)/2
    r1_values = torch.arange(R1/(n_radius+1), R1, R1/(n_radius+1))[:n_radius]
    r2_values = torch.arange(R2/(n_radius+1), R2, R2/(n_radius+1))[:n_radius]
    phi_values = torch.arange(0, 2 * pi, 2 * pi / n_theta)
    
    x1_values = torch.tensordot(r1_values, torch.cos(phi_values), dims=0) + R1
    x2_values = torch.tensordot(r2_values, torch.sin(phi_values), dims=0) + R2
    kernel1 = torch.arange(0, kernel_size[0], 1)
    kernel2 = torch.arange(0, kernel_size[1], 1)
    
    I = torch.zeros(n_radius, n_theta, kernel_size[0], kernel_size[1])
    for r in range(n_radius):
        for theta in range(n_theta):
            x1 = x1_values[r,theta]
            x2 = x2_values[r,theta]
            for y1 in kernel1:
                for y2 in kernel2:
                    if k == 0 or k == 1:
                        integer_x1, fraction_x1 = floor(x1), x1 - floor(x1)
                        integer_x2, fraction_x2 = floor(x2), x2 - floor(x2)
                        
                        def w(x,y):
                            return float(x>= 0.5) * float(y>=0.5) if k==0 else x*y
                        
                        if (y1,y2) == (integer_x1, integer_x2):
                            I[r,theta, y1, y2] = w(1-fraction_x1, 1-fraction_x2)
                        if (y1,y2) == (integer_x1+1, integer_x2):
                            I[r,theta, y1, y2] = w(fraction_x1, 1-fraction_x2)
                        if (y1,y2) == (integer_x1, integer_x2+1):
                            I[r,theta, y1, y2] = w(1 - fraction_x1, fraction_x2)
                        if (y1,y2) == (integer_x1+1, integer_x2+1):
                            I[r,theta, y1, y2] = w(fraction_x1, fraction_x2)
                    
                    elif k>1:
                        f = torch.zeros((len(kernel1), len(kernel2)))
                        f[y1,y2] = 1
                        spline = RectBivariateSpline(kernel1, kernel2, f, kx=k, ky=k)
                        I[r,theta, y1, y2] = spline(x1_values[r,theta], x2_values[r,theta])[0, 0]
    
    return I.type(torch.cfloat)
    

#######################################################################################################################
################################################# Fint Matrix #########################################################
#######################################################################################################################

def get_Fint_matrix(kernel_size, n_radius, n_theta, max_m, interpolation_type=1):
    R1 = (kernel_size[0]-1)/2
    R2 = (kernel_size[1]-1)/2
    r1_values = (torch.arange(R1/(n_radius+1), R1, R1/(n_radius+1))[:n_radius]).type(torch.cfloat)
    r2_values = (torch.arange(R2/(n_radius+1), R2, R2/(n_radius+1))[:n_radius]).type(torch.cfloat)
    
    # Interpolation
    I = get_interpolation_matrix(kernel_size, n_radius, n_theta, min(interpolation_type, min(kernel_size)-1))
    
    # Fourier Transform Matrix
    FT = (torch.fft.fft(torch.eye(max_m, n_theta)) / sqrt(n_theta))
    
    Fint = torch.einsum('r, mt, rtxy -> mrxy', torch.sqrt(r1_values*r2_values), FT, I)
    
    return Fint

# def get_Fint_matrix(kernel_size, n_radius, n_theta, max_m, interpolation_type='cubic'):
#     R = (min(kernel_size)-1)/2
#     r_values = (torch.arange(R/(n_radius+1), R, R/(n_radius+1))[:n_radius])

#     x_range = torch.arange(-kernel_size[0]/2, kernel_size[0]/2, 1) + 0.5  # Example range for x-coordinate
#     y_range = torch.arange(-kernel_size[1]/2, kernel_size[1]/2, 1) + 0.5  # Example range for y-coordinate
#     X, Y = torch.meshgrid(x_range, y_range, indexing='xy')
#     theta = torch.arctan2(Y, X)
#     norm = torch.sqrt(X**2 + Y**2).reshape(kernel_size[0], kernel_size[1])
#     tau_r = torch.exp(-((r_values.reshape(-1,1,1) - norm)/2) ** 2).type(torch.cfloat)

#     #Fint = torch.stack([torch.exp( m * 1j * theta) for m in range(max_m)], dim = 0)
#     #Fint = torch.einsum('rxy, mxy-> mrxy', tau_r, Fint)

#     I = get_interpolation_matrix(kernel_size, n_radius, n_theta, interpolation_type)
#     FT = (torch.fft.fft(torch.eye(max_m, n_theta)) / sqrt(n_theta))
#     Fint = torch.einsum('rxy, mt, rtxy -> mrxy', tau_r, FT, I)

#     return Fint

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

def rotate_2D_image(image, degree):
    
    def rotate_slice_image(image_slice, degree):
        image_slice = torch.from_numpy(rotate(image_slice, degree, (1,0), reshape=False, order = 5))
        
        return image_slice
    
    image_shape = image.shape
    image = image.reshape(-1, image_shape[-2], image_shape[-1])
    image = torch.vstack([rotate_slice_image(image[i], degree) for i in range(image.shape[0])]).view(*image_shape)
    
    return image
