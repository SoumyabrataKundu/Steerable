import torch
import pyshtools
from math import pi, floor
from sympy.physics.quantum.cg import CG
from scipy.ndimage.interpolation import rotate
from scipy.special import sph_harm

#######################################################################################################################
############################################ Interpolation Matrix #####################################################
#######################################################################################################################

def get_interpolation_matrix_3D(kernel_size, n_radius, n_theta, n_phi, interpolation_type=1):
    R1, R2, R3 = (kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2, (kernel_size[2] - 1) / 2
    r1_values = torch.arange(R1 / (n_radius+1), R1, R1 / (n_radius+1))[:n_radius]
    r2_values = torch.arange(R2 / (n_radius+1), R2, R2 / (n_radius+1))[:n_radius]
    r3_values = torch.arange(R3 / (n_radius+1), R3, R3 / (n_radius+1))[:n_radius]
    
    theta_values = torch.arange(0, pi, pi / n_theta)
    phi_values = torch.arange(0, 2 * pi, 2 * pi / n_phi)

    def w(x,y,z):
        if interpolation_type == 0:
            return float(x>=0.5)*float(y>=0.5)*float(z>=0.5)
        elif interpolation_type == 1:
            return x*y*z
        # return x**2*y**2*z**2*(9 - 6*x - 6*y - 6*z + 4*x*y*z)

    I = torch.zeros(n_radius, n_theta, n_phi, kernel_size[0], kernel_size[1], kernel_size[2], dtype=torch.cfloat)
    x_new = torch.tensordot(torch.tensordot(r1_values, torch.sin(theta_values), dims=0), torch.cos(phi_values), dims = 0) + R1
    y_new = torch.tensordot(torch.tensordot(r2_values, torch.sin(theta_values), dims=0), torch.sin(phi_values), dims = 0) + R2
    z_new = torch.tensordot(r3_values, torch.cos(theta_values), dims=0).unsqueeze(2).repeat(1, 1, n_phi) + R3

    for r in range(n_radius):
        for theta in range(n_theta):
            for phi in range(n_phi):
                integer_x, fraction_x = floor(x_new[r, theta, phi]), x_new[r, theta, phi] - floor(x_new[r, theta, phi])
                integer_y, fraction_y = floor(y_new[r, theta, phi]), y_new[r, theta, phi] - floor(y_new[r, theta, phi])
                integer_z, fraction_z = floor(z_new[r, theta, phi]), z_new[r, theta, phi] - floor(z_new[r, theta, phi])

                I[r, theta, phi, integer_x, integer_y, integer_z] = w((1-fraction_x), (1-fraction_y), (1-fraction_z))
                I[r, theta, phi, integer_x + 1, integer_y, integer_z] = w(fraction_x, (1-fraction_y), (1-fraction_z))
                I[r, theta, phi, integer_x, integer_y + 1, integer_z] = w((1 - fraction_x), fraction_y, (1-fraction_z))
                I[r, theta, phi, integer_x + 1, integer_y + 1, integer_z] = w(fraction_x, fraction_y, (1-fraction_z))

                I[r, theta, phi, integer_x, integer_y, integer_z + 1] = w((1-fraction_x), (1-fraction_y), fraction_z)
                I[r, theta, phi, integer_x + 1, integer_y, integer_z + 1] = w(fraction_x, (1-fraction_y), fraction_z)
                I[r, theta, phi, integer_x, integer_y + 1, integer_z + 1] = w((1 - fraction_x), fraction_y, fraction_z)
                I[r, theta, phi, integer_x + 1, integer_y + 1, integer_z + 1] = w(fraction_x, fraction_y, fraction_z)

    I = I.flatten(1,2) # n_radius x n_theta * n_phi x *kernel
    return I

######################################################################################################################
######################################### Spherical Harmonic Transform ###############################################
######################################################################################################################
def get_sh_transform_matrix(n_theta, n_phi, maxl):
    f = torch.zeros(n_theta, n_phi)
    FT = [torch.zeros(2*l + 1, n_theta * n_phi, dtype=torch.cfloat) for l in range(maxl + 1)]
    index = 0

    for theta in range(n_theta):
        for phi in range(n_phi):
            torch.zero_(f)
            f[theta, phi] = 1

            sh_transform = torch.from_numpy(pyshtools.expand.SHExpandDHC(f))
            sh_transform = torch.hstack((torch.fliplr(sh_transform[1])[:, :-1], sh_transform[0]))

            for l in range(maxl+1):
                FT[l][:, index] = sh_transform[l,((n_theta//2 -1) - l):((n_theta//2 -1) + l + 1)]
            index += 1
    return FT # 2l+1 x n_theta * n_phi


#########################################################################################################################
############################################### Clebsch Gordan Tensors ##################################################
#########################################################################################################################

def get_CGtensor(l,l1,l2):
  CG_tensor = torch.zeros(2*l+1,2*l1+1, 2*l2+1)
  if l>= abs(l1-l2)  and l<= l1+l2:
    a = l1+l2-l
    for m1 in range(2*l1+1):
      for m2 in range(2*l2+1):
          m = m1+m2-a
          if m>=0 and m<=2*l:
            CG_tensor[m,m1,m2] =  float(CG(l1,m1-l1,l2,m2-l2,l,m-l).doit())
  return CG_tensor



###########################################################################################################################
################################################### Fint Matrix ###########################################################
###########################################################################################################################

def get_Fint_matrix(kernel_size, n_radius, n_theta, n_phi, maxl, interpolation_type=1):
    R1, R2, R3 = (kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2, (kernel_size[2] - 1) / 2
    r1_values = torch.arange(R1 / (n_radius+1), R1, R1 / (n_radius+1))[:n_radius]
    r2_values = torch.arange(R2 / (n_radius+1), R2, R2 / (n_radius+1))[:n_radius]
    r3_values = torch.arange(R3 / (n_radius+1), R3, R3 / (n_radius+1))[:n_radius]
    
    if interpolation_type == -1:
        x_range = torch.arange(-kernel_size[0]/2, kernel_size[0]/2, 1) + 0.5  # Example range for x-coordinate
        y_range = torch.arange(-kernel_size[1]/2, kernel_size[1]/2, 1) + 0.5  # Example range for y-coordinate
        z_range = torch.arange(-kernel_size[2]/2, kernel_size[2]/2, 1) + 0.5  # Example range for z-coordinate
        X, Y, Z = torch.meshgrid(x_range, y_range, z_range, indexing='xy')
        
        norm = torch.sqrt(X**2 + Y**2 + Z**2)
        theta = torch.arctan2(Y, X)
        phi = torch.acos(torch.clamp(Z / norm, -1.0, 1.0))
        tau_r = torch.exp(-(( (torch.arange(n_radius)+1).reshape(-1,1,1,1) - norm)**2)/2).type(torch.cfloat)
        
        Fint = []
        for l in range(maxl+1):
            Y_lm_stack = []
            for m in range(-l, l + 1):
                # Compute spherical harmonics using scipy
                Y_lm = torch.from_numpy(sph_harm(m, l, theta.numpy(), phi.numpy())).type(torch.cfloat)
                Y_lm_stack.append(torch.complex(torch.nan_to_num(Y_lm.real, nan=0.0),
                                                torch.nan_to_num(Y_lm.real, nan=0.0)))
            Fint.append(torch.stack(Y_lm_stack, dim=0).reshape(-1, 1, *kernel_size)*tau_r)
        
        
    elif 0 <= interpolation_type and interpolation_type<=5 and type(interpolation_type) == int:
        I = get_interpolation_matrix_3D(kernel_size, n_radius, n_theta, n_phi, interpolation_type) # Interpolation Matrix
        SHT = get_sh_transform_matrix(n_theta, n_phi, maxl) # Spherical Harmonic Transform Matrix
        Fint = [torch.einsum('r, lt, rtxyz -> lrxyz', r1_values*r2_values, SHT[l], I) for l in range(maxl+1)] # Fint Matrix
    else:
        raise ValueError("'interpolation_type' integer takes values between -1 and 1.")

    return Fint

def get_CFint_matrix(kernel_size, n_radius, n_theta, n_phi, maxl, maxl1, maxl2, interpolation_type=1):
    Fint = get_Fint_matrix(kernel_size, n_radius, n_theta, n_phi, maxl2, interpolation_type)

    # CG Tensor
    C =[[[(2*l1+1) * (2*l2+1) * get_CGtensor(l, l1, l2)/(2*l+1)
                  for l2 in range(maxl2+1)]
              for l1 in range(maxl1+1)]
         for l in range(maxl+1)]

    # Fint Matrix
    CFint = [[torch.stack([torch.einsum('lmn, nrxyz -> lrmxyz', C[l][l1][l2].type(torch.cfloat), Fint[l2])
                  for l2 in range(maxl2+1)], dim=2)
              for l in range(maxl+1)] for l1 in range(maxl1+1)]

    return CFint

#########################################################################################################################
################################################ Rotate 3D Image ########################################################
#########################################################################################################################


def rotate_3D_image(image, degree, order=5):
    
    def rotate_slice_image(image_slice, degree):
        image_slice = torch.from_numpy(rotate(image_slice, degree[0], (1,0), reshape=False, order = order))
        image_slice = torch.from_numpy(rotate(image_slice, degree[1], (0,2), reshape=False, order = order))
        image_slice = torch.from_numpy(rotate(image_slice, degree[2], (1,0), reshape=False, order = order))
        
        return image_slice
    
    image_shape = image.shape
    image = image.reshape(-1, *image_shape[-3:])
    image = torch.vstack([rotate_slice_image(image[i], degree) for i in range(image.shape[0])]).view(*image_shape)
    
    return image

#########################################################################################################################
####################################### Merge and Split Channels ########################################################
#########################################################################################################################


def merge_channel_dim(x, channel_last=False):
    if not channel_last:
        if type(x) is list:
            channels = [part.shape[2] for part in x]
            parts = [part.flatten(1,2) for part in x]
            x = torch.cat(parts, dim=1)
        else:
            channels = x.shape[-4]
    else:
        if type(x) is list:
            channels = [part.shape[-1] for part in x]
            parts = [part.flatten(-2,-1) for part in x]
            x = torch.cat(parts, dim=-1)
        else:
            channels = x.shape[-1]
    return x, channels

def split_channel_dim(x, channels, channel_last=False):
    split_index = 0 
    result = []
    if not channel_last:    
        for l,c in enumerate(channels):
            result.append(x[:, split_index:split_index + (2*l+1)*c].reshape(x.shape[0], 2*l+1, c, *x.shape[2:]))
            split_index += (2*l+1)*c
    else:
        for l,c in enumerate(channels):
            result.append(x[..., split_index:split_index + (2*l+1)*c].reshape(*x.shape[0], 2*l+1, c, *x.shape[2:]))
            split_index += (2*l+1)*c
    
    return result
    
#########################################################################################################################
################################################ Positional Encoding ####################################################
#########################################################################################################################
    
def get_pos_encod(kernel_size, maxl):
    # Define the range for each dimension
    x_range = torch.arange(0, kernel_size[0], 1)  # Example range for x-coordinate
    y_range = torch.arange(0, kernel_size[1], 1)  # Example range for y-coordinate
    z_range = torch.arange(0, kernel_size[2], 1)  # Example range for z-coordinate

    # Create the mesh grid
    X, Y, Z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Flatten the mesh grid to get coordinates
    coordinates = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)

    # Compute Pairwise differences
    num_points = coordinates.shape[0]
    pairwise_diffs = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
    pairwise_diffs = pairwise_diffs.view(-1, 3)  # Flatten the pairwise differences

    x, y, z = pairwise_diffs[:, 0], pairwise_diffs[:, 1], pairwise_diffs[:, 2]

    # Convert to polar coordinates
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(z / r)
    phi = torch.arctan2(y, x)
    phi_r = [torch.exp(-(r**2 - l)) for l in range(maxl+1)]

    result = []
    for l in range(maxl+1):
        part = torch.cat([(sph_harm(m, l, phi, theta) * phi_r[l]).unsqueeze(-1) for m in range(-l, l+1)], dim=-1)
        part = part.reshape(num_points, num_points, 2*l+1).transpose(-2,-1).unsqueeze(-2)
        part = torch.nan_to_num(part.real, nan=0.0) + 1j * torch.nan_to_num(part.imag, nan=0.0)
        part = part.type(torch.cfloat)

        result.append(torch.conj(part))

    return result

