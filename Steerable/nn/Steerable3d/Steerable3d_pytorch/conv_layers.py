import torch
import torch.nn as nn
from numpy import prod
try:
    import gelib
except:
    pass
    #raise ImportError("GElib is not installed. SE3CGNonLinearity only works with GElib backend.")

from Steerable.nn.utils import get_CFint_matrix
from Steerable.nn.utils import merge_channel_dim, split_channel_dim

##################################################################################################################################
########################################################### First Layer ##########################################################
##################################################################################################################################

class SE3Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_angle, interpolation_type=1,
                 dilation=1, padding=0, stride=1, restricted=False, conv_first=False):
        super(SE3Conv, self).__init__()
        
        # Layer Design
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if type(kernel_size) is not tuple else kernel_size
        self.dilation = (dilation, dilation, dilation) if type(kernel_size) is not tuple else dilation
        self.padding = padding if type(padding) is tuple or type(padding) is str else (padding, padding, padding)
        self.stride = (stride, stride, stride) if type(stride) is not tuple else stride

        self.n_radius = n_radius
        self.n_angle = n_angle
        self.in_channels = [in_channels] if type(in_channels) is not list and type(in_channels) is not tuple else in_channels
        self.out_channels = [out_channels] if type(out_channels) is not list and type(out_channels) is not tuple else out_channels
        self.conv_first = conv_first
        
        # Fint Matrix
        self.Fint = get_CFint_matrix(self.kernel_size, n_radius, n_angle, max(len(self.out_channels), len(self.in_channels))-1, interpolation_type)

        if conv_first:
            if restricted:
                # Layer Parameters
                self.weights = nn.ParameterList([nn.Parameter(
                                            torch.randn(self.out_channels[l], sum(self.in_channels) * n_radius, dtype = torch.cfloat))
                                            for l in range(len(self.out_channels))])
                self.Fint = [torch.cat([t2.sum(dim=2) for t2 in t1], dim=0).flatten(0,1) for t1 in self.Fint]
            
            else:
                # Layer Parameters
                self.weights = nn.ParameterList([nn.Parameter(
                                            torch.randn(self.out_channels[l], sum(self.in_channels) * (max(len(self.out_channels), len(self.in_channels))-1) * n_radius, dtype = torch.cfloat))
                                            for l in range(len(self.out_channels))])
                self.Fint = [torch.cat(t, dim=0).flatten(0,2) for t in self.Fint]
                
        else:
            if restricted:
                self.weights = nn.ParameterList([nn.Parameter(
                                                torch.randn(self.out_channels[l], 1, self.in_channels[l1], n_radius, dtype = torch.cfloat))
                                                for l in range(len(self.out_channels)) for l1 in range(len(self.in_channels))])
                self.Fint = [[t2.flatten(-3).sum(dim=2).transpose(1,2).unsqueeze(1) for t2 in t1] for t1 in self.Fint]
            else:
                self.weights = nn.ParameterList([nn.Parameter(
                                            torch.randn(self.out_channels[l], 1, self.in_channels[l1], max(len(self.out_channels), len(self.in_channels))*n_radius, dtype = torch.cfloat))
                                            for l in range(len(self.out_channels)) for l1 in range(len(self.in_channels))])
                self.Fint = [[t2.flatten(-3).flatten(1,2).transpose(1,2).unsqueeze(1) for t2 in t1] for t1 in self.Fint]

        
    def forward(self, x):
        if self.conv_first:
            x = [x] if type(x) is not list and type(x) is not tuple else x
            batch_size = x[0].shape[0]
            parts = []
            
            # Convolution
            for l1, (part, c) in enumerate(zip(x, self.in_channels)):
                part = part.reshape(batch_size, -1, c, *part.shape[-3:]).transpose(1,2).flatten(0,1)
                part = torch.conv3d(part, self.Fint[l1].to(part.device), stride=self.stride, padding = self.padding, dilation=self.dilation)
                out_res = part.shape[-3:]
                part = part.reshape(batch_size, c, len(self.out_channels)**2, -1, prod(out_res)).transpose(1,2).flatten(2,3)
                parts.append(part)
            parts = torch.cat(parts, dim = 2)
            
            # Weight Multiplication
            result = []
            for l, c in enumerate(self.out_channels):
                result.append((self.weights[l] @ parts[:, l**2:(l+1)**2]).reshape(batch_size, 2*l+1, c, *out_res))
            
        else:
            kernel = self.get_kernel()
            x, _ = merge_channel_dim(x)
            x = torch.conv3d(x, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)
            result = split_channel_dim(x, self.out_channels)
        
        return result
    
    def get_kernel(self):
        kernel = []
        for l in range(len(self.out_channels)):
            kernels = []
            for l1 in range(len(self.in_channels)):
                index = l1 + l*len(self.in_channels)
                kernel_block = self.weights[index] @ self.Fint[l1][l].to(self.weights[index].device)
                kernel_block = kernel_block.reshape((2*l+1) * self.out_channels[l], -1, *self.kernel_size)
                kernels.append(kernel_block)
                
            kernel.append(torch.cat(kernels, dim=1))
        kernel = torch.cat(kernel, dim=0)
        
        return kernel


########################################################################################################################
#################################################### Non-linearity #####################################################
########################################################################################################################

class SE3CGNonLinearity(nn.Module):
    def __init__(self, in_channels):
        super(SE3CGNonLinearity, self).__init__()

        self.in_channels = [in_channels] if type(in_channels) is not list and type(in_channels) is not tuple else in_channels
        self.maxl = len(in_channels) - 1
        
        size = [sum([int(abs(l1-l2) <= l<= l1+l2)  for l1 in range(self.maxl+1) for l2 in range(self.maxl+1)]) for l in range(self.maxl+1)]
        hidden_dim = min(in_channels)
        self.weights1 = nn.ParameterList([nn.Parameter(
                                        torch.randn(in_channels[l], hidden_dim, dtype = torch.cfloat))
                                         for l in range(self.maxl + 1)])
        self.weights2 = nn.ParameterList([nn.Parameter(
                                        torch.randn(hidden_dim * size[l], in_channels[l], dtype = torch.cfloat))
                                         for l in range(self.maxl + 1)])
    def forward(self, x):
        inputs = gelib.SO3vecArr()
        inputs.parts = [x[l].permute(0,3,4,5,1,2) @ self.weights1[l] for l in range(self.maxl+1)]
        inputs = gelib.DiagCGproduct(inputs,inputs,self.maxl)
        result = [(inputs.parts[l] @ self.weights2[l]).permute(0,4,5,1,2,3) for l in range(self.maxl+1)]
        return result

class SE3NormNonLinearity(nn.Module):
    def __init__(self, in_channels):
        super(SE3NormNonLinearity, self).__init__()
        
        self.non_linearity = torch.nn.GELU()
        self.eps = 1e-5
        self.in_channels = [in_channels] if type(in_channels) is not list and type(in_channels) is not tuple else in_channels
        self.b = nn.ParameterList([nn.Parameter(
                                    torch.randn(self.in_channels[l], dtype = torch.float))
                                  for l in range(len(self.in_channels))])
    def forward(self, x):
        result = []
        for l,part in enumerate(x):
            magnitude = torch.linalg.vector_norm(part, dim=1, keepdim=True) + self.eps
            bias = self.b[l].reshape(1,1,self.in_channels[l], *[1]*len(part.shape[3:]))
            factor = self.non_linearity(magnitude + bias) / magnitude
            result.append(part * factor)
        
        return result

class SE3GatedNonLinearity(nn.Module):
    def __init__(self, in_channels, kernel_size, n_radius, n_angle):
        super(SE3GatedNonLinearity, self).__init__()
        
        self.non_linearity = torch.nn.Sigmoid()
        self.layer = SE3Conv(in_channels, sum(in_channels), kernel_size, n_radius, n_angle, padding='same')
        self.in_channels = self.layer.in_channels
        
    def forward(self, x):
        factor = self.non_linearity(self.layer(x)[0])
        result = []
        split_index = 0
        for l,part in enumerate(x):
            magnitude = factor[:, :, split_index: split_index + self.in_channels[l]]
            result.append(magnitude * part)
            split_index += self.in_channels[l]
        
        return result
  
#######################################################################################################################
################################################### Batch Normalization ###############################################
#######################################################################################################################

class SE3BatchNorm(nn.Module):
    def __init__(self):
        super(SE3BatchNorm, self).__init__()
        self.eps = 1e-5

    def forward(self, x):
        result = []
        for part in x:
            magnitude = torch.linalg.vector_norm(part, dim=1, keepdim=True) + self.eps
            part = (part.real/magnitude) + 1j*(part.imag/magnitude)
            result.append(part)
        
        return result

#######################################################################################################################
################################################## Average Pooling Layer ##############################################
#######################################################################################################################  

class SE3AvgPool(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SE3AvgPool, self).__init__()
        self.pool = nn.AvgPool3d(*args, **kwargs)

    def forward(self, x):
        x, channels = merge_channel_dim(x)
        x = self.pool(x.real) + 1j*self.pool(x.imag)
        return split_channel_dim(x, channels)

#######################################################################################################################
################################################### Invariant Layers ##################################################
#######################################################################################################################

class SE3NormFlatten(nn.Module):
    def __init__(self):
        super(SE3NormFlatten, self).__init__()

    def forward(self, x):
        parts = []
        for part in x:
            part = torch.mean(part.flatten(3), dim = (-1,))
            part = torch.linalg.vector_norm(part, dim = (1,))
            parts.append(part)
            
        result = torch.cat(parts, dim=1)
        return result