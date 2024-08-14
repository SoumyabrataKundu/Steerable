import torch
import torch.nn as nn
import gelib

from Steerable.nn.Steerable3d.utils import get_CFint_matrix, merge_channel_dim, split_channel_dim


##################################################################################################################################
########################################################### First Layer ##########################################################
##################################################################################################################################

class SE3Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_theta,
                 dilation=1, padding=0, stride=1, restricted=False, conv_first=False):
        super(SE3Conv, self).__init__()
        
        # Layer Design
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if type(kernel_size) is not tuple else kernel_size
        self.dilation = (dilation, dilation, dilation) if type(kernel_size) is not tuple else dilation
        self.padding = padding if type(padding) is tuple or type(padding) is str else (padding, padding, padding)
        self.stride = (stride, stride, stride) if type(kernel_size) is not tuple else stride

        self.n_radius = n_radius
        self.n_theta = n_theta
        self.in_channels = [in_channels] if type(in_channels) is not list and type(in_channels) is not tuple else in_channels
        self.out_channels = [out_channels] if type(out_channels) is not list and type(out_channels) is not tuple else out_channels
        self.conv_first = conv_first
        
        # Fint Matrix
        self.Fint = get_CFint_matrix(self.kernel_size, n_radius, n_theta, n_theta, len(self.out_channels)-1, len(self.in_channels)-1, len(self.in_channels)-1)

        if conv_first:
            if restricted:
                # Layer Parameters
                self.weights = nn.ParameterList([nn.Parameter(
                                            torch.randn(sum(self.in_channels) * n_radius, self.out_channels[l], dtype = torch.cfloat))
                                            for l in range(len(self.out_channels))])
                self.Fint = [torch.cat([t2.sum(dim=2) for t2 in t1], dim=0).flatten(0,1) for t1 in self.Fint]
            
            else:
                # Layer Parameters
                self.weights = nn.ParameterList([nn.Parameter(
                                            torch.randn(sum(self.in_channels) * len(self.in_channels) * n_radius, self.out_channels[l], dtype = torch.cfloat))
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
                                            torch.randn(self.out_channels[l], 1, self.in_channels[l1], len(self.in_channels)*n_radius, dtype = torch.cfloat))
                                            for l in range(len(self.out_channels)) for l1 in range(len(self.in_channels))])
                self.Fint = [[t2.flatten(-3).flatten(1,2).transpose(1,2).unsqueeze(1) for t2 in t1] for t1 in self.Fint]

        
    def forward(self, x):
        result = gelib.SO3vecArr()

        if type(x) is not gelib.SO3vecArr:
            inputs = gelib.SO3vecArr()
            inputs.parts.append(x.unsqueeze(4))
        else:
            inputs = x

        if self.conv_first:
            batch_size = inputs.parts[0].shape[0]
            parts = []
            
            # Convolution
            for l1, (part, c) in enumerate(zip(inputs.parts, self.in_channels)):
                part = part.permute(0,5,4,1,2,3).flatten(0,1)
                part = torch.conv3d(part, self.Fint[l1].to(part.device), stride=self.stride, padding = self.padding, dilation=self.dilation)
                part = part.reshape(batch_size, c, len(self.out_channels)**2, -1, *part.shape[-3:]).permute(0,4,5,6,2,1,3).flatten(-2)
                parts.append(part)
            parts = torch.cat(parts, dim = -1)
            
            # Weight Multiplication
            for l, c in enumerate(self.out_channels):
                result.parts.append((parts[..., l**2:(l+1)**2,:] @ self.weights[l]))
            
        else:
            kernel = self.get_kernel()
            x, _ = merge_channel_dim(inputs.parts, channel_last=True)
            x = torch.conv3d(x.permute(0,4,1,2,3), kernel, stride=self.stride, padding=self.padding, dilation=self.dilation).permute(0,2,3,4,1)
            result.parts = split_channel_dim(x, self.out_channels, channel_last=True)
            
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

class CGNonLinearity3D(nn.Module):
    def __init__(self, in_channels):
        super(CGNonLinearity3D, self).__init__()

        self.in_channels = [in_channels] if type(in_channels) is not list and type(in_channels) is not tuple else in_channels
        self.maxl = len(in_channels) - 1
        size = [sum([int(abs(l1-l2) <= l<= l1+l2)  for l1 in range(self.maxl+1) for l2 in range(self.maxl+1)]) for l in range(self.maxl+1)]
        self.weights = nn.ParameterList([nn.Parameter(
                                        torch.randn(in_channels[l] * size[l], in_channels[l], dtype = torch.cfloat))
                                         for l in range(self.maxl + 1)])
    def forward(self, x):
        x = gelib.DiagCGproduct(x,x,self.maxl)
        x.parts = [x.parts[l] @ self.weights[l] for l in range(self.maxl+1)]
        return x

class HNonLinearity3D(nn.Module):
    def __init__(self, in_channels):
        super(HNonLinearity3D, self).__init__()
        
        self.non_linearity = torch.nn.ReLU()
        self.eps = 1e-5
        self.in_channels = [in_channels] if type(in_channels) is not list and type(in_channels) is not tuple else in_channels
        self.b = nn.ParameterList([nn.Parameter(
                                    torch.randn(self.in_channels[l], dtype = torch.float))
                                  for l in range(len(self.in_channels))])
    def forward(self, x):
        result = gelib.SO3vecArr()
        for l,part in enumerate(x.parts):
            magnitude = torch.linalg.vector_norm(part, dim=-2, keepdim=True) + self.eps
            factor = self.non_linearity(magnitude + self.b[l]) / magnitude
            result.parts.append(part * factor)
        
        return result
    
class GatedNonLinearity3D(nn.Module):
    def __init__(self, in_channels, kernel_size, n_radius, n_theta):
        super(GatedNonLinearity3D, self).__init__()
        
        self.non_linearity = torch.nn.Sigmoid()
        self.layer = SE3Conv(in_channels, sum(in_channels), kernel_size, n_radius, n_theta, padding='same')
        self.in_channels = self.layer.in_channels
        
    def forward(self, x):
        factor = self.non_linearity(self.layer(x).parts[0])
        result = gelib.SO3vecArr()
        split_index = 0
        for l,part in enumerate(x.parts):
            magnitude = factor[..., split_index: split_index + self.in_channels[l]]
            result.parts.append(magnitude * part)
            split_index += self.in_channels[l]
        
        return result
            
            
#######################################################################################################################
################################################### Batch Normalization ###############################################
#######################################################################################################################

class SteerableBatchNorm3D(nn.Module):
    def __init__(self):
        super(SteerableBatchNorm3D, self).__init__()
        self.eps = 1e-5

    def forward(self, x):
        result = gelib.SO3vecArr()
        for part in x.parts:
            magnitude = torch.linalg.vector_norm(part, dim=-2, keepdim=True) + self.eps
            part = (part.real/magnitude) + 1j*(part.imag/magnitude)
            result.parts.append(part)
        
        return result
    
#######################################################################################################################
################################################## Average Pooling Layer ##############################################
#######################################################################################################################  

class SE3AvgPool(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SE3AvgPool, self).__init__()
        self.pool = nn.AvgPool3d(*args, **kwargs)

    def forward(self, x):
        result = gelib.SO3vecArr()
        x, channels = merge_channel_dim(x.parts, channel_last=True)
        x = x.permute(0,4,1,2,3)
        x = (self.pool(x.real) + 1j*self.pool(x.imag)).permute(0,2,3,4,1)
        result.parts = split_channel_dim(x, channels, channel_last=True)
        return result


#######################################################################################################################
################################################### Invariant Layers ##################################################
#######################################################################################################################

class NormFlatten(nn.Module):
    def __init__(self):
        super(NormFlatten, self).__init__()

    def forward(self, x):
        parts = []
        for part in x.parts:
            part = torch.mean(part, dim = (1,2,3))
            part = torch.linalg.vector_norm(part, dim = (1,))
            parts.append(part)
            
        result = torch.cat(parts, dim=1)
        return result
