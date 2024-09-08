import torch
import torch.nn as nn
from numpy import sqrt, prod
from Steerable.nn.Steerable2d.utils import get_Fint_matrix, get_CG_matrix

#######################################################################################################################
################################################ Convolution Layers ###################################################
#######################################################################################################################

class _SE2Conv(nn.Module):
    '''
    SE(2) Convolution Base Class.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_theta, max_m, 
                 interpolation_type='linear',dilation=1, padding=0, stride=1, conv_first=False):
        super(_SE2Conv, self).__init__()

        # Layer Design
        self.kernel_size = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
        self.dilation = dilation if type(dilation) is tuple else (dilation, dilation)
        self.padding = padding if type(padding) is tuple or type(padding) is str else (padding, padding)
        self.stride = stride if type(stride) is tuple else(stride, stride)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_m = max_m
        self.n_radius = n_radius
        self.n_theta = n_theta
        self.conv_first = conv_first
        self.groups = 1

        # Layer Parameters
        self.weights = None
        self.kernel = None

        # Fint Matrix
        self.Fint = get_Fint_matrix(self.kernel_size, n_radius, n_theta, self.max_m, interpolation_type)
        

    def forward(self, x):
        if x.shape[-3] != self.in_channels:
            raise ValueError(f"Number of channels in the input ({x.shape[2]}) must match in_channels ({self.in_channels}).")    
        
       
        if self.conv_first:
            # Convolution
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1, self.in_channels, *x.shape[-2:]).transpose(1,2).flatten(0,1)
            x = torch.conv2d(x, self.Fint.to(x.device), stride=self.stride, padding = self.padding, dilation=self.dilation, groups=self.groups)
            out_res = x.shape[-2:]
            x = x.reshape(batch_size, -1, self.max_m, self.n_radius, prod(out_res)).transpose(1,2).flatten(2,3)
            
            # Weight Multiplication
            x = self.weights @ x
            x = x.reshape(batch_size, self.max_m, self.out_channels, *out_res)
            
        else:
            # Steerable Kernel
            self.kernel = self.get_kernel()
            
            # Convolution
            x = x.reshape(x.shape[0], -1, *x.shape[-2:])
            x = torch.conv2d(x, self.kernel.to(x.device), stride=self.stride, padding = self.padding, dilation=self.dilation)
            x = x.reshape(x.shape[0], self.max_m, self.out_channels, *x.shape[-2:])
            

        return x
    
    def get_kernel(self):
        # Kernel Preparation
        kernel = (self.weights @ self.Fint.to(self.weights.device)).reshape(self.max_m * self.out_channels, -1, *self.kernel_size)
        return kernel

class SE2ConvType1(_SE2Conv):
    '''
    SE(2) Convolution in first Layer.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_theta, max_m, 
                 interpolation_type='linear', dilation=1, padding=0, stride=1, conv_first=False):
        super(SE2ConvType1, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           n_radius=n_radius, n_theta=n_theta, max_m=max_m, 
                                           interpolation_type=interpolation_type,
                                           dilation=dilation, padding=padding, stride=stride, conv_first=conv_first)

        # Layer Parameters
        if self.conv_first:
            self.weights = nn.Parameter(torch.randn(max_m, out_channels, in_channels * n_radius, dtype=torch.cfloat))
            self.Fint = self.Fint.reshape(self.max_m*n_radius, 1, *self.kernel_size)
        else:
            self.weights = nn.Parameter(torch.randn(max_m, out_channels * in_channels, n_radius, dtype=torch.cfloat))
            self.Fint = self.Fint.reshape(self.max_m, n_radius, -1)
        

    
class SE2ConvType2(_SE2Conv):
    '''
    SE(2) Convolution in higher Layer.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_theta, max_m, 
                 interpolation_type='linear',  dilation=1, padding=0, stride=1, restricted=False, conv_first=False) -> None:
        super(SE2ConvType2, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           n_radius=n_radius, n_theta=n_theta, max_m=max_m, 
                                           interpolation_type=interpolation_type,
                                           dilation=dilation, padding=padding, stride=stride, conv_first=conv_first)
        # CFint Matrix
        CG_Matrix = get_CG_matrix(max_m=max_m)
        self.Fint = torch.einsum('lmn, nrxy -> lmrxy', CG_Matrix, self.Fint)
        
        if conv_first:
            self.Fint = self.Fint.transpose(1,2).flatten(0,1)
            if restricted:
                self.weights = nn.Parameter(torch.randn(max_m, out_channels, in_channels * n_radius, dtype=torch.cfloat))
            else:
                self.weights = nn.Parameter(torch.randn(max_m, out_channels, max_m * in_channels * n_radius, dtype=torch.cfloat))
                self.Fint = self.Fint.transpose(0,1).flatten(0,1).unsqueeze(1)
                self.groups = self.max_m
        else:
            self.Fint = self.Fint.reshape(max_m, 1, max_m, n_radius, -1)
            if restricted:
                self.weights = nn.Parameter(torch.randn(max_m, out_channels, 1, in_channels, n_radius, dtype=torch.cfloat))
            else:
                self.weights = nn.Parameter(torch.randn(max_m, out_channels, max_m, in_channels, n_radius, dtype=torch.cfloat))
        
        
        
#######################################################################################################################
############################################## DeConvolution Layers ###################################################
#######################################################################################################################

class _SE2DeConv(_SE2Conv):
    '''
    SE(2) DeConvolution Base Class.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_theta, max_m, 
                 interpolation_type='linear',  dilation=1, padding=0, stride=1, output_padding=0) -> None:
        super(_SE2DeConv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           n_radius=n_radius, n_theta=n_theta, max_m=max_m, 
                                           interpolation_type=interpolation_type,
                                           dilation=dilation, padding=padding, stride=stride)

        # Layer Design
        self.output_padding = stride if type(output_padding) is tuple else(output_padding, output_padding)
        

    def forward(self, x):
        if x.shape[-3] != self.in_channels:
            raise ValueError(f"Number of channels in the input ({x.shape[2]}) must match in_channels ({self.in_channels}).")    
        
        # Convolution
        x = x.reshape(x.shape[0], -1, *x.shape[-2:])
        x = torch.nn.functional.conv_transpose2d(x, self.kernel.to(x.device).transpose(0,1), stride=self.stride, padding = self.padding, 
                                                 dilation=self.dilation, output_padding=self.output_padding)
        x = x.reshape(x.shape[0], self.max_m, self.out_channels, *x.shape[-2:])

        return x

class SE2DeConvType1(_SE2DeConv, SE2ConvType1):
    '''
    SE(2) Convolution in first Layer.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_theta, max_m, 
                 interpolation_type='linear',  dilation=1, padding=0, stride=1, output_padding=0,
                 restricted=False) -> None:
        
        _SE2DeConv.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           n_radius=n_radius, n_theta=n_theta, max_m=max_m, 
                                           interpolation_type=interpolation_type,
                                           dilation=dilation, padding=padding, stride=stride, output_padding=output_padding)
        
        SE2ConvType1.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           n_radius=n_radius, n_theta=n_theta, max_m=max_m, 
                                           interpolation_type=interpolation_type,
                                           dilation=dilation, padding=padding, stride=stride, restricted=restricted)

    
class SE2DeConvType2(_SE2DeConv, SE2ConvType2):
    '''
    SE(2) Convolution in higher Layer.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_theta, max_m, 
                 interpolation_type='linear',  dilation=1, padding=0, stride=1, output_padding=0,
                 restricted=False) -> None:
        
        _SE2DeConv.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           n_radius=n_radius, n_theta=n_theta, max_m=max_m, 
                                           interpolation_type=interpolation_type,
                                           dilation=dilation, padding=padding, stride=stride, output_padding=output_padding)
        
        SE2ConvType2.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           n_radius=n_radius, n_theta=n_theta, max_m=max_m, 
                                           interpolation_type=interpolation_type,
                                           dilation=dilation, padding=padding, stride=stride, restricted=restricted)
            
   
   
#######################################################################################################################
################################################### Non-linearity #####################################################
#######################################################################################################################

class SE2CGNonLinearity(nn.Module):
    '''
    The Clebsch Gordan Non-Linearity Module
    '''
    def __init__(self, max_m):
        super(SE2CGNonLinearity, self).__init__()
        self.max_m = max_m
        self.CG_Matrix = get_CG_matrix(max_m)
        
    def forward(self, x):
        CG_Matrix = self.CG_Matrix.to(x.device)
        x = torch.einsum('lmn, bmoxy, bnoxy -> bloxy', CG_Matrix, x,x)
        return x


class SE2NonLinearity(nn.Module):
    '''
    This module takes the tensor to the Time domain, performs a non-linearity
    there and then transforms it back to Fourier domain.
    The default non-linearity is ReLU.
    '''
    def __init__(self, max_m, nonlinearity = nn.ReLU()):
        super(SE2NonLinearity, self).__init__()
        self.FT = torch.fft.fft(torch.eye(max_m, max_m)) / sqrt(max_m)
        self.IFT = torch.fft.ifft(torch.eye(max_m, max_m)) / sqrt(max_m)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        x_shape = x.shape
        FT = self.FT.to(x.device)
        IFT = self.IFT.to(x.device)
        
        x = IFT @ x.flatten(2)
        x = self.nonlinearity(x.real) + 1j*self.nonlinearity(x.imag)
        x = (FT @ x).reshape(*x_shape)
        return x
    

class SE2NormNonLinearity(nn.Module):
    '''
    This module takes the tensor applies non-linearity to absolute value of the tensor
    The default non-linearity is ReLU.
    '''
    def __init__(self, in_channels, max_m, nonlinearity = torch.nn.ReLU()):
        super(SE2NormNonLinearity, self).__init__()
        self.eps = 1e-5
        self.nonlinearity = nonlinearity
        self.b = nn.Parameter(torch.randn(max_m, in_channels))

    def forward(self, x):
        magnitude = x.abs() + self.eps
        b = self.b.reshape(1,*self.b.shape, *[1]*len(x.shape[3:]))
        factor = self.nonlinearity(magnitude + b) / magnitude
        x = factor * x
        return x
    
#######################################################################################################################
################################################### Batch Normalization ###############################################
#######################################################################################################################

class SE2BatchNorm(nn.Module):
    def __init__(self):
        super(SE2BatchNorm, self).__init__()
        self.eps = 1e-5

    def forward(self, x):
        #factor = x.abs() + self.eps
        factor = torch.linalg.vector_norm(x, dim = (1,), keepdim=True) + self.eps
        x = (x.real/factor) + 1j*(x.imag/factor)

        return x
    
#######################################################################################################################
################################################## Average Pooling Layer ##############################################
#######################################################################################################################  

class SE2AvgPool(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SE2AvgPool, self).__init__()
        self.pool = nn.AvgPool2d(*args, **kwargs)

    def forward(self, x):
        x_shape = x.shape
        x = self.pool(x.real.flatten(1,2)) + 1j*self.pool(x.imag.flatten(1,2))
        x = x.reshape(*x_shape[:3], *x.shape[2:])
        
        return x


#######################################################################################################################
################################################### Invariant Layers ##################################################
#######################################################################################################################

class SE2NormFlatten(nn.Module):
    def __init__(self):
        super(SE2NormFlatten, self).__init__()

    def forward(self, x):
        x = torch.mean(x.flatten(3), dim = (3,))
        x = torch.linalg.vector_norm(x, dim = (1,))
        return x

class SE2Pooling(nn.Module):
    def __init__(self):
        super(SE2Pooling, self).__init__()

    def forward(self, x):
        x = torch.mean(x.flatten(3), dim = (3,))
        x = torch.sum(x.abs(), dim = 1)
        return x
