import torch
from math import sqrt, ceil
from Steerable.nn.utils import get_Fint_matrix, get_CFint_matrix, get_CG_matrix

#######################################################################################################################
################################################ Convolution Layers ###################################################
#######################################################################################################################

class _SE2Conv(torch.nn.Module):
    '''
    SE(2) Convolution Base Class.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, freq_cutoff, n_radius=None, n_angle=None, dilation=1, padding=0, stride=1):
        super(_SE2Conv, self).__init__()

        # Layer Design
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
        self.freq_cutoff = freq_cutoff
        self.n_radius = n_radius if n_radius else ceil(max(self.kernel_size) / 2)
        self.n_angle = n_angle if n_angle else freq_cutoff
        self.dilation = dilation if type(dilation) is tuple else (dilation, dilation)
        self.padding = padding if type(padding) is tuple or type(padding) is str else (padding, padding)
        self.stride = stride if type(stride) is tuple else(stride, stride)
        

    def forward(self, x):
        if x.shape[-3] != self.in_channels:
            raise ValueError(f"Number of channels in the input ({x.shape[2]}) must match in_channels ({self.in_channels}).")    
        
        # Steerable Kernel
        self.kernel = self.get_kernel()
        
        # Convolution
        x = x.reshape(x.shape[0], -1, *x.shape[-2:])
        x = torch.conv2d(x, self.kernel, stride=self.stride, padding = self.padding, dilation=self.dilation)
        x = x.reshape(x.shape[0], self.freq_cutoff, self.out_channels, *x.shape[-2:])

        return x
    
    def get_kernel(self):
        # Kernel Preparation
        kernel = (self.weights @ self.Fint).reshape(self.freq_cutoff * self.out_channels, -1, *self.kernel_size)
        return kernel

class SE2ConvType1(_SE2Conv):
    '''
    SE(2) Convolution in first Layer.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, freq_cutoff, n_radius=None, n_angle=None, dilation=1, padding=0, stride=1, interpolation_type=1):
        super(SE2ConvType1, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, freq_cutoff=freq_cutoff,
                                           n_radius=n_radius, n_angle=n_angle, dilation=dilation, padding=padding, stride=stride)
        
        # Fint Matrix
        Fint = get_Fint_matrix(self.kernel_size, self.n_radius, self.n_angle, self.freq_cutoff, interpolation_type)
        Fint = Fint.reshape(self.freq_cutoff, self.n_radius, -1)
        self.register_buffer("Fint", Fint, persistent=False)
        self.weights = torch.nn.Parameter(torch.randn(self.freq_cutoff, self.out_channels * self.in_channels, self.n_radius, dtype=torch.cfloat))

    
class SE2ConvType2(_SE2Conv):
    '''
    SE(2) Convolution in higher Layer.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, freq_cutoff, n_radius=None, n_angle=None, dilation=1, padding=0, stride=1, interpolation_type=1) -> None:
        super(SE2ConvType2, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, freq_cutoff=freq_cutoff,
                                           n_radius=n_radius, n_angle=n_angle, dilation=dilation, padding=padding, stride=stride)
        # CFint Matrix
        Fint = get_CFint_matrix(self.kernel_size, self.n_radius, self.n_angle, self.freq_cutoff, interpolation_type)
        Fint = Fint.reshape(self.freq_cutoff, 1, self.freq_cutoff, self.n_radius, -1)
        self.register_buffer("Fint", Fint, persistent=False)
        self.weights = torch.nn.Parameter(torch.randn(self.freq_cutoff, self.out_channels, self.freq_cutoff, self.in_channels, self.n_radius, dtype=torch.cfloat))
        
        
        
#######################################################################################################################
############################################## DeConvolution Layers ###################################################
#######################################################################################################################

class _SE2DeConv(_SE2Conv):
    '''
    SE(2) DeConvolution Base Class.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, n_radius, n_angle, freq_cutoff, 
                 dilation=1, padding=0, stride=1, output_padding=0) -> None:
        super(_SE2DeConv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, freq_cutoff=freq_cutoff, 
                                           n_radius=n_radius, n_angle=n_angle, dilation=dilation, padding=padding, stride=stride)
        # Layer Design
        self.output_padding = stride if type(output_padding) is tuple else(output_padding, output_padding)
        
    def forward(self, x):
        if x.shape[-3] != self.in_channels:
            raise ValueError(f"Number of channels in the input ({x.shape[2]}) must match in_channels ({self.in_channels}).")    
        
        # Convolution
        x = x.reshape(x.shape[0], -1, *x.shape[-2:])
        x = torch.nn.functional.conv_transpose2d(x, self.kernel.transpose(0,1), stride=self.stride, padding = self.padding, 
                                                 dilation=self.dilation, output_padding=self.output_padding)
        x = x.reshape(x.shape[0], self.freq_cutoff, self.out_channels, *x.shape[-2:])

        return x

class SE2DeConvType1(_SE2DeConv, SE2ConvType1):
    '''
    SE(2) Convolution in first Layer.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, freq_cutoff, n_radius=None, n_angle=None, dilation=1, padding=0, stride=1, output_padding=0, interpolation_type=1):
        
        _SE2DeConv.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, freq_cutoff=freq_cutoff,
                            n_radius=n_radius, n_angle=n_angle, dilation=dilation, padding=padding, stride=stride, output_padding=output_padding)
        
        SE2ConvType1.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, freq_cutoff=freq_cutoff, 
                              n_radius=n_radius, n_angle=n_angle, interpolation_type=interpolation_type, dilation=dilation, padding=padding, stride=stride)

    
class SE2DeConvType2(_SE2DeConv, SE2ConvType2):
    '''
    SE(2) Convolution in higher Layer.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, freq_cutoff, 
                 n_radius=None, n_angle=None, interpolation_type=1,  dilation=1, padding=0, stride=1, output_padding=0) -> None:
        
        _SE2DeConv.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, freq_cutoff=freq_cutoff,
                            n_radius=n_radius, n_angle=n_angle, dilation=dilation, padding=padding, stride=stride, output_padding=output_padding)
        
        SE2ConvType2.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, freq_cutoff=freq_cutoff, 
                              n_radius=n_radius, n_angle=n_angle, dilation=dilation, padding=padding, stride=stride, interpolation_type=interpolation_type)
        
#######################################################################################################################
################################################### Non-linearity #####################################################
#######################################################################################################################

class SE2CGNonLinearity(torch.nn.Module):
    '''
    The Clebsch Gordan Non-Linearity Module
    '''
    def __init__(self, freq_cutoff, n_angle=None):
        super(SE2CGNonLinearity, self).__init__()
        self.freq_cutoff = freq_cutoff
        CG_Matrix = torch.tensor(get_CG_matrix(dimension=2, freq_cutoff=freq_cutoff, n_angle=n_angle), dtype=torch.cfloat)
        self.register_buffer("CG_Matrix", CG_Matrix, persistent=False)
        self.weight = torch.nn.Parameter(torch.randn(freq_cutoff, freq_cutoff, dtype=torch.cfloat))
        
    def forward(self, x):
        x = torch.einsum('lm, lmn, bmoxy, bnoxy -> bloxy', self.weight, self.CG_Matrix, x,x)
        return x


class SE2NonLinearity(torch.nn.Module):
    '''
    This module takes the tensor to the Time domain, performs a non-linearity
    there and then transforms it back to Fourier domain.
    The default non-linearity is ReLU.
    '''
    def __init__(self, freq_cutoff, n_angle = None, nonlinearity = torch.nn.ReLU()):
        super(SE2NonLinearity, self).__init__()
        self.n_angle = n_angle if n_angle else freq_cutoff
        self.nonlinearity = nonlinearity
        
        self.register_buffer("IFT", torch.fft.ifft(torch.eye(self.n_angle, freq_cutoff), dim=0, norm='ortho'), persistent=False)
        self.register_buffer("FT", torch.fft.fft(torch.eye(freq_cutoff, self.n_angle), norm='ortho'), persistent=False)
        

    def forward(self, x):
        x_shape = x.shape
        x = self.IFT @ x.flatten(2)
        x = self.nonlinearity(x.real) + 1j*self.nonlinearity(x.imag)
        x = (self.FT @ x).reshape(*x_shape)
        return x
    

class SE2NormNonLinearity(torch.nn.Module):
    '''
    This module takes the tensor applies non-linearity to absolute value of the tensor
    The default non-linearity is GELU.
    '''
    def __init__(self, in_channels, freq_cutoff, nonlinearity = torch.nn.ReLU()):
        super(SE2NormNonLinearity, self).__init__()
        self.eps = 1e-5
        self.nonlinearity = nonlinearity
        self.b = torch.nn.Parameter(torch.randn(freq_cutoff, in_channels))

    def forward(self, x):
        magnitude = x.abs() + self.eps
        b = self.b.reshape(1,*self.b.shape, *[1]*len(x.shape[3:]))
        factor = self.nonlinearity(magnitude + b) / magnitude
        x = (x.real * factor) + 1j*(x.imag * factor)
        return x
    
#######################################################################################################################
################################################### Batch Normalization ###############################################
#######################################################################################################################

class SE2BatchNorm(torch.nn.Module):
    def __init__(self):
        super(SE2BatchNorm, self).__init__()
        self.eps = 1e-5

    def forward(self, x):
        factor = torch.linalg.vector_norm(x, dim = (1,), keepdim=True) + self.eps
        x = (x.real/factor) + 1j*(x.imag/factor)

        return x

#######################################################################################################################
################################################## Average Pooling Layer ##############################################
#######################################################################################################################  

class SE2AvgPool(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(SE2AvgPool, self).__init__()
        self.pool = torch.nn.AvgPool2d(*args, **kwargs)

    def forward(self, x):
        x_shape = x.shape
        x = self.pool(x.real.flatten(1,2)) + 1j*self.pool(x.imag.flatten(1,2))
        x = x.reshape(*x_shape[:3], *x.shape[2:])
        
        return x


#######################################################################################################################
################################################### Invariant Layers ##################################################
#######################################################################################################################

class SE2NormFlatten(torch.nn.Module):
    def __init__(self):
        super(SE2NormFlatten, self).__init__()

    def forward(self, x):
        x = torch.mean(x.flatten(3), dim = (3,))
        x = torch.linalg.vector_norm(x, dim = (1,))
        return x

class SE2Pooling(torch.nn.Module):
    def __init__(self):
        super(SE2Pooling, self).__init__()

    def forward(self, x):
        x = torch.mean(x.flatten(3), dim = (3,))
        x = torch.max(x.abs(), dim = 1)[0]
        return x

#######################################################################################################################
################################################### Complex Dropout ###################################################
#######################################################################################################################

class ComplexDropout(torch.nn.Module):
    def __init__(self, p: float = 0.1, inplace: bool = False):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"dropout probability has to be in [0,1), got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        if not torch.is_complex(x):
            return torch.nn.functional.dropout(x, self.p, training=True, inplace=self.inplace)

        mask = (torch.rand_like(x.real) > self.p).to(x.dtype) / (1 - self.p)
        if self.inplace:
            return x.mul_(mask)
        return x * mask

    def extra_repr(self):
        return f"p={self.p}, inplace={self.inplace}"