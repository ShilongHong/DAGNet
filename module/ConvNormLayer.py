import torch
from torch import nn
from .utils import get_activation

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class DWConv(nn.Module):
    """Depth-wise layer with optional pooling and activation."""
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=None, bias=False, act=None):
        """
        Initialize a Depth-wise layer.

        Args:
            ch_in (int): Number of input channels.
            ch_out (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel. Default is 3.
            stride (int): Stride of the convolution. Default is 1.
            padding (int, optional): Padding for the convolution. Auto-calculated if None.
            bias (bool): Whether to use bias in convolution. Default is False.
            act (str or nn.Module, optional): Activation function. Default is None (Identity).
            use_wt (bool): Placeholder for potential weight usage. Default is False.
        """
        super().__init__()
        # Auto padding if not specified
        padding = (kernel_size - 1) // 2 if padding is None else padding
        
        # Depth-wise convolution
        self.Dconv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, 
                               groups=ch_in, bias=bias)
        # Point-wise convolution
        self.Wconv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=bias)
        
        # Batch normalization
        self.norm = nn.BatchNorm2d(ch_out)
        
        # Activation function
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        """
        Forward pass through the Depth-wise layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        x = self.Dconv(x)  # Depth-wise convolution
        x = self.Wconv(x)  # Point-wise convolution
        x = self.norm(x)   # Batch normalization
        x = self.act(x)    # Activation
        return x