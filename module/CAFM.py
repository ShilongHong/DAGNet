from .ConvNormLayer import ConvNormLayer, DWConv
from .CBAM import CBAM
import torch
import torch.nn as nn

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu",
                 useglu=False,
                 use_wt=False,
                 use_Dcn=False):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, 
                                   bias=bias, act=act)
        self.bottlenecks = DWConv(in_channels, hidden_channels)
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, 
                                       bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        # x = self.conv1(x)
        x_1 = self.bottlenecks(x)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

class CAFM(nn.Module):
    def __init__(self, int_c, out_c, depth_mult=1, act='silu', expansion=1):
        super(CAFM, self).__init__()
        self.attn = CBAM(int_c//2)     
        self.bn1 = nn.BatchNorm2d(int_c)
        self.csp = CSPRepLayer(int_c, out_c, 1, act=act, expansion=expansion, use_Dcn=False, use_wt=False)

    def forward(self, h_x, l_x):
        h_x = self.attn(h_x)
        l_x = self.attn(l_x)
        res = torch.cat((h_x, l_x), dim=1).contiguous()
        
        res = self.csp(res)
        return res
        