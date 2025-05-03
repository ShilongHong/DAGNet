import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        return x


class FDIM2(nn.Module):
    def __init__(self, channel, ):
        super(FDIM, self).__init__()

        self.freq_ehance = DynamicFilter(channel, size=64)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x, y):
        _, C, H, W = x.shape

        x = self.freq_ehance(x) + x
        y = self.freq_ehance(y) + y
        return x, y


class DynamicFilterV2(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        self.pwconv0 = nn.Linear(dim, dim, bias=bias)

        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()

        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)

        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)

        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x, y):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape

        x_mean = x.mean(dim=(1, 2))  # [B, C]
        y_mean = y.mean(dim=(1, 2))  # [B, C]
        x_mean = x_mean * torch.sigmoid(self.pwconv0(y_mean)) + x_mean

        routing_weights = self.reweight(x_mean).view(B, self.num_filters, -1).softmax(dim=1)

        complex_weights = torch.view_as_complex(self.complex_weights)
        routing_weights = routing_weights.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routing_weights, complex_weights)
        weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        x_enhanced = self._apply_frequency_enhancement(x, weight, H, W)

        return x_enhanced

    def _apply_frequency_enhancement(self, input_tensor, weight, H, W):
        freq = self.pwconv1(input_tensor)
        freq = self.act1(freq)
        freq = freq.to(torch.float32)

        freq = torch.fft.rfft2(freq, dim=(1, 2), norm='ortho')

        freq = freq * weight

        freq = torch.fft.irfft2(freq, s=(H, W), dim=(1, 2), norm='ortho')
        freq = self.act2(freq)
        freq = self.pwconv2(freq)

        freq = freq.permute(0, 3, 1, 2)  # [B, C, H, W]
        return freq


class FDIM(nn.Module):

    def __init__(self, channel, size=64):
        super(FDIM, self).__init__()
        self.freq_enhance = DynamicFilterV2(channel, size=size)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x, y):
        x_enhanced = self.freq_enhance(x, y) + x
        y_enhanced = self.freq_enhance(y, x) + y

        return x_enhanced, y_enhanced


if __name__ == '__main__':
    batch_size, channels, height, width = 3, 32, 64, 64

    model = FDIM(channels, size=height)
    x_input = torch.rand(batch_size, channels, height, width)
    y_input = torch.rand(batch_size, channels, height, width)

    x_output, y_output = model(x_input, y_input)

    print(f"输入尺寸: x={x_input.size()}, y={y_input.size()}")
    print(f"输出尺寸: x_out={x_output.size()}, y_out={y_output.size()}")