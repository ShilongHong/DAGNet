import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class VanillaModel(nn.Module):

    def __init__(self, num_classes=15, dropout=0.3, backbone="resnet"):

        super(VanillaModel, self).__init__()
        if backbone == "resnet":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)


            self.features = nn.Sequential(resnet.conv1,
                                        resnet.bn1,
                                        resnet.relu,
                                        resnet.maxpool)
                
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

            output_channels = [256, 512, 1024, 2048]
        elif backbone == "resnext":
            resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(resnext.conv1,
                                        resnext.bn1,
                                        resnext.relu,
                                        resnext.maxpool)
                
            self.layer1 = resnext.layer1
            self.layer2 = resnext.layer2
            self.layer3 = resnext.layer3
            self.layer4 = resnext.layer4

            output_channels = [256, 512, 1024, 2048]
        elif backbone == "regnet":
            regnet = models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(
                regnet.stem[0],  # conv
                regnet.stem[1],  # bn
                regnet.stem[2],  # act
            )
            self.layer1 = regnet.trunk_output.block1
            self.layer2 = regnet.trunk_output.block2
            self.layer3 = regnet.trunk_output.block3
            self.layer4 = regnet.trunk_output.block4
            output_channels = [96, 192, 432, 1008]
        elif backbone == "convnext":
            backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
            backbone = backbone.features
            self.features = nn.Sequential(
                backbone[0],
            )
            self.layer1 = backbone[1]
            self.layer2 = backbone[2:4]
            self.layer3 = backbone[4:6]
            self.layer4 = backbone[6:]
            output_channels = [96, 192, 384, 768]
        else:
            raise ValueError("Unsupported backbone: {}".format(backbone))
        output_dim = output_channels[3]
        self.classifier = nn.Sequential(
            nn.AvgPool2d(8, stride=1),
            nn.Flatten(),
            nn.Linear(output_dim, num_classes)
        )

    def forward(self, image_ol, image_sd):
        # 提取OL图像特征
        bf_ol = self.features(image_ol)
        f_l1_o1 = self.layer1(bf_ol)
        f_l2_ol = self.layer2(f_l1_o1)
        f_l3_ol = self.layer3(f_l2_ol)
        f_l4_ol = self.layer4(f_l3_ol)

        # 提取SD图像特征
        bf_sd = self.features(image_sd)
        f_l1_sd = self.layer1(bf_sd)
        f_l2_sd = self.layer2(f_l1_sd)
        f_l3_sd = self.layer3(f_l2_sd)
        f_l4_sd = self.layer4(f_l3_sd)

        # 单独分类得到两个预测结果
        ol_output = self.classifier(f_l4_ol)
        sd_output = self.classifier(f_l4_sd)
        
        return ol_output, sd_output

if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaModel(num_classes=15).to(device)

    flops, params = profile(model, (torch.randn(1, 3, 224, 224).to(device), torch.randn(1, 3, 224, 224).to(device)))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops)
    print(params)
