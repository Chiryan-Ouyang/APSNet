import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import os
import torchvision.models as models


__all__ = ['APSnet18', 'APSnet34', 'APSnet50']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class de_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(de_conv, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//scale, in_ch//scale, kernel_size=3, stride=1, padding=1)

        self.conv = de_conv(in_ch, out_ch)
        self.dropout = nn.Dropout()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, math.ceil(diffY / 2),
                   diffX // 2, math.ceil(diffX / 2)), "constant", 0)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.sigmoid(attn)
        return x * attn


class TransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_heads=8):
        super(TransformerBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Spatial reduction if needed (similar to stride in conv)
        self.reduction = None
        if stride != 1:
            self.reduction = nn.MaxPool2d(kernel_size=stride, stride=stride)
        
        # Channel adjustment
        self.conv_adjust = None
        if inplanes != planes:
            self.conv_adjust = nn.Sequential(
                conv1x1(inplanes, planes),
                norm_layer(planes)
            )
        
        # Multi-Head Self-Attention
        self.norm1 = norm_layer(planes if self.conv_adjust else inplanes)
        self.attention = nn.MultiheadAttention(
            embed_dim=planes if self.conv_adjust else inplanes, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # FFN
        self.norm2 = norm_layer(planes if self.conv_adjust else inplanes)
        self.ffn = nn.Sequential(
            nn.Conv2d(planes if self.conv_adjust else inplanes, 
                      (planes if self.conv_adjust else inplanes) * 4, 
                      kernel_size=1),
            nn.GELU(),
            nn.Conv2d((planes if self.conv_adjust else inplanes) * 4, 
                      planes if self.conv_adjust else inplanes, 
                      kernel_size=1)
        )
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        # Apply spatial reduction if needed
        if self.reduction is not None:
            x = self.reduction(x)
        
        # Apply channel adjustment if needed
        if self.conv_adjust is not None:
            x = self.conv_adjust(x)
        
        # First normalization
        x_norm = self.norm1(x)
        
        # Reshape for multi-head attention
        b, c, h, w = x_norm.shape
        x_reshape = x_norm.flatten(2).permute(0, 2, 1)  # B, HW, C
        
        # Self-attention
        attn_out, _ = self.attention(x_reshape, x_reshape, x_reshape)
        attn_out = attn_out.permute(0, 2, 1).reshape(b, c, h, w)
        
        # First residual connection
        x = x + attn_out
        
        # Second normalization
        x_norm = self.norm2(x)
        
        # FFN
        ffn_out = self.ffn(x_norm)
        
        # Second residual connection
        out = x + ffn_out
        
        # Downsample the identity if needed
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # Final residual connection
        out = out + identity
        
        return out


class APSnet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2, use_se=True, use_dual_path=True, num_heads=8):
        super(APSnet, self).__init__()
        self.inplanes = 64
        self.use_se = use_se
        self.use_dual_path = use_dual_path
        self.num_heads = num_heads
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # SE模块，只有在use_se为True时才使用
        if self.use_se:
            self.se1 = SEBlock(64)
            self.se2 = SEBlock(64)
            self.se3 = SEBlock(128)
            self.se4 = SEBlock(256)

        # 只在启用dual-path时创建解码器相关模块
        if self.use_dual_path:
            self.up4 = up(512 + 256, 256)
            self.up3 = up(256 + 128, 128)
            self.up2 = up(128 + 64, 64) 
            self.up1 = up(64 + 64, 64, 1)
            self.outconv = conv3x3(64, 1)

            # 多尺度特征融合模块
            self.seg_to_cls_256 = nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False)
            )
            self.seg_to_cls_128 = nn.Sequential(
                nn.Conv2d(128, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False)
            )
            self.seg_to_cls_64 = nn.Sequential(
                nn.Conv2d(64, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False)
            )

            # 多尺度注意力模块
            self.attn_256 = MultiScaleAttention(256)
            self.attn_128 = MultiScaleAttention(128)
            self.attn_64 = MultiScaleAttention(64)

            # 特征融合后的分类器
            self.cls_fusion = nn.Sequential(
                nn.Conv2d(128 * 3 + 512, 512, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=False)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if block == TransformerBlock:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, num_heads=self.num_heads))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, num_heads=self.num_heads))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()  # 保存输入尺寸
        # 编码器路径
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        
        if self.use_se:
            x1 = self.se1(x1)

        x2 = self.layer1(x1)
        if self.use_se:
            x2 = self.se2(x2)
            
        x3 = self.layer2(x2)
        if self.use_se:
            x3 = self.se3(x3)
            
        x4 = self.layer3(x3)
        if self.use_se:
            x4 = self.se4(x4)
            
        x5 = self.layer4(x4)

        if not self.use_dual_path:
            # 如果不使用dual-path，直接进行分类
            cls_feat = self.avgpool(x5)
            cls_feat = cls_feat.view(cls_feat.size(0), -1)
            cls_out = self.fc(cls_feat)
            # 创建一个与输入图像大小相同的全零张量作为分割输出
            seg_out = torch.zeros(input_size[0], 1, input_size[2] // 4 , input_size[3] // 4, device=x.device)
            return cls_out, seg_out

        # 以下是dual-path的处理逻辑
        # 解码器路径（分割）
        y4 = self.up4(x5, x4)  # 256 channels
        y3 = self.up3(y4, x3)  # 128 channels
        y2 = self.up2(y3, x2)  # 64 channels
        y1 = self.up1(y2, x1)  # 64 channels
        seg_out = self.outconv(y1)

        # 多尺度特征提取和注意力
        f4 = self.attn_256(y4)  # 256维特征
        f3 = self.attn_128(y3)  # 128维特征
        f2 = self.attn_64(y2)   # 64维特征

        # 特征转换到相同维度
        f4_trans = self.seg_to_cls_256(f4)
        f3_trans = self.seg_to_cls_128(f3)
        f2_trans = self.seg_to_cls_64(f2)

        # 将所有特征上采样到相同大小
        target_size = f2_trans.shape[2:]
        f4_up = F.interpolate(f4_trans, size=target_size, mode='bilinear', align_corners=True)
        f3_up = F.interpolate(f3_trans, size=target_size, mode='bilinear', align_corners=True)
        x5_up = F.interpolate(x5, size=target_size, mode='bilinear', align_corners=True)

        # 特征融合
        fused_features = torch.cat([x5_up, f4_up, f3_up, f2_trans], dim=1)
        fused_features = self.cls_fusion(fused_features)

        # 分类头
        cls_feat = self.avgpool(fused_features)
        cls_feat = cls_feat.view(cls_feat.size(0), -1)
        cls_out = self.fc(cls_feat)

        return cls_out, seg_out


def load_pretrained(model, premodel):
    pretrained_dict = premodel.state_dict()
    model_dict = model.state_dict()
      
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def APSnet18(pretrained=False, use_se=True, use_dual_path=True, block_type='resnet', num_heads=8):
    if block_type.lower() == 'transformer':
        model = APSnet(TransformerBlock, [2, 2, 2, 2], use_se=use_se, use_dual_path=use_dual_path, num_heads=num_heads)
    else:  # default to resnet
        model = APSnet(BasicBlock, [2, 2, 2, 2], use_se=use_se, use_dual_path=use_dual_path)
    
    if pretrained and block_type.lower() != 'transformer':
        premodel = models.resnet18(pretrained=True)
        premodel.fc = nn.Linear(512, 2)
        model = load_pretrained(model, premodel)
    return model


def APSnet34(pretrained=False, use_se=True, use_dual_path=True, block_type='resnet', num_heads=8):
    if block_type.lower() == 'transformer':
        model = APSnet(TransformerBlock, [3, 4, 6, 3], use_se=use_se, use_dual_path=use_dual_path, num_heads=num_heads)
    else:  # default to resnet
        model = APSnet(BasicBlock, [3, 4, 6, 3], use_se=use_se, use_dual_path=use_dual_path)
    
    if pretrained and block_type.lower() != 'transformer':
        premodel = models.resnet34(pretrained=True)
        premodel.fc = nn.Linear(512, 2)
        model = load_pretrained(model, premodel)
    return model


def APSnet50(pretrained=False, use_se=True, use_dual_path=True, block_type='resnet', num_heads=8):
    if block_type.lower() == 'transformer':
        model = APSnet(TransformerBlock, [4, 8, 8, 4], use_se=use_se, use_dual_path=use_dual_path, num_heads=num_heads)
    else:  # default to resnet
        model = APSnet(BasicBlock, [4, 8, 8, 4], use_se=use_se, use_dual_path=use_dual_path)
    
    if pretrained and block_type.lower() != 'transformer':
        premodel = models.resnet50(pretrained=True)
        premodel.fc = nn.Linear(512, 2)
        model = load_pretrained(model, premodel)
    return model


#if __name__ == '__main__':
#    images = torch.randn(1, 1, 256, 256)
#    model = APSnet18(pretrained=True)
#    out_class, out_seg = model(images)
#    print(out_class.shape, out_seg.shape)
