import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from .vmamba import *


import matplotlib.pyplot as plt
import numpy as np

#Conv Block
class DoubleConv(nn.Module):
    def __init__(self, cin, cout, k=3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, padding=k//2),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=k, padding=k//2),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H=None, W=None):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x


class FFTM(nn.Module):
    def __init__(self, dim, h=16, w=16) -> None:
        super().__init__()
        self.dwconv = DWConv(dim*2)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=1),
            nn.BatchNorm2d(dim*2),
            nn.GELU()
        )
        self.complex_weight = nn.Parameter(torch.randn(2, dim, h, w, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        # x1, x2 = torch.chunk(x, 2, dim=1)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        xreal, ximag = x.real, x.imag
        xstack = torch.stack((xreal, ximag), dim=1).view((x.shape[0], -1, x.shape[-2], x.shape[-1]))
        xstack_ = self.conv1x1(xstack).view((x.shape[0], 2, x.shape[1], x.shape[2], x.shape[3])).permute(0,2,3,4,1)
        xstack__ = torch.complex(xstack_[...,0], xstack_[...,1])
        weight = F.interpolate(self.complex_weight, size=x.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        x = xstack__ * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        
        
        return x

class FM(nn.Module):
    def __init__(self, 
                    dim,
                    out_dim,
                    attn_drop=0.,
                    drop_path=0.,
                    drop_rate=0., 
                    attn_drop_rate=0.,
                    drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, 
                    downsample=None, 
                    upsample=None,
                    use_checkpoint=False, 
                    d_state=16,
                    **kwargs) -> None:
        super().__init__()
        self.fftm = FFTM(dim//2)
        self.norm1 = nn.LayerNorm(dim)
        self.down = downsample
        self.up = upsample
        self.downd = nn.MaxPool2d(self.down)
        self.norm2 = nn.LayerNorm(dim)


        self.patch_embed = PatchEmbed2D(patch_size=2, in_chans=dim//2, embed_dim=dim//2,
            norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.vss = VSSLayer(dim//2, depth=1)
        self.prj = nn.Conv2d(dim,out_dim,1)
        self.mlp = DWConv(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        input = x
        x = self.norm1(x.permute(0,2,3,1)).permute(0, 3, 1, 2)

        x1, x2 = torch.chunk(x, 2, dim=1)

        if self.down is not None:
            # x1 = self.downd(x1)
            x2 = self.downd(x2)
            input = self.downd(input)
            
        x1 = self.patch_embed(x1)
        x1 = self.pos_drop(x1)
        x1 = self.vss(x1).permute(0, 3, 1, 2)

        x2 = self.fftm(x2)


        x = torch.cat((x1, x2), dim=1)
        x = x + input

        input2 = x
        x = self.norm2(x.permute(0,2,3,1)).permute(0, 3, 1, 2)
        x = self.mlp(x)
        x = x + input2
        x = self.prj(x)

        return x

#ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
#频域SS2D
class FreqVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim*2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        B, H, W, C = input.shape
        x_1 = torch.fft.rfft2(input, dim=(1, 2), norm='ortho')                 # [B, H, W//2+1, C]
        x_1 = torch.stack((x_1.real, x_1.imag), dim=4).view((x_1.shape[0], x_1.shape[1], x_1.shape[2], -1))
        att = self.self_attention(x_1).view((x_1.shape[0], x_1.shape[1], x_1.shape[2], -1, 2))
        att = torch.complex(att[...,0], att[...,1])
        x_1 = torch.fft.irfft2(att, s=(H, W), dim=(1, 2), norm='ortho')
        x = input + self.drop_path(x_1)

        return x

#频空域SS2D
class FreqVSS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.SpaVss = VSSBlock(dim)
        self.FreqVss = FreqVSSBlock(dim)
        # self.patch_embed = PatchEmbed2D(patch_size=1, in_chans=dim, embed_dim=dim,
        #     norm_layer=nn.LayerNorm)
        # self.pos_drop = nn.Dropout(p=0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        input = x
        # x = self.patch_embed(x)
        # x = self.pos_drop(x)

        x_spa = self.norm1(x.permute(0, 2, 3, 1))
        # print("Xs1:", x_spa.shape)
        x_spa = self.SpaVss(x_spa).permute(0, 3, 1, 2)
        # print("Xs2:", x_spa.shape)
        x_freq = self.norm2(x.permute(0, 2, 3, 1))
        # print("Xf1:", x_freq.shape)
        x_freq = self.FreqVss(x_freq).permute(0, 3, 1, 2)
        # print("Xf2:", x_freq.shape)
        x = x_spa + x_freq

        return x


#FreqHL_h5l5
# class FreqBG(nn.Module):
#     def __init__(self, dim, h=16, w=16):
#         super().__init__()
#         self.dim = dim
#         self.norm1 = nn.LayerNorm(dim)
#         self.complex_weight_high = nn.Parameter(torch.randn(2, dim//2, h, w, dtype=torch.float32) * 0.02)       #原
#         self.complex_weight_low = nn.Parameter(torch.randn(2, dim//2, h, w, dtype=torch.float32) * 0.02)        #原 
#         # self.complex_weight_high = nn.Parameter(torch.randn(2, int(dim*0.8), h, w, dtype=torch.float32) * 0.02)        #split消融
#         # self.complex_weight_low = nn.Parameter(torch.randn(2, dim-int(dim*0.8), h, w, dtype=torch.float32) * 0.02)     #split消融
#         # self.learnable_radius = nn.Parameter(torch.tensor(5,dtype=torch.float32))
#         self.prj = nn.Sequential(
#             nn.Conv2d(dim//2, dim, 1),  #原
#             # nn.Conv2d(dim, dim, 1), #split消融
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         # radius = H//4    #原
#         radius = H//8    #消融2/4/6/8/10
        
#         x_h, x_l = torch.chunk(x, 2, 1)   #原
        
#         # # split消融
#         # # 计算按照4:6比例分割的索引位置
#         # total_size = x.size(1)  # 获取要分割维度上的总大小（这里维度索引为1）
#         # split_index = int(total_size * 0.)  # 按照4:6比例计算分割的索引位置，这里取40%的位置作为分割点
#         # # 使用torch.split进行分割
#         # x_h, x_l = torch.split(x, [split_index, total_size - split_index], dim=1)

#         x_1 = torch.fft.rfft2(x_h, dim=(2, 3), norm='ortho')    # [B, C, H, W//2+1]
#         filter_size = x_1.shape[-2:]
#         center_y, center_x = filter_size[0] // 2, filter_size[1] // 2
#         filter_tensor = torch.ones(filter_size)
#         y, x = torch.meshgrid(torch.arange(filter_size[0]), torch.arange(filter_size[1]))
#         distance_from_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
#         filter_tensor[distance_from_center < radius] = 0
#         weight_high = F.interpolate(self.complex_weight_high, size=x_1.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
#         weight_high = torch.view_as_complex(weight_high.contiguous())
#         x1 = x_1 * filter_tensor.cuda() * weight_high
#         x1 = torch.fft.irfft2(x1, s=(H, W), dim=(2, 3), norm='ortho')


#         x_2 = torch.fft.rfft2(x_l, dim=(2, 3), norm='ortho')    # [B, C, H, W//2+1]
#         filter_size = x_2.shape[-2:]
#         center_y, center_x = filter_size[0] // 2, filter_size[1] // 2
#         filter_tensor = torch.ones(filter_size)
#         y, x = torch.meshgrid(torch.arange(filter_size[0]), torch.arange(filter_size[1]))
#         distance_from_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
#         filter_tensor[distance_from_center > radius] = 0
#         weight_low = F.interpolate(self.complex_weight_low, size=x_2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
#         weight_low = torch.view_as_complex(weight_low.contiguous())
#         x2 = x_2 * filter_tensor.cuda() * weight_low
#         x2 = torch.fft.irfft2(x2, s=(H, W), dim=(2, 3), norm='ortho')

#         # x = torch.concat((x1, x2), dim=1)
#         x = x1 + x2               #原
#         # x = torch.cat((x1, x2), dim=1)    #split消融
#         x = self.prj(x)

        
        # return x


#FreqHL
class FreqBG(nn.Module):
    def __init__(self, dim, h=16, w=16):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        # self.complex_weight_high = nn.Parameter(torch.randn(2, dim//2, h, w, dtype=torch.float32) * 0.02)       #原
        # self.complex_weight_low = nn.Parameter(torch.randn(2, dim//2, h, w, dtype=torch.float32) * 0.02)        #原 
        self.complex_weight_high = nn.Parameter(torch.randn(2, int(dim*0.8), h, w, dtype=torch.float32) * 0.02)        #split消融
        self.complex_weight_low = nn.Parameter(torch.randn(2, dim-int(dim*0.8), h, w, dtype=torch.float32) * 0.02)     #split消融
        # self.learnable_radius = nn.Parameter(torch.tensor(5,dtype=torch.float32))
        self.prj = nn.Sequential(
            # nn.Conv2d(dim//2, dim, 1),  #原
            nn.Conv2d(dim, dim, 1), #split消融
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # radius = H//4    #原
        radius = H//8    #消融2/4/6/8/10
        
        # x_h, x_l = torch.chunk(x, 2, 1)   #原
        
        # split消融
        # 计算按照4:6比例分割的索引位置
        total_size = x.size(1)  # 获取要分割维度上的总大小（这里维度索引为1）
        split_index = int(total_size * 0.8)  # 按照4:6比例计算分割的索引位置，这里取40%的位置作为分割点
        # 使用torch.split进行分割
        x_h, x_l = torch.split(x, [split_index, total_size - split_index], dim=1)

        x_1 = torch.fft.rfft2(x_h, dim=(2, 3), norm='ortho')    # [B, C, H, W//2+1]
        filter_size = x_1.shape[-2:]
        center_y, center_x = filter_size[0] // 2, filter_size[1] // 2
        filter_tensor = torch.ones(filter_size)
        y, x = torch.meshgrid(torch.arange(filter_size[0]), torch.arange(filter_size[1]))
        distance_from_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        filter_tensor[distance_from_center < radius] = 0
        weight_high = F.interpolate(self.complex_weight_high, size=x_1.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight_high = torch.view_as_complex(weight_high.contiguous())
        x1 = x_1 * filter_tensor.cuda() * weight_high
        x1 = torch.fft.irfft2(x1, s=(H, W), dim=(2, 3), norm='ortho')


        x_2 = torch.fft.rfft2(x_l, dim=(2, 3), norm='ortho')    # [B, C, H, W//2+1]
        filter_size = x_2.shape[-2:]
        center_y, center_x = filter_size[0] // 2, filter_size[1] // 2
        filter_tensor = torch.ones(filter_size)
        y, x = torch.meshgrid(torch.arange(filter_size[0]), torch.arange(filter_size[1]))
        distance_from_center = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        filter_tensor[distance_from_center > radius] = 0
        weight_low = F.interpolate(self.complex_weight_low, size=x_2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight_low = torch.view_as_complex(weight_low.contiguous())
        x2 = x_2 * filter_tensor.cuda() * weight_low
        x2 = torch.fft.irfft2(x2, s=(H, W), dim=(2, 3), norm='ortho')

        # x = torch.concat((x1, x2), dim=1)
        # x = x1 + x2               #原
        x = torch.cat((x1, x2), dim=1)    #split消融
        x = self.prj(x)

        
        return x

#DDVM
class FreqSM(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim//2)
        self.norm2 = nn.LayerNorm(dim//2)
        self.eca = eca_layer(dim//2, k_size=3)
        self.dwconv = DWConv(dim//2)
        self.vss = FreqVSS(dim//2)
        self.freqbg = FreqBG(dim//2)
        self.prj = nn.Conv2d(dim,out_dim,1)

    def forward(self, x):
        # print("x:", x.shape)
        x1, x2 = torch.chunk(x, 2, 1)
        # print("x1:", x1.shape)
        # print("x2:", x2.shape)
        x1 = self.vss(x1) + x1
        x2 = self.freqbg(x2) + x2

        x1 = self.dwconv(self.eca(self.norm1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))) + x1
        x2 = self.dwconv(self.eca(self.norm1(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))) + x2


        x = torch.cat((x1, x2), dim=1)

        x = self.prj(x)

        return x

class FreqSMamba(nn.Module):
    def __init__(self, num_classes=1, 
                 input_channels=3, 
                 dims=[96, 192, 384, 768],
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None
                 ):
        super().__init__()

        # dims = [8, 16, 32, 64, 128]        #Flops: 0.07 G, Params: 0.27 M
        # dims = [16, 32, 64, 128, 256]
        # dims = [32, 64, 128, 256, 512]    #Flops: 0.97 G, Params: 4.09 M
        # dims = [48, 96, 192, 384, 768]  #Flops: 2.16 G, Params: 9.17 M
        dims = [64, 128, 256, 512, 1024]  #原 Flops: 3.81 G, Params: 16.26 M

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.e1 = nn.Sequential(
            DoubleConv(input_channels, dims[0]),
        )
        self.e2 = nn.Sequential(
            DoubleConv(dims[0], dims[1]),
            # FreqSM(dims[0], dims[1]),
        )
        self.e3 = nn.Sequential(
            DoubleConv(dims[1], dims[2]),
            # FreqSM(dims[1], dims[2]),
        )
        self.e4 = nn.Sequential(
            FreqSM(dims[2], dims[3]),
        )
        self.e5 = nn.Sequential(
            FreqSM(dims[3], dims[4]),
        )

        self.d5 = nn.Sequential(
            FreqSM(dims[4], dims[3]),
        )
        self.d4 = nn.Sequential(
            FreqSM(dims[3], dims[2]),
        )
        self.d3 = nn.Sequential(
            DoubleConv(dims[2],dims[1]),
            # FreqSM(dims[2], dims[1]),
        )
        self.d2 = nn.Sequential(
            DoubleConv(dims[1],dims[0]),
            # FreqSM(dims[1], dims[0]),
        )
        self.d1 = nn.Sequential(
            DoubleConv(dims[0],dims[0]//2),
        )
        self.out = nn.Conv2d(dims[0]//2, num_classes, 1)


    def forward(self, x):
        x1 = self.e1(self.maxpool(x))
        x2 = self.e2(self.maxpool(x1))
        x3 = self.e3(self.maxpool(x2))
        x4 = self.e4(self.maxpool(x3))
        x5 = self.e5(self.maxpool(x4))

        
        d5 = self.d5(x5)
        d5 = self.upsample(d5) + x4
        d4 = self.d4(d5)
        d4 = self.upsample(d4) + x3
        d3 = self.d3(d4)
        d3 = self.upsample(d3) + x2
        d2 = self.d2(d3)
        d2 = self.upsample(d2) + x1
        d1 = self.d1(d2)
        d1 = self.upsample(d1)

        out = self.out(d1)

        # spectrum = np.fft.fft2(torch.sum(d4.squeeze(),0).detach().cpu().numpy())
        # # spectrum = np.fft.fft2(x[0][0].detach().cpu().numpy())
        # magnitude_shifted = np.fft.fftshift(spectrum)
        # magnitude = np.abs(magnitude_shifted)
        # plt.imshow(np.log1p(magnitude), cmap='viridis')
        # # plt.colorbar()
        # plt.tight_layout()
        # plt.axis('off')
        # plt.savefig('a1111.png')

        return out


if __name__ == '__main__':
    ras = FreqSMamba(3).cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()
    # from PIL import Image
    # img = Image.open('datasets/ph2/test/images/IMD017.bmp').resize((224,224))
    # if img.mode!= "RGB":
    #     img = img.convert("RGB")
    # tensor_img = torch.tensor(list(img.getdata())).float() / 255.0
    # tensor_img = tensor_img.reshape(1, 3, img.height, img.width)
    # input_tensor = tensor_img.cuda()
    output = ras(input_tensor)
    print(output.shape)

    from thop import profile
    flops, params = profile(ras, (input_tensor,))
    print('Flops: %.2f G, Params: %.2f M' % (flops / 1e9, params / 1e6))

    # import time
    # num_iterations = 100
    # # 开始计时
    # start_time = time.time()

    # # 进行多次推理
    # with torch.no_grad():
    #     for _ in range(num_iterations):
    #         output = ras(input_tensor)

    # # 结束计时
    # end_time = time.time()

    # # 计算总时间
    # total_time = end_time - start_time

    # # 计算每秒完成的推理次数
    # fps = num_iterations / total_time

    # print(f"FPS: {fps}")
