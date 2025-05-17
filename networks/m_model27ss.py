"""
做两次实验：第一次实验-验证swinunet架构-使用多模块psa-mlla的有效性
第二次实验，验证unetr架构使用多模块psa-mlla的有效性
第三次实验：实验emdeing-dims的选择##emde_dims[48,96,192,384,768]
第四次实验：确定PSA中的多头注意力机制个数
第五次实验：测试瓶颈层是否有用
第六次实验：测试PSA中是否需要两分支的激活函数以及dwconv
"""

"""
0.9283
0.9296
再用xrmodel27_5跑一次glas对比两种架构哪个更好
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import warnings
from thop import profile

try:
    # raise Exception("not work")
    from tvdcn.ops import PackedDeformConv2d, PackedDeformConv3d
    print("tvdcn is installed, using it for deformable convolution")
    
    class DeformConv2d(PackedDeformConv2d):
        def __init__(self, in_channels, out_channels, kernel_size, 
                    stride=1, padding=0, dilation=1, groups=1, 
                    offset_groups=1, mask_groups=1, bias=True, 
                    generator_bias: bool = False, 
                    deformable: bool = True, modulated: bool = False):
            super().__init__(in_channels, out_channels, kernel_size, 
                            stride, padding, dilation, groups, offset_groups, 
                            mask_groups, bias, generator_bias, deformable, modulated)

        def forward(self, x):
            return super().forward(x)
        
except:
    pass

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        # x = x.flatten(2).transpose(1, 2)
        # print(f'steam-x-return-shape:{x.shape}')
        return x



class DePE(nn.Module):
    def __init__(self, dim_in, dim_out, k_sizes=[3, 5], conv_op=nn.Conv2d, groups=None):
        super().__init__()
        if groups is None:
            self.groups = dim_out
        else:
            self.groups = groups

        self.proj_convs = nn.ModuleList()
        for k_size in k_sizes:
            self.proj_convs.append(conv_op(dim_out, dim_out, k_size, 1, k_size//2, groups=self.groups))

        if dim_in != dim_out:
            self.input_conv = conv_op(dim_in, dim_out, 1, 1, 0)
        else:
            self.input_conv = nn.Identity()

    def forward(self, x):
        x = self.input_conv(x)
        for proj in self.proj_convs:
            x = x + proj(x)

        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

###这里的psa不像标准的transformer结构
###是动态调整PSA的多头注意力机制还是直接设置成8？
class PSA(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.act=nn.GELU()
        self.c = int(c1 * e)
        self.norm1=nn.LayerNorm(self.c)
        self.norm2=nn.LayerNorm(self.c)
        self.dwc=nn.Conv2d(self.c,self.c,3,1,padding=1,groups=self.c)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=8)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        ##x:[B,C,H,W]
      
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        a=self.act(a)
        b=self.act(self.dwc(b))
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
###编码器所用的卷积结构
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,img_size):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        return self.conv(input)
    
###解码器所用的卷积结构
class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,img_size):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
    


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, expansion=4):
        super().__init__()
      
        hidden_dim = int(inp * expansion)
        self.proj=nn.Conv2d(inp,oup,1,1,0)
        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
          shortcut=x
          shortcut=self.proj(x)
          return shortcut+self.conv(x)
        
        

class PSA_MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim,out_channels, input_resolution, mlp_ratio=4.,  drop=0.1, drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim

        self.input_resolution = input_resolution

        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        
        # self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.atten=PSA(dim,dim,0.5)
      
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.convblock=Conv(dim,out_channels,k=3,s=1,p=1) ##使用3*3卷积将通道降下来
    def forward(self, x):
        """先假设x的输入为B,L,C-->这样在PSA模块使用之前可以引入什么patch_embeding,以及位置嵌入"""
      
        H, W = self.input_resolution
     
        x=x.flatten(2).permute(0,2,1).contiguous()
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))###右半边部分
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2)))
   
        # Linear Attention
        x = self.atten(x)
      
        x=x.flatten(2).permute(0,2,1).contiguous()##[B,L,C]
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x=x.reshape(B,H,W,C).permute(0,3,1,2).contiguous()
        x=self.convblock(x)##目的是将x的通道降下来
        # x=x.flatten(2).permute(0,2,1).contiguous()
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"
    

###确认一下卷积块与PSA_MLLA模块使用比例

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # self.m = SoftPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))





class Rolling_Unet_S(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size=224,deep_supervision=False,
                 patch_size=4,
                 embed_dims=[64,128,256,512],
                 en_deepths=[2, 2, 8, 2],
                 de_deepths=[2,2,2,2],
              ):
        super().__init__()
        
        # self.hidden_dim=48
        self.refconv=Conv(input_channels,embed_dims[0],k=3,s=1,p=1)
        ##stage0-H/4,W/4,32
        self.patch_embeding=Stem(img_size=img_size,patch_size=patch_size,
                                 in_chans=input_channels,
                                 embed_dim=embed_dims[0])
        ##这里的stem其实就相当于linear emdeding 64
        self.deform_pos=DePE(embed_dims[0],embed_dims[0],[3,5],nn.Conv2d,groups=embed_dims[0])
        self.sppf=SPPF(embed_dims[0],embed_dims[0])
    
        self.psa_mlla_block0=self._make_layer(DoubleConv,embed_dims[0],embed_dims[0],en_deepths[0],
                                              (img_size//4,img_size//4))
        self.downsample1=nn.MaxPool2d(2)
     

        ##stage3-H/32,W/32,C3 512
        self.psa_mlla_block1=self._make_layer(DoubleConv,embed_dims[0],embed_dims[1],en_deepths[1],
                                              (img_size//8,img_size//8))
        self.downsample2=nn.MaxPool2d(2)
  
       

        self.psa_mlla_block2=self._make_layer(PSA_MLLABlock,embed_dims[1],embed_dims[2],en_deepths[2],
                                              (img_size//16,img_size//16))
        # self.sppf1=SPPF(embed_dims[3],embed_dims[3])
        self.downsample3=nn.MaxPool2d(2)

        self.psa_mlla_block3=self._make_layer(PSA_MLLABlock,embed_dims[2],embed_dims[3],
                                             en_deepths[3],(img_size//32,img_size//32))
        ###decoder patchexpand可以将特征图的尺寸增大，通道降低
        ##decoder3

        ###需要验证这个瓶颈层是否有用？
        # self.psa_mlla_block4=self._make_layer(PSA_MLLABlock,embed_dims[4],embed_dims[4],
        #                                       deepths[3],(img_size//32,img_size//32))

        self.psa_mlla_block3_up=self._make_layer(PSA_MLLABlock,embed_dims[3],embed_dims[2],
                                                 de_deepths[0],(img_size//16,img_size//16))
    
        ##decoder2
        self.psa_mlla_block2_up=self._make_layer(D_DoubleConv,embed_dims[2],embed_dims[1],
                                                 de_deepths[1],(img_size//8,img_size//8))
        ##decoder1
      
       
        self.psa_mlla_block1_up=self._make_layer(D_DoubleConv,embed_dims[1],embed_dims[0],
                                                 de_deepths[2],(img_size//4,img_size//4))
        ###decoder0--卷积层 64
      # ##48
        ###与x经过一个卷积层之后的结果进行相加
        self.last_conv=Conv(embed_dims[0],8,k=3,s=1,p=1) ##32
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)
        
        self.eca3=BottleneckCSP(embed_dims[2],embed_dims[2],shortcut=True)
        self.eca2=BottleneckCSP(embed_dims[1],embed_dims[1],shortcut=True)
        self.eca1=BottleneckCSP(embed_dims[0],embed_dims[0],shortcut=True)

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])

        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size))
            else:
                layers.append(block(oup, oup, image_size))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

            
        x_conv=self.refconv(x)##
        
        x_steam=self.patch_embeding(x)
        x_steam=self.deform_pos(x_steam)
        x_steam=self.sppf(x_steam)
        x0=self.psa_mlla_block0(x_steam)## H//4,W//4,C*4

        t0=x0
       
        x1=self.downsample1(x0)
        x1=self.psa_mlla_block1(x1) ##stage1,H//8,W//8,C*8 
        t1=x1
    
       
        x2=self.downsample2(x1)
        x2=self.psa_mlla_block2(x2)  ##stage2 H//16,W//16,C*16
        t2=x2
        
        
        x3=self.downsample3(x2)
       
        x3=self.psa_mlla_block3(x3) ##stage3 H//32,W//32,C*32

        ###decoder-interploate
        out3=F.interpolate(x3, scale_factor=(2, 2), mode='bilinear')##对最后一个特征图进行上采样
        ###图像尺寸变化，但是通道并没有下降
        out3=self.psa_mlla_block3_up(out3)##将上采样过后的特征图，通道降低
        out3=torch.add(out3,t2)

        out3=self.eca3(out3)

        out2=F.interpolate(out3,scale_factor=(2,2),mode='bilinear')
        out2=self.psa_mlla_block2_up(out2)
        out2=torch.add(out2,t1)

        out2=self.eca2(out2)

        out1=F.interpolate(out2,scale_factor=(2,2),mode='bilinear')
        out1=self.psa_mlla_block1_up(out1)
        out1=torch.add(out1,t0)

        out1=self.eca1(out1)
        out0=F.interpolate(out1,scale_factor=(4,4),mode='bilinear')

        out0=torch.add(out0,x_conv)
        out=self.last_conv(out0)
        out=self.final(out)
        return out



if __name__=='__main__':
     # print('1')
   
    input=torch.rand(2,3,224,224)
    model=Rolling_Unet_S(num_classes=1,input_channels=3,
                         deep_supervision=False,img_size=224)
    out=model(input)
    # print(f'out.shape:{out.shape}')
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
