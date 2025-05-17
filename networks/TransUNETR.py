"""
conv+psa-mlla-csp-deforbed-sppf
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
from timm.models.layers import DropPath, trunc_normal_


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
        # print(f'stem-x.shape:{x.shape}')
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        # x = x.flatten(2).transpose(1, 2)
        # print(f'steam-x-return-shape:{x.shape}')
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
class PSA(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.act=nn.GELU()
        self.c = int(c1 * e)
        self.dwc=nn.Conv2d(self.c,self.c,3,1,padding=1,groups=self.c)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 32)
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
    def __init__(self, in_ch, out_ch):
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
    


#######===========MHSA code


class MHSA_Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MHSA_Attention(nn.Module):
    """Attention module that can take tensor with [B, N, C] or [B, C, H, W] as input.
    Modified from: 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, 'dim should be divisible by head_dim'
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            N = H * W
            x = torch.flatten(x, start_dim=2).transpose(-2, -1) # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

##########without 
class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    --dim: embedding dim
    --token_mixer: token mixer module
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim,
                out_dim, 
                 token_mixer=nn.Identity, 
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()
        self.convblock=Conv(dim,out_dim,k=3,s=1,p=1)
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MHSA_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))

            x=self.convblock(x)
           
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            x=self.convblock(x)
          
        return x



class MSPE(nn.Module):
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

#####MSFI
class MSFI(nn.Module):
    def __init__(self,in_ch,out_ch,):
        super(MSFI,self).__init__()
        self.mspe=MSPE(in_ch,out_ch)
        self.sppf=SPPF(out_ch,out_ch,)
    def forward(self,x):
        x=self.mspe(x)
        x=self.sppf(x)
        return x


###GPSA
class GPSABlock(nn.Module):
    def __init__(self,dim,out_channels, input_resolution, mlp_ratio=4., 
     drop=0., drop_path=0.,  act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(GPSABlock,self).__init__()

        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.atten=PSA(dim,dim,0.5)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.convblock=Conv(dim,out_channels,k=3,s=1,p=1) #
    def forward(self, x):
       
        H, W = self.input_resolution
        x=x.flatten(2).permute(0,2,1).contiguous()
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        ###这里目的是应该是设计位置编码！
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
        return x

    
    
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

class CSPI(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
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
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


###M:32, 64, 128, 256
###S:16,32,64,128
###L:64,128,256,512

####实验1: MHSA分数测试 不使用MSFI
#####实验2：PSA分数测试 不使用MSFI 不使用门控机制

class Rolling_Unet_S(nn.Module):
    def __init__(self,num_classes, input_channels=3, 
            deep_supervision=False,
            img_size=224,
                 patch_size=4,
                 embed_dims=[64,128,256,512],):
        super().__init__()
        
        self.ds=deep_supervision
        self.refconv=Conv(input_channels,embed_dims[0],k=3,s=1,p=1)
        ##stage0-H/4,W/4,C0
        self.patch_embeding=Stem(img_size=img_size,patch_size=patch_size,
                                 in_chans=input_channels,
                                 embed_dim=embed_dims[0])

        ####MSPE
        
        # self.msfi=MSFI(embed_dims[0],embed_dims[0])

        self.encoder0=DoubleConv(embed_dims[0],embed_dims[0])
        
    
        ##stage1-H/8,W/8,C1
    
        self.downsample1=nn.MaxPool2d(2)
    
        self.encoder1=DoubleConv(embed_dims[0],embed_dims[1])
        ##stage2-H/16,W/16,C2      
      
        self.downsample2=nn.MaxPool2d(2)
 
        self.encoder2=DoubleConv(embed_dims[1],embed_dims[2])
        ##stage3-H/32,W/32,C3
     
        self.downsample3=nn.MaxPool2d(2)

        ###HATB1
        # self.gpsa3=GPSABlock(embed_dims[2],embed_dims[3],
        #                                    (img_size//32,img_size//32),
        #                                  mlp_ratio=4,drop=0.1,drop_path=0.1)
        
        self.gpsa3=MetaFormerBlock(embed_dims[2],embed_dims[3],
        token_mixer=MHSA_Attention,)
        # self.psa_mlla_block3=D_DoubleConv(embed_dims[2],embed_dims[3])

        ###HATB2
        # self.gpsa3_up=GPSABlock(embed_dims[3],embed_dims[2],
        #                                       (img_size//16,img_size//16),
        #                                       mlp_ratio=4,drop=0.1,drop_path=0.1)

        self.gpsa3_up=MetaFormerBlock(embed_dims[3],embed_dims[2],
        token_mixer=MHSA_Attention,)
        # self.psa_mlla_block3_up=D_DoubleConv(embed_dims[3],embed_dims[2])

        # # ##decoder2
        self.encoder2_up=DoubleConv(embed_dims[2],embed_dims[1])

        ##decoder1
        self.encoder1_up=DoubleConv(embed_dims[1],embed_dims[0])
        
        ###decoder0--卷积层
        self.last_conv=Conv(embed_dims[0],8,k=3,s=1,p=1)
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        ####cspi layer
        # self.eca3=CSPI(embed_dims[2],embed_dims[2],shortcut=True)
        # self.eca2=CSPI(embed_dims[1],embed_dims[1],shortcut=True)
        # self.eca1=CSPI(embed_dims[0],embed_dims[0],shortcut=True)

        if self.ds:
            self.ds_conv2=nn.Sequential(
                nn.Conv2d(embed_dims[1], num_classes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.01),)

            self.ds_conv1=nn.Sequential(
                nn.Conv2d(embed_dims[0], num_classes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.01),)

    def forward(self, x):
        B,C,H,W=x.shape
        ##H,W,C
       
        x_conv=self.refconv(x)##与解码器合成的特征图
        
        x_steam=self.patch_embeding(x)###stem 层
        # x_steam=self.msfi(x_steam)
        x0=self.encoder0(x_steam)## H//4,W//4,C*4

        t0=x0
       
        x1=self.downsample1(x0)
        x1=self.encoder1(x1) ##stage1,H//8,W//8,C*8 
        t1=x1
    
       
        x2=self.downsample2(x1)
        x2=self.encoder2(x2)  ##stage2 H//16,W//16,C*16
        t2=x2
        
        
        x3=self.downsample3(x2)
       
        x3=self.gpsa3(x3) ##stage3 H//32,W//32,C*32

        ###decoder-interploate
        out3=F.interpolate(x3, scale_factor=(2, 2), mode='bilinear')##对最后一个特征图进行上采样
        ###图像尺寸变化，但是通道并没有下降
        out3=self.gpsa3_up(out3)##将上采样过后的特征图，通道降低
        out3=torch.add(out3,t2)

        # out3=self.eca3(out3)

        out2=F.interpolate(out3,scale_factor=(2,2),mode='bilinear')
        out2=self.encoder2_up(out2)
        out2=torch.add(out2,t1)

        # out2=self.eca2(out2)

        out1=F.interpolate(out2,scale_factor=(2,2),mode='bilinear')
        out1=self.encoder1_up(out1)
        out1=torch.add(out1,t0)

        # out1=self.eca1(out1)
        out0=F.interpolate(out1,scale_factor=(4,4),mode='bilinear')
        out0=torch.add(out0,x_conv)
        out=self.last_conv(out0)
        out=self.final(out)

        if self.ds:
            out1=F.interpolate(out1,scale_factor=(4,4),mode='bilinear')
            out2=F.interpolate(out2,scale_factor=(8,8),mode='bilinear')
           
            out1=self.ds_conv1(out1)
            out2=self.ds_conv2(out2)
            return out,out2,out1

        return out


if __name__=='__main__':
     # print('1')
   
    input=torch.rand(1,3,352,352)
    model=Rolling_Unet_S(num_classes=1,input_channels=3,
                         deep_supervision=True,img_size=352)
    out=model(input)
    # print(f'out.shape:{out.shape}')
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
