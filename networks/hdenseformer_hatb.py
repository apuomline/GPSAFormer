import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from thop import profile
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DenseForward(nn.Module):
    def __init__(self, dim, hidden_dim, outdim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, outdim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Dense_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x = torch.cat(x, 2)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class DensePreConv_AttentionBlock(nn.Module):
    def __init__(self, out_channels, height, width, growth_rate=32,  depth=4, heads=8,  dropout=0.5, attention=Dense_Attention):
        super().__init__()
        mlp_dim = growth_rate * 2
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Linear(out_channels + i * growth_rate, growth_rate),
                PreNorm(growth_rate, attention(growth_rate, heads = heads, dim_head = (growth_rate) // heads, dropout = dropout, num_patches=(height,width))),
                PreNorm(growth_rate, DenseForward(growth_rate, mlp_dim,growth_rate, dropout = dropout))
            ]))
        self.out_layer = DenseForward(out_channels + depth * growth_rate, mlp_dim,out_channels, dropout = dropout)
            
    def forward(self, x):
        features = [x]
        for l, attn, ff in self.layers:
            x = torch.cat(features, 2)
            x = l(x)
            x = attn(x) + x
            x = ff(x) + x
            features.append(ff(x))
        x = torch.cat(features, 2)
        x = self.out_layer(x)
        return x


class Dense_TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, growth_rate=32, patch_size=16, depth=6, heads=8, dropout=0.5, attention=DensePreConv_AttentionBlock):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.outsize = (image_height // patch_size, image_width// patch_size)
        h = image_height // patch_height
        w = image_width // patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        mlp_dim = out_channels * 2
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, out_channels))

        self.blocks = nn.ModuleList([])
        for i in range(depth):
            self.blocks.append(nn.ModuleList([
                attention(out_channels, height=h, width=w, growth_rate=growth_rate)
            ]))
        
        self.re_patch_embedding = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 1, p2 = 1, h = h)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img):
        x = self.patch_embeddings(img)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        x = self.dropout(embeddings)
        print(f'dense-conv-x.shape:{x.shape}')
        for block, in self.blocks:
            # print(block)
            x = block(x)
            print(f'dense-block-atten-x.shape:{x.shape}')
        x = self.re_patch_embedding(x)
        return F.interpolate(x, self.outsize)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x



############HATB code
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

class PSA_MLLABlock(nn.Module):

    def __init__(self, dim,out_channels, input_resolution, mlp_ratio=4.,  drop=0., drop_path=0.,
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
        # self.convblock=Conv(dim,out_channels,k=3,s=1,p=1) ##使用3*3卷积将通道降下来
    def forward(self, x):
        """先假设x的输入为B,L,C-->这样在PSA模块使用之前可以引入什么patch_embeding,以及位置嵌入"""
      
        H, W = self.input_resolution
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

        ####不需要将通道变换
        return x



class Dense_TransformerBlock_HATB(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, growth_rate=32, patch_size=16, depth=6, heads=8, dropout=0.5, 
    attention=PSA_MLLABlock):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.outsize = (image_height // patch_size, image_width// patch_size)
        h = image_height // patch_height
        w = image_width // patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        mlp_dim = out_channels * 2
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, out_channels))

        self.blocks = nn.ModuleList([])
        for i in range(depth):
            self.blocks.append(nn.ModuleList([
                attention(out_channels,out_channels,(h,w))
            ]))
        
        self.re_patch_embedding = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 1, p2 = 1, h = h)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img):
        x = self.patch_embeddings(img)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        x = self.dropout(embeddings)###B,1024,128

        for block, in self.blocks:
            # print(block)
            x = block(x)
        
        x = self.re_patch_embedding(x)
        return F.interpolate(x, self.outsize)

#===============================================

class HDenseFormer_2D_HATB(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters, image_size=(384,384), transformer_depth=12):
        super(HDenseFormer_2D_HATB, self).__init__()
        self.in_channels = in_channels
        self.n_cls = n_cls
        self.n_filters = n_filters

        self.attns = nn.ModuleList(
            [Dense_TransformerBlock_HATB(in_channels=1,out_channels=4 * n_filters,image_size=image_size,
            patch_size=16,depth=transformer_depth//4,attention=PSA_MLLABlock) for _ in range(self.in_channels)] 
            )

        # self.deep_conv = BasicConv2d(4 * n_filters * 3, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.deep_conv = UpConv(4 * n_filters * self.in_channels, 8 * n_filters)

        self.up1 = UpConv(8 * n_filters,4 * n_filters)
        self.up2 = UpConv(4 * n_filters,2 * n_filters)
        self.up3 = UpConv(2 * n_filters,1 * n_filters)

        self.block_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        self.conv1x1_d1 = nn.Conv2d(2 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)
        self.conv1x1_d2 = nn.Conv2d(4 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)
        self.conv1x1_d3 = nn.Conv2d(8 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        attnall = torch.cat([self.attns[i](x[:,i:i+1,:,:]) for i in range(self.in_channels)],1)
        attnout = self.deep_conv(attnall)  # 256, 1/8

        at1 = self.up1(attnout)   # 128, 1/4
        at2 = self.up2(at1)  # 64, 1/2
        at3 = self.up3(at2)

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds0 = ds0+at3
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds1 = ds1+at2
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        ds2 = ds2+at1
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        x = x+attnout

        out3 = self.conv1x1_d3(x)
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        out2 = self.conv1x1_d2(x)
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        out1 = self.conv1x1_d1(x)
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)
        # return [x,out1,out2,out3] ###如何将深度监督这种东西应用上?
        return x


###=============================================
class HDenseFormer_2D(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters, image_size=(384,384), transformer_depth=12):
        super(HDenseFormer_2D, self).__init__()
        self.in_channels = in_channels
        self.n_cls = n_cls
        self.n_filters = n_filters

        self.attns = nn.ModuleList(
            [Dense_TransformerBlock(in_channels=1,out_channels=4 * n_filters,image_size=image_size,
            patch_size=16,depth=transformer_depth//4,attention=DensePreConv_AttentionBlock) for _ in range(self.in_channels)] 
            )

        # self.deep_conv = BasicConv2d(4 * n_filters * 3, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.deep_conv = UpConv(4 * n_filters * self.in_channels, 8 * n_filters)

        self.up1 = UpConv(8 * n_filters,4 * n_filters)
        self.up2 = UpConv(4 * n_filters,2 * n_filters)
        self.up3 = UpConv(2 * n_filters,1 * n_filters)

        self.block_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        self.conv1x1_d1 = nn.Conv2d(2 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)
        self.conv1x1_d2 = nn.Conv2d(4 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)
        self.conv1x1_d3 = nn.Conv2d(8 * n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        attnall = torch.cat([self.attns[i](x[:,i:i+1,:,:]) for i in range(self.in_channels)],1)
        attnout = self.deep_conv(attnall)  # 256, 1/8

        at1 = self.up1(attnout)   # 128, 1/4
        at2 = self.up2(at1)  # 64, 1/2
        at3 = self.up3(at2)

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds0 = ds0+at3
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds1 = ds1+at2
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        ds2 = ds2+at1
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        x = x+attnout

        out3 = self.conv1x1_d3(x)
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        out2 = self.conv1x1_d2(x)
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        out1 = self.conv1x1_d1(x)
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)
        # return [x,out1,out2,out3] ###如何将深度监督这种东西应用上?
        return x

def Rolling_Unet_S(num_classes, input_channels, deep_supervision,image_size, transformer_depth=6):
    return HDenseFormer_2D(in_channels=input_channels, n_cls=num_classes, image_size=image_size,n_filters=32, transformer_depth=transformer_depth)

def HDenseFormer_2D_16(in_channels, n_cls, image_size, transformer_depth):
    return HDenseFormer_2D(in_channels=in_channels, n_cls=n_cls, image_size=image_size, n_filters=16,transformer_depth=transformer_depth)

def Rolling_Unet_S_HATB(num_classes, input_channels, deep_supervision,image_size, transformer_depth=6):
    return HDenseFormer_2D_HATB(in_channels=input_channels, n_cls=num_classes, image_size=image_size,n_filters=32, transformer_depth=transformer_depth)


if __name__=='__main__':
     # print('1')
   
    input=torch.rand(1,3,512,512)
    model=Rolling_Unet_S_HATB(1,3,False,512)
    out=model(input)
    # print(f'out.shape:{out.shape}')
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
