import torch
from torch import nn
from collections import OrderedDict
from weights_loader import load_pretrained_from_url


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stem_norm=False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6) if stem_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))

class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len):
        super().__init__()
        gate_dim = dim // 2
        self.norm = nn.LayerNorm(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)

class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(self, seq_len, in_features, channel_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, channel_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)

        assert channel_dim % 2 == 0, 'channels are not splittable'
        self.gate = SpatialGatingUnit(channel_dim, seq_len)

        self.fc2 = nn.Linear(channel_dim // 2, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.gate(x)
        x = self.drop2(self.fc2(x))
        return x

class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, mlp_ratio=4, drop=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = GatedMlp(seq_len, dim, channel_dim, drop)

    def forward(self, x):
        x = x + self.mlp_channels(self.norm(x))
        return x

class gMLP(nn.Module):
    r"""MLP-Mixer
            A PyTorch impl of : MLP-Mixer: An all-MLP Architecture for Vision- https://arxiv.org/abs/2105.01601

    Args:
        num_classes: 
    """
    def __init__(self, num_classes=1000, img_size=224, in_channels=3, patch_size=16,
                    num_blocks=30, embed_dim=256, mlp_ratio=6, drop_rate=0., stem_norm=False):
        super().__init__()
        assert (img_size % patch_size) == 0, 'Image size must be divisible by patch size'
        num_patches = (img_size // patch_size) ** 2

        self.stem = PatchEmbed(in_channels, embed_dim, patch_size, stem_norm)
        self.blocks = nn.Sequential(*[ # N x (Mixer Layer)
            SpatialGatingBlock(embed_dim, num_patches, mlp_ratio, drop=drop_rate)
            for _ in range(num_blocks)
            ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2) # (bs, embed_dim, H/ps, W/ps) -> (bs, num_patches, embed_dim)
        x = self.blocks(x) # retain shape, (bs, tokens length, channels)
        x = self.norm(x)
        x = x.mean(dim=1) # GlobalAvgPooling (bs, channels)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x) # (bs, num_classes)
        return x


def _create_mixer(variant, pretrained=False, **kwargs):
    gmlp = gMLP(**kwargs)

    if pretrained:
        weights_dict = load_pretrained_from_url(variant=variant, map_location='cpu')
        # Delete classification head weights
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        gmlp.load_state_dict(weights_dict, strict=False)

    return gmlp


def gmlp_ti16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128, mlp_ratio=6, **kwargs)
    model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
    return model

def gmlp_s16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, **kwargs)
    model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
    return model

def gmlp_b16_224(pretrained=False, **kwargs):
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=512, mlp_ratio=6, **kwargs)
    model = _create_mixer('gmlp_b16_224', pretrained=pretrained, **model_args)
    return model


if __name__ == '__main__':
    num_classes = 1000
    gmlp = gmlp_s16_224(pretrained=False, num_classes=num_classes)
    print(gmlp)

    input = torch.randn(1, 3, 224, 224)
    output = gmlp(input)
    print(output.shape) # (1, 1000) if correct