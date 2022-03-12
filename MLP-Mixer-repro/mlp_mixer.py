import torch
from torch import nn
from collections import OrderedDict
from weights_loader import load_pretrained_from_url

class Mlp(nn.Module):
    """Standard Mlp block inherited from ViT,
       follows the dense->gelu->drop->dense->drop fashion.
    """
    def __init__(self, input_dim, hidden_dim, drop) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x):
        # x: (bs, embed_dim, ps) for tokens-mixing or (bs, ps, embed_dim) for channels-mixing
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x

class MixerBlock(nn.Module):
    """Residual Block with tokens-mixing and channels-mixing MLP
    """
    def __init__(self, embed_dim=512, num_patches=196, mlp_ratio=(0.5, 4.), drop=0., drop_path=0.):
        super().__init__()
        tokens_hidden_dim, channels_hidden_dim = [int(x * embed_dim) for x in mlp_ratio]
        
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_tokens = Mlp(num_patches, hidden_dim=tokens_hidden_dim, drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_channels = Mlp(embed_dim, hidden_dim=channels_hidden_dim, drop=drop)

    def forward(self, x):
        # tokens mixing
        residual = x
        x = self.norm1(x)
        x = self.mlp_tokens(x.transpose(1, 2)).transpose(1, 2)
        x = x + residual
        # channels mixing
        residual = x
        x = self.norm2(x) #(bs, tokens, channels)
        x = self.mlp_channels(x)
        x = x + residual  # (bs, tokens, channels)
        return x

class MlpMixer(nn.Module):
    r"""MLP-Mixer
            A PyTorch impl of : MLP-Mixer: An all-MLP Architecture for Vision- https://arxiv.org/abs/2105.01601

    Args:
        num_classes: 
    """
    def __init__(self, num_classes=1000, img_size=224, in_channels=3, patch_size=16,
                    num_blocks=8, embed_dim=512, mlp_ratio=(0.5, 4.0), drop_rate=0., stem_norm=False):
        super().__init__()
        assert (img_size % patch_size) == 0, 'Image size must be divisible by patch size'
        num_patches = (img_size // patch_size) ** 2

        self.stem = nn.Sequential(OrderedDict([
                ('proj', nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)),
                ('norm', nn.LayerNorm(embed_dim, eps=1e-6) if stem_norm else nn.Identity())
            ]))
        self.blocks = nn.Sequential(*[ # N x (Mixer Layer)
            MixerBlock(embed_dim, num_patches, mlp_ratio, drop=drop_rate)
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
    mlp_mixer = MlpMixer(**kwargs)

    if pretrained:
        weights_dict = load_pretrained_from_url(variant=variant, map_location='cpu')
        # Delete classification head weights
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        mlp_mixer.load_state_dict(weights_dict, strict=False)

    return mlp_mixer

def mixer_s16_224(pretrained=False, **kwargs):
    """Small model, patch_size=16, img_size=224, pre-trained on ImageNet-1K
    """
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model

def mixer_b16_224_in21k(pretrained=False, **kwargs):
    """Base model, patch_size=16, img_size=224, pre-trained on ImageNet-21K
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_in21k', pretrained=pretrained, **model_args)
    return model

def mixer_l16_224(pretrained=False, **kwargs):
    """Large model, patch_size=16, img_size=224, pre-trained on ImageNet-1K
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model

def mixer_l16_224_in21k(pretrained=False, **kwargs):
    """Large model, patch_size=16, img_size=224, pre-trained on ImageNet-21K
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224_in21k', pretrained=pretrained, **model_args)
    return model


if __name__ == '__main__':
    num_classes = 1000
    mlp_mixer = mixer_b16_224_in21k(pretrained=True, num_classes=num_classes)
    print(mlp_mixer)

    input = torch.randn(1, 3, 224, 224)
    output = mlp_mixer(input)
    print(output.shape) # (1, 1000) if correct
    