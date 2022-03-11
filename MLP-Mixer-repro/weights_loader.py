import os
import torch
from tqdm import tqdm
from urllib.request import urlopen, Request


OFFICAL_BASE_URL = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/'
PRETRAINED_OFFICIAL_HUB = {
    'mixer_b16_224_in21k' : OFFICAL_BASE_URL + 'jx_mixer_b16_224_in21k-617b3de2.pth',
    'mixer_l16_224': OFFICAL_BASE_URL + 'jx_mixer_l16_224-92f9adc4.pth',
    'mixer_l16_224_in21k': OFFICAL_BASE_URL + 'jx_mixer_l16_224_in21k-846aa33c.pth',
}

AGENT_BASE_URL = 'https://d7.serctl.com/downloads8/'
PRETRAINED_AGENT_HUB = {
    'mixer_b16_224_in21k' : AGENT_BASE_URL +'2022-03-11-15-40-47-pytorch-image-models-jx_mixer_b16_224-76587d61.pth',
    'mixer_l16_224': AGENT_BASE_URL + '2022-03-11-15-45-08-pytorch-image-models-jx_mixer_l16_224-92f9adc4.pth',
    'mixer_l16_224_in21k': AGENT_BASE_URL + '2022-03-11-15-57-28-pytorch-image-models-jx_mixer_l16_224_in21k-846aa33c.pth',
}

def download_url_to_file(url, dst):
    req = Request(url)
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    f = open(dst, 'wb')
    with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        while True:
            buffer = u.read(8192)
            if len(buffer) == 0:
                break
            f.write(buffer)
            pbar.update(len(buffer))
    f.close()

def load_pretrained_from_url(variant='mixer_b16_224_in21k', map_location='cpu', dst_dir='./weights'):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, variant + '.pth')
    if os.path.exists(dst):
        print("Pre-trained %s have been downloaded to %s" % (variant, dst_dir))
    else:
        assert variant in PRETRAINED_AGENT_HUB.keys(), \
            f'Pre-trained weights for {variant} is not officially available yet!'
        url = PRETRAINED_AGENT_HUB[variant]
        print('Downloading {} to {}'.format(variant, dst_dir))
        download_url_to_file(url, dst=dst)
    print("Loading ", variant)
    
    return torch.load(dst, map_location=map_location)
    