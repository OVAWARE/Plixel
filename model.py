# FILE: model.py

from time_embed import get_timestamp_embedding, TIME_DIM
from text_embed import get_text_embedding, TEXT_EMBED_DIM
import torch
from torch import nn
from parameters import *

channels = [64, 128, 256]
last_channel = channels[-1]
embed_dim = 256
num_heads = 1

MODEL_NAME = "diffusion_text2img"
MODEL_PATH = "../models/" + MODEL_NAME + ".pt"

class EncoderBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.ch = ch

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(ch, ch, 3, padding='same')
        self.conv2 = nn.Conv2d(ch, ch, 3, padding='same')
        self.conv3 = nn.Conv2d(ch, ch, 3, padding='same')

        self.temb_to_ch = nn.Linear(TIME_DIM + TEXT_EMBED_DIM, ch)
    
    def forward(self, x, cond):
        h = x

        tembs2 = self.temb_to_ch(cond)
        assert tembs2.shape == (cond.shape[0], self.ch)

        h = self.conv1(h)
        h = self.act(h)

        # Add on conditional embedding
        h = h + tembs2[:, :, None, None]

        h = self.conv2(h)
        h = self.act(h)

        h = self.conv3(h)
        h = self.act(h)

        return h + x

class DecoderBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.ch = ch

        self.act = nn.ReLU()

        # Receives two copies; so in_ch * 2
        self.conv1 = nn.Conv2d(ch * 2, ch, 3, padding='same')
        self.conv2 = nn.Conv2d(ch, ch, 3, padding='same')
        self.conv3 = nn.Conv2d(ch, ch, 3, padding='same')

        self.temb_to_ch = nn.Linear(TIME_DIM + TEXT_EMBED_DIM, ch)
    
    def forward(self, x, y, cond):
        assert x.shape == y.shape

        h = torch.cat((x, y), dim=1)

        tembs2 = self.temb_to_ch(cond)
        assert tembs2.shape == (cond.shape[0], self.ch)

        h = self.conv1(h)
        h = self.act(h)

        # Add on conditional embedding
        h = h + tembs2[:, :, None, None]

        h = self.conv2(h)
        h = self.act(h)

        h = self.conv3(h)
        h = self.act(h)

        return h + x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def get_upsampler(in_ch, out_ch):
            # Exactly doubles size
            k = 2
            return nn.ConvTranspose2d(in_ch, out_ch, k, stride=2, padding=(k-2)//2)

        def get_downsampler(in_ch, out_ch):
            # Exactly halves size
            return nn.Conv2d(in_ch, out_ch, 2, stride=2)

        self.enc = nn.ModuleList([EncoderBlock(channels[i]) for i in range(len(channels))])
        self.dec = nn.ModuleList([DecoderBlock(channels[i]) for i in range(len(channels)-2, -1, -1)])

        assert len(self.enc) == len(self.dec) + 1
        
        self.downsamplers = nn.ModuleList([get_downsampler(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.upsamplers   = nn.ModuleList([get_upsampler  (channels[i], channels[i-1]) for i in range(len(channels)-1, 0, -1)])

        self.image_to = nn.Conv2d(NUM_CHANNELS, channels[0], 3, padding='same')
        self.to_image = nn.Conv2d(channels[0], NUM_CHANNELS, 1, padding='same')

        self.attention = nn.MultiheadAttention(num_heads * embed_dim, num_heads, batch_first=True)

        self.to_query = nn.Linear(last_channel, num_heads * embed_dim)
        self.to_key   = nn.Linear(last_channel, num_heads * embed_dim)
        self.to_value = nn.Linear(last_channel, num_heads * embed_dim)
        self.out_proj = nn.Linear(num_heads * embed_dim, last_channel)

        # New: Linear layer to combine time and text embeddings
        self.combine_embeddings = nn.Linear(TIME_DIM + TEXT_EMBED_DIM, TIME_DIM + TEXT_EMBED_DIM)
    
    def compute_attention(self, x):
        N,C,H,W = x.shape

        assert embed_dim == C, f"Expected embed dim ({embed_dim}) to equal channels ({C})."

        x = torch.flatten(x, start_dim=2)

        x = torch.permute(x, (0, 2, 1))
        queries = self.to_query(x)
        keys    = self.to_key(x)
        values  = self.to_value(x)

        x,_ = self.attention(queries, keys, values, need_weights=False)

        x = self.out_proj(x)

        x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x, (N, C, H, W))
        return x

    def forward(self, x, t, text_embed):
        assert x.shape[0] == t.shape[0] == text_embed.shape[0]

        temb = get_timestamp_embedding(t).to(device)
        
        # Combine time and text embeddings
        cond = torch.cat([temb, text_embed], dim=1)
        cond = self.combine_embeddings(cond)

        N_enc = len(self.enc)
        N_dec = len(self.dec)

        outputs = []

        h = x

        h = self.image_to(h)

        for i in range(N_enc):
            h = self.enc[i](h, cond)

            outputs.append(h.clone())

            if i != N_enc-1:
                h = self.downsamplers[i](h)
        
        att = self.compute_attention(h)
        h = h + att

        outputs = list(reversed(outputs[:-1]))

        for i in range(N_dec):
            h = self.upsamplers[i](h)
            h = self.dec[i](h, outputs[i], cond)
        
        h = self.to_image(h)

        return h