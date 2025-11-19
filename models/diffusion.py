import math
import torch
import torch.nn as nn
from models.model_utils import *

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch = config["model"]["ch"]
        out_ch = config["model"]["out_ch"]
        ch_mult = tuple(config["model"]["ch_mult"])
        num_res_blocks = config["model"]["num_res_blocks"]
        dropout = config["model"]["dropout"]
        in_channels = config["model"]["in_channels"]
        resolution = config["data"]["image_size"]
        resamp_with_conv = config["model"]["resamp_with_conv"]

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # --- timestep embedding ---
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(in_features=self.ch, out_features=self.temb_ch),
            nn.Linear(in_features=self.temb_ch, out_features=self.temb_ch)
        ])

        # --- downsampling ---
        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=self.ch,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        # --- middle ---
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                           dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # --- upsampling ---
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)

        # --- end ---
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(in_channels=block_in,
                                  out_channels=out_ch,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # --- timestep embedding ---
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = self.temb.dense[1](temb)

        # --- downsampling ---
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # --- middle ---
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # --- upsampling ---
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # --- end ---
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
