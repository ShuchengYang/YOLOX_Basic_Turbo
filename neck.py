import torch
from network_blocks import *
#delete
from backbone import *



class Neck(nn.Module):
    def __init__(self, scale=1):
        super(Neck, self).__init__()
        self.seq1 = CBL(384//scale, 128//scale, 1, 1, 0)
        self.seq2 = CBL(768//scale, 256//scale, 1, 1, 0)
        self.seq3 = CBL(1024//scale, 512//scale, 1, 1, 0)
        self.up1 = nn.Sequential(
            CBL(256//scale, 128//scale, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )
        self.up2 = nn.Sequential(
            CBL(512//scale, 256//scale, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, backbone_output):
        n3 = self.seq3(backbone_output[2])
        nm = self.up2(n3)
        n2 = self.seq2(torch.cat([backbone_output[1], nm], dim=1))
        nl = self.up1(n2)
        n1 = self.seq1(torch.cat([backbone_output[0], nl], dim=1))
        return [n1, n2, n3]

