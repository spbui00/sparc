import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralExpertAE(nn.Module):
    def __init__(self, in_channels, mid_channels=32):
        super(NeuralExpertAE, self).__init__()
        
        # Input: (B, C, T)
        self.encoder = nn.Sequential(
            # Layer 1: Extract features
            nn.Conv1d(in_channels, mid_channels, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True),
            
            # Layer 2: Downsample (T -> T/2)
            nn.Conv1d(mid_channels, mid_channels*2, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm1d(mid_channels*2),
            nn.ReLU(True),
            
            # Layer 3: Downsample (T/2 -> T/4) - The Bottleneck
            nn.Conv1d(mid_channels*2, mid_channels*4, kernel_size=4, padding=1, stride=2),
            nn.ReLU(True) 
        )
        
        # DECODER: Reconstruct
        self.decoder = nn.Sequential(
            # Layer 3: Upsample (T/4 -> T/2)
            nn.ConvTranspose1d(mid_channels*4, mid_channels*2, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm1d(mid_channels*2),
            nn.ReLU(True),
            
            # Layer 2: Upsample (T/2 -> T)
            nn.ConvTranspose1d(mid_channels*2, mid_channels, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(True),
            
            # Layer 1: Output
            nn.Conv1d(mid_channels, in_channels, kernel_size=5, padding=2, stride=1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


