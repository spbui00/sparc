import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAttentionSeparator(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_heads=8, dropout=0.1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.low_freq_conv = nn.Conv1d(in_channels, 64, 15, padding=7, dilation=1)   # Low frequencies
        self.mid_freq_conv = nn.Conv1d(in_channels, 64, 7, padding=3, dilation=2)    # Mid frequencies  
        self.high_freq_conv = nn.Conv1d(in_channels, 64, 3, padding=1, dilation=4)   # High frequencies
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(192, 128, 1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention for temporal modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable queries biased by frequency content
        self.neural_query = nn.Parameter(torch.randn(1, 1, 128))  # Biased toward low frequencies
        self.artifact_query = nn.Parameter(torch.randn(1, 1, 128))  # Biased toward high frequencies
        
        # Gating mechanism based on attention weights
        self.neural_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.artifact_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Reconstruction heads
        self.neural_decoder = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, out_channels, 3, padding=1)
        )
        
        self.artifact_decoder = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, out_channels, 3, padding=1)
        )
        
        # Initialize based on your wavelet findings
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize neural query to prefer low-frequency patterns
        nn.init.normal_(self.neural_query, mean=0.0, std=0.1)
        # Initialize artifact query to prefer high-frequency patterns  
        nn.init.normal_(self.artifact_query, mean=0.0, std=0.2)
        
    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        
        # Extract multi-scale features
        low_freq = torch.tanh(self.low_freq_conv(x))    # Emphasize smooth patterns
        mid_freq = torch.tanh(self.mid_freq_conv(x))    # Middle frequencies
        high_freq = torch.tanh(self.high_freq_conv(x))  # Emphasize transients
        
        # Ensure all feature maps have the same size by interpolating to the target length
        target_len = low_freq.size(2)
        if mid_freq.size(2) != target_len:
            mid_freq = F.interpolate(mid_freq, size=target_len, mode='linear', align_corners=False)
        if high_freq.size(2) != target_len:
            high_freq = F.interpolate(high_freq, size=target_len, mode='linear', align_corners=False)
        
        # Concatenate frequency bands
        multi_scale_features = torch.cat([low_freq, mid_freq, high_freq], dim=1)
        fused_features = self.feature_fusion(multi_scale_features)
        
        # Prepare for attention: (batch, seq_len, features)
        features_permuted = fused_features.permute(0, 2, 1)
        
        # Apply self-attention to capture temporal dependencies
        attended_features, attention_weights = self.attention(
            query=features_permuted,
            key=features_permuted, 
            value=features_permuted,
            need_weights=True
        )
        
        # Expand learnable queries for batch processing
        neural_queries = self.neural_query.expand(batch_size, -1, -1)
        artifact_queries = self.artifact_query.expand(batch_size, -1, -1)
        
        # Cross-attention: neural component focuses on low-freq patterns
        neural_features, neural_attn = self.attention(
            query=neural_queries,
            key=features_permuted,
            value=features_permuted
        )
        
        # Cross-attention: artifact component focuses on high-freq patterns  
        artifact_features, artifact_attn = self.attention(
            query=artifact_queries,
            key=features_permuted, 
            value=features_permuted
        )
        
        # Expand and apply gating
        neural_features_expanded = neural_features.expand(-1, seq_len, -1)
        artifact_features_expanded = artifact_features.expand(-1, seq_len, -1)
        
        neural_gate_weights = self.neural_gate(neural_features_expanded).permute(0, 2, 1)
        artifact_gate_weights = self.artifact_gate(artifact_features_expanded).permute(0, 2, 1)
        
        # Apply gating to separate components
        neural_component = attended_features * neural_gate_weights.permute(0, 2, 1)
        artifact_component = attended_features * artifact_gate_weights.permute(0, 2, 1)
        
        # Decode back to signal space
        neural_signal = self.neural_decoder(neural_component.permute(0, 2, 1))
        artifact_signal = self.artifact_decoder(artifact_component.permute(0, 2, 1))
        
        return neural_signal, artifact_signal, {
            'neural_attention': neural_attn,
            'artifact_attention': artifact_attn,
            'self_attention': attention_weights
        }