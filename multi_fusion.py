import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from cross_attention import CrossAttention
from AV_encoder import VideoModel,AudioModel


class AVFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.videoencoder = VideoModel()
        self.audioencoder = AudioModel()
        
        self.v2a_attn = CrossAttention(dim=256, num_heads=8)
        self.a2v_attn = CrossAttention(dim=256, num_heads=8)
        
        self.gate_v = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.Sigmoid()
        )
        self.gate_a = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.Sigmoid()
        )
        
        self.fc_out = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, video, audio):
        x1 = self.videoencoder(video)  
        x2 = self.audioencoder(audio)  
        x1 = self.v2a_attn(x1, x2)  
        x2 = self.a2v_attn(x2, x1)  
        # 动态门控融合
        gate_v = self.gate_v(x1.mean(dim=1, keepdim=True))  
        gate_a = self.gate_a(x2.mean(dim=1, keepdim=True))
        fused = gate_v * x1 + gate_a * x2  # [B, T, D]
        output = self.fc_out(fused.mean(dim=1))
        return output  