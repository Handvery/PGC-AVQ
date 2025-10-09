import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models


class VideoModel(nn.Module):
    def __init__(self, hidden_dim=256, nhead=8, num_layers=2):
        super(VideoModel, self).__init__()
        
        # --- CNN backbone (ResNet18) ---
        backbone = models.resnet18(pretrained=False)
        self.stage1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)  # 64-d
        self.stage2 = backbone.layer1  # 64-d
        self.stage3 = backbone.layer2  # 128-d
        self.stage4 = backbone.layer3  # 256-d
        self.stage5 = backbone.layer4  # 512-d

        # --- 1x1 conv 对齐到 hidden_dim ---
        self.lateral2 = nn.Conv2d(64, hidden_dim, kernel_size=1)
        self.lateral3 = nn.Conv2d(128, hidden_dim, kernel_size=1)
        self.lateral4 = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.lateral5 = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # --- Transformer 时序建模 ---
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Attention Pooling (选择关键尺度+时间片) ---
        self.attn_pool = nn.Linear(hidden_dim, 1)

        # --- Final MLP ---
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256)
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.size()
        multi_scale_feats = []

        for t in range(T):
            f = self.stage1(x[:, t])
            c2 = self.stage2(f)   # (B, 64, H/4, W/4)
            c3 = self.stage3(c2)  # (B, 128, H/8, W/8)
            c4 = self.stage4(c3)  # (B, 256, H/16, W/16)
            c5 = self.stage5(c4)  # (B, 512, H/32, W/32)

            # lateral conv -> unify to hidden_dim
            p2 = self.lateral2(c2)
            p3 = self.lateral3(c3)
            p4 = self.lateral4(c4)
            p5 = self.lateral5(c5)

            # 全局平均池化，每个层得到一个向量
            p2 = F.adaptive_avg_pool2d(p2, (1, 1)).squeeze(-1).squeeze(-1)  # (B, hidden_dim)
            p3 = F.adaptive_avg_pool2d(p3, (1, 1)).squeeze(-1).squeeze(-1)
            p4 = F.adaptive_avg_pool2d(p4, (1, 1)).squeeze(-1).squeeze(-1)
            p5 = F.adaptive_avg_pool2d(p5, (1, 1)).squeeze(-1).squeeze(-1)

            # 融合多个尺度 (concat)
            feat = (p2 + p3 + p4 + p5) / 4
            multi_scale_feats.append(feat)

        feats = torch.stack(multi_scale_feats, dim=1)  # (B, T, hidden_dim)

        # Transformer (时序建模)
        feats = self.transformer(feats)  # (B, T, hidden_dim)

        # Attention pooling (时间维度)
        attn_weights = torch.softmax(self.attn_pool(feats), dim=1)  # (B, T, 1)
        video_feat = torch.sum(attn_weights * feats, dim=1)  # (B, hidden_dim)

        out = self.fc(video_feat)  # (B, 256)
        return out
    
class AudioModel(nn.Module):
    def __init__(self, n_mels=64, input_channels=2, output_dim=256):
        super(AudioModel, self).__init__()

        # 梅尔频谱图转换
        #self.mel_transform = T.MelSpectrogram(
        #    sample_rate=48000,  # 假设音频采样率为16kHz
        #    n_mels=n_mels,
        #    n_fft=400,
        #    hop_length=160,
        #    win_length=400,
        #    center=False
        #)
        
        # 定义卷积层，用于提取特征
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        
        # Batch Normalization
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(128)
        
        # 全连接层
        self.fc1 = nn.Linear(28672, output_dim)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        # 将特征展平
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x