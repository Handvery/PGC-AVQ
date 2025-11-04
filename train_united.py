import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import pearsonr, spearmanr
from multi_fusion import AVFusionModel
import torchaudio
import torchaudio.transforms as T
import os
import numpy as np
import random

# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------- 数据集 --------------------
class MultiModalDataset(Dataset):
    def __init__(self, audio_dir, video_dir, label_file, transform=None):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.transform = transform
        self.pairs = self.load_csv(label_file)

    def load_csv(self, label_file):
        pairs = []
        with open(label_file, "r") as f:
            next(f)  # 跳过表头
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                audio_file, video_file, label = parts[0], parts[1], float(parts[2])
                audio_path = os.path.join(self.audio_dir, audio_file)
                video_path = os.path.join(self.video_dir, video_file)
                pairs.append((audio_path, video_path, label))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, video_path, label = self.pairs[idx]
        target_length = 900  # Mel 时间维度

        # ==== 音频处理 ====
        if os.path.basename(audio_path) == "0_0.wav":
            # 全零 Mel 特征
            waveform = torch.zeros(2, 64, target_length)
        else:
            waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
            # 如果是单声道，则补成 2 通道
            if waveform.size(0) == 1:
                waveform = waveform.repeat(2, 1)
            mel_transform = T.MelSpectrogram(
                sample_rate=48000, n_mels=64, n_fft=400, hop_length=160, win_length=400, center=False
            )
            waveform = mel_transform(waveform)
            current_length = waveform.size(2)
            if current_length > target_length:
                waveform = waveform[:, :, :target_length]
            else:
                padding = target_length - current_length
                channels = waveform.size(0)
                waveform = torch.cat([waveform, torch.zeros(channels, 64, padding)], dim=2)

        # ==== 视频处理 ====
        video_tensor = torch.load(video_path)

        # ==== transform 可选 ====
        if self.transform:
            waveform = self.transform(waveform)
            video_tensor = self.transform(video_tensor)

        return video_tensor, waveform, torch.tensor(label, dtype=torch.float32)

# -------------------- 加载数据 --------------------
def load_data(audio_dir, video_dir, label_file, batch_size=8, train_ratio=0.8):
    dataset = MultiModalDataset(audio_dir, video_dir, label_file)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] 数据集样本数: {len(dataset)}")
    return train_dataloader, test_dataloader

# -------------------- 训练与评估 --------------------
def evaluate_model(model, test_dataloader, device='cuda'):
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for video, audio, label in test_dataloader:
            video, audio, label = video.to(device), audio.to(device), label.to(device)
            outputs = model(video, audio).squeeze()
            loss = criterion(outputs, label)
            total_loss += loss.item()
            all_labels.extend(label.cpu().tolist())
            all_predictions.extend(outputs.cpu().tolist())

    avg_loss = total_loss / len(test_dataloader)
    plcc, _ = pearsonr(all_labels, all_predictions)
    srcc, _ = spearmanr(all_labels, all_predictions)
    return avg_loss, plcc, srcc

def train_model(model, train_dataloader, test_dataloader, num_epochs=200, lr=0.0001, device='cuda'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        for video, audio, label in train_dataloader:
            video, audio, label = video.to(device), audio.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(video, audio).squeeze()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_labels.extend(label.cpu().tolist())
            all_predictions.extend(outputs.detach().cpu().tolist())

        avg_train_loss = total_loss / len(train_dataloader)
        train_plcc, _ = pearsonr(all_labels, all_predictions)
        train_srcc, _ = spearmanr(all_labels, all_predictions)

        # 测试集评估
        test_loss, test_plcc, test_srcc = evaluate_model(model, test_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, PLCC: {train_plcc:.4f}, SRCC: {train_srcc:.4f} | "
              f"Test Loss: {test_loss:.4f}, PLCC: {test_plcc:.4f}, SRCC: {test_srcc:.4f}")

    print("Training complete!")

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    audio_dir = "/home/Users/zcy/Multimodal_assessment/dataset_video/audio/"
    video_dir = "/home/Users/zcy/Multimodal_assessment/dataset_video/video/"
    label_file = "labels_av.csv"

    num_epochs = 300
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader, test_dataloader = load_data(audio_dir, video_dir, label_file, batch_size=batch_size)

    model = AVFusionModel()
    train_model(model, train_dataloader, test_dataloader, num_epochs=num_epochs, lr=0.00001, device=device)
