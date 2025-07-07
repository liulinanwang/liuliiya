import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import pywt
from scipy.signal import butter, sosfiltfilt
from PyEMD import EMD
from scipy.signal import hilbert
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps

class View(nn.Module):
    """一个在 nn.Sequential 中重塑张量的辅助类。"""
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_H(nn.Module):
    def __init__(self, z_dim=32, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1), nn.ReLU(True),
            View((-1, 256*1*1)),
            nn.Linear(256, z_dim*2),
        )
        
        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4), nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),
        )

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def kl_divergence(mu, logvar):
    """计算KL散度损失。"""
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    return total_kld

# 2. 数据处理

def preprocess_signal(signal, cutoff_hz, fs):

    def filter_signal(s, btype):
        sos = butter(N=4, Wn=cutoff_hz / (0.5 * fs), btype=btype, analog=False, output='sos')
        filtered = sosfiltfilt(sos, s)
        return filtered

    high_freq_signal = filter_signal(signal, 'high')
    low_freq_signal = filter_signal(signal, 'low')
    
    return high_freq_signal, low_freq_signal

def run_cwt_analysis_and_save(signal, sampling_rate, fault_name, output_dir, file_index, freq_range=(800, 4000)):
    """对(已预处理的)高频信号执行CWT分析并保存图像。"""
    class_output_dir = os.path.join(output_dir, fault_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    sampling_period = 1.0 / sampling_rate
    frequencies_to_analyze = np.linspace(freq_range[0], freq_range[1], 256)
    wavelet_type = 'morl'
    pywt_scales = pywt.central_frequency(wavelet_type) / (frequencies_to_analyze * sampling_period)
    cwt_img, _ = pywt.cwt(signal, scales=pywt_scales, wavelet=wavelet_type, sampling_period=sampling_period)
    
    fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=50) # 生成 64x64 的图像
    ax.imshow(np.abs(cwt_img), aspect='auto', cmap='viridis', interpolation='bicubic')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    save_path = os.path.join(class_output_dir, f"{fault_name}_{file_index:04d}.png")
    fig.savefig(save_path)
    plt.close(fig)

def calculate_theoretical_frequencies():
    """计算并打印轴承故障的理论特征频率作为参考。"""
    rpm = 1200 
    fr = rpm / 60
    bpfo = 59.35
    bpfi = 90.25
    bsf = 67.18
    freqs = {'Rotation': fr, 'Outer': bpfo, 'Inner': bpfi, 'Ball': bsf}
    
    print("\n--- 理论故障频率 (参考) ---")
    for name, freq in freqs.items():
        print(f"{name} Frequency (Hz): {freq:.2f}")
    print("--------------------------------\n")

def run_hht_analysis(signal, time_vector, fs, fault_type, top_n=4):
    """对(已预处理的)低频信号执行完整的HHT分析并绘图。"""
    print(f"\n--- 正在对 [{fault_type}] 的低频分量进行HHT分析 ---")
    
    # 1. EMD分解
    emd = EMD()
    imfs = emd(signal)
    n_imfs = imfs.shape[0]
    print(f"信号被分解为 {n_imfs - 1} 个IMF 和 1 个残差。")
    
    # 2. 根据能量选择Top N个IMF
    imf_energies = [np.sum(imf**2) for imf in imfs[:-1]]
    sorted_energy_indices = np.argsort(imf_energies)
    num_to_select = min(top_n, n_imfs - 1)
    top_indices = np.sort(sorted_energy_indices[-num_to_select:])
    print(f"根据能量选出前 {num_to_select} 个IMF (索引): {top_indices + 1}")
    
    # 3. 对被选中的IMF进行希尔伯特变换并绘制瞬时频率
    plt.figure(figsize=(15, 2 * num_to_select), dpi=100)
    plt.suptitle(f'HHT - {fault_type}的瞬时频率', fontsize=16)
    for i, imf_index in enumerate(top_indices):
        imf = imfs[imf_index]
        analytic_signal = hilbert(imf)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
        
        plt.subplot(num_to_select, 1, i + 1)
        plt.plot(time_vector[:-1], instantaneous_frequency, 'b')
        plt.title(f"IMF {imf_index + 1} 的瞬时频率")
        plt.xlabel("时间 (s)")
        plt.ylabel("频率 (Hz)")
        plt.ylim(0, 500)
        plt.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

class CWTImageDataset(Dataset):
    """用于加载CWT图像的PyTorch Dataset类。"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                self.image_paths.append(os.path.join(cls_path, img_name))
                self.labels.append(self.class_to_idx[cls_name])
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================================================================
# 3. 主执行流程
# =============================================================================
def main():
    # --- 参数配置 ---
    # 路径配置
    DATA_PATH = "E:/研究生/CNN/HDU Bearing Dataset"
    CWT_IMAGE_DIR = "./cwt_images_filtered"
    MODEL_SAVE_PATH = "cwt_filtered_vae_model.pth"
    
    # 信号处理配置
    SIGNAL_LENGTH = 1024
    FS = 10000
    CUTOFF_HZ = 2000.0
    
    # 流程控制开关
    RUN_HHT_ANALYSIS = True
    GENERATE_IMAGES = True
    FORCE_TRAIN = True
    
    # VAE模型与训练配置
    Z_DIM = 32
    BETA = 4
    IMAGE_SIZE = 64
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    LR = 1e-4

    # 数据文件定义
    fault_files = {
        'inner': os.path.join('Simple Fault', '2 IF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'outer': os.path.join('Simple Fault', '3 OF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'ball': os.path.join('Simple Fault', '4 BF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
    }
    
    # --- 阶段一：预处理与分析 ---
    if RUN_HHT_ANALYSIS:
        calculate_theoretical_frequencies()

    if GENERATE_IMAGES:
        print("\n--- 阶段一：正在进行预处理并生成CWT图像 ---")
        if os.path.exists(CWT_IMAGE_DIR):
            print(f"检测到旧的图像目录，正在清理: {CWT_IMAGE_DIR}")
            shutil.rmtree(CWT_IMAGE_DIR)
        print("清理完成，开始生成新图像。")
        
        global_index = 0
        for fault_type, file_path in fault_files.items():
            print(f"\n正在处理故障类型: {fault_type}")
            full_path = os.path.join(DATA_PATH, file_path)
            mat_data = scipy.io.loadmat(full_path)
            col_index = 4 if fault_type == 'inner' else 3
            signal = mat_data['Data'][:, col_index].ravel()
            num_segments = len(signal) // SIGNAL_LENGTH
            
            for i in tqdm(range(num_segments), desc=f"处理 {fault_type} 段"):
                segment = signal[i * SIGNAL_LENGTH : (i + 1) * SIGNAL_LENGTH]
                
                # 步骤1: 对每个信号段进行统一的预处理
                high_freq_segment, low_freq_segment = preprocess_signal(segment, CUTOFF_HZ, FS)
                
                # 步骤2: 对预处理后的高频分量进行CWT分析并保存图像
                run_cwt_analysis_and_save(high_freq_segment, FS, fault_type, CWT_IMAGE_DIR, global_index)
                global_index += 1
                
                # 步骤3: (仅对第一个段)对预处理后的低频分量进行HHT分析
                if i == 0 and RUN_HHT_ANALYSIS:
                    time_vector = np.arange(SIGNAL_LENGTH) / FS
                    run_hht_analysis(low_freq_segment, time_vector, FS, fault_type)
        print("所有分析与图像生成已完成。")

    # --- 阶段二：VAE模型训练 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    dataset = CWTImageDataset(root_dir=CWT_IMAGE_DIR, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = BetaVAE_H(z_dim=Z_DIM).to(device)

    if FORCE_TRAIN or not os.path.exists(MODEL_SAVE_PATH):
        print("\n--- 阶段二：正在训练VAE模型 ---")
        optimizer = optim.Adam(model.parameters(), lr=LR)
        for epoch in range(NUM_EPOCHS):
            pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
            for x, _ in pbar:
                x = x.to(device)
                x_recon, mu, logvar = model(x)
                
                recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(BATCH_SIZE)
                total_kld = kl_divergence(mu, logvar)
                loss = recon_loss + BETA * total_kld
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.2f}")
        
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"模型训练完成并保存至 {MODEL_SAVE_PATH}")
    else:
        print(f"\n--- 阶段二：跳过训练，加载现有模型 {MODEL_SAVE_PATH} ---")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # --- 阶段三：VAE特征提取与可视化 ---
    print("\n--- 阶段三：VAE特征提取并进行t-SNE可视化 ---")
    model.eval()
    all_features = []
    all_labels = []
    
    analysis_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for x, y in tqdm(analysis_loader, desc="提取VAE特征"):
            x_gpu = x.to(device)
            _, mu, _ = model(x_gpu)
            all_features.append(mu.cpu().numpy())
            all_labels.append(y.numpy())
            
    features_np = np.concatenate(all_features)
    labels_np = np.concatenate(all_labels)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(features_np)

    df = pd.DataFrame({
        'tsne-1': tsne_results[:, 0],
        'tsne-2': tsne_results[:, 1],
        'label': [dataset.idx_to_class[l] for l in labels_np]
    })
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="tsne-1", y="tsne-2",
        hue="label",
        palette=sns.color_palette("hsv", len(dataset.classes)),
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.title('从(预处理后)CWT图像中提取的VAE特征的t-SNE可视化', fontsize=16)
    plt.xlabel("t-SNE维度1")
    plt.ylabel("t-SNE维度2")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()