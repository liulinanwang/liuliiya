import os
import numpy as np
import scipy.io
from scipy.signal import hilbert, butter, sosfiltfilt
import matplotlib.pyplot as plt
import pywt
from PyEMD import EMD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
from scipy.ndimage import zoom
from sklearn.manifold import TSNE
import pandas as pd

# =============================================================================
# 工具函数
# =============================================================================
def cuda(tensor, use_cuda):
    """将张量移动到 GPU（如果可用）"""
    return tensor.cuda() if use_cuda and torch.cuda.is_available() else tensor

def kaiming_init(m):
    """Kaiming 权重初始化"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# =============================================================================
# 定制 VAE 模型
# =============================================================================
class CustomVAE(nn.Module):
    """为 CWT 时频图设计的变分自编码器"""
    def __init__(self, z_dim=10, nc=1):
        super(CustomVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # [B, 32, 32, 32]
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # [B, 32, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # [B, 64, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # [B, 64, 4, 4]
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Linear(256, z_dim * 2)  # 输出 mu 和 logvar
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # [B, 64, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # [B, 32, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # [B, 32, 32, 32]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # [B, nc, 64, 64]
        )
        
        # 初始化权重
        self.apply(kaiming_init)
    
    def reparametrize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """前向传播"""
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def encode(self, x):
        """仅编码"""
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        return mu

# =============================================================================
# 滤波器函数 和 对比图函数（保持不变）
# =============================================================================
def filter_signal(signal, cutoff_hz, fs, filter_type='low'):
    sos = butter(N=4, Wn=cutoff_hz / (0.5 * fs), btype=filter_type, analog=False, output='sos')
    filtered_signal = sosfiltfilt(sos, signal)
    return filtered_signal

def plot_filtered_components(original, low, high, time_vector, fault_type, cutoff_hz):
    plt.figure(figsize=(15, 8), dpi=150)
    plt.subplot(3, 1, 1)
    plt.plot(time_vector, original, 'k-', label='Original Signal')
    plt.title(f'Signal Separation for {fault_type} (Cutoff: {cutoff_hz} Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(time_vector, low, 'b-', label='Low-Frequency Component (for HHT)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(time_vector, high, 'r-', label='High-Frequency Component (for CWT)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 原有的预处理和数据增强函数（保留但不使用）
# =============================================================================
def preprocess_signal(signal, fault_type, time_vector):
    print("--- [Note] Generic preprocess_signal function is defined but not used in this workflow. ---")
    return signal

def augment_data(signal, method='add_noise', noise_level=0.02):
    return signal

# =============================================================================
# CWT 分析函数（保持不变）
# =============================================================================
def run_cwt_analysis(signal, sampling_rate, fault_name, freq_range=(800, 4000)):
    print(f"\n--- Running CWT Analysis on High-Freq Component [{fault_name}] ---")
    sampling_period = 1.0 / sampling_rate
    frequencies_to_analyze = np.linspace(freq_range[0], freq_range[1], 200)
    wavelet_type = 'morl'
    central_frequency = pywt.central_frequency(wavelet_type)
    pywt_scales = central_frequency / (frequencies_to_analyze * sampling_period)
    cwt_img, cwt_freq = pywt.cwt(
        signal,
        scales=pywt_scales,
        wavelet=wavelet_type,
        sampling_period=sampling_period
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    im = ax.imshow(
        np.abs(cwt_img),
        aspect='auto',
        extent=[0, len(signal) * sampling_period, cwt_freq[-1], cwt_freq[0]],
        cmap='viridis'
    )
    ax.set_title(f'CWT Time-Frequency Map - {fault_name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(im, ax=ax, label='Magnitude')
    plt.tight_layout()
    plt.show()
    return cwt_img, cwt_freq
    
# ============= ================================================================
# HHT 分析函数（保持不变）
# =============================================================================
def calculate_theoretical_frequencies():
    rpm = 1000
    fr = rpm / 60
    bpfo = 59.35
    bpfi = 90.25
    bsf = 67.18
    freqs = {'Rotation': fr, 'Outer': bpfo, 'Inner': bpfi, 'Ball': bsf}
    print("--- Theoretical Fault Frequencies (for reference) ---")
    for name, freq in freqs.items():
        print(f"{name} Frequency (Hz): {freq:.2f}")
    print("---------------------------------------------------\n")
    return freqs

def run_hht_analysis(signal, time_vector, fs, fault_type, top_n=4):
    print(f"\n--- Running HHT Analysis on Low-Freq Component [{fault_type}] ---")
    emd = EMD()
    imfs = emd(signal)
    n_imfs = imfs.shape[0]
    print(f"Signal decomposed into {n_imfs - 1} IMFs and 1 Residual.")
    imf_energies = [np.sum(imf**2) for imf in imfs[:-1]]
    sorted_energy_indices = np.argsort(imf_energies)
    num_to_select = min(top_n, n_imfs - 1)
    top_indices = sorted_energy_indices[-num_to_select:]
    top_indices_sorted = np.sort(top_indices)
    print(f"Selected top {num_to_select} IMFs by energy (indices sorted for plotting): {top_indices_sorted + 1}")
    selected_imfs = imfs[top_indices_sorted]
    plt.figure(figsize=(15, 2 * (num_to_select + 1)), dpi=100)
    plt.suptitle(f'EMD Decomposition - Top {num_to_select} IMFs for: {fault_type}', fontsize=16)
    plt.subplot(num_to_select + 1, 1, 1)
    plt.plot(time_vector, signal, 'r')
    plt.title("Low-Frequency Component Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    for i, imf_index in enumerate(top_indices_sorted):
        plt.subplot(num_to_select + 1, 1, i + 2)
        plt.plot(time_vector, selected_imfs[i], 'g')
        plt.title(f"Selected IMF {imf_index + 1} (Energy-based Selection)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("Performing Hilbert-Huang Transform on selected IMFs...")
    plt.figure(figsize=(15, 2 * num_to_select), dpi=100)
    plt.suptitle(f'HHT - Instantaneous Frequencies for: {fault_type}', fontsize=16)
    for i, imf_index in enumerate(top_indices_sorted):
        imf = selected_imfs[i]
        analytic_signal = hilbert(imf)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
        plt.subplot(num_to_select, 1, i + 1)
        plt.plot(time_vector[:-1], instantaneous_frequency, 'b')
        plt.title(f"Instantaneous Frequency of Selected IMF {imf_index + 1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(0, 500)
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# VAE 相关函数
# =============================================================================
def preprocess_cwt_for_vae(cwt_img, target_size=(64, 64)):
    """预处理 CWT 时频图为 VAE 输入"""
    cwt_img = np.abs(cwt_img)
    cwt_img = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-8)
    scale_x = target_size[1] / cwt_img.shape[1]
    scale_y = target_size[0] / cwt_img.shape[0]
    cwt_img = zoom(cwt_img, (scale_y, scale_x))
    cwt_img = torch.from_numpy(cwt_img).float().unsqueeze(0).unsqueeze(0)
    return cwt_img

def train_vae(cwt_images, args):
    """训练 VAE 模型"""
    vae = CustomVAE(z_dim=args.z_dim, nc=1)
    vae = cuda(vae, args.cuda)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    print("Training Custom VAE...")
    for epoch in range(args.max_iter // (len(cwt_images) // args.batch_size + 1)):
        np.random.shuffle(cwt_images)
        for i in range(0, len(cwt_images), args.batch_size):
            batch = torch.cat(cwt_images[i:i+args.batch_size], dim=0)
            batch = Variable(cuda(batch, args.cuda))
            x_recon, mu, logvar = vae(batch)
            recon_loss = F.mse_loss(x_recon, batch, reduction='sum') / batch.size(0)
            total_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
            loss = recon_loss + args.beta * total_kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i // args.batch_size) % 10 == 0:
                print(f"Epoch {epoch}, Batch {i // args.batch_size}, Loss: {loss.item():.3f}, "
                      f"Recon: {recon_loss.item():.3f}, KLD: {total_kld.item():.3f}")
    torch.save(vae.state_dict(), os.path.join(args.output_dir, "custom_vae.pth"))
    print(f"VAE model saved to {args.output_dir}/custom_vae.pth")
    return vae

def extract_semantic_features(cwt_img, vae_model, use_cuda):
    """提取语义特征"""
    vae_model.eval()
    with torch.no_grad():
        cwt_img = Variable(cuda(cwt_img, use_cuda))
        mu = vae_model.encode(cwt_img)
    return mu.cpu().numpy().squeeze()

def visualize_latent_space(latent_features, fault_labels, output_dir, method='scatter'):
    """Visualize latent space using scatter plot or t-SNE."""
    plt.figure(figsize=(10, 8), dpi=150)
    
    if method == 'tsne' and latent_features.shape[1] > 2:
        n_samples = latent_features.shape[0]
        if n_samples < 5:
            print(f"Warning: Too few samples ({n_samples}) for t-SNE. Skipping t-SNE visualization.")
            return
        # Adjust perplexity to be less than n_samples
        perplexity = min(5, n_samples - 1)  # Ensure perplexity < n_samples
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            latent_2d = tsne.fit_transform(latent_features)
        except ValueError as e:
            print(f"Error in t-SNE: {e}. Falling back to scatter plot.")
            latent_2d = latent_features[:, :2]
    else:
        latent_2d = latent_features[:, :2]
    
    unique_labels = sorted(set(fault_labels))
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(fault_labels) if l == label]
        plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1], 
                    label=label, c=[colors(i)], alpha=0.6, s=50)
    
    plt.title(f"Latent Space Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"latent_space_{method}.png"))
    plt.show()
# =============================================================================
# 主执行函数
# =============================================================================
def main():
    ANALYSIS_SWITCH = {
        'CWT': True,
        'HHT': True,
        'VAE': True,
    }
    cutoff_hz = 2000.0
    data_path = "E:/研究生/CNN/HDU Bearing Dataset"
    fault_files = {
        'normal': os.path.join('Simple Fault', '1 Normal', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'inner': os.path.join('Simple Fault', '2 IF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'outer': os.path.join('Simple Fault', '3 OF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'ball': os.path.join('Simple Fault', '4 BF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'inner_outer': os.path.join('Compound Fault', '1 IF&OF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'inner_ball': os.path.join('Compound Fault', '2 IF&BF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'outer_ball': os.path.join('Compound Fault', '3 OF&BF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
        'inner_outer_ball': os.path.join('Compound Fault', '5 IF&OF&BF', 'f600', 'rpm1200_f600','rpm1200_f600_01.mat')
    }
    fs = 10000
    num_points = 4096
    class VAEArgs:
        z_dim = 10
        cuda = torch.cuda.is_available()
        max_iter = 10000
        batch_size = 8
        lr = 1e-4
        beta1 = 0.9
        beta2 = 0.999
        beta = 4.0
        output_dir = "./vae_outputs"
    args = VAEArgs()
    os.makedirs(args.output_dir, exist_ok=True)
    cwt_images = []
    fault_labels = []
    latent_features = []
    if ANALYSIS_SWITCH['HHT']:
        calculate_theoretical_frequencies()
    for fault_type, file_path in fault_files.items():
        full_path = os.path.join(data_path, file_path)
        print(f"\n==================================================")
        print(f"Processing file for fault type: '{fault_type}'")
        print(f"==================================================")
        try:
            mat_data = scipy.io.loadmat(full_path)
        except FileNotFoundError:
            print(f"!!! ERROR: File not found at {full_path}")
            continue
        col_index = 4 if fault_type == 'inner' else 3
        signal = mat_data['Data'][:, col_index].ravel()
        signal_segment = signal[:num_points]
        time_vector = np.arange(num_points) / fs
        low_freq_signal = filter_signal(signal_segment, cutoff_hz, fs, 'low')
        high_freq_signal = filter_signal(signal_segment, cutoff_hz, fs, 'high')
        #plot_filtered_components(signal_segment, low_freq_signal, high_freq_signal, time_vector, fault_type, cutoff_hz)
        if ANALYSIS_SWITCH['CWT']:
            cwt_freq_range = (cutoff_hz, fs / 2.5)
            cwt_img, cwt_freq = run_cwt_analysis(high_freq_signal, fs, fault_type, freq_range=cwt_freq_range)
            cwt_tensor = preprocess_cwt_for_vae(cwt_img)
            cwt_images.append(cwt_tensor)
            fault_labels.append(fault_type)
        if ANALYSIS_SWITCH['HHT']:
            run_hht_analysis(low_freq_signal, time_vector, fs, fault_type)
    if ANALYSIS_SWITCH['VAE'] and cwt_images:
        print("\n==================================================")
        print("Starting Custom VAE Training and Semantic Feature Extraction")
        print("==================================================")
        vae_model = train_vae(cwt_images, args)
        for cwt_img, fault_type in zip(cwt_images, fault_labels):
            latent_z = extract_semantic_features(cwt_img, vae_model, args.cuda)
            latent_features.append(latent_z)
            print(f"Extracted latent features for {fault_type}: {latent_z}")
        latent_features = np.array(latent_features)
        np.save(os.path.join(args.output_dir, "latent_features.npy"), latent_features)
        np.save(os.path.join(args.output_dir, "fault_labels.npy"), np.array(fault_labels))
        visualize_latent_space(latent_features, fault_labels, args.output_dir, method='scatter')
        if latent_features.shape[1] > 2:
            visualize_latent_space(latent_features, fault_labels, args.output_dir, method='tsne')

if __name__ == '__main__':
    main()