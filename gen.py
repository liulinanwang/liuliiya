import os
import numpy as np
import scipy.io
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import pywt
from PyEMD import EMD
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import seaborn as sns
import random # 导入 random 库

VAE_PARAMS = {
    'latent_dim': 3,      # 潜在空间维度。设置为2或3可以直接可视化，更高维度需要t-SNE降维
    'hidden_dim': 128,    # 隐藏层维度
    'beta': 4.0,          # beta值，控制解耦强度，beta > 1.0
    'epochs': 100,        # 训练轮数
    'batch_size': 64,     # 批量大小
    'learning_rate': 1e-3,# 学习率
    'segment_len': 4096,  # 从每个文件中切分的信号段长度
    'segment_step': 1024, # 切分时的步长（重叠）
    'RANDOM_SEED': 42     # <<<< 新增：用于保证结果可复现的随机种子
}
# 设置PyTorch设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- VAE will run on: {DEVICE} ---")


# =============================================================================
# 新增：设置随机种子的函数
# =============================================================================
def set_seed(seed):
    """
    设置所有相关库的随机种子以确保结果的可复现性。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # 确保CUDA操作的确定性，可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Random seed set to {seed} for reproducibility ---")


# =============================================================================
# 第1部分: 为VAE提取特征的函数
# =============================================================================
# 增强版的特征提取函数
def extract_features_for_vae(signal_segment, fs):
    """
    从信号段中提取时域 + 频域特征，为VAE提供更丰富的信息。
    """
    # --- 1. 时域特征 (与之前相同) ---
    rms = np.sqrt(np.mean(signal_segment**2))
    p2p = np.max(signal_segment) - np.min(signal_segment)
    kurt = kurtosis(signal_segment)
    skewness = skew(signal_segment)
    crest_factor = np.max(np.abs(signal_segment)) / rms if rms > 0 else 0
    shape_factor = rms / np.mean(np.abs(signal_segment)) if np.mean(np.abs(signal_segment)) > 0 else 0
    time_features = np.array([rms, p2p, kurt, skewness, crest_factor, shape_factor])

    # --- 2. 频域特征 ---
    N = len(signal_segment)
    fft_vals = np.fft.fft(signal_segment)
    fft_mag = np.abs(fft_vals[:N // 2]) # 只取正频率部分
    freqs = np.fft.fftfreq(N, 1/fs)[:N // 2]

    # a) 谱峭度 (Spectral Kurtosis): 对冲击性故障很敏感
    spectral_kurtosis = kurtosis(fft_mag)

    # b) 谱偏度 (Spectral Skewness)
    spectral_skewness = skew(fft_mag)

    # c) 谱质心 (Spectral Centroid): 频谱的“重心”
    spectral_centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag) if np.sum(fft_mag) > 0 else 0

    # d) 关键频带能量 (Energy in key frequency bands)
    # 我们可以根据先验知识定义一些频带，例如低频、中频、高频（共振频带）
    low_freq_energy = np.sum(fft_mag[freqs <= 500]**2) / np.sum(fft_mag**2) if np.sum(fft_mag) > 0 else 0
    mid_freq_energy = np.sum(fft_mag[(freqs > 500) & (freqs <= 2500)]**2) / np.sum(fft_mag**2) if np.sum(fft_mag) > 0 else 0
    high_freq_energy = np.sum(fft_mag[freqs > 2500]**2) / np.sum(fft_mag**2) if np.sum(fft_mag) > 0 else 0

    freq_features = np.array([
        spectral_kurtosis, spectral_skewness, spectral_centroid, 
        low_freq_energy, mid_freq_energy, high_freq_energy
    ])

    # --- 3. 组合所有特征 ---
    # 将时域和频域特征拼接起来
    feature_vector = np.concatenate([time_features, freq_features])

    return feature_vector

# =============================================================================
# 第2部分: β-VAE 模型定义
# =============================================================================
class BetaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(BetaVAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim) # 输出层没有激活函数
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 这个操作的随机性受 torch.manual_seed() 控制
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

def vae_loss_function(reconstructed_x, x, mu, logvar, beta):
    """计算β-VAE的损失函数"""
    # 1. 重建损失 (使用均方误差)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    
    # 2. KL散度损失
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kld_loss
    return total_loss

# =============================================================================
# 第3部分: VAE训练与可视化函数
# =============================================================================
def train_and_visualize_vae(data_path, fault_files):
    """
    主函数，用于数据准备、模型训练和潜在空间可视化。
    """
    print("\n\n==================================================")
    print("--- Starting β-VAE Semantic Extraction Process ---")
    print("==================================================")
    
    # --- 1. 数据准备 ---
    print("\nStep 1: Preparing dataset for VAE...")
    all_features = []
    all_labels = []
    fault_type_map = {name: i for i, name in enumerate(fault_files.keys())}
    
    for fault_type, file_path in fault_files.items():
        full_path = os.path.join(data_path, file_path)
        print(f"   - Loading and segmenting: {fault_type}")
        try:
            mat_data = scipy.io.loadmat(full_path)
            col_index = 4 if fault_type == 'inner' else 3
            signal = mat_data['Data'][:, col_index].ravel()

            # 滑动窗口切分数据
            for i in range(0, len(signal) - VAE_PARAMS['segment_len'], VAE_PARAMS['segment_step']):
                segment = signal[i : i + VAE_PARAMS['segment_len']]
                features = extract_features_for_vae(segment, fs=10000)
                all_features.append(features)
                all_labels.append(fault_type_map[fault_type])
        
        except FileNotFoundError:
            print(f"   !!! WARNING: File not found, skipping: {full_path}")
            continue

    if not all_features:
        print("\n!!! ERROR: No data was loaded. Cannot train VAE. Check data paths.")
        return

    features_np = np.array(all_features)
    labels_np = np.array(all_labels)

    # 数据归一化 (非常重要)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_np)
    
    # 转换为PyTorch Tensor
    features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
    dataset = TensorDataset(features_tensor)
    # DataLoader中的shuffle也受torch种子控制
    dataloader = DataLoader(dataset, batch_size=VAE_PARAMS['batch_size'], shuffle=True)
    
    print(f"Dataset prepared: {features_np.shape[0]} samples, {features_np.shape[1]} features each.")

    # --- 2. 模型训练 ---
    print("\nStep 2: Training the β-VAE model...")
    input_dim = features_np.shape[1]
    # 模型初始化受torch种子控制
    model = BetaVAE(
        input_dim=input_dim, 
        hidden_dim=VAE_PARAMS['hidden_dim'],
        latent_dim=VAE_PARAMS['latent_dim']
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_PARAMS['learning_rate'])
    
    model.train()
    for epoch in range(VAE_PARAMS['epochs']):
        total_loss = 0
        for batch_features, in dataloader:
            optimizer.zero_grad()
            reconstructed_x, mu, logvar = model(batch_features)
            loss = vae_loss_function(reconstructed_x, batch_features, mu, logvar, VAE_PARAMS['beta'])
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{VAE_PARAMS['epochs']}], Average Loss: {avg_loss:.4f}")
    
    print("Training complete.")

    # --- 3. 可视化潜在空间 (语义信息) ---
    print("\nStep 3: Visualizing the learned semantic latent space...")
    model.eval()
    with torch.no_grad():
        # 使用编码器获取所有数据的潜在向量表示 (mu)
        latent_vectors_mu, _ = model.encode(features_tensor)
    
    latent_vectors_np = latent_vectors_mu.cpu().numpy()
    
    # 创建一个DataFrame方便绘图
    label_names = list(fault_files.keys())
    df = pd.DataFrame(data=latent_vectors_np, columns=[f'Z_{i+1}' for i in range(VAE_PARAMS['latent_dim'])])
    df['label_id'] = labels_np
    df['label_name'] = df['label_id'].apply(lambda x: label_names[x])
    
    plt.style.use('default') # 恢复默认绘图风格

    if VAE_PARAMS['latent_dim'] == 2:
        plt.figure(figsize=(12, 10), dpi=150)
        sns.scatterplot(
            x='Z_1', y='Z_2',
            hue='label_name',
            palette=sns.color_palette("hsv", n_colors=len(fault_type_map)),
            data=df,
            legend='full',
            alpha=0.7
        )
        plt.title('2D Latent Space Visualization of Fault Semantics (β-VAE)', fontsize=16)
        plt.xlabel('Latent Dimension 1', fontsize=12)
        plt.ylabel('Latent Dimension 2', fontsize=12)
    
    elif VAE_PARAMS['latent_dim'] == 3:
        fig = plt.figure(figsize=(14, 12), dpi=150)
        ax = fig.add_subplot(projection='3d')
        colors = sns.color_palette("hsv", n_colors=len(fault_type_map))
        for i, label_name in enumerate(label_names):
            subset = df[df['label_name'] == label_name]
            ax.scatter(subset['Z_1'], subset['Z_2'], subset['Z_3'], c=[colors[i]], label=label_name, s=30, alpha=0.7)
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        ax.set_title('3D Latent Space Visualization of Fault Semantics (β-VAE)', fontsize=16)
        ax.legend()
        
    else: # 使用 t-SNE 降维进行可视化
        print(f"Latent dimension is {VAE_PARAMS['latent_dim']}. Using t-SNE for 2D visualization.")
        plt.figure(figsize=(12, 10), dpi=150)
        
        # <<<< 修改：为TSNE添加random_state以确保可复现性
        tsne = TSNE(
            n_components=2, 
            verbose=1, 
            perplexity=40, 
            n_iter=300, 
            random_state=VAE_PARAMS['RANDOM_SEED']
        )
        
        tsne_results = tsne.fit_transform(latent_vectors_np)
        df['tsne-2d-one'] = tsne_results[:,0]
        df['tsne-2d-two'] = tsne_results[:,1]

        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label_name",
            palette=sns.color_palette("hsv", n_colors=len(fault_type_map)),
            data=df,
            legend="full",
            alpha=0.7
        )
        plt.title('t-SNE Visualization of Latent Space (β-VAE)', fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)

    plt.grid(True)
    plt.legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# =============================================================================
# 原始的CWT/HHT分析函数 (完整无省略)
# =============================================================================
def filter_signal(signal, cutoff_hz, fs, filter_type='low'):
    """
    对信号应用一个零相位的高通或低通巴特沃斯滤波器。
    """
    sos = butter(N=4, Wn=cutoff_hz / (0.5 * fs), btype=filter_type, analog=False, output='sos')
    filtered_signal = sosfiltfilt(sos, signal)
    return filtered_signal

def plot_filtered_components(original, low, high, time_vector, fault_type, cutoff_hz):
    """
    可视化原始信号、低频分量和高频分量。
    """
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
    
def run_cwt_analysis(signal, sampling_rate, fault_name, freq_range=(800, 4000)):
    """
    对给定的信号执行CWT分析并绘图 (修改后专注于指定频率范围)。
    """
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

def calculate_theoretical_frequencies():
    """计算轴承故障的理论特征频率"""
    rpm = 1000 
    fr = rpm / 60
    
    bpfo = 59.35
    bpfi = 90.25
    bsf = 67.18

    freqs = {
        'Rotation': fr, 'Outer': bpfo, 'Inner': bpfi, 'Ball': bsf
    }
    print("--- Theoretical Fault Frequencies (for reference) ---")
    for name, freq in freqs.items():
        print(f"{name} Frequency (Hz): {freq:.2f}")
    print("---------------------------------------------------\n")
    return freqs

def run_hht_analysis(signal, time_vector, fs, fault_type, top_n=4):
    """
    对给定的信号执行完整的HHT分析 (根据新版代码逻辑)。
    1. EMD分解
    2. 根据能量选择Top N个IMF
    3. 绘制原始信号和被选中的IMF
    4. 对被选中的IMF进行希尔伯特变换并绘制瞬时频率
    """
    print(f"\n--- Running HHT Analysis on Low-Freq Component [{fault_type}] (New Method) ---")
    
    # 1. EMD分解
    emd = EMD()
    imfs = emd(signal)
    n_imfs = imfs.shape[0]
    print(f"Signal decomposed into {n_imfs - 1} IMFs and 1 Residual.")
    
    # 2. 根据能量选择Top N个IMF
    imf_energies = [np.sum(imf**2) for imf in imfs[:-1]] # 排除残差分量
    sorted_energy_indices = np.argsort(imf_energies)
    
    num_to_select = min(top_n, n_imfs - 1)
    if num_to_select <= 0:
        print("No IMFs were generated. Skipping HHT analysis.")
        return
        
    top_indices = sorted_energy_indices[-num_to_select:]
    top_indices_sorted = np.sort(top_indices) # 按IMF序号排序
    
    print(f"Selected top {num_to_select} IMFs by energy (indices sorted for plotting): {top_indices_sorted + 1}")
    selected_imfs = imfs[top_indices_sorted]

    # 3. 绘制原始信号和被选中的IMF
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

    # 4. 对被选中的IMF进行希尔伯特变换并绘制瞬时频率
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
        plt.ylim(0, 500) # 设置合理的Y轴范围以清晰观察低频特征
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =============================================================================
# Main Execution Block (完整无省略)
# =============================================================================
def main():
    """
    主执行函数
    """
    # <<<< 修改：在主函数开始时设置随机种子
    set_seed(VAE_PARAMS['RANDOM_SEED'])
    
    # --- 1. 分析开关与设置 ---
    ANALYSIS_SWITCH = {
        'CWT': False,     # 设为True来运行原始的CWT分析
        'HHT': False,     # 设为True来运行原始的HHT分析
        'VAE': True       # <<<< 设为True来运行新增的VAE语义提取
    }
    
    cutoff_hz = 2000.0
    
    # !!!重要!!! 请务必将此路径修改为您自己的数据集路径
    # 例如: "D:/data/HDU_Bearing_Dataset" 或 "/home/user/data/HDU_Bearing_Dataset"
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
    num_points_for_plot = 4096 # 仅用于原始的绘图

    # --- 2. 运行原始的CWT/HHT分析 (如果开关为True) ---
    if ANALYSIS_SWITCH['CWT'] or ANALYSIS_SWITCH['HHT']:
        print("--- Running original CWT/HHT analysis ---")
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
                print("!!! Please check your data_path variable and file structure.\n")
                continue

            col_index = 4 if fault_type == 'inner' else 3
            signal = mat_data['Data'][:, col_index].ravel()
            
            signal_segment = signal[:num_points_for_plot]
            time_vector = np.arange(num_points_for_plot) / fs

            print(f"Loaded original signal. Analyzing the first {num_points_for_plot} points.")
            print(f"\nSeparating signal with a cutoff frequency of {cutoff_hz} Hz...")
            low_freq_signal = filter_signal(signal_segment, cutoff_hz, fs, 'low')
            high_freq_signal = filter_signal(signal_segment, cutoff_hz, fs, 'high')
            print("Signal separation complete.")
            
            plot_filtered_components(signal_segment, low_freq_signal, high_freq_signal, time_vector, fault_type, cutoff_hz)

            if ANALYSIS_SWITCH['CWT']:
                cwt_freq_range = (cutoff_hz, fs / 2.5)
                run_cwt_analysis(high_freq_signal, fs, fault_type, freq_range=cwt_freq_range)
                
            if ANALYSIS_SWITCH['HHT']:
                run_hht_analysis(low_freq_signal, time_vector, fs, fault_type)
    
    # --- 3. 运行新增的VAE语义提取与可视化 (如果开关为True) ---
    if ANALYSIS_SWITCH['VAE']:
        train_and_visualize_vae(data_path, fault_files)


if __name__ == '__main__':
    main()