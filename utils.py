import numpy as np
import pywt
from PyEMD import CEEMDAN, EMD
from scipy.stats import kurtosis, skew, entropy
import pyfftw
from scipy import signal


def wavelet_packet_energy(signal_data, wavelet='db4', level=3):
    """
    小波包分解提取能量特征，适合捕捉球体故障的非平稳特性
    """
    wp = pywt.WaveletPacket(data=signal_data, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = [node.path for node in wp.get_level(level)]

    # 提取节点能量特征并归一化
    energy_features = {}
    total_energy = 0
    for node in nodes:
        node_data = wp[node].data
        energy = np.sum(node_data ** 2)
        energy_features[f'wp_energy_{node}'] = energy
        total_energy += energy

    # 归一化能量特征
    if total_energy > 0:
        for key in energy_features:
            energy_features[key] /= total_energy

    return energy_features


def emd_feature_extraction(signal_data, num_imfs=5):
    """
    使用经验模态分解提取特征，适合球体故障的非线性特性
    """
    emd = EMD()
    imfs = emd(signal_data, max_imf=num_imfs)

    features = {}
    for i, imf in enumerate(imfs[:num_imfs]):
        # 提取每个IMF的统计特征
        features[f'imf{i}_mean'] = np.mean(imf)
        features[f'imf{i}_std'] = np.std(imf)
        features[f'imf{i}_kurtosis'] = kurtosis(imf)
        features[f'imf{i}_skewness'] = skew(imf)

        # 提取IMF的频域特征
        fft_mag = np.abs(np.fft.fft(imf))
        features[f'imf{i}_peak_freq'] = np.argmax(fft_mag[:len(fft_mag) // 2]) / len(fft_mag) * 2
        features[f'imf{i}_energy'] = np.sum(fft_mag ** 2) / len(fft_mag)

    return features


def cyclic_spectral_coherence(signal_data, fs, fmax=None, alpha_max=None):
    """
    循环谱相干分析，特别适合提取调制信号特征
    """
    if fmax is None:
        fmax = fs / 2  # 奈奎斯特频率
    if alpha_max is None:
        alpha_max = fs / 4  # 最大循环频率

    # 分段计算
    nperseg = min(1024, len(signal_data) // 10)
    noverlap = nperseg // 2

    # 使用Welch方法估计循环谱密度
    f, alpha, Sxx = signal.cyclic_spectrum(signal_data, fs, nperseg=nperseg, noverlap=noverlap,
                                           fmax=fmax, alpha_max=alpha_max)

    # 提取循环谱特征
    features = {}

    # 找出显著的循环频率
    alpha_energy = np.sum(np.abs(Sxx), axis=0)
    top_alpha_indices = np.argsort(alpha_energy)[-5:]  # 提取5个最显著的循环频率

    for i, idx in enumerate(top_alpha_indices):
        features[f'cyclic_freq_{i}'] = alpha[idx]
        features[f'cyclic_energy_{i}'] = alpha_energy[idx]

    # 集成整体特征
    features['cyclic_total_energy'] = np.sum(alpha_energy)
    features['cyclic_max_energy'] = np.max(alpha_energy)

    return features


def teager_kaiser_energy(signal_data):
    """
    Teager-Kaiser能量算子，能有效捕捉瞬态变化
    """
    n = len(signal_data)
    tk_energy = np.zeros(n - 2)

    for i in range(1, n - 1):
        tk_energy[i - 1] = signal_data[i] ** 2 - signal_data[i - 1] * signal_data[i + 1]

    # 提取TK能量特征
    features = {
        'tk_mean': np.mean(tk_energy),
        'tk_std': np.std(tk_energy),
        'tk_max': np.max(tk_energy),
        'tk_kurtosis': kurtosis(tk_energy)
    }

    return features


def self_correlation_features(signal_data, max_lag=100):
    """
    自相关特征提取，适合周期性故障特征
    """
    correlation = np.correlate(signal_data, signal_data, mode='full')
    correlation = correlation[len(correlation) // 2:len(correlation) // 2 + max_lag]

    # 归一化
    correlation = correlation / correlation[0]

    # 找出自相关的峰值
    peaks, _ = signal.find_peaks(correlation, height=0.2, distance=5)

    features = {}
    if len(peaks) > 0:
        features['corr_first_peak_pos'] = peaks[0]
        features['corr_first_peak_val'] = correlation[peaks[0]]
        features['corr_peak_mean'] = np.mean(correlation[peaks])
        features['corr_peak_std'] = np.std(correlation[peaks])

        # 计算峰值间距
        if len(peaks) > 1:
            peak_distances = np.diff(peaks)
            features['corr_peak_dist_mean'] = np.mean(peak_distances)
            features['corr_peak_dist_std'] = np.std(peak_distances)

    return features


def extract_advanced_features(signal_data, fs):
    """
    综合高级特征提取函数
    """
    features = {}

    # 1. 小波包能量特征
    wp_features = wavelet_packet_energy(signal_data)
    features.update(wp_features)

    # 2. EMD特征
    emd_features = emd_feature_extraction(signal_data)
    features.update(emd_features)

    # 3. Teager-Kaiser能量特征
    tk_features = teager_kaiser_energy(signal_data)
    features.update(tk_features)

    # 4. 自相关特征
    corr_features = self_correlation_features(signal_data)
    features.update(corr_features)

    # 5. 循环谱特征（计算密集，可选）
    # cyclic_features = cyclic_spectral_coherence(signal_data, fs)
    # features.update(cyclic_features)

    return features