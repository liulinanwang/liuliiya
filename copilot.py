import os
import numpy as np
import scipy.io
from scipy.signal import hilbert, butter, sosfiltfilt
import matplotlib.pyplot as plt
import pywt
from PyEMD import EMD
import pandas as pd

# =============================================================================
# 滤波器函数 和 对比图函数 (保持不变)
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

# =============================================================================
# 原有的预处理和数据增强函数 (保留在此处，但主流程中不使用)
# =============================================================================
def preprocess_signal(signal, fault_type, time_vector):
    """
    对给定的信号执行完整的数据预处理流程。
    """
    print("--- [Note] Generic preprocess_signal function is defined but not used in this workflow. ---")
    return signal

def augment_data(signal, method='add_noise', noise_level=0.02):
    """
    数据增强函数 (用于训练机器学习模型)。
    """
    return signal
    
# =============================================================================
# CWT (Continuous Wavelet Transform) Analysis Function (保持不变)
# =============================================================================
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

# =============================================================================
# HHT (Hilbert-Huang Transform) Analysis Functions (已替换为新版)
# =============================================================================
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
# Main Execution Block (保持不变)
# =============================================================================
def main():
    """
    主执行函数
    """
    # --- 1. 分析开关与设置 ---
    ANALYSIS_SWITCH = {
        'CWT': True,
        'HHT': True,
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

    # --- 3. 打印理论频率 (为HHT提供参考) ---
    if ANALYSIS_SWITCH['HHT']:
        calculate_theoretical_frequencies()

    # --- 4. 循环处理每个文件 ---
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
        
        signal_segment = signal[:num_points]
        time_vector = np.arange(num_points) / fs

        print(f"Loaded original signal. Analyzing the first {num_points} points.")
        
        # --- 5. 执行滤波分离 ---
        print(f"\nSeparating signal with a cutoff frequency of {cutoff_hz} Hz...")
        low_freq_signal = filter_signal(signal_segment, cutoff_hz, fs, 'low')
        high_freq_signal = filter_signal(signal_segment, cutoff_hz, fs, 'high')
        print("Signal separation complete.")
        
        plot_filtered_components(signal_segment, low_freq_signal, high_freq_signal, time_vector, fault_type, cutoff_hz)

        # --- 6. 根据开关执行选择的分析 ---
        if ANALYSIS_SWITCH['CWT']:
            cwt_freq_range = (cutoff_hz, fs / 2.5)
            run_cwt_analysis(high_freq_signal, fs, fault_type, freq_range=cwt_freq_range)
            
        if ANALYSIS_SWITCH['HHT']:
            run_hht_analysis(low_freq_signal, time_vector, fs, fault_type)

if __name__ == '__main__':
    main()