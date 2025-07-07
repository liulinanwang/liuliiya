import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import hilbert, welch
from scipy.fft import fft, fftfreq

# 数据集路径配置
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

# 采样频率 (根据您的数据集实际情况设置)
fs = 12000  # 假设采样频率为12kHz

def load_vibration_data(file_path, fault_type):
    """加载振动数据，特别处理inner故障在第5列的情况"""
    mat_data = loadmat(file_path)
    if fault_type == 'inner':
        vibration_data = mat_data['Data'][:, 4]  # 内圈故障在第5列
    else:
        vibration_data = mat_data['Data'][:, 3]  # 其他故障在第4列
    return vibration_data

def compute_envelope_spectrum(signal, fs):
    """
    计算信号的包络谱
    :param signal: 输入信号
    :param fs: 采样频率
    :return: 频率轴, 包络谱幅值
    """
    # 1. 计算解析信号(希尔伯特变换)
    analytic_signal = hilbert(signal)
    
    # 2. 计算包络信号
    envelope = np.abs(analytic_signal)
    
    # 3. 对包络信号进行FFT得到包络谱
    n = len(envelope)
    envelope_fft = fft(envelope)
    envelope_spectrum = 2/n * np.abs(envelope_fft[:n//2])
    freq = fftfreq(n, 1/fs)[:n//2]
    
    return freq, envelope_spectrum

def plot_time_domain(signal, fs, title):
    """绘制时域信号"""
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title(f'Time Domain Signal - {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_frequency_domain(signal, fs, title):
    """绘制频域信号"""
    n = len(signal)
    fft_result = fft(signal)
    magnitude = 2/n * np.abs(fft_result[:n//2])
    freq = fftfreq(n, 1/fs)[:n//2]
    
    plt.figure(figsize=(10, 4))
    plt.plot(freq, magnitude)
    plt.title(f'Frequency Domain - {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

def plot_envelope_spectrum(freq, envelope_spectrum, title, max_freq=1000):
    """绘制包络谱"""
    plt.figure(figsize=(10, 4))
    plt.plot(freq[freq <= max_freq], envelope_spectrum[freq <= max_freq])
    plt.title(f'Envelope Spectrum - {title}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def analyze_fault(fault_type, file_path, segment_length=8192):
    """分析特定故障类型的信号"""
    print(f"\nAnalyzing {fault_type} fault...")
    
    # 1. 加载数据
    vibration_data = load_vibration_data(file_path, fault_type)
    
    # 2. 截取一段数据进行分析
    segment = vibration_data[:segment_length]
    
    # 3. 时域分析
    plot_time_domain(segment, fs, fault_type)
    
    # 4. 频域分析
    plot_frequency_domain(segment, fs, fault_type)
    
    # 5. 包络谱分析
    freq, envelope_spectrum = compute_envelope_spectrum(segment, fs)
    plot_envelope_spectrum(freq, envelope_spectrum, fault_type)
    
    return freq, envelope_spectrum

def main():
    # 分析所有故障类型
    all_envelope_spectra = {}
    
    for fault_type, relative_path in fault_files.items():
        file_path = os.path.join(data_path, relative_path)
        if os.path.exists(file_path):
            freq, envelope_spectrum = analyze_fault(fault_type, file_path)
            all_envelope_spectra[fault_type] = (freq, envelope_spectrum)
        else:
            print(f"File not found: {file_path}")
    
    # 比较所有故障的包络谱
    plt.figure(figsize=(12, 8))
    max_freq = 1000  # 只显示0-1000Hz范围
    
    for fault_type, (freq, envelope_spectrum) in all_envelope_spectra.items():
        plt.plot(freq[freq <= max_freq], envelope_spectrum[freq <= max_freq], label=fault_type)
    
    plt.title('Comparison of Envelope Spectra for Different Fault Types')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
