import os
import numpy as np
import scipy.io
from scipy import signal
from scipy.stats import kurtosis
import pandas as pd


# --- 1. 特征提取函数定义 ---

def calculate_mean_abs(data):
    """
    计算平均峰值 (此处解释为信号绝对值的平均值)
    """
    return np.mean(np.abs(data))


def calculate_std(data):
    """
    计算标准差
    """
    return np.std(data)


def calculate_peak_to_peak(data):
    """
    计算峰峰值
    """
    return np.ptp(data)


def calculate_impulse_factor(data):
    """
    计算冲击因子: 峰值 / 绝对值的平均值
    """
    mean_abs = np.mean(np.abs(data))
    if mean_abs == 0:
        return 0
    return np.max(np.abs(data)) / mean_abs


def calculate_envelope_kurtosis(data):
    """
    计算包络峭度
    """
    # 使用希尔伯特变换获取解析信号
    analytic_signal = signal.hilbert(data)
    # 获取包络线
    envelope = np.abs(analytic_signal)
    # 计算包络线的峭度
    return kurtosis(envelope)


def calculate_spectral_entropy(data):
    """
    计算频谱熵
    """
    # 计算傅里叶变换和功率谱
    fft_vals = np.fft.fft(data)
    ps = np.abs(fft_vals) ** 2

    # 将功率谱归一化为概率分布
    ps_norm = ps / np.sum(ps)

    # 计算熵
    # 添加一个极小值 1e-12 来避免 log(0)
    entropy = -np.sum(ps_norm * np.log2(ps_norm + 1e-12))
    return entropy


def calculate_mean_psd(data, fs=12800):
    """
    计算平均功率谱密度 (PSD)
    使用Welch方法计算PSD，fs是采样频率。
    根据HDU轴承数据集的常用配置，采样频率通常为 12800 Hz。
    """
    _, psd = signal.welch(data, fs=fs)
    return np.mean(psd)


# --- 2. 数据路径和配置 ---

# !!!重要提示: 请确保此路径是您系统上的正确路径!!!
data_path = "E:/研究生/CNN/HDU Bearing Dataset"

# 定义故障文件相对路径
fault_files = {
    'normal': os.path.join('Simple Fault', '1 Normal', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
    'inner': os.path.join('Simple Fault', '2 IF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
    'outer': os.path.join('Simple Fault', '3 OF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
    'ball': os.path.join('Simple Fault', '4 BF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
    'inner_outer': os.path.join('Compound Fault', '1 IF&OF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
    'inner_ball': os.path.join('Compound Fault', '2 IF&BF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
    'outer_ball': os.path.join('Compound Fault', '3 OF&BF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat'),
    'inner_outer_ball': os.path.join('Compound Fault', '5 IF&OF&BF', 'f600', 'rpm1200_f600', 'rpm1200_f600_01.mat')
}

# 采样频率 (根据HDU数据集文档)
SAMPLING_FREQUENCY = 12800

# --- 3. 主处理循环 ---

results = []
print("--- 开始提取特征 ---")

for fault_type, file_name in fault_files.items():
    full_path = os.path.join(data_path, file_name)
    print(f"正在处理: {full_path}...")

    try:
        # 加载 .mat 文件
        mat_data = scipy.io.loadmat(full_path)

        # 动态查找数据键名 (通常与文件名相同)
        file_key = os.path.splitext(os.path.basename(file_name))[0]
        if file_key not in mat_data:
            # 如果找不到，尝试使用文件中的第一个非系统变量名
            potential_keys = [k for k in mat_data if not k.startswith('__')]
            if not potential_keys:
                raise ValueError("在MAT文件中未找到数据键。")
            file_key = potential_keys[0]
            print(
                f"  > 警告: 未找到键 '{os.path.splitext(os.path.basename(file_name))[0]}', 将使用键 '{file_key}' 代替。")

        vibration_data = mat_data[file_key]

        # 根据故障类型选择正确的列
        if fault_type == 'inner':
            # 内圈故障使用第五列 (索引为4)
            signal_data = vibration_data[:, 4].flatten()
        else:
            # 其他故障使用第四列 (索引为3)
            signal_data = vibration_data[:, 3].flatten()

        # --- 4. 计算所有特征 ---
        features = {
            '故障类型': fault_type,
            '平均峰值': calculate_mean_abs(signal_data),
            '标准差': calculate_std(signal_data),
            '频谱熵': calculate_spectral_entropy(signal_data),
            '峰峰值': calculate_peak_to_peak(signal_data),
            '冲击因子': calculate_impulse_factor(signal_data),
            '包络峭度': calculate_envelope_kurtosis(signal_data),
            '平均功率谱密度': calculate_mean_psd(signal_data, fs=SAMPLING_FREQUENCY)
        }
        results.append(features)
        print(f"  > {fault_type} 特征提取完成。")

    except FileNotFoundError:
        print(f"  > 错误: 文件未找到 {full_path}。请检查您的 `data_path` 和文件名是否正确。")
    except Exception as e:
        print(f"  > 错误: 处理 {file_name} 时发生错误: {e}")

# --- 5. 显示结果 ---
if results:
    df_results = pd.DataFrame(results).set_index('故障类型')
    print("\n\n--- 提取的特征汇总 ---")
    # 使用 to_string() 以确保所有列都能完整显示
    print(df_results.to_string())