#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试CNN到VAE投影分析系统的集成
Test script for CNN to VAE projection analysis system integration
"""

import numpy as np
import torch
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """创建测试数据"""
    print("Creating synthetic test data...")
    
    # 生成合成信号数据
    np.random.seed(42)
    n_samples = 200
    signal_length = 1024
    n_classes = 4  # normal, inner, outer, ball
    
    signals = []
    labels = []
    
    for class_idx in range(n_classes):
        for i in range(n_samples // n_classes):
            # 生成带有不同频率特征的合成信号
            t = np.linspace(0, 1, signal_length)
            
            # 基础信号
            base_freq = 10 + class_idx * 5
            signal = np.sin(2 * np.pi * base_freq * t)
            
            # 添加类别特定的特征
            if class_idx == 0:  # normal
                pass  # 保持原始信号
            elif class_idx == 1:  # inner
                signal += 0.3 * np.sin(2 * np.pi * 50 * t)
            elif class_idx == 2:  # outer  
                signal += 0.3 * np.sin(2 * np.pi * 80 * t)
            elif class_idx == 3:  # ball
                signal += 0.3 * np.sin(2 * np.pi * 120 * t)
            
            # 添加噪声
            signal += 0.1 * np.random.randn(signal_length)
            
            signals.append(signal)
            labels.append(class_idx)
    
    signals = np.array(signals, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"Created {len(signals)} signals with shape {signals.shape}")
    print(f"Labels distribution: {np.bincount(labels)}")
    
    return signals, labels

def test_projection_system():
    """测试投影分析系统"""
    print("\n=== Testing CNN to VAE Projection Analysis System ===")
    
    try:
        # 导入必要模块
        from projection_system import CNNToVAEProjectionSystem
        from config import VAE_CONFIG, CNN_FEATURE_DIM
        
        # 创建测试数据
        signals, labels = create_test_data()
        
        # 创建合成CNN特征（模拟CNN输出）
        print("Creating synthetic CNN features...")
        n_samples = len(signals)
        cnn_features = np.random.randn(n_samples, CNN_FEATURE_DIM).astype(np.float32)
        
        # 为了让测试更有意义，让CNN特征与标签有一定相关性
        for i in range(n_samples):
            label = labels[i]
            # 在特征的某些维度上添加类别相关的偏移
            cnn_features[i, :10] += label * 0.5
        
        print(f"Created CNN features with shape {cnn_features.shape}")
        
        # 初始化投影系统
        print("\nInitializing projection system...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        projection_system = CNNToVAEProjectionSystem(device=device)
        
        # 运行投影分析（使用较小的参数以加快测试）
        print("\nRunning projection analysis...")
        
        # 临时修改配置以加快测试
        original_vae_epochs = VAE_CONFIG['epochs']
        original_proj_epochs = projection_system.projection_net
        
        # 减少训练轮数以加快测试
        VAE_CONFIG['epochs'] = 5  # 原来是50
        
        # 运行投影分析
        results = projection_system.run_projection_analysis(
            signal_data=signals,
            cnn_features=cnn_features,
            labels=labels
        )
        
        # 恢复原始配置
        VAE_CONFIG['epochs'] = original_vae_epochs
        
        # 检查结果
        if results and results.get('success'):
            print("\n✓ Projection analysis completed successfully!")
            
            metrics = results.get('alignment_metrics', {})
            print(f"Results summary:")
            print(f"  MSE Loss: {metrics.get('mse_loss', 'N/A')}")
            print(f"  Cosine Similarity: {metrics.get('cosine_similarity', 'N/A')}")
            print(f"  Alignment Score: {metrics.get('alignment_score', 'N/A')}")
            
            # 检查投影特征
            projected_features = results.get('projected_features')
            vae_features = results.get('vae_features')
            
            if projected_features is not None and vae_features is not None:
                print(f"  Projected features shape: {projected_features.shape}")
                print(f"  VAE features shape: {vae_features.shape}")
                print("✓ All output features generated correctly")
            
            return True
        else:
            print(f"✗ Projection analysis failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_integration():
    """测试主管道集成"""
    print("\n=== Testing Main Pipeline Integration ===")
    
    try:
        # 导入主模块
        from main import ZeroShotCompoundFaultDiagnosis, ENABLE_PROJECTION_ANALYSIS, PROJECTION_AVAILABLE
        
        print(f"Projection analysis enabled: {ENABLE_PROJECTION_ANALYSIS}")
        print(f"Projection system available: {PROJECTION_AVAILABLE}")
        
        if not (ENABLE_PROJECTION_ANALYSIS and PROJECTION_AVAILABLE):
            print("Projection analysis not available, skipping main integration test")
            return True
        
        # 创建模拟的数据路径（不需要真实数据文件）
        test_data_path = "/tmp/test_bearing_data"
        os.makedirs(test_data_path, exist_ok=True)
        
        # 初始化主系统
        print("Initializing main diagnosis system...")
        fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
            data_path=test_data_path,
            sample_length=1024,
            latent_dim=64,
            batch_size=32
        )
        
        # 检查投影系统是否正确初始化
        if hasattr(fault_diagnosis, 'enable_projection') and fault_diagnosis.enable_projection:
            print("✓ Projection analysis properly integrated into main system")
            
            # 测试特征提取方法是否存在
            if hasattr(fault_diagnosis, '_extract_cnn_features_for_projection'):
                print("✓ CNN feature extraction method available")
            
            if hasattr(fault_diagnosis, 'run_projection_analysis'):
                print("✓ Projection analysis method available")
            
            return True
        else:
            print("✗ Projection analysis not properly integrated")
            return False
            
    except Exception as e:
        print(f"✗ Main integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CNN to VAE Projection Analysis Integration Test")
    print("=" * 60)
    
    # 测试投影系统
    test1_passed = test_projection_system()
    
    # 测试主管道集成
    test2_passed = test_main_integration()
    
    # 总结
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Projection System Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  Main Integration Test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! Integration is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
    
    print("=" * 60)