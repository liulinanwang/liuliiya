#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义投影网络测试脚本

用于验证semantic_projection.py的各个组件功能
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

def test_semantic_projection():
    """测试语义投影网络基本功能"""
    print("=" * 50)
    print("测试1: 语义投影网络基本功能")
    print("=" * 50)
    
    import semantic_projection as sp
    
    # 测试不同的维度配置
    test_configs = [
        (512, 16),   # 默认配置
        (256, 8),    # 较小配置
        (1024, 32),  # 较大配置
    ]
    
    for cae_dim, vae_dim in test_configs:
        print(f"\n测试配置: CAE={cae_dim}维 -> VAE={vae_dim}维")
        
        # 创建模型
        model = sp.SemanticProjection(cae_dim=cae_dim, vae_dim=vae_dim)
        
        # 测试前向传播
        batch_size = 32
        test_input = torch.randn(batch_size, cae_dim)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  输入形状: {test_input.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 验证输出形状
        assert output.shape == (batch_size, vae_dim), f"输出形状不匹配"
        print("  ✓ 前向传播测试通过")
    
    print("\n✓ 语义投影网络测试完成")


def test_data_generator():
    """测试数据生成器"""
    print("\n" + "=" * 50)
    print("测试2: 数据生成器功能")
    print("=" * 50)
    
    import semantic_projection as sp
    
    # 测试不同的配置
    test_configs = [
        {'cae_feature_dim': 512, 'vae_semantic_dim': 16},
        {'cae_feature_dim': 256, 'vae_semantic_dim': 8},
    ]
    
    for config in test_configs:
        print(f"\n测试配置: {config}")
        
        data_gen = sp.DataGenerator(config)
        
        # 测试VAE语义特征加载
        print(f"  VAE语义特征类型数: {len(data_gen.vae_semantics)}")
        print(f"  故障类型: {list(data_gen.vae_semantics.keys())}")
        
        # 测试CAE特征生成
        fault_type = 'normal'
        num_samples = 100
        cae_features = data_gen.generate_cae_features(fault_type, num_samples)
        
        print(f"  生成的CAE特征形状: {cae_features.shape}")
        assert cae_features.shape == (num_samples, config['cae_feature_dim'])
        print("  ✓ CAE特征生成测试通过")
        
        # 测试数据集生成
        train_ds, val_ds, train_labels, val_labels = data_gen.generate_dataset(
            num_samples_per_type=50, train_split=0.8
        )
        
        print(f"  训练集大小: {len(train_ds)}")
        print(f"  验证集大小: {len(val_ds)}")
        print(f"  训练标签数: {len(train_labels)}")
        print(f"  验证标签数: {len(val_labels)}")
        print("  ✓ 数据集生成测试通过")
    
    print("\n✓ 数据生成器测试完成")


def test_trainer():
    """测试训练器功能"""
    print("\n" + "=" * 50)
    print("测试3: 训练器功能")
    print("=" * 50)
    
    import semantic_projection as sp
    from torch.utils.data import DataLoader
    
    # 快速训练配置
    config = sp.CONFIG.copy()
    config['epochs'] = 5
    config['batch_size'] = 32
    config['early_stopping_patience'] = 3
    
    # 生成小量数据
    data_gen = sp.DataGenerator(config)
    train_ds, val_ds, _, _ = data_gen.generate_dataset(num_samples_per_type=100)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
    # 创建模型和训练器
    model = sp.SemanticProjection()
    trainer = sp.SemanticProjectionTrainer(model, config=config)
    
    print("开始快速训练测试...")
    history = trainer.train(train_loader, val_loader, verbose=False)
    
    print(f"  训练完成，总轮数: {len(history['train_losses'])}")
    print(f"  最终训练损失: {history['train_losses'][-1]:.6f}")
    print(f"  最终验证损失: {history['val_losses'][-1]:.6f}")
    print(f"  最佳验证损失: {history['best_val_loss']:.6f}")
    print("  ✓ 训练器测试通过")
    
    print("\n✓ 训练器测试完成")


def test_visualizer():
    """测试可视化器功能"""
    print("\n" + "=" * 50)
    print("测试4: 可视化器功能")
    print("=" * 50)
    
    import semantic_projection as sp
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    vae_features = np.random.randn(n_samples, 16)
    cae_features = np.random.randn(n_samples, 16)
    labels = ['normal', 'inner', 'outer', 'ball'] * (n_samples // 4)
    
    visualizer = sp.SemanticVisualizer()
    
    # 测试特征对齐分析
    print("  测试特征对齐分析...")
    try:
        alignment_scores = visualizer.visualize_feature_alignment(
            vae_features, cae_features, labels, 
            save_path='test_alignment.png'
        )
        print(f"    对齐分析完成，包含{len(alignment_scores)}个故障类型")
        print("    ✓ 特征对齐分析测试通过")
    except Exception as e:
        print(f"    ✗ 特征对齐分析测试失败: {e}")
    
    # 测试语义空间可视化
    print("  测试语义空间可视化...")
    try:
        visualizer.visualize_semantic_space(
            vae_features[:50], cae_features[:50], labels[:50],
            save_path='test_semantic_space.png'
        )
        print("    ✓ 语义空间可视化测试通过")
    except Exception as e:
        print(f"    ✗ 语义空间可视化测试失败: {e}")
    
    # 测试训练历史可视化
    print("  测试训练历史可视化...")
    try:
        train_losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        val_losses = [0.9, 0.7, 0.5, 0.3, 0.1]
        visualizer.visualize_training_history(
            train_losses, val_losses,
            save_path='test_training_history.png'
        )
        print("    ✓ 训练历史可视化测试通过")
    except Exception as e:
        print(f"    ✗ 训练历史可视化测试失败: {e}")
    
    print("\n✓ 可视化器测试完成")


def test_integration():
    """集成测试"""
    print("\n" + "=" * 50)
    print("测试5: 集成测试")
    print("=" * 50)
    
    import semantic_projection as sp
    from torch.utils.data import DataLoader
    
    # 完整流程测试
    print("  执行完整流程测试...")
    
    # 配置
    config = sp.CONFIG.copy()
    config['epochs'] = 3
    config['batch_size'] = 32
    
    try:
        # 1. 数据生成
        data_gen = sp.DataGenerator(config)
        train_ds, val_ds, train_labels, val_labels = data_gen.generate_dataset(
            num_samples_per_type=80
        )
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
        
        # 2. 模型训练
        model = sp.SemanticProjection()
        trainer = sp.SemanticProjectionTrainer(model, config=config)
        history = trainer.train(train_loader, val_loader, verbose=False)
        
        # 3. 模型评估
        model.eval()
        test_cae_features = []
        test_vae_features = []
        
        with torch.no_grad():
            for cae_feat, vae_feat in val_loader:
                projected_feat = model(cae_feat.to(sp.DEVICE))
                test_cae_features.append(projected_feat.cpu().numpy())
                test_vae_features.append(vae_feat.numpy())
        
        test_cae_features = np.vstack(test_cae_features)
        test_vae_features = np.vstack(test_vae_features)
        
        # 4. 可视化（仅测试不保存）
        visualizer = sp.SemanticVisualizer(config)
        alignment_scores = visualizer.visualize_feature_alignment(
            test_vae_features, test_cae_features, val_labels,
            save_path='test_integration_alignment.png'
        )
        
        print(f"    训练完成，验证损失: {history['val_losses'][-1]:.6f}")
        print(f"    特征对齐度: {np.mean([s['mean_similarity'] for s in alignment_scores]):.6f}")
        print("    ✓ 集成测试通过")
        
    except Exception as e:
        print(f"    ✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ 集成测试完成")


def main():
    """主测试函数"""
    print("开始语义投影网络测试套件...")
    print("设备:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 执行所有测试
    test_semantic_projection()
    test_data_generator()
    test_trainer()
    test_visualizer()
    test_integration()
    
    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)
    
    # 清理测试文件
    import os
    test_files = [
        'test_alignment.png',
        'test_semantic_space.png', 
        'test_training_history.png',
        'test_integration_alignment.png'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    print("测试文件已清理")


if __name__ == "__main__":
    main()