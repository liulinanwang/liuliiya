#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义投影网络演示脚本

演示如何使用semantic_projection.py进行CAE特征到VAE语义空间的投影
包括训练、测试和可视化的完整流程。

使用方法:
python demo_semantic_projection.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端避免显示问题

# 导入我们的语义投影模块
import semantic_projection as sp
from torch.utils.data import DataLoader

def main():
    """主演示函数"""
    print("=" * 60)
    print("CAE特征到VAE语义空间的投影网络 - 演示程序")
    print("=" * 60)
    
    # 1. 设置演示参数
    demo_config = sp.CONFIG.copy()
    demo_config['epochs'] = 50  # 减少训练轮数用于演示
    demo_config['batch_size'] = 64
    demo_config['learning_rate'] = 1e-3
    demo_config['early_stopping_patience'] = 10
    
    print(f"演示配置:")
    print(f"  - CAE特征维度: {demo_config['cae_feature_dim']}")
    print(f"  - VAE语义维度: {demo_config['vae_semantic_dim']}")
    print(f"  - 训练轮数: {demo_config['epochs']}")
    print(f"  - 批处理大小: {demo_config['batch_size']}")
    print(f"  - 学习率: {demo_config['learning_rate']}")
    
    # 2. 生成演示数据
    print("\n" + "=" * 60)
    print("步骤1: 生成演示数据")
    print("=" * 60)
    
    data_generator = sp.DataGenerator(demo_config)
    
    # 显示VAE语义特征信息
    print(f"VAE语义特征信息:")
    for fault_type, semantic_vector in data_generator.vae_semantics.items():
        print(f"  {fault_type}: {semantic_vector.shape}, 范数={np.linalg.norm(semantic_vector):.3f}")
    
    # 生成数据集
    train_dataset, val_dataset, train_labels, val_labels = data_generator.generate_dataset(
        num_samples_per_type=300,  # 每个故障类型300个样本
        train_split=0.8
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=demo_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=demo_config['batch_size'], shuffle=False)
    
    print(f"数据集统计:")
    print(f"  - 训练集大小: {len(train_dataset)}")
    print(f"  - 验证集大小: {len(val_dataset)}")
    print(f"  - 故障类型数: {len(data_generator.fault_types)}")
    
    # 3. 创建和训练模型
    print("\n" + "=" * 60)
    print("步骤2: 创建和训练语义投影网络")
    print("=" * 60)
    
    # 创建模型
    model = sp.SemanticProjection(
        cae_dim=demo_config['cae_feature_dim'],
        vae_dim=demo_config['vae_semantic_dim'],
        dropout_rate=demo_config['dropout_rate']
    )
    
    print(f"模型信息:")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 模型结构: {model}")
    
    # 创建训练器
    trainer = sp.SemanticProjectionTrainer(model, sp.DEVICE, demo_config)
    
    # 开始训练
    print("\n开始训练...")
    training_history = trainer.train(train_loader, val_loader, verbose=True)
    
    # 4. 评估模型
    print("\n" + "=" * 60)
    print("步骤3: 评估模型性能")
    print("=" * 60)
    
    # 加载最佳模型
    trainer.load_model('best_model.pth')
    model.eval()
    
    # 在验证集上评估
    test_cae_features = []
    test_vae_features = []
    test_labels_numeric = []
    
    with torch.no_grad():
        for i, (cae_feat, vae_feat) in enumerate(val_loader):
            cae_feat = cae_feat.to(sp.DEVICE)
            projected_feat = model(cae_feat)
            
            test_cae_features.append(projected_feat.cpu().numpy())
            test_vae_features.append(vae_feat.numpy())
            
            # 计算对应的标签
            batch_size = cae_feat.size(0)
            start_idx = i * demo_config['batch_size']
            end_idx = min(start_idx + batch_size, len(val_labels))
            batch_labels = val_labels[start_idx:end_idx]
            test_labels_numeric.extend(batch_labels)
    
    # 合并结果
    test_cae_features = np.vstack(test_cae_features)
    test_vae_features = np.vstack(test_vae_features)
    
    # 计算性能指标
    print("性能评估:")
    
    # 计算重构误差
    reconstruction_errors = np.mean((test_cae_features - test_vae_features) ** 2, axis=1)
    print(f"  - 平均重构误差: {np.mean(reconstruction_errors):.6f}")
    print(f"  - 重构误差标准差: {np.std(reconstruction_errors):.6f}")
    
    # 计算余弦相似度
    cosine_similarities = []
    for i in range(len(test_cae_features)):
        cae_feat = test_cae_features[i]
        vae_feat = test_vae_features[i]
        similarity = np.dot(cae_feat, vae_feat) / (np.linalg.norm(cae_feat) * np.linalg.norm(vae_feat) + 1e-8)
        cosine_similarities.append(similarity)
    
    print(f"  - 平均余弦相似度: {np.mean(cosine_similarities):.6f}")
    print(f"  - 余弦相似度标准差: {np.std(cosine_similarities):.6f}")
    
    # 5. 可视化结果
    print("\n" + "=" * 60)
    print("步骤4: 生成可视化结果")
    print("=" * 60)
    
    visualizer = sp.SemanticVisualizer(demo_config)
    
    # 创建结果保存目录
    results_dir = 'demo_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1) 训练历史可视化
    print("生成训练历史可视化...")
    visualizer.visualize_training_history(
        training_history['train_losses'],
        training_history['val_losses'],
        save_path=os.path.join(results_dir, 'training_history.png')
    )
    
    # 2) 语义空间可视化
    print("生成语义空间可视化...")
    visualizer.visualize_semantic_space(
        test_vae_features,
        test_cae_features,
        test_labels_numeric,
        save_path=os.path.join(results_dir, 'semantic_space_comparison.png'),
        title="CAE特征到VAE语义空间的投影效果"
    )
    
    # 3) 特征对齐分析
    print("生成特征对齐分析...")
    alignment_scores = visualizer.visualize_feature_alignment(
        test_vae_features,
        test_cae_features,
        test_labels_numeric,
        save_path=os.path.join(results_dir, 'feature_alignment_analysis.png')
    )
    
    # 6. 保存演示结果
    print("\n" + "=" * 60)
    print("步骤5: 保存演示结果")
    print("=" * 60)
    
    # 保存结果摘要
    results_summary = {
        'config': demo_config,
        'training_history': {
            'final_train_loss': training_history['train_losses'][-1],
            'final_val_loss': training_history['val_losses'][-1],
            'best_val_loss': training_history['best_val_loss'],
            'total_epochs': len(training_history['train_losses']),
            'training_time': training_history['total_time']
        },
        'performance_metrics': {
            'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
            'std_reconstruction_error': float(np.std(reconstruction_errors)),
            'mean_cosine_similarity': float(np.mean(cosine_similarities)),
            'std_cosine_similarity': float(np.std(cosine_similarities))
        },
        'alignment_analysis': {
            'fault_types': [score['fault_type'] for score in alignment_scores],
            'mean_similarities': [float(score['mean_similarity']) for score in alignment_scores],
            'std_similarities': [float(score['std_similarity']) for score in alignment_scores]
        }
    }
    
    # 保存到文件
    import json
    with open(os.path.join(results_dir, 'demo_results_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"演示结果已保存到: {results_dir}/")
    print(f"  - 训练历史图: training_history.png")
    print(f"  - 语义空间对比图: semantic_space_comparison.png")
    print(f"  - 特征对齐分析图: feature_alignment_analysis.png")
    print(f"  - 结果摘要: demo_results_summary.json")
    
    # 7. 演示如何使用训练好的模型
    print("\n" + "=" * 60)
    print("步骤6: 演示模型使用")
    print("=" * 60)
    
    print("演示如何使用训练好的模型进行特征投影:")
    
    # 创建一个示例CAE特征
    example_cae_feature = np.random.randn(1, demo_config['cae_feature_dim']).astype(np.float32)
    example_cae_tensor = torch.FloatTensor(example_cae_feature).to(sp.DEVICE)
    
    # 进行投影
    with torch.no_grad():
        projected_feature = model(example_cae_tensor)
        projected_feature_numpy = projected_feature.cpu().numpy()
    
    print(f"  - 输入CAE特征形状: {example_cae_feature.shape}")
    print(f"  - 输出VAE语义特征形状: {projected_feature_numpy.shape}")
    print(f"  - 输出特征前5个元素: {projected_feature_numpy[0][:5]}")
    
    # 8. 总结
    print("\n" + "=" * 60)
    print("演示完成 - 总结")
    print("=" * 60)
    
    print("本演示成功展示了:")
    print("  1. ✓ 语义投影网络的创建和训练")
    print("  2. ✓ CAE特征到VAE语义空间的投影")
    print("  3. ✓ 特征对齐效果的评估")
    print("  4. ✓ 多种可视化分析方法")
    print("  5. ✓ 模型的保存和加载")
    print("  6. ✓ 训练好的模型的使用")
    
    print(f"\n最终性能指标:")
    print(f"  - 最佳验证损失: {training_history['best_val_loss']:.6f}")
    print(f"  - 平均特征对齐度: {np.mean([score['mean_similarity'] for score in alignment_scores]):.6f}")
    print(f"  - 训练时间: {training_history['total_time']:.2f}秒")
    
    print("\n" + "=" * 60)
    print("感谢使用CAE特征到VAE语义空间的投影网络！")
    print("=" * 60)
    
    return model, trainer, visualizer, results_summary

if __name__ == "__main__":
    try:
        model, trainer, visualizer, results = main()
        print("\n✓ 演示程序成功完成！")
    except Exception as e:
        print(f"\n✗ 演示程序出现错误: {e}")
        import traceback
        traceback.print_exc()