#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义投影网络使用示例

展示如何使用训练好的语义投影网络进行CAE特征到VAE语义空间的投影
"""

import numpy as np
import torch
import semantic_projection as sp

def inference_example():
    """推理示例"""
    print("=" * 60)
    print("语义投影网络推理示例")
    print("=" * 60)
    
    # 1. 加载训练好的模型
    print("1. 加载训练好的模型...")
    model = sp.SemanticProjection(cae_dim=512, vae_dim=16)
    trainer = sp.SemanticProjectionTrainer(model)
    
    try:
        trainer.load_model('best_model.pth')
        print("   ✓ 模型加载成功")
    except FileNotFoundError:
        print("   ⚠ 未找到预训练模型，将使用随机初始化的模型")
    
    model.eval()  # 设置为推理模式
    
    # 2. 准备测试数据
    print("\n2. 准备测试数据...")
    
    # 模拟CAE特征（实际使用时，这些特征来自于训练好的CAE模型）
    batch_size = 5
    cae_features = np.random.randn(batch_size, 512).astype(np.float32)
    
    # 为每个样本分配故障类型标签（仅用于演示）
    fault_types = ['normal', 'inner', 'outer', 'ball', 'inner_outer']
    
    print(f"   输入CAE特征形状: {cae_features.shape}")
    print(f"   测试样本数: {batch_size}")
    print(f"   故障类型: {fault_types}")
    
    # 3. 进行特征投影
    print("\n3. 进行特征投影...")
    
    # 转换为PyTorch张量
    cae_tensor = torch.FloatTensor(cae_features)
    
    # 推理
    with torch.no_grad():
        projected_features = model(cae_tensor)
        projected_features_np = projected_features.cpu().numpy()
    
    print(f"   输出VAE语义特征形状: {projected_features_np.shape}")
    print(f"   投影成功完成")
    
    # 4. 分析投影结果
    print("\n4. 分析投影结果...")
    
    for i in range(batch_size):
        cae_feat = cae_features[i]
        vae_feat = projected_features_np[i]
        fault_type = fault_types[i]
        
        # 计算特征统计
        cae_norm = np.linalg.norm(cae_feat)
        vae_norm = np.linalg.norm(vae_feat)
        
        print(f"   样本 {i+1} ({fault_type}):")
        print(f"     CAE特征范数: {cae_norm:.4f}")
        print(f"     VAE特征范数: {vae_norm:.4f}")
        print(f"     VAE特征预览: [{vae_feat[0]:.3f}, {vae_feat[1]:.3f}, {vae_feat[2]:.3f}, ...]")
    
    # 5. 与参考VAE语义特征对比
    print("\n5. 与参考VAE语义特征对比...")
    
    # 加载参考VAE语义特征
    data_gen = sp.DataGenerator()
    reference_semantics = data_gen.vae_semantics
    
    print("   参考VAE语义特征:")
    for fault_type in fault_types:
        if fault_type in reference_semantics:
            ref_feat = reference_semantics[fault_type]
            proj_feat = projected_features_np[fault_types.index(fault_type)]
            
            # 计算余弦相似度
            similarity = np.dot(ref_feat, proj_feat) / (
                np.linalg.norm(ref_feat) * np.linalg.norm(proj_feat) + 1e-8
            )
            
            print(f"     {fault_type}: 相似度 = {similarity:.4f}")
    
    # 6. 批量处理示例
    print("\n6. 批量处理示例...")
    
    # 生成更多测试数据
    large_batch_size = 100
    large_cae_features = np.random.randn(large_batch_size, 512).astype(np.float32)
    
    # 批量推理
    with torch.no_grad():
        large_cae_tensor = torch.FloatTensor(large_cae_features)
        large_projected = model(large_cae_tensor)
        large_projected_np = large_projected.cpu().numpy()
    
    print(f"   批量处理: {large_batch_size} 个样本")
    print(f"   输入形状: {large_cae_features.shape}")
    print(f"   输出形状: {large_projected_np.shape}")
    
    # 计算批量统计
    mean_norm = np.mean(np.linalg.norm(large_projected_np, axis=1))
    std_norm = np.std(np.linalg.norm(large_projected_np, axis=1))
    
    print(f"   输出特征范数: 均值={mean_norm:.4f}, 标准差={std_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("推理示例完成")
    print("=" * 60)
    
    return projected_features_np

def integration_example():
    """集成示例：展示如何将投影网络集成到完整的故障诊断流程中"""
    print("\n" + "=" * 60)
    print("集成应用示例")
    print("=" * 60)
    
    # 1. 初始化组件
    print("1. 初始化系统组件...")
    
    # 加载预训练模型
    projection_model = sp.SemanticProjection(cae_dim=512, vae_dim=16)
    trainer = sp.SemanticProjectionTrainer(projection_model)
    
    try:
        trainer.load_model('best_model.pth')
        print("   ✓ 语义投影网络加载成功")
    except FileNotFoundError:
        print("   ⚠ 使用随机初始化的投影网络")
    
    projection_model.eval()
    
    # 初始化数据生成器（模拟CAE和VAE）
    data_gen = sp.DataGenerator()
    print("   ✓ 数据生成器初始化完成")
    
    # 2. 模拟故障诊断流程
    print("\n2. 模拟故障诊断流程...")
    
    # 模拟输入信号处理
    print("   步骤1: 信号预处理...")
    # 这里通常是从传感器获取的原始信号
    raw_signal = np.random.randn(8192)  # 模拟原始信号
    print(f"     原始信号长度: {len(raw_signal)}")
    
    # 模拟CAE特征提取
    print("   步骤2: CAE特征提取...")
    # 这里通常使用训练好的CAE模型提取特征
    cae_features = np.random.randn(1, 512).astype(np.float32)
    print(f"     CAE特征形状: {cae_features.shape}")
    
    # 使用投影网络转换到VAE语义空间
    print("   步骤3: 语义空间投影...")
    with torch.no_grad():
        cae_tensor = torch.FloatTensor(cae_features)
        vae_semantics = projection_model(cae_tensor)
        vae_semantics_np = vae_semantics.cpu().numpy()
    
    print(f"     VAE语义特征形状: {vae_semantics_np.shape}")
    
    # 在VAE语义空间中进行故障分类
    print("   步骤4: 故障分类...")
    
    # 计算与各故障类型参考语义的相似度
    fault_scores = {}
    for fault_type, ref_semantic in data_gen.vae_semantics.items():
        similarity = np.dot(vae_semantics_np[0], ref_semantic) / (
            np.linalg.norm(vae_semantics_np[0]) * np.linalg.norm(ref_semantic) + 1e-8
        )
        fault_scores[fault_type] = similarity
    
    # 找到最相似的故障类型
    predicted_fault = max(fault_scores, key=fault_scores.get)
    max_score = fault_scores[predicted_fault]
    
    print(f"     预测故障类型: {predicted_fault}")
    print(f"     置信度: {max_score:.4f}")
    
    print("   故障类型相似度排序:")
    sorted_scores = sorted(fault_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (fault_type, score) in enumerate(sorted_scores[:5]):
        print(f"     {i+1}. {fault_type}: {score:.4f}")
    
    # 3. 结果可视化
    print("\n3. 结果可视化...")
    
    # 创建可视化器
    visualizer = sp.SemanticVisualizer()
    
    # 准备可视化数据
    all_vae_features = np.array([data_gen.vae_semantics[ft] for ft in data_gen.fault_types])
    projected_feature = vae_semantics_np
    
    # 合并特征用于可视化
    combined_features = np.vstack([all_vae_features, projected_feature])
    combined_labels = data_gen.fault_types + [f'predicted_{predicted_fault}']
    
    print("   生成语义空间可视化...")
    try:
        # 为可视化准备数据
        sample_labels = [f'ref_{ft}' for ft in data_gen.fault_types]
        visualizer.visualize_semantic_space(
            all_vae_features, 
            projected_feature.reshape(1, -1), 
            sample_labels,
            save_path='inference_result.png',
            title='故障诊断结果 - 语义空间可视化'
        )
        print("   ✓ 可视化结果已保存到: inference_result.png")
    except Exception as e:
        print(f"   ⚠ 可视化生成失败: {e}")
        # 生成简单的统计信息作为替代
        print("   生成统计信息替代可视化...")
        print(f"   预测特征与各参考特征的距离:")
        for i, ft in enumerate(data_gen.fault_types):
            distance = np.linalg.norm(projected_feature[0] - all_vae_features[i])
            print(f"     {ft}: {distance:.4f}")
    
    print("\n" + "=" * 60)
    print("集成应用示例完成")
    print("=" * 60)
    
    return {
        'predicted_fault': predicted_fault,
        'confidence': max_score,
        'fault_scores': fault_scores,
        'vae_semantics': vae_semantics_np
    }

def main():
    """主函数"""
    print("语义投影网络使用示例\n")
    
    # 运行推理示例
    projected_features = inference_example()
    
    # 运行集成示例
    diagnosis_result = integration_example()
    
    print("\n" + "=" * 60)
    print("使用示例总结")
    print("=" * 60)
    
    print("本示例展示了：")
    print("1. ✓ 如何加载预训练的语义投影网络")
    print("2. ✓ 如何进行单样本和批量特征投影")
    print("3. ✓ 如何分析投影结果")
    print("4. ✓ 如何将投影网络集成到故障诊断流程")
    print("5. ✓ 如何进行结果可视化")
    
    print(f"\n诊断结果: {diagnosis_result['predicted_fault']} (置信度: {diagnosis_result['confidence']:.4f})")
    
    print("\n使用提示:")
    print("- 确保CAE特征的维度与网络输入维度匹配")
    print("- 在实际应用中，使用训练好的CAE模型提取特征")
    print("- 可以根据具体需求调整相似度计算方法")
    print("- 建议在验证集上评估投影质量")

if __name__ == "__main__":
    main()