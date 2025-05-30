import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

# 创建最终解决方案目录
solution_dir = 'E:\\pythonProject17'
os.makedirs(solution_dir, exist_ok=True)

# 设置路径
data_dir = 'E:\\pythonProject17\\preprocessed_data\\preprocessed_data'
model_dir = os.path.join('E:\\pythonProject17\\preprocessed_data\\preprocessed_data', 'models')
semantic_dir = os.path.join('E:\\pythonProject17\\preprocessed_data\\preprocessed_data', 'semantics')

# 故障类型
fault_types = ['ball', 'inner', 'outer', 'normal', 'inner_ball', 'inner_outer', 'outer_ball', 'inner_outer_ball']
single_faults = ['ball', 'inner', 'outer', 'normal']
compound_faults = ['inner_ball', 'inner_outer', 'outer_ball', 'inner_outer_ball']


# 加载预处理后的数据
def load_data():
    all_data = {}
    all_labels = {}

    # 加载所有数据文件
    for fault_type in fault_types:
        file_path = os.path.join(data_dir, f"{fault_type}_preprocessed.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)
            all_data[fault_type] = data
            print(f"Loaded {fault_type}_preprocessed.npy: {data.shape}")

            # 创建标签
            if fault_type in single_faults:
                label = single_faults.index(fault_type)
            else:
                label = compound_faults.index(fault_type)
            all_labels[fault_type] = label

    return all_data, all_labels


# 准备训练和测试数据
def prepare_data(all_data, all_labels):
    # 单一故障数据用于训练
    train_data = []
    train_labels = []

    # 复合故障数据用于测试
    test_data = []
    test_labels = []

    # 处理训练数据（单一故障）
    for fault_type in single_faults:
        if fault_type in all_data:
            data = all_data[fault_type]
            label = all_labels[fault_type]

            train_data.append(data)
            train_labels.extend([label] * len(data))

    # 处理测试数据（复合故障）
    for fault_type in compound_faults:
        if fault_type in all_data:
            data = all_data[fault_type]
            label = all_labels[fault_type]

            test_data.append(data)
            test_labels.extend([label] * len(data))

    # 合并数据
    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)

    # 转换为numpy数组
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # 重塑数据以适应CNN输入 (samples, height, width, channels)
    train_data = train_data.reshape(-1, train_data.shape[1], 1)
    test_data = test_data.reshape(-1, test_data.shape[1], 1)

    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    return train_data, train_labels, test_data, test_labels


# 加载语义向量
def load_semantics():
    fused_semantics_path = os.path.join(semantic_dir, 'fused_semantics.npy')
    if not os.path.exists(fused_semantics_path):
        print(f"Error: Fused semantics not found at {fused_semantics_path}")
        return None

    fused_semantics = np.load(fused_semantics_path, allow_pickle=True).item()
    print("Available semantic keys:", list(fused_semantics.keys()))
    return fused_semantics


# 创建零样本复合轴承故障诊断模型
class ZeroShotBearingFaultDiagnosisModel:
    def __init__(self, input_shape, semantic_dim, num_classes):
        self.input_shape = input_shape
        self.semantic_dim = semantic_dim
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        # 信号输入
        signal_input = layers.Input(shape=self.input_shape, name='signal_input')

        # 语义输入
        semantic_input = layers.Input(shape=(self.semantic_dim,), name='semantic_input')

        # 信号处理通道
        x1 = layers.Conv1D(32, 16, activation='relu', padding='same')(signal_input)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling1D(2, padding='same')(x1)

        x1 = layers.Conv1D(64, 8, activation='relu', padding='same', dilation_rate=2)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling1D(2, padding='same')(x1)

        x1 = layers.Conv1D(128, 4, activation='relu', padding='same')(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.GlobalAveragePooling1D()(x1)

        # 语义处理通道
        x2 = layers.Dense(128, activation='relu')(semantic_input)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dense(128, activation='relu')(x2)

        # 融合两个通道
        merged = layers.Concatenate()([x1, x2])

        # 全连接层
        x = layers.Dense(256, activation='relu')(merged)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # 输出层
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        # 创建模型
        model = Model(inputs=[signal_input, semantic_input], outputs=outputs)

        # 编译模型
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, train_data, train_labels, train_semantics, val_data=None, val_labels=None, val_semantics=None,
              epochs=50, batch_size=32):
        # 创建回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        # 训练模型
        if val_data is not None and val_labels is not None and val_semantics is not None:
            history = self.model.fit(
                [train_data, train_semantics], train_labels,
                validation_data=([val_data, val_semantics], val_labels),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                [train_data, train_semantics], train_labels,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )

        return history

    def evaluate(self, test_data, test_labels, test_semantics):
        # 评估模型
        loss, accuracy = self.model.evaluate([test_data, test_semantics], test_labels)
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")

        # 预测
        predictions = self.model.predict([test_data, test_semantics])
        predicted_labels = np.argmax(predictions, axis=1)

        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, predicted_labels)

        # 生成分类报告
        report = classification_report(test_labels, predicted_labels)
        print("Classification Report:")
        print(report)

        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(solution_dir, 'confusion_matrix.png'))
        plt.close()

        return accuracy, predicted_labels, cm, report

    def save(self, model_path):
        self.model.save(model_path)
        print(f"Model saved to {model_path}")


# 主函数
def main():
    print("Starting Zero-Shot Compound Bearing Fault Diagnosis...")

    # 1. 加载数据
    all_data, all_labels = load_data()

    # 2. 准备训练和测试数据
    train_data, train_labels, test_data, test_labels = prepare_data(all_data, all_labels)

    # 3. 加载语义向量
    fused_semantics = load_semantics()
    if fused_semantics is None:
        print("Error: Could not load semantic vectors. Exiting.")
        return

    # 4. 准备语义输入
    # 创建单一故障到复合故障的映射
    fault_mapping = {
        'ball': 'ball',
        'inner': 'inner',
        'outer': 'outer',
        'normal': 'normal',
        0: 'ball',  # 单一故障标签映射
        1: 'inner',
        2: 'outer',
        3: 'normal',
        4: 'inner_ball',  # 复合故障标签映射
        5: 'inner_outer',
        6: 'outer_ball',
        7: 'inner_outer_ball'
    }

    # 准备训练和测试语义
    train_semantics = np.array([fused_semantics[fault_mapping[label]] for label in train_labels])
    test_semantics = np.array([fused_semantics[fault_mapping[label + 4]] for label in test_labels])  # 复合故障从索引4开始

    print(f"Train semantics shape: {train_semantics.shape}")
    print(f"Test semantics shape: {test_semantics.shape}")

    # 5. 创建和训练模型
    input_shape = train_data.shape[1:]
    semantic_dim = train_semantics.shape[1]
    num_classes = len(np.unique(test_labels))

    print(f"Input shape: {input_shape}")
    print(f"Semantic dimension: {semantic_dim}")
    print(f"Number of classes: {num_classes}")

    # 创建模型
    model = ZeroShotBearingFaultDiagnosisModel(input_shape, semantic_dim, num_classes)

    # 打印模型摘要
    model.model.summary()

    # 训练模型
    history = model.train(train_data, train_labels, train_semantics, epochs=50, batch_size=32)

    # 6. 评估模型
    accuracy, predicted_labels, cm, report = model.evaluate(test_data, test_labels, test_semantics)

    # 7. 保存模型
    model.save(os.path.join(solution_dir, 'zero_shot_model.h5'))

    # 8. 保存评估结果
    with open(os.path.join(solution_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # 9. 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(solution_dir, 'training_history.png'))
    plt.close()

    print(f"Zero-Shot Compound Bearing Fault Diagnosis completed with accuracy: {accuracy:.4f}")
    print(f"Results saved to {solution_dir}")

    return accuracy


if __name__ == "__main__":
    main()
