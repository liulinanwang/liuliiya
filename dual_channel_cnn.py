import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置随机种子，确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 设置工作目录
data_dir = 'E:\\pythonProject17\\preprocessed_data\\preprocessed_data'  # 修改为正确的数据目录
output_dir = 'E:\\pythonProject17\\preprocessed_data\\preprocessed_data'
model_dir = os.path.join(output_dir, 'models')
semantic_dir = os.path.join(output_dir, 'semantics')

# 创建输出目录
os.makedirs(model_dir, exist_ok=True)

# 文件列表
single_fault_files = ['ball_preprocessed.npy', 'inner_preprocessed.npy', 'outer_preprocessed.npy',
                      'normal_preprocessed.npy']
compound_fault_files = ['inner_ball_preprocessed.npy', 'inner_outer_preprocessed.npy', 'outer_ball_preprocessed.npy',
                        'inner_outer_ball_preprocessed.npy']

# 故障类型映射
fault_type_mapping = {
    'normal_preprocessed.npy': [0, 0, 0],  # 正常
    'inner_preprocessed.npy': [1, 0, 0],  # 内圈故障
    'outer_preprocessed.npy': [0, 1, 0],  # 外圈故障
    'ball_preprocessed.npy': [0, 0, 1],  # 滚动体故障
    'inner_ball_preprocessed.npy': [1, 0, 1],  # 内圈+滚动体故障
    'inner_outer_preprocessed.npy': [1, 1, 0],  # 内圈+外圈故障
    'outer_ball_preprocessed.npy': [0, 1, 1],  # 外圈+滚动体故障
    'inner_outer_ball_preprocessed.npy': [1, 1, 1]  # 内圈+外圈+滚动体故障
}


# 加载预处理后的数据
def load_preprocessed_data():
    """加载预处理后的数据"""
    all_data = {}

    # 加载单一故障数据
    for filename in single_fault_files:
        file_path = os.path.join(data_dir, filename)
        all_data[filename] = np.load(file_path)
        print(f"Loaded {filename}: {all_data[filename].shape}")

    # 加载复合故障数据
    for filename in compound_fault_files:
        file_path = os.path.join(data_dir, filename)
        all_data[filename] = np.load(file_path)
        print(f"Loaded {filename}: {all_data[filename].shape}")

    return all_data


# 准备训练和测试数据
def prepare_data(all_data):
    """准备训练和测试数据"""
    # 单一故障数据用于训练
    train_data = []
    train_labels = []
    train_conditions = []

    # 复合故障数据用于测试
    test_data = []
    test_labels = []
    test_conditions = []

    # 处理训练数据（单一故障）
    for filename in single_fault_files:
        data = all_data[filename]
        label_idx = single_fault_files.index(filename)
        condition = fault_type_mapping[filename]

        # 添加到训练集
        train_data.append(data)
        train_labels.extend([label_idx] * len(data))
        train_conditions.extend([condition] * len(data))

    # 处理测试数据（复合故障）
    for filename in compound_fault_files:
        data = all_data[filename]
        label_idx = compound_fault_files.index(filename)
        condition = fault_type_mapping[filename]

        # 添加到测试集
        test_data.append(data)
        test_labels.extend([label_idx] * len(data))
        test_conditions.extend([condition] * len(data))

    # 合并数据
    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)

    # 转换为numpy数组
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_conditions = np.array(train_conditions)
    test_conditions = np.array(test_conditions)

    # 重塑数据以适应CNN输入 (samples, height, width, channels)
    train_data = train_data.reshape(-1, train_data.shape[1], 1)
    test_data = test_data.reshape(-1, test_data.shape[1], 1)

    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Train conditions shape: {train_conditions.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Test conditions shape: {test_conditions.shape}")

    return train_data, train_labels, train_conditions, test_data, test_labels, test_conditions


# 加载语义向量
def load_semantics():
    """加载语义向量"""
    # 加载融合后的语义
    fused_semantics_path = os.path.join(semantic_dir, 'fused_semantics.npy')
    if not os.path.exists(fused_semantics_path):
        print(f"Error: Fused semantics not found at {fused_semantics_path}")
        print("Please run semantic_construction.py first.")
        return None

    fused_semantics = np.load(fused_semantics_path, allow_pickle=True).item()
    # 打印语义向量的键，用于调试
    print("Available semantic keys:", list(fused_semantics.keys()))
    return fused_semantics


# 双通道卷积网络
class DualChannelCNN:
    def __init__(self, input_shape, semantic_dim, num_classes):
        self.input_shape = input_shape
        self.semantic_dim = semantic_dim
        self.num_classes = num_classes
        self.model = None
        self.build_model()

    def build_model(self):
        # 通道1：原始信号输入
        signal_input = layers.Input(shape=self.input_shape, name='signal_input')

        # 通道1处理：使用大卷积核进行特征提取和降噪
        x1 = layers.Conv1D(32, 16, activation='relu', padding='same')(signal_input)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling1D(2, padding='same')(x1)

        # 引入空洞卷积，扩大感受野
        x1 = layers.Conv1D(64, 8, activation='relu', padding='same', dilation_rate=2)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling1D(2, padding='same')(x1)

        # 多尺度卷积
        # 分支1：小卷积核
        branch1 = layers.Conv1D(32, 3, activation='relu', padding='same')(x1)
        # 分支2：中卷积核
        branch2 = layers.Conv1D(32, 5, activation='relu', padding='same')(x1)
        # 分支3：大卷积核
        branch3 = layers.Conv1D(32, 7, activation='relu', padding='same')(x1)

        # 合并多尺度特征
        x1 = layers.Concatenate()([branch1, branch2, branch3])
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling1D(2, padding='same')(x1)

        # 通道2：语义输入
        semantic_input = layers.Input(shape=(self.semantic_dim,), name='semantic_input')

        # 通道2处理：通过全连接层映射到更高维度
        x2 = layers.Dense(128, activation='relu')(semantic_input)
        x2 = layers.BatchNormalization()(x2)

        # 添加通道注意力（SE模块）
        x2 = layers.Dense(64, activation='relu')(x2)
        x2 = layers.Dense(128, activation='sigmoid')(x2)

        # 重塑为与x1兼容的形状以便后续融合
        x1_shape = tf.keras.backend.int_shape(x1)
        x2 = layers.Reshape((1, 128))(x2)
        x2 = layers.UpSampling1D(x1_shape[1])(x2)

        # 融合两个通道的特征
        x = layers.Concatenate()([x1, x2])

        # 使用1x1卷积进行跨模态信息交互
        x = layers.Conv1D(128, 1, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # 最终特征提取
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)

        # 全连接层
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # 输出层
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        # 定义模型
        self.model = Model([signal_input, semantic_input], outputs)

        # 编译模型
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_data, train_semantics, train_labels,
              val_data=None, val_semantics=None, val_labels=None,
              batch_size=32, epochs=50):
        """训练模型"""
        # 准备训练数据
        train_inputs = [train_data, train_semantics]

        # 准备验证数据
        validation_data = None
        if val_data is not None and val_semantics is not None and val_labels is not None:
            validation_data = ([val_data, val_semantics], val_labels)

        # 定义学习率调度器 - 余弦退火
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )

        # 定义早停
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )

        # 训练模型
        history = self.model.fit(
            train_inputs, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[lr_scheduler, early_stopping],
            shuffle=True
        )

        return history

    def evaluate(self, test_data, test_semantics, test_labels):
        """评估模型"""
        test_inputs = [test_data, test_semantics]
        loss, accuracy = self.model.evaluate(test_inputs, test_labels)
        return loss, accuracy

    def predict(self, data, semantics):
        """预测"""
        inputs = [data, semantics]
        return self.model.predict(inputs)

    def save_model(self, model_path):
        """保存模型"""
        self.model.save(model_path)

    def load_model(self, model_path):
        """加载模型"""
        self.model = tf.keras.models.load_model(model_path)


# 自定义损失函数
def create_custom_loss():
    """创建自定义损失函数，包括对比损失和特征-语义一致性损失"""

    def triplet_loss(y_true, y_pred, margin=0.5):
        """Triplet Loss实现"""
        # 这里简化实现，实际应用中需要动态选择anchor, positive, negative
        # 在模型训练中通过自定义层或回调来实现

        # 获取嵌入向量
        embeddings = y_pred

        # 计算成对距离
        # 这里使用欧氏距离平方
        def squared_distance(x):
            return tf.reduce_sum(tf.square(x[0] - x[1]), axis=1)

        # 计算anchor-positive距离
        pos_dist = squared_distance([embeddings, embeddings])

        # 计算anchor-negative距离
        neg_dist = squared_distance([embeddings, tf.roll(embeddings, shift=1, axis=0)])

        # 计算triplet loss
        basic_loss = tf.maximum(0., margin + pos_dist - neg_dist)
        loss = tf.reduce_mean(basic_loss)

        return loss

    def feature_semantic_consistency_loss(y_true, y_pred, features, semantics, beta=0.01):
        """特征-语义一致性损失"""
        # 计算特征和语义之间的余弦相似度
        features_norm = tf.nn.l2_normalize(features, axis=1)
        semantics_norm = tf.nn.l2_normalize(semantics, axis=1)
        cosine_sim = tf.reduce_sum(tf.multiply(features_norm, semantics_norm), axis=1)

        # 余弦相似度越高越好，所以取负值
        consistency_loss = tf.reduce_mean(1 - cosine_sim)

        # 添加L2正则化
        l2_loss = tf.reduce_sum(tf.square(features))

        return consistency_loss + beta * l2_loss

    def combined_loss(y_true, y_pred):
        """组合损失函数"""
        # 分类损失（交叉熵）
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        ce_loss = tf.reduce_mean(ce_loss)

        # 组合损失
        # 注意：在实际应用中，triplet_loss和feature_semantic_consistency_loss
        # 需要访问模型中间层的输出，这里简化处理
        # 完整实现需要自定义训练循环或模型层

        return ce_loss

    return combined_loss


# 可视化训练历史
def plot_training_history(history):
    """可视化训练历史"""
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dual_cnn_training_history.png'))
    plt.close()


# 可视化混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names):
    """可视化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 打印分类报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)

    # 保存分类报告到文件
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)


# 主函数
def main():
    # 加载预处理后的数据
    all_data = load_preprocessed_data()

    # 准备训练和测试数据
    train_data, train_labels, train_conditions, test_data, test_labels, test_conditions = prepare_data(all_data)

    # 加载语义向量
    fused_semantics = load_semantics()
    if fused_semantics is None:
        return

    # 创建文件名到语义键的映射
    file_to_key_map = {
        'ball_preprocessed.npy': 'ball',
        'inner_preprocessed.npy': 'inner',
        'outer_preprocessed.npy': 'outer',
        'normal_preprocessed.npy': 'normal',
        'inner_ball_preprocessed.npy': 'inner_ball',
        'inner_outer_preprocessed.npy': 'inner_outer',
        'outer_ball_preprocessed.npy': 'outer_ball',
        'inner_outer_ball_preprocessed.npy': 'inner_outer_ball'
    }

    # 准备语义输入
    train_semantics = np.array([fused_semantics[file_to_key_map[single_fault_files[label]]] for label in train_labels])
    test_semantics = np.array([fused_semantics[file_to_key_map[compound_fault_files[label]]] for label in test_labels])

    print(f"Train semantics shape: {train_semantics.shape}")
    print(f"Test semantics shape: {test_semantics.shape}")

    # 创建双通道CNN模型
    input_shape = train_data.shape[1:]  # (signal_length, 1)
    semantic_dim = train_semantics.shape[1]  # 语义向量维度
    num_classes = len(np.unique(test_labels))  # 复合故障类别数

    print(f"Input shape: {input_shape}")
    print(f"Semantic dimension: {semantic_dim}")
    print(f"Number of classes: {num_classes}")

    # 创建模型
    dual_cnn = DualChannelCNN(input_shape, semantic_dim, num_classes)

    # 打印模型摘要
    dual_cnn.model.summary()

    # 分割训练集为训练和验证
    from sklearn.model_selection import train_test_split
    train_data_split, val_data, train_semantics_split, val_semantics, train_labels_split, val_labels = train_test_split(
        train_data, train_semantics, train_labels, test_size=0.2, random_state=42
    )

    # 训练模型
    history = dual_cnn.train(
        train_data_split, train_semantics_split, train_labels_split,
        val_data, val_semantics, val_labels,
        batch_size=32, epochs=50
    )

    # 可视化训练历史
    plot_training_history(history)

    # 评估模型
    test_loss, test_accuracy = dual_cnn.evaluate(test_data, test_semantics, test_labels)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 预测
    y_pred_prob = dual_cnn.predict(test_data, test_semantics)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 可视化混淆矩阵
    class_names = [f.split('_')[0] for f in compound_fault_files]
    plot_confusion_matrix(test_labels, y_pred, class_names)

    # 保存模型
    dual_cnn.save_model(os.path.join(model_dir, 'dual_channel_cnn.h5'))

    print("Dual Channel CNN training and evaluation completed.")


if __name__ == "__main__":
    main()
