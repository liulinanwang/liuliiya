import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# 设置随机种子，确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 设置工作目录
data_dir = 'E:\\pythonProject17\\preprocessed_data\\preprocessed_data'
output_dir = 'E:\\pythonProject17\\preprocessed_data\\preprocessed_data'
model_dir = os.path.join(output_dir, 'models')

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


# 条件自编码器模型
class ConditionalAutoencoder:
    def __init__(self, input_shape, condition_shape, latent_dim=64):
        self.input_shape = input_shape
        self.condition_shape = condition_shape
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.build_model()

    def build_model(self):
        # 编码器
        input_signal = layers.Input(shape=self.input_shape, name='input_signal')
        input_condition = layers.Input(shape=self.condition_shape, name='input_condition')

        # 条件向量扩展到与信号相同的长度
        condition_repeated = layers.RepeatVector(self.input_shape[0])(input_condition)
        condition_reshaped = layers.Reshape((self.input_shape[0], self.condition_shape[0]))(condition_repeated)

        # 将信号和条件拼接
        x = layers.Concatenate(axis=2)([input_signal, condition_reshaped])

        # 编码器层
        x = layers.Conv1D(32, 8, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding='same')(x)

        x = layers.Conv1D(64, 8, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding='same')(x)

        x = layers.Conv1D(128, 8, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding='same')(x)

        # 展平
        x = layers.Flatten()(x)

        # 潜在空间表示
        latent = layers.Dense(self.latent_dim, name='latent_space')(x)

        # 定义编码器模型
        self.encoder = Model([input_signal, input_condition], latent, name='encoder')

        # 解码器
        latent_input = layers.Input(shape=(self.latent_dim,), name='latent_input')
        decoder_condition = layers.Input(shape=self.condition_shape, name='decoder_condition')

        # 解码器层
        y = layers.Dense(128 * (self.input_shape[0] // 8))(latent_input)
        y = layers.Reshape((self.input_shape[0] // 8, 128))(y)

        # 将条件向量扩展并拼接
        condition_repeated_decoder = layers.RepeatVector(self.input_shape[0] // 8)(decoder_condition)
        y = layers.Concatenate(axis=2)([y, condition_repeated_decoder])

        y = layers.Conv1D(128, 8, activation='relu', padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.UpSampling1D(2)(y)

        y = layers.Conv1D(64, 8, activation='relu', padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.UpSampling1D(2)(y)

        y = layers.Conv1D(32, 8, activation='relu', padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.UpSampling1D(2)(y)

        # 输出层
        decoded = layers.Conv1D(1, 8, activation='tanh', padding='same', name='decoded')(y)

        # 定义解码器模型
        self.decoder = Model([latent_input, decoder_condition], decoded, name='decoder')

        # 完整的自编码器模型
        encoded = self.encoder([input_signal, input_condition])
        decoded = self.decoder([encoded, input_condition])

        self.autoencoder = Model([input_signal, input_condition], decoded, name='autoencoder')

        # 编译模型
        self.autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )

    def train(self, train_data, train_conditions, validation_data=None, validation_conditions=None,
              batch_size=32, epochs=50):
        """训练条件自编码器"""
        # 准备训练数据
        train_input = [train_data, train_conditions]
        train_output = train_data

        # 准备验证数据
        validation_input = None
        validation_output = None
        if validation_data is not None and validation_conditions is not None:
            validation_input = [validation_data, validation_conditions]
            validation_output = validation_data

        # 训练模型
        history = self.autoencoder.fit(
            train_input, train_output,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(validation_input, validation_output) if validation_input is not None else None,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_input is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_input is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
        )

        return history

    def encode(self, data, conditions):
        """使用编码器提取特征"""
        return self.encoder.predict([data, conditions])

    def decode(self, latent, conditions):
        """使用解码器重构信号"""
        return self.decoder.predict([latent, conditions])

    def reconstruct(self, data, conditions):
        """重构输入信号"""
        return self.autoencoder.predict([data, conditions])

    def save_models(self, model_dir):
        """保存模型"""
        self.encoder.save(os.path.join(model_dir, 'encoder.h5'))
        self.decoder.save(os.path.join(model_dir, 'decoder.h5'))
        self.autoencoder.save(os.path.join(model_dir, 'autoencoder.h5'))

    def load_models(self, model_dir):
        """加载模型"""
        self.encoder = tf.keras.models.load_model(os.path.join(model_dir, 'encoder.h5'))
        self.decoder = tf.keras.models.load_model(os.path.join(model_dir, 'decoder.h5'))
        self.autoencoder = tf.keras.models.load_model(os.path.join(model_dir, 'autoencoder.h5'))


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

    # 绘制学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cae_training_history.png'))
    plt.close()


# 可视化重构结果
def visualize_reconstructions(model, test_data, test_conditions, n_samples=5):
    """可视化重构结果"""
    # 随机选择样本
    indices = np.random.choice(len(test_data), n_samples, replace=False)
    samples = test_data[indices]
    conditions = test_conditions[indices]

    # 重构样本
    reconstructions = model.reconstruct(samples, conditions)

    # 可视化
    plt.figure(figsize=(15, 10))
    for i in range(n_samples):
        # 原始信号
        plt.subplot(n_samples, 2, 2 * i + 1)
        plt.plot(samples[i].flatten())
        plt.title(f'Original Signal {i + 1}')
        plt.grid(True)

        # 重构信号
        plt.subplot(n_samples, 2, 2 * i + 2)
        plt.plot(reconstructions[i].flatten())
        plt.title(f'Reconstructed Signal {i + 1}')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cae_reconstructions.png'))
    plt.close()


# 可视化潜在空间
def visualize_latent_space(model, train_data, train_conditions, train_labels,
                           test_data, test_conditions, test_labels):
    """可视化潜在空间"""
    # 提取特征
    train_features = model.encode(train_data, train_conditions)
    test_features = model.encode(test_data, test_conditions)

    # 使用t-SNE降维
    from sklearn.manifold import TSNE

    # 合并训练和测试特征
    all_features = np.vstack([train_features, test_features])
    all_labels = np.concatenate([train_labels, test_labels + len(np.unique(train_labels))])

    # 应用t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    all_features_2d = tsne.fit_transform(all_features)

    # 分离回训练和测试特征
    train_features_2d = all_features_2d[:len(train_features)]
    test_features_2d = all_features_2d[len(train_features):]

    # 可视化
    plt.figure(figsize=(12, 10))

    # 绘制训练数据
    plt.subplot(2, 1, 1)
    for i in range(len(np.unique(train_labels))):
        idx = train_labels == i
        plt.scatter(train_features_2d[idx, 0], train_features_2d[idx, 1],
                    label=f'Train: {single_fault_files[i].split("_")[0]}')
    plt.title('Latent Space - Training Data (Single Faults)')
    plt.legend()
    plt.grid(True)

    # 绘制测试数据
    plt.subplot(2, 1, 2)
    for i in range(len(np.unique(test_labels))):
        idx = test_labels == i
        plt.scatter(test_features_2d[idx, 0], test_features_2d[idx, 1],
                    label=f'Test: {compound_fault_files[i].split("_")[0]}')
    plt.title('Latent Space - Testing Data (Compound Faults)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cae_latent_space.png'))
    plt.close()


# 主函数
def main():
    # 加载预处理后的数据
    all_data = load_preprocessed_data()

    # 准备训练和测试数据
    train_data, train_labels, train_conditions, test_data, test_labels, test_conditions = prepare_data(all_data)

    # 创建条件自编码器模型
    input_shape = train_data.shape[1:]  # (signal_length, 1)
    condition_shape = (train_conditions.shape[1],)  # (condition_dim,)
    latent_dim = 64

    print(f"Input shape: {input_shape}")
    print(f"Condition shape: {condition_shape}")

    cae = ConditionalAutoencoder(input_shape, condition_shape, latent_dim)

    # 打印模型摘要
    cae.encoder.summary()
    cae.decoder.summary()
    cae.autoencoder.summary()

    # 训练模型
    # 分割训练集为训练和验证
    from sklearn.model_selection import train_test_split
    train_data_split, val_data, train_conditions_split, val_conditions, train_labels_split, val_labels = train_test_split(
        train_data, train_conditions, train_labels, test_size=0.2, random_state=42
    )

    history = cae.train(
        train_data_split, train_conditions_split,
        validation_data=val_data, validation_conditions=val_conditions,
        batch_size=32, epochs=50
    )

    # 可视化训练历史
    plot_training_history(history)

    # 可视化重构结果
    visualize_reconstructions(cae, test_data, test_conditions)

    # 可视化潜在空间
    visualize_latent_space(cae, train_data, train_conditions, train_labels,
                           test_data, test_conditions, test_labels)

    # 保存模型
    cae.save_models(model_dir)

    print("Conditional Autoencoder training and evaluation completed.")


if __name__ == "__main__":
    main()
