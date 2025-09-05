import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Приховати AVX та інші інфо-повідомлення

import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
try:
    # Перевірка доступності GPU (Metal)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {gpus}")
        # Спробуємо просту операцію на GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print("GPU test operation successful:", c.numpy())
    else:
        print("No GPU found by TensorFlow (tensorflow-metal might not be working or not installed). Will use CPU.")
        # Спробуємо просту операцію на CPU
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("CPU test operation successful:", c.numpy())

except Exception as e:
    print(f"Error during TensorFlow test: {e}")
