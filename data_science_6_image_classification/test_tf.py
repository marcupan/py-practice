import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
# На DirectML це зазвичай False — це нормально
is_cuda = getattr(tf.test, 'is_built_with_cuda', lambda: False)()
print(f"Built with CUDA: {is_cuda}")

# Інформація про збірку (CUDA/cuDNN)
try:
    build_info = tf.sysconfig.get_build_info()
    print(f"CUDA version used by TensorFlow: {build_info.get('cuda_version')}")
    print(f"cuDNN version used by TensorFlow: {build_info.get('cudnn_version')}")
except Exception:
    print("Could not get CUDA build info")

# Перелік фізичних/логічних пристроїв: CUDA GPU та DirectML (DML)
physical_gpus = tf.config.list_physical_devices('GPU')
logical_gpus = tf.config.list_logical_devices('GPU')
physical_dml = []
logical_dml = []
try:
    physical_dml = tf.config.list_physical_devices('DML')
    logical_dml = tf.config.list_logical_devices('DML')
except Exception:
    # На звичайному TF без DirectML цей тип недоступний — ігноруємо
    pass

print(f"Physical GPU devices: {len(physical_gpus)}")
print(f"Physical DML devices: {len(physical_dml)}")
print(f"Logical GPU devices: {len(logical_gpus)}")
print(f"Logical DML devices: {len(logical_dml)}")
if physical_gpus:
    print(f"GPU devices: {physical_gpus}")
if physical_dml:
    print(f"DML devices: {physical_dml}")

# Дозволити поступове виділення пам'яті (може не підтримуватись на DML)
try:
    for dev in physical_gpus + physical_dml:
        tf.config.experimental.set_memory_growth(dev, True)
    if physical_gpus or physical_dml:
        print("Memory growth enabled (where supported)")
except RuntimeError as e:
    print(f"Memory growth setting failed: {e}")
except Exception:
    # Якщо на DirectML це не підтримується — тихо пропустимо
    pass

# Вибір цільового пристрою: спершу CUDA GPU, інакше DirectML, інакше CPU
target_device = None
if physical_gpus:
    target_device = '/GPU:0'
elif physical_dml:
    target_device = '/DML:0'

# Тестова операція
try:
    if target_device:
        with tf.device(target_device):
            print(f"Testing operation on {target_device}...")
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print(f"Device {target_device} matmul OK. Result shape: {c.shape}")
    else:
        print("No GPU/DML devices found. Testing CPU operation...")
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("CPU operation successful:", c.numpy())
except Exception as e:
    print(f"Operation failed: {e}")
