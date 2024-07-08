import tensorflow as tf

print("tensorflow: " + tf.__version__)
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print("device: ", device)

if device == "GPU":
    print("GPU devices:")
    for gpu in tf.config.list_physical_devices('GPU'):
        print("  ", gpu)
        print("    Name:", tf.test.gpu_device_name())
        print("    Memory:", tf.DeviceSpec.from_string(gpu.name).memory_limit)