import tensorflow as tf

print("1. Membaca otak AI versi PC (model_bisindo.h5)...")
model = tf.keras.models.load_model('model_bisindo.h5')

print("2. Memulai proses kompresi ke format Mobile (TFLite)...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS    
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

print("3. Menyimpan hasil...")

with open('model_bisindo.tflite', 'wb') as f:
    f.write(tflite_model)

print("BERHASIL! File 'model_bisindo.tflite' sudah jadi dan siap dimasukkan ke dalam project aplikasi Anda!")

