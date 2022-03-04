from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # Untuk membangun GPU di Memory
config.log_device_placement = True # Untuk menyimpan log
sess = tf.Session(config=config)
set_session(sess)