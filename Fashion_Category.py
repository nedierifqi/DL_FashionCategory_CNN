# Mengimpor Library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from keras.datasets import fashion_mnist
(X_train, y_train),(X_test, y_test) = fashion_mnist.load_data()

# Jenis Kategori
# 0 = T-shirt/top
# 1 = Trouser
# 2 = Pullover
# 3 = Dress
# 4 = Coat
# 5 = Sandal
# 6 = Shirt
# 7 = Sneaker
# 8 = Bag
# 9 = Ankle Boot
kategori = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

i = random.randint(1, len(X_train))
plt.figure()
plt.imshow(X_train[i,:,:], cmap='gray')
plt.title('Item ke {} - Kategori = {}'.format(i, kategori[y_train[i]]))
plt.show()

# Menampilkan banyak gambar sekaligus
nrow = 10
ncol = 10
fig, axes = plt.subplots(nrow, ncol)
axes = axes.ravel() # Gunakan ravel jika ncol > 1 dan nrow > 1
ntraining = len(X_train)
for i in np.arange(0, nrow*ncol):
    indexku = np.random.randint(0, ntraining)
    axes[i].imshow(X_train[indexku, :, :], cmap='gray')
    axes[i].set_title(int(y_train[indexku]), fontsize=8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)

# Normalisasi dataset
X_train = X_train/255
X_test = X_test/255

# Membagi dataset menjadi training dan validate set
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

# Merubah dimensi dataset
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

# Mengimpor library Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam

# Mendefinisikan model CNN
classifier = Sequential()
classifier.add(Conv2D(32,(3,3), input_shape=(28, 28, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

# Memulai flattening dan membuat FC-NN
classifier.add(Flatten())
classifier.add(Dense(activation='relu', units=32))
classifier.add(Dense(activation='sigmoid', units=10))
classifier.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

classifier.summary()

# Visualisasi model NN
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model_fashion.png', show_shapes=True, show_layer_names=False)

# Training model
run_model = classifier.fit(X_train, y_train, batch_size = 500, nb_epoch = 30, verbose = 1, validation_data = (X_validate, y_validate))

# Parameter apa saja yang disimpan selama proses training
print(run_model.history.keys())

# Proses plotting accuracy selama training
plt.plot(run_model.history['accuracy'])
plt.plot(run_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# Proses plotting cost function selama training
plt.plot(run_model.history['loss'])
plt.plot(run_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# Mengevaluasi model CNN
evaluasi = classifier.evaluate(X_test, y_test)
print('Test accuracy = {:.2f}%'.format(evaluasi[1]*100))

# Save model
classifier.save('model_cnn_fashion.hd5', include_optimizer=True)
print('Model sudah disimpan')

# Load model
'''
from keras.models import load_model
classifier = load_model('model_cnn_fashion.hd5')
'''

# Memprediksi kategori di test set
hasil_prediksi = classifier.predict_classes(X_test)

# Membuat plot hasil prediksi
fig, axes = plt.subplots(5, 5)
axes = axes.ravel()
for i in np.arange(0, 5*5):
    axes[i].imshow(X_test[i].reshape(28,28), cmap='gray')
    axes[i].set_title('Hasil Prediksi = {}\n Label Asli = {}'.format(hasil_prediksi[i], y_test[i]))
    axes[i].axis('off')

# Confusion matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
cm = confusion_matrix(y_test, hasil_prediksi)
cm_label = pd.DataFrame(cm, columns = np.unique(y_test), index = np.unique(y_test))
cm_label.index.name = 'Asli'
cm_label.columns.name = 'Prediksi'
plt.figure(figsize=(14, 10))
sns.heatmap(cm_label, annot=True, fmt='g')

# Summary performa model
from sklearn.metrics import classification_report
jumlah_kategori = 10
target_names = ['Kategori {}'.format(i) for i in range(jumlah_kategori)]
print(classification_report(y_test, hasil_prediksi, 
                            target_names = target_names))