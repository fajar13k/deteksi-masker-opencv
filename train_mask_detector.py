# Impor semua modul
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Menginisialisasi Learning Rate, Epochs yang akan dilatih, dan jumlah yang akan dilatih dalam batch
INIT_LR = 1e-4 # Nilai scientific 0.0001 atau 1 banding 10,000
EPOCHS = 20
BS = 32

# Lokasi file yang akan dilatih
DIR = r"C:\Ngoding\Face-Mask-Detection\dataset"

# Kategori yang dilatih; Memakai masker dan Tidak memakai masker
CATEGORIES = ["with_mask", "without_mask"]

# Mengambil daftar gambar yang ada di dataset DIR, lalu inisialisasi array
# Daftar data-nya (contoh: gambar) dan class gambar
print("[Info] Memuat gambar...")

data = []
labels = []

# Melakukan loop dalam pengambilan gambar di setiap kategori dan melabelinya
for category in CATEGORIES:
	path = os.path.join(DIR, category)
	for img in os.listdir(path):
		img_path = os.path.join(path, img)
		image = load_img(img_path, target_size = (224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)

        # Masukkan gambar yang sudah jadi Array dan melakukan Preprocess ke dalam Array data yang dibuat
		data.append(image)
        # Gambar dilabeli dengan "with_mask" atau "without_mask"
		labels.append(category)

# Melakukan encoding untuk label, data menjadi binary 101010010101001
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Melakukan pelabelan di numpy, data dilakukan dengan parsing array float32
data = np.array(data, dtype = "float32")
labels = np.array(labels)

# Memisahkan array data menjadi train dan test subset
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.2, stratify = labels, random_state = 42)

# Melakukan generate gambar tensor dengan proyeksi data real-time
aug = ImageDataGenerator (
    rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest"
)

# Memuat jaringan MobileNetV2, dan memastikan layerset dasar sudah siap digunakan
# Shape 224, 224 diinput menyesuaikan dengan gambar yang dimiliki dalam dataset
baseModel = MobileNetV2(
    weights = "imagenet",
    include_top = False,
    input_tensor = Input(shape = (224, 224, 3))
)

# Membuat model untuk mengenali kepala
# Model kepala ini akan diletakkan di atas layer dasar baseModel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7, 7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = "softmax")(headModel)

# Taruh layer headModel di atas layer dasar
# Model ini juga yang nantinya yang akan kita latih
# Untuk membaca
model = Model(inputs = baseModel.input, outputs = headModel)

# Melakukan loop kepada semua layer dasar dan freeze
# Hal ini dilakukan supaya layer tidak berubah-ubah bentuk ketika
# Melakukan pelatihan pengenalan pattern
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
print("[Info] Meng-compile model...")
opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS) # Optimisasi menggunakan algoritma Adam(menerima input Learning Rate, dan decay)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

# Melatih network head
print("[Info] Melatih head network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size = BS),
    steps_per_epoch = len(trainX) // BS,
    validation_data = (testX, testY),
    validation_steps = len(testX) // BS,
    epochs = EPOCHS
)

# Membuat prediksi dari testing set yang dilakukan
print("[Info] Meninjau ulang network...")
predIdxs = model.predict(testX, batch_size = BS)

# Untuk setiap gambar yang berada dalam testing set, kita perlu mencari index label
# dengan kemungkinan prediksi terbesar
predIdxs = np.argmax(predIdxs, axis = 1)

# Menampilkan laporan klasifikasi hasil pelatihan yang telah diformat
print(classification_report(testY.argmax(axis = 1), predIdxs, target_names = lb.classes_))

# Serialize model menjadi disk
print("[Info] Menyimpan model pendeteksi mask...")
model.save("mask_detector.model", save_format = "h5")

# Membuat Plot nilai latihan (Loss dan Akurat)
N = EPOCHS
plt.style.use("seaborn-dark")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Iterasi #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("tingkat_akurasi_prediksi.png")