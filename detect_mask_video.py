# Impor semua modul
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import imutils
import time
import cv2
import os

def deteksi_dan_prediksi_masker(frame, faceNet, maskNet):
    # Ambil dimensi frame dan membuat blob dari image yang diambil (ambil height dan width)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Pass blob ke jaringan dan mendapatkan deteksi muka
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # Inisialisasi daftar muka, lokasinya, dan prediksi yang tersedia dari mask muka
    faces = []
    locs = []
    predictions = []

    # Loop deteksi
    for i in range(0, detections.shape[2]):
        # Ekstraksi kemungkinan deteksi
        confidence = detections[0, 0, i, 2]

        # Filter deteksi yang lemah dan pastikan kemungkinan terbesar tercapai
        if confidence > 0.5:
            # Kalkulasi (x, y) - koordinate bounding box dari objek
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Memastikan bounding box berada di dalam dimensi frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Ekstraksi ROI dalam muka, konversi warna dari BGR menjadi RGB channel
            # Disortir, lalu diatur ulang menjadi 224x224, dan melakukan preprocess
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Tambahkan muka dan bounding box ke dalam list
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Prediksi hanya akan dilakukan jika setidaknya terdapat satu muka terdeteksi
    if len(faces) > 0:
        # Untuk performa lebih cepat, langsung membuat prediksi dalam batch
        # DIbanding melakukan prediksi pada muka satu-per-satu
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size = 32)

    # Mengembalikan nilai 2-tokasi muka dan lokasinya
    return (locs, predictions)

# Memuat serialisasi pendeteksi muka dari disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Memuat model mask pendeteksi muka dari disk
maskNet = load_model("mask_detector.model")

# Inisialisasi siaran video satu kamera
print("[Info] Memulai siaran video...")
vs = VideoStream(src = 1).start()

# Untuk kamera eksternal dan kameranya dua
# vs = VideoStream(src = 1).start()

# Melakukan looping selama siaran berlangsung untuk merender frame per detik
while True:
    # Ambil frame dari siaran video dan mengatur dimensinya menjadi maksimal 400px
    frame = vs.read()
    frame = imutils.resize(frame, width = 1000)

    # Deteksi muka dalam frame dan mengambil keputusan apakah memakai masker atau tidak
    (locs, predictions) = deteksi_dan_prediksi_masker(frame, faceNet, maskNet)

    # Melakukan loop berulang untuk muka yang terdeteksi beserta output lokasinya
    for (box, pred) in zip(locs, predictions):
        # Buka bounding box dan prediksinya
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Menentukan label dan warna yang akan kita render di bounding box beserta teksnya
        label = ""
        if mask > withoutMask:
            label = "Bermasker"
            color = (0, 255, 0)
        else:
            label = "Tidak Bermasker"
            color = (0, 0, 255)
            # playsound('audio/bleep-fail.wav')
            print('Tidak menggunakan masker')

        # Muat tingkat prediksi ke dalam label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Tampilkan label dan bounding box dalam frame
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Tampilkan frame output
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Ketika 'q' ditekan, hentikan loop
    if key == ord("q"):
        break

# Cleanup jika program dihentikan
cv2.destroyAllWindows()
vs.stop()