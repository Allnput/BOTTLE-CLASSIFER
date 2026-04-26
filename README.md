Bottle Classifier
Overview

Bottle Classifier adalah aplikasi berbasis Python yang dapat mendeteksi botol, memeriksa kondisi tutup botol, dan mengklasifikasikan level air dalam botol menggunakan kombinasi YOLOv8 dan model CNN Keras. Aplikasi ini dirancang untuk real-time inference melalui kamera komputer.

Demo
[https://colab.research.google.com/drive/1vBjNPH1faUeilesNfnP-uXJJ0nFDBvM7?usp=sharing]

Catatan: Pada klasifikasi level air, pengguna perlu menyediakan API key Kaggle (kaggle.json) saat diminta. Upload file tersebut ke Colab sesuai instruksi.

Struktur Folder
Bottle-Classifier/
│
├─ main.py                 # Script utama untuk menjalankan deteksi dan klasifikasi
├─ yolov8n.pt              # Model YOLOv8 pretrained
├─ model_water_level.keras # CNN untuk klasifikasi level air
├─ model_condition_botle.keras # CNN untuk klasifikasi kondisi botol
├─ model_cap_bottley.pt     # YOLO 2-class untuk deteksi tutup botol
└─ README.md               # Dokumentasi

Persiapan Lingkungan
Pastikan Python terinstall (gueh pake versi 3.13.0).

Install dependency:
[pip install opencv-python opencv-python-headless numpy tensorflow ultralytics]

Cara Menjalankan
-Buka folder project di terminal (klik kanan → Open with Terminal).
Jalankan script utama:
py main.py
-Aplikasi akan membuka kamera, mendeteksi botol, menampilkan bounding box botol, dan menampilkan klasifikasi:
Cap: kondisi tutup botol (Proper atau Missing/Defect)
Botol: kondisi botol (Proper atau Defect)
Air: level air (Penuh, Kurang, Overflow)
