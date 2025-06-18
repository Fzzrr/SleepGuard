# Sleep Guard: Sistem Deteksi Kantuk Berkendara

**Sleep Guard** adalah sistem deteksi kantuk berbasis AI yang dirancang untuk membantu pengemudi mendeteksi tanda-tanda kantuk dan kelelahan saat berkendara. Sistem ini menggunakan kamera untuk mendeteksi status mata pengemudi secara real-time dan memberikan alarm jika pengemudi berisiko tertidur.

## Gambaran Umum

Proyek ini memanfaatkan **Convolutional Neural Network (CNN)** untuk mendeteksi apakah mata pengemudi terbuka atau tertutup. Ketika sistem mendeteksi mata tertutup lebih dari 3 detik, sistem akan memicu alarm untuk memberi peringatan kepada pengemudi. Sistem ini menggunakan **OpenCV** untuk deteksi wajah dan mata, serta **TensorFlow** untuk model klasifikasi.

## Model yang Digunakan

Inti dari sistem ini adalah model **Convolutional Neural Network (CNN)** yang mengklasifikasikan gambar mata pengemudi ke dalam dua kategori: "Terbuka" dan "Tertutup." Model ini dilatih menggunakan dataset gambar mata terbuka dan tertutup, lalu digunakan untuk prediksi secara real-time untuk mendeteksi apakah pengemudi menunjukkan tanda-tanda kantuk.

### Arsitektur CNN:
- Model ini terdiri dari beberapa lapisan konvolusi yang diikuti dengan pooling dan lapisan fully connected. Model ini dilatih untuk mengenali status mata dengan mempelajari fitur visual dari gambar masukan.
- Model ini dikompilasi menggunakan **Adam optimizer** dan menggunakan **binary cross-entropy** sebagai fungsi loss, yang menjadikannya cocok untuk klasifikasi biner (mata terbuka/tertutup).

### Masalah yang Diselesaikan:
- **Deteksi Kelelahan**: Model ini mendeteksi apakah mata pengemudi tertutup terlalu lama, yang merupakan tanda kantuk saat berkendara.
- **Sistem Peringatan Real-time**: Jika kantuk terdeteksi, alarm akan berbunyi untuk memberi peringatan kepada pengemudi agar tetap terjaga dan waspada.

## Dataset yang Digunakan

Dataset yang digunakan untuk melatih model ini terdiri dari gambar mata dalam dua kondisi: "Terbuka" dan "Tertutup." Dataset ini digunakan untuk melatih model Convolutional Neural Network (CNN) untuk mengklasifikasikan status mata.

### Link Dataset:
- Dataset dapat diunduh dari repositori yang telah Anda unggah di sini: [Eye_Dataset.zip](sandbox:/mnt/data/Eye_Dataset.zip).
- Dataset harus disusun sebagai berikut:
