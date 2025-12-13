# 👁️ Computer Vision Learning Projects

Repositori ini berisi kumpulan proyek pembelajaran saya dalam mendalami **Computer Vision** menggunakan **Python** dan library **OpenCV**. Di sini terdapat tiga eksperimen dasar yang mencakup deteksi wajah, pengenalan warna, dan pelacakan objek bergerak.

## 🛠️ Teknologi yang Digunakan
*   **Python 3.x**
*   **OpenCV** (`cv2`)
*   **NumPy**

## 📂 Daftar Proyek

Berikut adalah detail dari ketiga modul yang ada dalam repositori ini:

### 1. 👤 Face, Eye, and Mouth Detection
**Deskripsi:**
Proyek ini saya buat untuk mendeteksi bagian wajah seperti mata dan mulut menggunakan OpenCV. Dari sini saya belajar cara kerja deteksi objek dasar dengan model **Haar Cascade**.

**Fitur Utama:**
*   Deteksi Wajah (Face Detection) secara real-time.
*   Deteksi Mata (Eye Detection) di dalam area wajah.
*   Deteksi Mulut (Mouth Detection).
*   Menghitung jumlah wajah, mata, dan mulut yang muncul di kamera

---

### 2. 🟦 Blue Color Detection
**Deskripsi:**
Program sederhana untuk mengenali dan menandai objek berwarna biru pada kamera. Proyek ini membantu saya memahami cara memproses warna menggunakan ruang warna **HSV (Hue, Saturation, Value)** untuk memisahkan warna tertentu dari background.

**Fitur Utama:**
*   Konversi warna frame dari BGR ke HSV.
*   Pembuatan Masking untuk warna biru.
*   Menandai objek terdeteksi dengan kotak pembatas (Bounding Box).

---

### 3. ⚽ Ball Tracking
**Deskripsi:**
Proyek ini digunakan untuk melacak pergerakan bola secara real-time. Melalui proyek ini, saya belajar tentang teknik **Contour Detection** dan bagaimana mengikuti (tracking) koordinat X dan Y dari objek bola yang bergerak.

**Fitur Utama:**
*   Deteksi bentuk lingkaran/bola.
*   Menghitung titik tengah (centroid) objek.
*   Menggambar jejak lintasan pergerakan bola.

---
