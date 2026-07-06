# Aplikasi Klasterisasi Zona RTH Kabupaten Sukabumi (Streamlit)

Aplikasi ini adalah **terjemahan Python/Streamlit** dari script R `RTH.R`.
Karena Streamlit hanya berjalan di atas Python (bukan R), seluruh logika
analisis (data cleaning, normalisasi Min-Max, evaluasi K dengan
Elbow/Silhouette/DBI, K-Means, labeling zona, scatter plot, dan peta
interaktif) sudah ditulis ulang di `app.py` menggunakan `pandas`,
`scikit-learn`, `plotly`, dan `folium` — hasil analisisnya setara dengan
script R aslinya.

## Isi Folder
```
rth_app/
├── app.py               # Aplikasi utama Streamlit
├── requirements.txt      # Daftar library Python yang dibutuhkan
├── Dataset_rth.csv       # Dataset bawaan (bisa diganti lewat upload di sidebar)
└── README.md             # Panduan ini
```

## 1. Menjalankan Secara Lokal (opsional, untuk uji coba)
Pastikan Python 3.9+ sudah terpasang, lalu jalankan di terminal:

```bash
cd rth_app
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`.

## 2. Deploy ke Streamlit Community Cloud (Gratis)

**Langkah 1 — Unggah ke GitHub**
1. Buat repository baru di GitHub (contoh: `klasterisasi-rth-sukabumi`).
2. Upload 3 file ini ke repository: `app.py`, `requirements.txt`, `Dataset_rth.csv`.
   (Bisa lewat web GitHub "Add file → Upload files", atau via `git push`.)

**Langkah 2 — Deploy di Streamlit Cloud**
1. Buka https://share.streamlit.io/ dan login dengan akun GitHub.
2. Klik **"New app"**.
3. Pilih repository, branch (`main`), dan file utama: `app.py`.
4. Klik **"Deploy"**.
5. Tunggu beberapa menit — Streamlit Cloud akan otomatis membaca
   `requirements.txt` dan menginstal semua library yang dibutuhkan.
6. Setelah selesai, Anda akan mendapat URL publik, misal:
   `https://klasterisasi-rth-sukabumi.streamlit.app`

## 3. Fitur Aplikasi
- **Sidebar**: unggah dataset CSV sendiri (opsional) dan atur jumlah klaster (K).
- **Evaluasi K**: grafik Elbow (WCSS), Silhouette, dan Davies-Bouldin Index (K=2–8).
- **Pemodelan K-Means**: default K=3 (sesuai script asli), bisa diubah dari sidebar.
- **Interpretasi**: tabel centroid asli per klaster & pelabelan otomatis
  (Zona RTH Tinggi/Sedang/Rendah — hanya berlaku saat K=3, sesuai logika asli).
- **Visualisasi**: scatter plot interaktif dan peta spasial interaktif (Folium).
- **Unduh hasil**: file CSV hasil clustering dan file peta HTML interaktif.

## Catatan Penting
- Format dataset CSV harus mengikuti struktur `Dataset_rth.csv` bawaan:
  kolom `Nama Kecamatan, latitude, longitude, Luas kecamatan, Luas ruang terbuka, Jumlah Penduduk`,
  dengan format angka gaya Indonesia (misal: `"30,448"` untuk luas kecamatan,
  `"101,3"` untuk luas RTH).
- Kecamatan **Cibitung** memiliki koreksi luas khusus (88.84 km²), mengikuti
  logika di script R asli.
