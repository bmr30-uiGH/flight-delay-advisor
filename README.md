---
title: Flight Delay + Passenger Rights Advisor
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# Flight Delay + Passenger Rights Advisor

Aplikasi ini menggabungkan dua fungsi untuk membantu penumpang penerbangan di Amerika Serikat:

1. memperkirakan probabilitas keberangkatan terlambat sekurang-kurangnya 15 menit (`DEP_DELAY >= 15`) menggunakan pola historis BTS; dan
2. menjawab pertanyaan tentang hak penumpang dengan mencari informasi pada dokumen U.S. DOT dan ringkasan kebijakan maskapai yang disimpan secara lokal.

Aplikasi dibuat dengan Streamlit, scikit-learn, MLflow, retrieval TF-IDF, Groq Responses API, dan Docker Compose.

> **Batas penggunaan:** prediksi bukan status penerbangan langsung dan bukan jaminan hasil aktual. Dokumen hak penumpang dalam proyek ini berfokus pada yurisdiksi Amerika Serikat. Jawaban chatbot bersifat informasional, bukan nasihat hukum.

## Fitur

- Mengunduh dan membersihkan data bulanan U.S. DOT Bureau of Transportation Statistics (BTS) On-Time Performance.
- Membuat label `DELAY_15` dari `DEP_DELAY >= 15` dan menyimpan data terproses dalam Parquet.
- Membandingkan kandidat model machine learning dan mencatat parameter serta metrik ke MLflow.
- Mendaftarkan model terpilih ke MLflow Model Registry dengan alias `champion`.
- Menyediakan prediksi melalui tool `predict_flight_delay`.
- Mengindeks dokumen hak penumpang lokal dan mencari kutipan relevan melalui tool `search_passenger_rights`.
- Menggunakan Groq Responses API untuk memilih tool sesuai pertanyaan pengguna.
- Menyediakan pre-filter untuk menghapus sejumlah instruksi manipulatif yang dikenali. Pre-filter ini merupakan perlindungan tambahan yang terbatas, bukan jaminan terhadap seluruh serangan prompt injection.

## Arsitektur singkat

1. **Data ingestion:** arsip bulanan BTS diunduh dari `https://transtats.bts.gov/PREZIP/`.
2. **Data processing:** data dibersihkan dan disimpan sebagai `data/processed/flights.parquet`.
3. **Model training:** kandidat baseline, Logistic Regression, dan Extra Trees dievaluasi. Metrik serta model dicatat di MLflow.
4. **Model serving:** aplikasi lebih dahulu mencoba memuat `models:/flight_delay_model@champion`; jika Registry tidak dapat diakses, aplikasi menggunakan `models/flight_delay_model.joblib` sebagai fallback.
5. **Passenger-rights retrieval:** file teks pada `documents/` dipecah menjadi chunks dan disimpan di `data/processed/passenger_rights_chunks.jsonl`.
6. **AI agent:** Groq Responses API memilih tool prediksi, retrieval, keduanya, atau tidak memanggil tool.
7. **User interface:** Streamlit menyediakan formulir detail penerbangan dan chatbot.

## Persyaratan

- Docker Desktop dengan Docker Compose.
- Koneksi internet untuk mengunduh arsip BTS saat bootstrap pertama dan untuk mengakses Groq.
- Groq API key untuk mode AI Agent penuh. Tanpa key, aplikasi dapat berjalan dalam mode fallback lokal dengan kemampuan terbatas.
- Ruang disk yang cukup untuk arsip BTS, data Parquet, model, dan artefak MLflow.

## Menjalankan aplikasi

### 1. Siapkan environment

Dari folder proyek, buat file `.env` berdasarkan template:

**PowerShell:**

```powershell
Copy-Item .env.example .env
```

Buka `.env`, lalu isi `GROQ_API_KEY` dengan API key milik Anda. Jangan masukkan API key ke README, source code, screenshot, atau arsip proyek yang dibagikan.

### 2. Build dan jalankan layanan

```powershell
docker compose up -d --build
```

Pada bootstrap pertama, layanan dapat mengunduh data BTS, memproses data, membangun indeks dokumen, melatih model, dan mendaftarkan model ke MLflow. Proses pertama dapat memerlukan waktu dan bandwidth.

### 3. Periksa layanan

```powershell
docker compose ps
```

Buka alamat berikut setelah layanan siap:

- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000

### 4. Menggunakan aplikasi

1. Isi maskapai, bandara asal, bandara tujuan, tanggal, dan jam keberangkatan terjadwal pada bagian **Detail penerbangan**.
2. Tekan **Simpan detail penerbangan**.
3. Pada chatbot, masukkan `Perkirakan risiko delay untuk penerbangan saya.`
4. Untuk hak penumpang, ajukan pertanyaan yang menyebutkan jenis penerbangan, kejadian, dan hak yang ingin diperiksa.

Prediksi menggunakan data yang disimpan dari formulir. Model tidak mengakses status penerbangan, cuaca, atau operasi bandara secara real-time.

## Environment variables

| Variable | Keterangan | Nilai contoh |
|---|---|---|
| `GROQ_API_KEY` | API key Groq; rahasia, jangan dibagikan | kosong pada `.env.example` |
| `LLM_BASE_URL` | Endpoint API Groq yang kompatibel | `https://api.groq.com/openai/v1` |
| `OPENAI_MODEL` | Nama model yang dipakai melalui SDK OpenAI-compatible | `openai/gpt-oss-120b` |
| `BTS_MONTHS` | Jumlah bulan BTS yang diminta saat bootstrap | `3` |
| `MAX_TRAIN_ROWS` | Batas jumlah baris training | `50000` |
| `MLFLOW_TRACKING_URI` | URI tracking MLflow; Compose mengatur `http://mlflow:5000` untuk container | `http://mlflow:5000` |
| `MODEL_NAME` | Nama model di MLflow Registry | `flight_delay_model` |

Nama `OPENAI_MODEL` dipertahankan karena merupakan nama variabel konfigurasi dalam kode. Pada konfigurasi default proyek ini, provider yang digunakan adalah Groq.

## Perintah Docker yang sering digunakan

**Menyalakan layanan yang sudah dibuat:**

```powershell
docker compose up -d
```

**Melihat status container:**

```powershell
docker compose ps
```

**Melihat log aplikasi:**

```powershell
docker compose logs -f app
```

**Melihat log MLflow:**

```powershell
docker compose logs -f mlflow
```

**Memuat ulang aplikasi setelah perubahan `.env`:**

```powershell
docker compose up -d --force-recreate app
```

**Build ulang image jika dependency atau Dockerfile berubah:**

```powershell
docker compose up -d --build
```

**Menghentikan layanan tanpa menghapus volume/data:**

```powershell
docker compose down
```

Jangan gunakan `docker compose down -v` kecuali memang ingin menghapus volume yang dikelola Docker.

## Training dan MLflow

MLflow menyimpan eksperimen, metrik, artefak, dan Model Registry pada direktori `storage/`.

Untuk melatih ulang dari dataset Parquet yang sudah tersedia (MLflow harus aktif):

```powershell
docker compose run --rm bootstrap python -u -m src.train_model
```

Untuk memeriksa alias model:

```powershell
docker compose exec mlflow python -c "from mlflow.tracking import MlflowClient; c=MlflowClient(tracking_uri='http://localhost:5000'); m=c.get_model_version_by_alias('flight_delay_model','champion'); print(f'Nama: {m.name} | Versi: {m.version} | Source: {m.source}')"
```

Jalankan training ulang hanya jika memang ingin memperbarui model atau pipeline. Menyalakan aplikasi sehari-hari cukup dengan `docker compose up -d`.

## Pengujian

Jalankan pengujian unit di dalam container aplikasi saat container `app` sedang berjalan:

```powershell
docker compose exec app python -m pytest -q
```

Atau jalankan di lingkungan Python lokal setelah memasang dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

Gunakan `python -m pytest`, bukan hanya `pytest`, agar direktori proyek ditambahkan dengan benar pada import path di sebagian lingkungan.

Pengujian unit memeriksa fungsi seperti routing tool, pre-filter prompt injection, pemrosesan data, ingest dokumen, dan referensi maskapai/bandara. Pengujian unit ini tidak menggantikan pengujian end-to-end pada Groq dan UI Streamlit. Hasil uji manual beserta batasannya dicatat pada [`tests/prompt_injection_test_report.md`](tests/prompt_injection_test_report.md).

## Hasil evaluasi model yang tercatat

Pada salah satu proses training yang tercatat di MLflow, kandidat `extra_trees` terpilih dengan metrik berikut:

| Metrik | Nilai |
|---|---:|
| Accuracy | 0.5920 |
| Precision | 0.4503 |
| Recall | 0.5592 |
| F1-score | 0.4989 |
| ROC-AUC | 0.6173 |

Metrik tersebut adalah hasil eksperimen pada data dan konfigurasi training saat itu, bukan jaminan performa pada periode atau rute lain. Evaluasi menggunakan random train/test split; evaluasi berbasis waktu dan pemeriksaan kalibrasi probabilitas masih merupakan pekerjaan lanjutan. Jangan menafsirkan probabilitas sekitar 50% sebagai kepastian penerbangan akan delay atau tepat waktu.

## Cakupan dan keterbatasan sumber hak penumpang

- Dataset penerbangan berasal dari BTS dan berfokus pada penerbangan Amerika Serikat.
- `documents/us_dot_refunds.txt` adalah ringkasan lokal dari halaman resmi DOT, bukan salinan verbatim.
- `documents/us_dot_airline_commitments_american.txt` merangkum komitmen American Airlines yang ditampilkan pada dashboard DOT. Ringkasan lokal mencatat bahwa teks alternatif dashboard yang digunakan menunjukkan tanggal pembaruan 3 Desember 2024. Periksa dashboard dan customer service plan terbaru sebelum menyatakan komitmen tersebut sebagai kebijakan terkini.
- Ambang refund DOT untuk keterlambatan kedatangan domestik dan waktu tunggu 3 jam untuk meal assistance pada komitmen maskapai adalah dua hal berbeda.
- Batas tarmac delay hanya berlaku pada keadaan pesawat di darat ketika penumpang tidak mendapat kesempatan turun; jangan menerapkannya pada semua keterlambatan saat menunggu di terminal.
- Jika dokumen yang diindeks tidak mendukung suatu klaim, chatbot seharusnya menyatakan bahwa sumber yang cukup belum ditemukan, bukan mengarang aturan.

## Keamanan

- Simpan `GROQ_API_KEY` hanya di `.env` atau secret manager platform deployment.
- Jangan sertakan `.env`, kredensial, atau `storage/` ke repositori atau arsip yang dibagikan.
- `.gitignore` mengabaikan `.env` untuk Git, tetapi **tidak** mencegah `.env` masuk ke RAR/ZIP yang dibuat secara manual. Periksa isi arsip sebelum membagikannya.
- Pre-filter prompt injection menggunakan pola tertentu. Mekanisme ini adalah defense-in-depth sederhana dan tidak menjamin deteksi seluruh prompt injection.

## Deployment publik

Konfigurasi Docker Compose saat ini disusun untuk pengembangan dan demo lokal. Deployment publik belum dianggap selesai hanya dengan mempublikasikan Streamlit. Lingkungan deployment harus menyediakan koneksi yang benar ke model/artefak, indeks dokumen, penyimpanan persisten MLflow, environment variables, dan akses API LLM. Jangan gunakan filesystem container sementara sebagai satu-satunya tempat menyimpan model atau database.

## Deployment publik dengan Hugging Face Spaces (Docker)

Konfigurasi Space menggunakan satu container yang menjalankan Streamlit pada port `8501` dan MLflow secara internal pada `127.0.0.1:5000`. Dengan cara ini, aplikasi dapat tetap memuat `models:/flight_delay_model@champion` dari registry lokal yang ikut dikemas bersama aplikasi. Dockerfile menggunakan pengguna UID `1000` dan menyiapkan kepemilikan folder `storage/` agar proses MLflow dapat mengakses database SQLite dan artefaknya di lingkungan Spaces.

### File yang wajib ikut di repository Space

- `data/processed/flights.parquet` — referensi rute dan jarak penerbangan.
- `data/processed/passenger_rights_chunks.jsonl` — indeks retrieval hak penumpang.
- `models/flight_delay_model.joblib` — fallback model lokal.
- `storage/mlflow.db` dan seluruh `storage/mlartifacts/` — metadata registry, alias `champion`, dan artefak model MLflow.
- `documents/`, `src/`, `app.py`, `requirements.txt`, `Dockerfile`, dan `supervisord.conf`.

Arsip mentah `data/raw/` tidak diperlukan untuk menjalankan aplikasi yang sudah diproses dan sengaja dikecualikan dari build. Jangan unggah `.env`.

### Langkah deploy

1. Buat Space baru di Hugging Face Hub dan pilih SDK **Docker**.
2. Pastikan repository Space memakai metadata `sdk: docker` dan `app_port: 8501` di bagian paling atas README (sudah disediakan pada file ini).
3. Salin/unggah isi proyek ke repository Space. Pastikan file model, file Parquet, indeks JSONL, database MLflow, dan folder artefak MLflow ikut terunggah.
4. Pada Space → **Settings** → **Variables and secrets**, tambahkan secret `GROQ_API_KEY` dengan nilai API key pribadi. Jangan menaruh key pada README, kode, atau repository.
5. Tambahkan variable `LLM_BASE_URL` dengan nilai `https://api.groq.com/openai/v1` dan `OPENAI_MODEL` dengan nilai `openai/gpt-oss-120b`.
6. Tunggu proses build selesai, lalu buka URL Space. Uji satu prediksi penerbangan dan satu pertanyaan Passenger Rights.

### Catatan operasional

- MLflow berjalan secara internal untuk membaca registry dan alias `champion`; dashboard MLflow tidak dipublikasikan sebagai layanan terpisah.
- Registry dan artefak awal dimasukkan ke image dari repository. Perubahan MLflow yang dibuat saat runtime pada Space gratis tidak dijamin bertahan setelah container dibuat ulang. Jalankan training dan registrasi model di lingkungan lokal, lalu commit/sinkronkan database dan artefak MLflow yang baru bila model diperbarui.
- Space dengan hardware gratis dapat masuk keadaan tidur ketika tidak digunakan. Permintaan pertama setelah periode idle dapat membutuhkan waktu lebih lama.
- Deployment belum dianggap selesai sebelum URL publik berhasil dibuka dan kedua fungsi utama (prediksi dan retrieval) diuji pada Space.

## Checklist sebelum final submission

- [x] Dataset BTS diproses dan file Parquet tersedia pada lingkungan yang diperiksa.
- [x] Kandidat model baseline, Logistic Regression, dan Extra Trees dibandingkan.
- [x] Model `flight_delay_model` versi 1 terdaftar di MLflow dengan alias `champion` (berdasarkan log training dan verifikasi pengguna, 31 Oktober 2026).
- [x] Aplikasi berhasil memanggil tool prediksi dan menampilkan probabilitas.
- [x] Agent berhasil memanggil retrieval untuk pertanyaan hak penumpang.
- [x] Uji manual penolakan permintaan regulasi DOT fiktif berhasil.
- [ ] Pastikan aplikasi memuat `models:/flight_delay_model@champion` tanpa fallback ke `.joblib` pada log runtime.
- [ ] Simpan screenshot/bukti eksperimen MLflow, prediksi, retrieval, dan prompt injection.
- [ ] Lengkapi dan jalankan ulang seluruh skenario pengujian yang masih berstatus catatan pada laporan tes.
- [x] Jalankan `docker compose exec app python -m pytest -q` di container proyek: `15 passed in 3.67s` (31 Oktober 2026).
- [ ] Uji `docker compose up -d --build` dari clone bersih pada mesin lain.
- [ ] Deploy seluruh komponen yang diperlukan ke URL publik dan uji URL tersebut.
- [ ] Pastikan `.env` dan kredensial tidak masuk ke repositori atau arsip submit.

## Struktur proyek

```text
app.py
src/
  agent.py
  bootstrap.py
  config.py
  data_pipeline.py
  ingest_documents.py
  predictor.py
  reference_data.py
  retrieval.py
  train_model.py
documents/                    # ringkasan lokal sumber hak penumpang
data/                         # data mentah dan terproses (diabaikan oleh Git)
models/                       # model lokal (diabaikan oleh Git)
storage/                      # database dan artefak MLflow (diabaikan oleh Git)
tests/                        # unit tests dan laporan prompt injection
compose.yaml
Dockerfile
requirements.txt
.env.example
```
