# Flight Delay + Passenger Rights Advisor

Aplikasi ini menggabungkan prediksi risiko keterlambatan penerbangan dan Passenger Rights Advisor berbasis retrieval-augmented generation (RAG).

Fungsi utama:
1. Memperkirakan probabilitas keberangkatan terlambat sekurang-kurangnya 15 menit (`DEP_DELAY >= 15`) berdasarkan pola historis U.S. DOT Bureau of Transportation Statistics (BTS).
2. Menjawab pertanyaan hak penumpang penerbangan Amerika Serikat berdasarkan ringkasan U.S. DOT dan eCFR.
3. Menjawab pertanyaan hak penumpang penerbangan Indonesia berdasarkan PM 89/2015 dan UU Nomor 1 Tahun 2009.

Teknologi yang digunakan meliputi Streamlit, Python, scikit-learn, TF-IDF, MLflow, Groq Responses API, dan Docker Compose.

> **Batas penggunaan:** prediksi bukan status penerbangan real-time dan bukan jaminan hasil aktual. Dataset prediksi berasal dari BTS dan berfokus pada penerbangan Amerika Serikat. Jawaban hak penumpang bersifat informasional, bukan nasihat hukum. Periksa sumber resmi untuk keputusan aktual.

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

1. **Data ingestion:** pipeline mencari dan mengunduh arsip bulanan BTS dari `https://transtats.bts.gov/PREZIP/`.
2. **Data processing:** data dibersihkan, diberi target `DELAY_15`, lalu disimpan dalam `data/processed/flights.parquet`.
3. **Model training:** kandidat model dievaluasi menggunakan train/test split. Parameter dan metrik dicatat di MLflow; model terpilih didaftarkan dengan alias `champion`.
4. **Model serving:** Docker lokal memprioritaskan `models:/flight_delay_model@champion` dan menyediakan fallback `models/flight_delay_model.joblib`.
5. **Retrieval Amerika Serikat:** dokumen U.S. DOT/eCFR diproses oleh `src/ingest_documents.py` menjadi `data/processed/passenger_rights_chunks.jsonl`.
6. **Retrieval Indonesia:** `src/indonesia_rights.py` membaca langsung `id_pm_89_2015.txt` dan `id_uu_1_2009.txt`, kemudian melakukan TF-IDF retrieval. Dokumen Indonesia tidak digabung ke indeks JSONL Amerika Serikat.
7. **AI Agent:** routing lokal menentukan tool yang relevan; Groq Responses API memilih di antara tool yang tersedia.
8. **User interface:** Streamlit menyediakan formulir penerbangan dan chatbot.

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
| `GROQ_API_KEY` | API key Groq; rahasia, jangan dibagikan | Kosong pada `.env.example` |
| `LLM_BASE_URL` | Endpoint API Groq | `https://api.groq.com/openai/v1` |
| `GROQ_MODEL` | Model Groq | `qwen/qwen3.8-27b` |
| `OPENAI_MODEL` | Model alternatif untuk provider OpenAI-compatible non-Groq | `gpt-4.1-mini` |
| `MODEL_SOURCE` | Sumber model prediksi | `mlflow` untuk Docker; `joblib` untuk Streamlit Cloud |
| `BTS_MONTHS` | Jumlah bulan arsip BTS yang diminta saat bootstrap | `3` |
| `MAX_TRAIN_ROWS` | Batas baris training sesuai konfigurasi pipeline | `50000` |
| `MLFLOW_TRACKING_URI` | URI MLflow di Docker | `http://mlflow:5000` |
| `MODEL_NAME` | Nama model di MLflow Registry | `flight_delay_model` |

Untuk Groq, agent membaca `GROQ_MODEL`. Jika tidak diatur, default aplikasi adalah `qwen/qwen3.8-27b`. `OPENAI_MODEL` tidak digunakan untuk memilih model Groq.

Docker lokal menggunakan `MODEL_SOURCE=mlflow`. Streamlit Community Cloud menggunakan `MODEL_SOURCE=joblib` karena deployment tersebut tidak menjalankan service MLflow Compose.

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

Hasil pengujian 3 November 2026:

- Repositori pengembangan: **176 passed**.
- Clean clone GitHub commit a7afbef: **161 passed, 1 skipped**. Satu test V3 dilewati karena dependensi eksperimen opsional tidak tersedia.
- Clean clone berhasil menjalankan Bootstrap, Streamlit, MLflow, prediksi ML, dan AI Agent Groq dengan retrieval dokumen Indonesia.
- Health Streamlit dan MLflow menghasilkan HTTP 200.

Pengujian clean clone menggunakan flights.parquet yang sudah tersedia di GitHub. Pengunduhan data BTS mentah dari nol belum diuji pada clean clone. Pengujian otomatis tidak menggantikan pengujian manual Groq. Laporan keamanan tersedia di tests/prompt_injection_test_report.md.

## Hasil evaluasi model yang tercatat

Eksperimen model-selection memilih kandidat `extra_trees` dengan metrik berikut:

| Metrik | Nilai |
|---|---:|
| Accuracy | 0.5920 |
| Precision | 0.4503 |
| Recall | 0.5592 |
| F1-score | 0.4989 |
| ROC-AUC | 0.6173 |

Metrik berasal dari random train/test split dan bukan jaminan performa pada periode atau rute lain. Evaluasi berbasis waktu dan kalibrasi probabilitas masih menjadi keterbatasan.

Audit MLflow lingkungan pengembangan mencatat **16 eksperimen dengan 122 runs**. Eksperimen development validation bukan otomatis hasil final holdout.

Pada clean clone, bootstrap melatih model tersendiri karena Registry awalnya kosong. `logistic_regression` terpilih dengan ROC-AUC 0.7010. Model tersebut tidak menggantikan Champion lingkungan pengembangan utama.

## Cakupan dan keterbatasan sumber hak penumpang

- Dataset prediksi BTS berfokus pada penerbangan Amerika Serikat dan bukan sumber status real-time.
- Dokumen hak penumpang AS menggunakan ringkasan U.S. DOT/eCFR yang diindeks ke JSONL.
- Dokumen Indonesia menggunakan `id_pm_89_2015.txt` dan `id_uu_1_2009.txt`, yang dibaca langsung oleh `src/indonesia_rights.py` melalui TF-IDF retrieval.
- Dokumen lokal merupakan ringkasan untuk retrieval, bukan pengganti pemeriksaan sumber resmi.
- Ringkasan komitmen American Airlines pada dashboard DOT mencatat label pembaruan 3 Desember 2024. Periksa versi terbaru sebelum menyatakan kebijakannya masih berlaku.
- Ambang refund DOT dan waktu tunggu meal assistance merupakan dua ketentuan berbeda.

## Keamanan

- Simpan `GROQ_API_KEY` hanya di `.env` atau Streamlit Secrets.
- Jangan unggah `.env`, kredensial, atau `storage/` ke GitHub.
- `.gitignore` tidak mencegah `.env` masuk arsip ZIP/RAR manual.
- Pre-filter prompt injection memiliki keterbatasan dan tidak menjamin pencegahan semua variasi serangan.
- Laporan pengujian tersedia di `tests/prompt_injection_test_report.md`.

## Deployment publik — Streamlit Community Cloud

Aplikasi publik menggunakan Streamlit Community Cloud, Groq Responses API, dan model lokal melalui `MODEL_SOURCE=joblib`. MLflow Registry digunakan pada Docker lokal.

Prediksi menggunakan `models/flight_delay_model.joblib`. Dokumen Indonesia dibaca langsung dari folder `documents/`.

Atur `GROQ_API_KEY`, `LLM_BASE_URL`, `GROQ_MODEL`, dan `MODEL_SOURCE=joblib` pada Streamlit Secrets. `app.py` memuat Secrets sebelum mengimpor AI Agent.

Panduan deployment: [README_STREAMLIT_CLOUD.md](README_STREAMLIT_CLOUD.md).

### Hasil pengujian clean clone

- Docker build dan bootstrap berhasil dari clone GitHub.
- Streamlit dan MLflow health menghasilkan HTTP 200.
- Model Champion berhasil dimuat dan prediksi berhasil dijalankan.
- AI Agent Groq berhasil menjawab pertanyaan hukum Indonesia menggunakan retrieval.
- Test suite clean clone commit `a7afbef`: **161 passed, 1 skipped**.
- Test suite pengembangan: **176 passed** pada 3 November 2026.

Clean clone menggunakan `flights.parquet` yang sudah tersedia di GitHub. Pengunduhan BTS mentah dari nol belum diuji pada clean clone.

### Checklist final submission

- [x] Data pipeline BTS dan penyimpanan Parquet.
- [x] Model ML, evaluasi, dan prediksi aplikasi.
- [x] AI Agent dan retrieval hak penumpang AS serta Indonesia.
- [x] Docker clean clone dan bootstrap.
- [x] Deployment publik telah diuji.
- [x] Eksperimen dan Model Registry MLflow.
- [x] Pengujian prompt injection dan dokumentasinya.
- [ ] Kumpulkan screenshot prediksi, chatbot, MLflow, Docker, dan pengujian keamanan.

## Struktur proyek

```text
app.py
src/
  agent.py
  bootstrap.py
  data_pipeline.py
  ingest_documents.py
  indonesia_rights.py
  predictor.py
  retrieval.py
  train_model.py
documents/
  id_pm_89_2015.txt
  id_uu_1_2009.txt
  us_*.txt
data/processed/
  flights.parquet
  passenger_rights_chunks.jsonl
models/
  flight_delay_model.joblib
storage/
tests/
compose.yaml
Dockerfile
requirements.txt
requirements-local.txt
.env.example
README_STREAMLIT_CLOUD.md
```

Jangan masukkan `.env`, API key, arsip BTS mentah, atau penyimpanan runtime MLflow ke GitHub.
