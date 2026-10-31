# Deploy Flight Delay Advisor ke Streamlit Community Cloud (gratis)

## Cara kerja deployment gratis

Deployment publik memakai:
- Streamlit Community Cloud untuk menjalankan `app.py`.
- Groq Responses API untuk AI Agent (API key dimasukkan melalui Streamlit Secrets).
- `models/flight_delay_model.joblib` untuk prediksi lokal di aplikasi publik.
- `data/processed/flights.parquet` untuk daftar rute dan referensi penerbangan.
- `data/processed/passenger_rights_chunks.jsonl` untuk retrieval hak penumpang.

MLflow Model Registry tetap tersedia pada lingkungan Docker lokal. Streamlit Community Cloud tidak menjalankan service `docker compose` pada konfigurasi ini, sehingga aplikasi publik diarahkan ke artifact `.joblib` yang sama. Jangan menyatakan bahwa MLflow server lokal sedang berjalan pada aplikasi publik.

## File yang harus masuk GitHub

- `app.py`
- `src/`
- `documents/`
- `data/processed/flights.parquet`
- `data/processed/passenger_rights_chunks.jsonl`
- `models/flight_delay_model.joblib`
- `requirements.txt`
- `tests/` (opsional tetapi disarankan)
- `README.md`, `.gitignore`, `requirements-local.txt`, `Dockerfile`, `compose.yaml`

Jangan unggah `.env`, arsip `data/raw/*.zip`, `storage/mlflow.db`, `storage/mlartifacts/`, folder `.venv/`, `.pytest_cache/`, atau file RAR/ZIP.

## Penting: versi scikit-learn

Model yang disimpan dalam artifact saat ini dibuat dengan scikit-learn **1.9.1**. `requirements.txt` mem-pin versi itu agar kompatibilitas model tetap terjaga. Pilih Python **3.11** pada Advanced settings Streamlit Community Cloud agar sejalan dengan image Docker lokal.

## Langkah deploy

1. Buat repository GitHub bernama `flight-delay-advisor`. Public boleh dipilih karena kode dan data aplikasi bukan rahasia; API key TIDAK BOLEH ada di repository.
2. Pastikan `.gitignore` dari paket ini sudah diterapkan.
3. Push folder proyek ke branch `main`.
4. Buka https://share.streamlit.io/ dan hubungkan akun GitHub.
5. Pilih **Create app**, repository `flight-delay-advisor`, branch `main`, dan main file path `app.py`.
6. Pada Advanced settings, pilih Python **3.11** dan tambahkan Secrets berikut:

```toml
GROQ_API_KEY = "ISI_API_KEY_GROQ_BARU_DI_SINI"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
OPENAI_MODEL = "openai/gpt-oss-120b"
MODEL_SOURCE = "joblib"
```

7. Deploy dan periksa logs jika build gagal.

## Uji setelah deployment

- Pastikan sidebar menampilkan bahwa API key Groq terdeteksi.
- Simpan data penerbangan dan minta: `Perkirakan risiko delay untuk penerbangan saya.`
- Ajukan pertanyaan hak penumpang untuk memastikan `search_passenger_rights` mengambil chunk lokal.
- Tes tanpa API key akan menggunakan local fallback dengan kemampuan agent yang terbatas.

## Catatan keamanan dan keterbatasan

- Jangan menulis API key di source code, README, screenshot, atau GitHub Issues.
- Jika API key pernah masuk ke archive atau repository, segera revoke/rotate key tersebut.
- Prediksi bersifat historis dan bukan status penerbangan real-time.
- Dataset dan dokumen yang digunakan berfokus pada konteks penerbangan Amerika Serikat.
- Probabilitas model bukan jaminan hasil aktual.
