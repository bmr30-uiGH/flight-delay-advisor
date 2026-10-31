# Flight Delay Advisor — Source Cleanup Patch

Patch ini merapikan jawaban dan daftar sumber Passenger Rights tanpa mengubah model ML, data BTS, dokumen retrieval, atau MLflow.

## File yang diperbarui

- `src/agent.py`
  - Menghapus penanda chunk/file internal dari jawaban LLM.
  - Menghapus bibliografi buatan LLM karena Streamlit menampilkan sumber berdasarkan metadata retrieval.
  - Mengelompokkan chunks berdasarkan URL agar satu halaman sumber tidak muncul berulang.
  - Menggabungkan bagian dan excerpt unik agar bukti yang ditemukan tetap terlihat.
  - Menghilangkan nama tool berulang pada label `Tool yang dijalankan`.
  - Mempertegas agar jawaban AA menjelaskan semua komitmen yang didukung retrieval.
- `app.py`
  - Menampilkan satu tautan per sumber, lalu bagian dokumen dan excerpt pendukungnya.
- `tests/test_source_presentation.py`
  - Lima tes baru untuk marker internal, footer sumber, dan deduplikasi metadata.

## Cara menerapkan

Dari root folder proyek, buat backup di temporary folder Windows:

```powershell
Copy-Item .\src\agent.py "$env:TEMP\agent.py.before-source-cleanup.bak"
Copy-Item .\app.py "$env:TEMP\app.py.before-source-cleanup.bak"
```

Ekstrak ZIP ke root folder proyek dengan `-Force`.

Lalu jalankan:

```powershell
docker compose exec app python -m pytest -q
docker compose restart app
```

Target dari test suite pada patch ini: `30 passed`.

Tidak perlu menjalankan ulang ingestion, download BTS, training model, atau registrasi MLflow. Setelah tes lokal dan pertanyaan Passenger Rights lolos, commit dan push perubahan ke GitHub untuk memperbarui Streamlit Community Cloud.

Jangan masukkan `.env`, API key, folder `storage`, atau arsip `data/raw` ke GitHub.
