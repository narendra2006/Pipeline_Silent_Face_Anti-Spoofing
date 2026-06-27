# Face Anti-Spoofing — ML Training Pipeline

Pipeline MLOps end-to-end untuk melatih, mengevaluasi, dan mengekspor model *liveness detection* ke format ONNX. Dirancang untuk mengamankan sistem presensi digital dari serangan biometrik palsu: foto cetak, replay layar, maupun topeng.

Output pipeline ini adalah model **ONNX (Opset 12)** yang ringan, bebas dependensi Python, dan siap di-deploy ke backend produksi.

---

## Daftar Isi

- [Arsitektur Model](#arsitektur-model)
- [Tahapan Pipeline](#tahapan-pipeline)
- [Struktur Dataset](#struktur-dataset)
- [Output Deployment](#output-deployment)

---

## Arsitektur Model

Sistem menggunakan pendekatan **Cascade Detection + Multi-Scale Ensemble Classification**.

### Face Detection (Cascade)

| Layer | Model | Peran |
|---|---|---|
| L1 | **YuNet** | Detektor utama — ringan, latensi rendah |
| L2 | **SCRFD** (InsightFace) | Fallback jika YuNet gagal di sudut ekstrem |

### Liveness Classification (Ensemble MiniFASNet)

Tiga model dijalankan paralel, masing-masing membaca informasi di level yang berbeda:

| Model | Skala | Yang Dianalisis |
|---|---|---|
| MiniFASNet-1 | 1.0× | Tekstur mikro: pantulan layar, piksel pecah, serat kertas |
| MiniFASNet-2 | 2.7× | Proporsi dan geometri 3D wajah |
| MiniFASNet-3 | 4.0× | Konteks tepi frame: bezel HP, jari yang memegang foto |

---

## Tahapan Pipeline

Seluruh pipeline dijalankan secara berurutan di Google Colab.

## ⚠️ Aturan Wajib Eksekusi (Google Colab)

Karena pipeline ini mengunci versi beberapa *library* spesifik (seperti ONNX dan NumPy) untuk menghindari *Environment Drift* bawaan Colab, Anda **wajib** mengikuti urutan eksekusi berikut agar tidak terjadi *error* (*crash*):

1. **Jalankan Step 0 (Environment Setup):** Eksekusi *cell* pertama yang berisi skrip instalasi (`!pip install...`) dan tunggu hingga proses selesai 100%.
2. **RESTART SESSION (Sangat Krusial):** Setelah instalasi selesai, **JANGAN** langsung menjalankan *cell* di bawahnya. Anda wajib menyegarkan memori Colab dengan cara klik menu bar di atas: **Runtime ➔ Restart session** (Mulai ulang sesi). 
3. **Lanjutkan Pipeline:** Setelah sesi berhasil di-*restart*, abaikan/lewati Step 0 (jangan dijalankan ulang), dan langsung jalankan *cell* **Import Library** dilanjutkan dengan step-step ke bawahnya secara berurutan hingga selesai.

### Step 1 — Data Ingestion & Workspace Setup

- Ekstrak dataset `.zip` dari Google Drive ke storage lokal Colab.
- Validasi keseimbangan kelas (*class balance*) antara data asli dan spoof.

### Step 2 — Preprocessing & Multi-Scale Crop

- Deteksi dan crop wajah otomatis ke ukuran `80×80` piksel.
- Catat foto korup atau yang tidak mengandung wajah ke log audit.
- **Anti-Leakage Split** — dataset dibagi Train/Val/Test (70/15/15%) menggunakan pengelompokan per identitas, sehingga wajah yang sama tidak tersebar lintas split.

### Step 3 — Training (GPU)

- Augmentasi on-the-fly: rotasi, flip horizontal, color jitter.
- Tiga model MiniFASNet dilatih secara independen.
- Dataset di-mount via symlink untuk menghindari duplikasi di disk.

### Step 4 — Evaluasi & Diagnostics

- Blind test pada Test Set yang belum pernah dilihat model.
- Kalkulasi metrik standar industri:

  | Metrik | Definisi | Target |
  |---|---|---|
  | **FAR** (False Acceptance Rate) | Spoof lolos sebagai wajah asli | < 1% |
  | **FRR** (False Rejection Rate) | Wajah asli ditolak sistem | < 5%  |

- Visualisasi Confusion Matrix dan pencarian threshold optimal.

### Step 5 — ONNX Export

- Bersihkan state dictionary PyTorch sebelum konversi.
- Export ke `.onnx` Opset 12 dengan dynamic axes.
- **Discrepancy validation** — bandingkan output numerik ONNX vs PyTorch untuk memastikan presisi tidak bergeser pasca-konversi.

---

## Struktur Dataset

Siapkan dataset dalam format `.zip` dengan dua kelas utama. Sub-folder di dalam setiap kelas bersifat opsional dan dapat disesuaikan dengan variasi kondisi pengambilan data.

```
dataset.zip
├── class_1_asli/
│   ├── normal/
│   ├── backlight/
│   └── masker/
└── class_0_spoof/
│   ├── normal/
│   ├── backlight/
    └── masker/
```

---

## Output Deployment

Pipeline menghasilkan 4 file ONNX yang secara otomatis dikirim ke Google Drive dan siap dipindahkan ke direktori `onnx_models/` di repository backend.

| File | Fungsi |
|---|---|
| `yunet.onnx` | Face detector utama |
| `anti_spoofing_1_80x80.onnx` | Classifier skala 1.0× |
| `anti_spoofing_2.7_80x80.onnx` | Classifier skala 2.7× |
| `anti_spoofing_4_80x80.onnx` | Classifier skala 4.0× |
