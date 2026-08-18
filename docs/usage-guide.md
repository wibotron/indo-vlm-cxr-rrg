# Cara Menjalankan Eksperimen

- Sebelum menjalankan eksperimen apapun, jalankan terlebih dahulu notebook preprocessing:
  - `notebooks/eda_and_preprocessing_chexpert-plus.ipynb`
  - Notebook ini menyiapkan data train, dev, dan test dalam format parquet yang digunakan oleh seluruh pipeline.

## Urutan Menjalankan Eksperimen

- Setiap eksperimen terdiri dari tiga tahap utama yang harus dijalankan secara berurutan.

### Tahap 1: Stage-B

- Stage-B bersifat sama untuk kedua eksperimen.
- Jalankan notebook stage_b dari salah satu eksperimen:
  - `notebooks/exp_001/stage_b.ipynb`
  - atau `notebooks/exp_002/stage_b.ipynb`
- Cukup jalankan salah satu, karena keduanya identik.

### Tahap 2: Stage-A

- Stage-A berbeda antara exp_001 dan exp_002 karena menggunakan alignment module yang berbeda.
- Untuk exp_001:
  - `notebooks/exp_001/stage_a.ipynb`
- Untuk exp_002:
  - `notebooks/exp_002/stage_a.ipynb`

### Tahap 3: Stage-MEETING

- Stage-MEETING membutuhkan checkpoint dari Stage-A dan Stage-B.
- Pastikan Anda mengarahkan konfigurasi ke checkpoint yang benar sesuai dengan eksperimen yang sedang dijalankan.
- Untuk exp_001:
  - `notebooks/exp_001/stage_meeting.ipynb`
- Untuk exp_002:
  - `notebooks/exp_002/stage_meeting.ipynb`

## Dependensi Eksternal

- Sebelum menjalankan evaluasi di Stage-MEETING, pastikan dependensi berikut tersedia.
- Clone repository CheXbert dari alamat berikut:
  - `https://github.com/stanfordmlgroup/CheXbert`
- Unduh file checkpoint chexbert.pth dari alamat berikut:
  - `https://huggingface.co/StanfordAIMI/RRG_scorers/tree/main`
- Letakkan file checkpoint pada lokasi berikut:
  - `external/chexbert/pretrained_weights/chexbert.pth`

## Aturan Perubahan Kode

### Prosedur saat terjadi error

- Cari referensi solusi. Boleh menggunakan bantuan LLM atau sumber lainnya.
- Ajukan izin dan kabarkan rencana perubahan sebelum menerapkan solusi pada codebase.
- Terapkan kode setelah mendapat persetujuan.

### Pelaporan

- Setiap perubahan pada kode atau hiperparameter, sekecil apa pun, wajib dilaporkan.

## Perbandingan Aspek Eksperimen

- Alignment Module:
  - Exp_001 (Default): Lightweight (nn.TransformerDecoder, cross-attention only, random init). Terinspirasi dari RaDialog.
  - Exp_002 (Conditional): Q-Former (Blip2QFormerModel, BERT-initialized, joint self-attention). Meniru RaDialog secara penuh.
- Loss Functions:
  - Exp_001 (Default): Contrastive + ITM only.
  - Exp_002 (Conditional): Contrastive + ITM. LM loss opsional melalui konfigurasi.
- Text Encoder:
  - Exp_001 (Default): BiomedBERT terpisah (frozen).
  - Exp_002 (Conditional): Internal Q-Former text branch (shared weights, bert-base-uncased).
- Target Kinerja:
  - Exp_001 (Default): Baseline cepat dan ringan.
  - Exp_002 (Conditional): Alignment lebih kaya, potensi performa lebih tinggi.