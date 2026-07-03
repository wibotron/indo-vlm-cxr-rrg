## Garis Besar Proyek
- Data masukkan: 
  - citra ronsen dada.
- Data keluaran yang diharapkan:
  - findings.
  - impressions.

## Dua Pendekatan
- Image to Text Retrieval
  - kata kunci: *matching*.
  - cara kerja: mencocokkan citra rontgen pasien dengan data laporan medis historis yang sudah ada.
  - arsitektur: encoder image + encoder text.
  - kelebihan:
    - Teks 100% tata bahasa benar dan menggunakan terminologi medis yang valid karena ditulis langsung oleh radiolog manusia di masa lalu.
    - Tidak ada risiko halusinasi klinis (model mengada-ada penyakit).
  - kekurangan:
    - Sangat kaku.
    - Jika pasien punya kombinasi kondisi yang langka atau unik.
    - Maka, gagal menemukan laporan lama yang exact matches dengan kondisi tersebut.
  - alur kerja (BiomedCLIP):
```
=========================================================================================
FASE 1: PRE-COMPUTATION & INDEXING (Dilakukan sekali di awal pada Database Laporan Medis)
=========================================================================================

  [ Kumpulan Laporan Medis Historis (Text) ] 
  (Contoh: Ribuan teks "Findings" & "Impression" dari MIMIC-CXR)
                    │
                    ▼
     ┌─────────────────────────────┐
     │   BiomedCLIP Text Encoder   │ <── (Menggunakan arsitektur PubMedBERT)
     └─────────────────────────────┘
                    │
                    ▼
    [ Matriks Vektor Teks (Dense Embeddings) ] 
    (Ukuran: [Jumlah_Laporan, D_dimensi], misal D = 512)
                    │
                    ▼
     ┌─────────────────────────────┐
     │      Database Vektor        │ (Disimpan di FAISS / ChromaDB / Qdrant)
     └─────────────────────────────┘


=========================================================================================
FASE 2: INFERENCE / RETRIEVAL RUNTIME (Saat ada Pasien Baru / Citra Rontgen Baru)
=========================================================================================

      [ Citra Rontgen Dada Baru (Query Image) ]
                    │
                    ▼
     ┌─────────────────────────────┐
     │  BiomedCLIP Image Encoder   │ <── (Menggunakan ViT - Vision Transformer)
     └─────────────────────────────┘
                    │
                    ▼
       [ Vektor Citra Pasien (x) ] ─── (Ukuran: [1, D_dimensi])
                    │
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────┐
 │ PROSES EVALUASI METRIK KEMIRIPAN (Similarity Matching)                              │
 │                                                                                     │
 │ Mengambil seluruh Vektor Teks (y_i) dari Database Vektor, lalu menghitung nilai     │
 │ Cosine Similarity antara Vektor Citra (x) dengan setiap Vektor Teks (y_i):          │
 │                                                                                     │
 │                       x · y_i                                                       │
 │    Similarity_i = ───────────────                                                   │
 │                    ||x|| ||y_i||                                                    │
 └─────────────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
     [ Array Nilai Similarity Score ] (Contoh: [0.89, 0.45, 0.12, 0.76, ...])
                    │
                    ▼
     ┌─────────────────────────────┐
     │   K-Nearest Neighbors (KNN) │ ─── (Mengurutkan skor tertinggi ke terendah)
     └─────────────────────────────┘
                    │
                    ▼
     [ Top-1 atau Top-K Match Text ]
                    │
                    ▼
  [ OUTPUT HASIL RETRIEVAL UNTUK PASIEN BARU ]
  - Findings: "Lungs are clear. Cardiomegaly is noted..."
  - Impression: "Cardiomegaly without acute pulmonary abnormality."
```


- Report Generation
  - kata kunci: (*generative*).
  - cara kerja: membuat teks kata demi kata dari representasi visual rontgen.
  - arsitektur: encoder image + decoder text.
  - kelebihan: Sangat fleksibel dan personal. Model mampu mendeskripsikan variasi posisi, ukuran, dan kombinasi temuan patologis secara granular.
  - kekurangan: Rentan terhadap halusinasi klinis (misal: rontgen normal, tapi model menulis ada pneumothorax).
  - alur kerja:
```
========================================================================================
ALUR INFERENCE / GENERASI (Proses End-to-End dari Citra Medis Menjadi Laporan Utuh)
========================================================================================

   [ Citra Rontgen Dada Baru (Input) ]
                   │
                   ▼
    ┌──────────────────────────────┐
    │     1. VISION ENCODER        │ ── (Contoh: ViT, ResNet, atau DenseNet)
    └──────────────────────────────┘
                   │
                   ▼
     [ Visual Feature Maps/Tokens ] ── (Representasi matriks/grid dari gambar,
                   │                    ukuran: [Batch, Num_Patches, Hidden_Dim])
                   │
                   ▼
    ┌──────────────────────────────┐
    │    2. PROYEKSI LINIER /      │ ── (Menyelaraskan dimensi fitur visual agar sesuai
    │    ALIGNMENT LAYER (MAPPING) │     dengan dimensi input Text Decoder / LLM)
    └──────────────────────────────┘
                   │
                   ▼
     [ Visual Tokens Aligned ]
                   │
                   ▼
    ┌──────────────────────────────┐
    │      3. TEXT DECODER         │ <── [ Prompt / Start Token ] (e.g., "<s>")
    │  (Transformer / LLM / GPT)   │
    └──────────────────────────────┘
                   │
                   ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │ 4. PROSES GENERASI AUTOREGRESIF (Looping Kata demi Kata)                           │
 │                                                                                    │
 │   T_1 : Generasi kata ke-1 ──> "The"                                               │
 │   T_2 : Input ("The") ──> Generasi kata ke-2 ──> "lungs"                           │
 │   T_3 : Input ("The lungs") ──> Generasi kata ke-3 ──> "are"                       │
 │   ...                                                                              │
 │   Loop berlanjut sampai model mengeluarkan End-of-Sequence token ("</s>")          │
 └────────────────────────────────────────────────────────────────────────────────────┘
                   │
                   ▼
       [ Raw Generated Text ] ─── (Berupa sekuens token yang digabungkan/detokenized)
                   │
                   ▼
    ┌──────────────────────────────┐
    │  5. POST-PROCESSING SPLITTER │
    └──────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
 [ Findings Text ]      [ Impression Text ]
(Deskripsi Detail)     (Kesimpulan Klinis)
```

## Eksperimen: Pure Report Generation

| Eksperimen | Vision Encoder | Text Decoder    | Vision Encoder Setup                          | Text Decoder Setup     |
|------------|-----------------|------------------|------------------------------------------------|--------------------------|
| A          | BioViL-T        | GPT-2 Base       | Freeze → Partial Unfreeze (last 1-2 layers)   | DAPT - Full Fine-tuning |
| B          | BioViL-T        | BioGPT (base)    | Freeze → Partial Unfreeze (last 1-2 layers)   | DAPT - PEFT (LoRA)      |
| C          | BioViL-T        | Qwen2.5-1.5B     | Freeze → Partial Unfreeze (last 1-2 layers)   | DAPT - PEFT (QLoRA)     |

Metrics Evaluation:
- Natural Language Generation (NLG) Metrics: [BLEU-N ; N = 1, 2, 3, and 4], ROUGE-L, and METEOR. 
- Clinical Efficacy (CE) utilizing CheXbert: precision, recall, macro F-1, and balanced accuracy.
- Optional Clinical Efficacy (CE): RadGraph F1
- Hallucination & Omission Rate

Resources:
- dataset: https://huggingface.co/datasets/cchitse/mimic-cxr-with-chexbert-labels (splitted already)
- CheXbert labeler for generated output for CE Metrics Evalution: https://github.com/stanfordmlgroup/CheXbert