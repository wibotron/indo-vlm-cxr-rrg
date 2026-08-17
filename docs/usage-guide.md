# RUNNING GUIDE: PANDUAN MENJALANKAN EKSPERIMEN

---

## 1. Prasyarat: Preprocessing / Prerequisites: Preprocessing (Wajib / Mandatory)

**INDONESIA:**  
Sebelum menjalankan eksperimen apa pun, WAJIB menjalankan notebook preprocessing terlebih dahulu:
* **File:** `D:\VLM-Research_Task-C\notebooks\eda_and_preprocessing_chexpert-plus.ipynb`
* **Fungsi:** Menyiapkan data `train`/`dev`/`test` dalam format `.parquet` serta menghasilkan kolom label yang dibutuhkan seluruh pipeline (Stage-B, Stage-A, dan Stage-MEETING).

**ENGLISH:**  
Before running any experiment, you MUST execute the preprocessing notebook first:
* **File:** `D:\VLM-Research_Task-C\notebooks\eda_and_preprocessing_chexpert-plus.ipynb`
* **Function:** Prepares the `train`/`dev`/`test` `.parquet` files and generates the label columns required for the entire pipeline (Stage-B, Stage-A, and Stage-MEETING).

---

## 2. Dependensi Eksternal / External Dependencies

**INDONESIA:**  
Pastikan struktur folder `external/chexbert/` sudah disiapkan sebelum menjalankan evaluasi Stage-MEETING:
* **Repository:** Clone repository CheXbert dari `https://github.com/stanfordmlgroup/CheXbert`
* **Checkpoint:** Unduh `chexbert.pth` dari `https://huggingface.co/StanfordAIMI/RRG_scorers/tree/main`
* **Lokasi File:** Letakkan checkpoint pada `external/chexbert/pretrained_weights/chexbert.pth`

**ENGLISH:**  
Ensure the `external/chexbert/` directory structure is set up before running Stage-MEETING evaluation:
* **Repository:** Clone the CheXbert repository from `https://github.com/stanfordmlgroup/CheXbert`
* **Checkpoint:** Download `chexbert.pth` from `https://huggingface.co/StanfordAIMI/RRG_scorers/tree/main`
* **File Location:** Place the checkpoint at `external/chexbert/pretrained_weights/chexbert.pth`

---

## 3. Alur Eksekusi / Execution Pipeline

**INDONESIA:**  
Tahapan running untuk setiap eksperimen (`exp_001` dan `exp_002`):

| Urutan | Tahap | Deskripsi |
| :---: | :--- | :--- |
| **1** | **Stage-B** | Melatih auxiliary classifier (BioViL-T + classification head) dan melakukan threshold tuning per kelas. |
| **2** | **Stage-A** | Melatih alignment module (Lightweight atau Q-Former) untuk menghasilkan soft visual tokens. |
| **3** | **Stage-MEETING** | Fine-tune BioGPT dengan LoRA menggunakan prompt gabungan soft tokens dan label CheXpert. |

*Catatan Dependency:* Stage-B dan Stage-A dapat dijalankan secara independen (paralel), tetapi Stage-MEETING membutuhkan checkpoint dari kedua stage tersebut.

**ENGLISH:**  
Execution steps for each experiment (`exp_001` and `exp_002`):

| Step | Stage | Description |
| :---: | :--- | :--- |
| **1** | **Stage-B** | Train auxiliary classifier (BioViL-T + classification head) and perform per-class threshold tuning. |
| **2** | **Stage-A** | Train alignment module (Lightweight or Q-Former) to generate soft visual tokens. |
| **3** | **Stage-MEETING** | Fine-tune BioGPT with LoRA using a combined prompt of soft tokens and CheXpert labels. |

*Dependency Note:* Stage-B and Stage-A can run independently (in parallel), but Stage-MEETING requires checkpoints from both prior stages.

---

## 4. Perbandingan Eksperimen / Experiment Comparison: `exp_001` vs `exp_002`

| Aspek / Aspect | `exp_001` (Default) | `exp_002` (Conditional) |
| :--- | :--- | :--- |
| **Alignment Module** | Lightweight (`nn.TransformerDecoder`, cross-attention only, random init). Terinspirasi / Inspired by RaDialog. | Q-Former (`Blip2QFormerModel`, BERT-initialized, joint self-attention). Meniru / Fully emulating RaDialog. |
| **Loss Functions** | Contrastive + ITM only. | Contrastive + ITM (LM loss opsional via konfigurasi / optional via config). |
| **Text Encoder** | BiomedBERT terpisah / separate (frozen). | Internal Q-Former text branch (shared weights, `bert-base-uncased`). |
| **Target Kinerja / Target** | Baseline cepat dan ringan / Fast, lightweight baseline. | Alignment lebih kaya, potensi performa lebih tinggi / Richer alignment, higher performance potential. |

**INDONESIA:**  
**Keputusan Eksekusi:** `exp_002` hanya dijalankan jika `exp_001` terbukti menjadi bottleneck (misalnya Recall@1 rendah atau cepat mengalami plateau).

**ENGLISH:**  
**Execution Decision:** `exp_002` is executed only if `exp_001` proves to be a bottleneck (e.g., low Recall@1 or early performance plateau).

---

## 5. Aturan Perubahan Kode / Code Change Rules

**INDONESIA:**  
1. **Pelaporan:** Setiap perubahan pada kode atau hiperparameter (sekecil apa pun) WAJIB dilaporkan.
2. **Prosedur Error / Debugging:**
   * Cari referensi solusi terlebih dahulu (misalnya via LLM).
   * Ajukan izin dan kabarkan rencana perubahan sebelum menerapkan solusi pada codebase.
   * Terapkan kode setelah disetujui.
3. **Dokumentasi:** Catat seluruh perubahan dan alasan keputusannya pada `docs/design-decisions-and-contributions.md` atau catatan eksperimen terkait.

**ENGLISH:**  
1. **Reporting:** Any modifications to the code or hyperparameters (no matter how small) MUST be reported.
2. **Error / Debugging Procedure:**
   * Attempt to find a solution first (e.g., via LLMs).
   * Request approval and explain the proposed fix before applying it to the codebase.
   * Apply changes only after approval.
3. **Documentation:** Document every change and rationale in `docs/design-decisions-and-contributions.md` or relevant experiment logs.

---

## 6. Ringkasan Eksekusi / Execution Summary

**INDONESIA:**  
1. **Langkah 1:** Jalankan Preprocessing (`eda_and_preprocessing_chexpert-plus.ipynb`).
2. **Langkah 2:** Jalankan **Stage-B** untuk eksperimen yang dipilih.
3. **Langkah 3:** Jalankan **Stage-A** untuk eksperimen yang dipilih.
4. **Langkah 4:** Jalankan **Stage-MEETING** menggunakan checkpoint Stage-B dan Stage-A.
5. **Langkah 5:** Evaluasi dan bandingkan hasil (`exp_001` vs `exp_002` jika dipicu).

**ENGLISH:**  
1. **Step 1:** Run Preprocessing (`eda_and_preprocessing_chexpert-plus.ipynb`).
2. **Step 2:** Run **Stage-B** for the selected experiment.
3. **Step 3:** Run **Stage-A** for the selected experiment.
4. **Step 4:** Run **Stage-MEETING** using checkpoints from Stage-B and Stage-A.
5. **Step 5:** Evaluate and compare results (`exp_001` vs `exp_002` if triggered).

---

*Dokumen ini disusun untuk menjaga konsistensi dan reproduktibilitas eksperimen.*  
*This document is maintained to ensure experimental consistency and reproducibility.*