## Pipeline Abstraction
- Task: Radiology Report Generation (RRG)
- Input: CXR Image (single view) 224 $\times$ 224
- Output:
  - Findings Text
  - Impression Text
  - Generation Strategy:
    - Option-A:
      - Method: Unified Autoregressive
      - Definition: Findings and Impression texts are generated as a single sequential text produced by a single decoder, concatenated into a unified sequence separated by plain-text section markers ("findings: ..." / "impression: ..."), not domain-specific special tokens.
    - Option-B:
      - Method: Dual Separated Generator (Dual Forward Pass / Dual Decoder Head)
      - Definition: Findings and Impression texts are generated independently or sequentially via two separate decoder heads/passes, decoupling the visual-to-text generation for each report section.
    - Hypothesis: The unified autoregressive paradigm (Option-A) is architecturally simpler and more effective at maintaining clinical context consistency between Findings and Impression.
- Dataset:
  - For Model and Architecture Development:
    - Official train set of CheXpert Plus
  - For Model Evaluation:
    - Official test set of CheXpert Plus
    - Official test set of MIMIC-CXR (for cross-dataset generalization)
- Model Architecture Components:
  - Image Auxiliary CXR Pathologies Classifier:
    - Pathology: ['No Finding', 'Support Devices', 'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Edema', 'Cardiomegaly', 'Consolidation', 'Pneumonia', 'Enlarged Cardiomediastinum', 'Pneumothorax', 'Lung Lesion', 'Fracture', 'Pleural Other']
    - Each pathology: 1.0 (Positive), 0.0 (Negative), -1.0 (Uncertain), NaN
    - Model: BioViL-T (same backbone as image encoder) + Classification Head (fully-connected + sigmoid, 14 outputs) + fine-tuned with weighted BCE loss with labels from df_train
  - Image Encoder:
    - BioViL-T
    - Two distinct checkpoints are maintained, not one shared instance: (1) a frozen pretrained BioViL-T used in Stage-A (alignment module training/inference), and (2) a separately fine-tuned BioViL-T used in Stage-B (auxiliary classifier). Do not reuse Stage-B's fine-tuned weights in Stage-A — this matches RaDialog's design, where the classifier's backbone is specialized independently of the frozen encoder feeding the alignment module.
  - Text Decoder:
    - SLM: BioGPT (fine-tuned with PEFT LoRA)
  - Alignment/Fusion Module:
    - RaDialog-inspired, lightweight variant — learnable query embeddings (i.e. 32 queries) which interact with visual features map through cross-attention to produce soft visual tokens aligned to BioGPT's language space. NOT BERT-initialized (random init) and uses cross-attention only, not joint self-attention with text — see Lower Level Detail for the full deviation rationale and future-work trigger.
- Full Pipeline Workflow:
  - Training:
    - Input: Dataset (CXR image + report pairs)
    - Stage-A (Alignment module training):
      - CXR image -> Image Encoder -> visual features in form of spatial grid map (embeddings) -> Alignment Module (BERT-based, Q-Former/BLIP-2 style), this type of alignment module has some learnable query embeddings which are learnt during training to interact with visual features through attention layers contained in its alignment module body -> n soft visual tokens -> these tokens are understandable in a language space of text-decoder-only
    - Stage-B (Image auxiliary classifier):
      - CXR image -> BioViL-T (fine-tuned) + classification head -> spatial grid map features -> global average pooling -> one vector which represents the visual features -> fully connected layer -> 14 outputs (one per CheXpert pathology) -> each output is forwarded to sigmoid activation (multilabel task) -> 14 probabilities for each pathology -> predicted pathology structural findings
      - threshold tuning is performed on `dev_internal` set to maximize macro-averaged F1 score.
      - the threshold will be used for prompt injection in stage-MEETING. 
    - Stage-MEETING:
      - Text-decoder-only is fine tuned with combined prompt: [task instruction] + [view either frontal or lateral] + [CheXpert labels predicted from auxiliary classifier] + [soft visual tokens].
      - Output target: findings text and impression text
  - Inference:
    - Input: CXR image
    - Stage-A: Input -> BioViL-T + alignment module -> soft visual tokens
    - Stage-B: Input -> image auxiliary classifier -> predicted pathology findings
    - Stage-MEETING: Prompt construction:  [task instruction] + [view either frontal or lateral] + [CheXpert labels predicted from auxiliary classifier] + [soft visual tokens] -> BioGPT (fine-tuned)
    - Output: findings text + impression text
  - Loss Function Used:
    - Alignment/Fusion Module: Contrastive loss + Image-Text matching loss. Language modeling loss is excluded for exp_001 (see Lower Level Detail for rationale) — a candidate addition for a future full-RaDialog-fidelity ablation, not a toggle available in the current implementation.
    - Image Auxiliary Classifier: Weighted binary cross entropy
    - BioGPT Fine-Tuning: Language modeling loss (autoregressive next token prediction)
    - Note: If resources are limited, the alignment module may need to be simplified into just a contrastive loss (similar to the GitHub repository https://github.com/anandr07/Medical-Report-Generator-from-Chest-X-Ray-Images/tree/main) instead of the three combined losses—this is a trade-off decision between training complexity and the potential for better alignment quality.
  - Evaluation Metrics:
    - stage-A:
      - loss history visualization (train vs. dev, contrastive loss and ITM loss plotted separately, since they operate on different scales and combining them into one curve would obscure whether one loss is stagnating while the other improves).
      - image-text retrieval accuracy on `dev_internal` (proxy metric, standard practice for evaluating contrastive alignment quality independent of any downstream task — same convention used to evaluate CLIP-style embeddings):
        - Recall@1 and Recall@5: for each image's pooled soft-visual-token embedding, rank all `target_report_text` embeddings in the dev batch by cosine similarity; check whether the correct pairing appears in the top-1 / top-5.
        - rationale: Stage-A's contrastive loss has no intuitive "good enough" threshold on its own (unlike Stage-B's macro-F1) — retrieval accuracy gives a concrete, interpretable number to track improvement across epochs before Stage-A is even plugged into stage-MEETING.
      - ITM head accuracy on `dev_internal`: classification accuracy of the ITM head on held-out (image, text) pairs — both true pairs and hard-negative pairs mined the same way as during training — to confirm the ITM head is learning a real matching signal, not degenerating to a trivial always-predict-match/no-match shortcut.
    - stage-B:
      - loss history visualization.
      - classification report.
      - balanced accuracy.
    - stage-MEETING:
      - note: evaluation is performed for each findings section, impression section, and full report section (findings + impression).
      - binary-multilabel ground truth:
        - findings section: `<pathology>_x`.
        - impression section: `<pathology>_y`.
        - full report section: `<pathology>_label`.
      - Natural Language Generation (NLG):
        - BLEU-n (n = 1, 2, 3, and 4)
        - ROUGE-L
        - METEOR
        - BERTScore
      - Clinical Efficacy (CE - CheXbert based):
        - macro & micro
        - recall, precision, F1, and balanced accuracy
      - Advanced Clinical Metrics:
        - SembScore, RadGraph, RaTEscore, GREEN, and CRIMSON
      - Qualitative:
        - Qualitative examples generated report vs ground truth (can be saved into csv)
- References:
  - https://arxiv.org/pdf/2311.18681v2
- Helpful Tools to Use:
  - https://github.com/stanfordmlgroup/CheXbert
  - https://huggingface.co/StanfordAIMI/RRG_scorers/tree/main
  - from health_multimodal.image import get_image_inference for BioViL-T
  - from health_multimodal.image.utils import ImageModelType for BioViL-T

## Lower Level Detail
- stage-A and stage-B are frozen during stage-MEETING in order to avoid catastrophic forgetting and circular dependency. BioGPT via LoRA adapter is the only one trainable during stage-MEETING.
- Weighted BCE for stage-B:
  - weights are calculated based on inverse frequency from positive labels on df_train_internal only.
- loss functions involve on stage-A:
  - Contrastive loss (image-text, in-batch negatives): symmetric cross-entropy over the cosine-similarity matrix between pooled soft visual tokens (mean over the `n` query outputs) and pooled `target_report_text` embeddings (from a frozen, general-purpose text encoder — decoupled from BioGPT, so Stage-A does not depend on the decoder's tokenizer/weights at all).
  - Image-Text Matching (ITM) loss: binary classification head on top of the concatenated (visual, text) pooled embeddings, trained with cross-entropy against match/no-match labels.
    - hard negatives are mined directly from the same similarity matrix already computed for the contrastive loss (highest-similarity *wrong* pair per sample, not a random in-batch pair) — this captures most of ITM's benefit without requiring a full Q-Former architecture (which would be needed for the canonical BLIP-2/RaDialog implementation, where queries and text tokens are jointly processed via shared self-attention).
  - Language Modeling (LM) loss: excluded, not a silent omission.
    - the full RaDialog recipe's LM loss requires the alignment module to also function as a small text generator, with its own causal decoding head over the query representations — a separate component from BioGPT (which already serves as the final generator in stage-MEETING).
    - implementing this properly is out of scope for exp_001; treated as a candidate follow-up (exp_002+) if the contrastive+ITM combination shows a plateau that a full 3-loss RaDialog-style alignment module might resolve.
  - documented deviation from strict RaDialog fidelity: RaDialog's Q-Former uses joint self-attention (queries + text tokens together) interleaved with cross-attention to the image. This project's `AlignmentModule` uses `nn.TransformerDecoder` (cross-attention only, queries never self-attend with text tokens) — lighter-weight and easier to read, at the cost of not benefiting from a BERT-initialized Q-Former's pretrained weights.
- how CheXpert predicted findings labels are formatted for prompt injection:
  - CXR image -> Stage-B classifier (frozen) -> 14 logits -> sigmoid -> 14 probabilities between 0.0 - 1.0.
  - - Thresholding (see "threshold tuning strategy" below for how the value is selected).
  - Contradiction/Ambiguity/Conflict Handling on "No Finding" (see resolution below).
  - Combine into a single string.
  - Inject into the prompt.
- resolve `No Finding` conflict with other pathologies for prompt injection format:
  - obtain 14 probabilities from Stage-B classifier (frozen).
  - define `threshold`.
  - do NOT use `p(No Finding)` for direct comparison with other pathology probabilities — since the 14 outputs are independent sigmoid logits from different neurons, their magnitudes are NOT calibrated across classes. Comparing them directly is statistically unsound and will arbitrarily bias decisions toward classes with wider dynamic ranges.
  - Step-by-step algorithm:
    1. Inspect the 13 other pathologies (excluding `No Finding`).
    2. Collect all pathologies with probability $\geq$ `threshold`.
    3. If one or more pathologies cross the threshold:
       $\rightarrow$ Suppress "No Finding" implicitly (do not include it in the prompt).
       $\rightarrow$ Collect all active other pathologies with probability $\geq$ `threshold`.
       $\rightarrow$ Format them as a comma-separated string.
       $\rightarrow$ prompt output: `"chexpert findings: {list_of_all_active_pathologies}"`.
    4. If no other pathology crosses the threshold:
       $\rightarrow$ prompt output: `"chexpert findings: no significant findings"`.
  - note:
    - `threshold` is treated as a tunable hyperparameter and will be validated on dev_internal.
    - This approach preserves maximum recall for all confidently detected findings, while the strict threshold prevents low-confidence noise from polluting the prompt.
- prompt order:
  - [task instruction] + [view either frontal or lateral] + [CheXpert labels predicted from auxiliary classifier] + [soft visual tokens].
- threshold tuning strategy for prompt injection:
  - performed exclusively on the `dev_internal` set — the official `test` split remains untouched until final evaluation.
  - Step 1 — probability extraction:
    - run inference of Stage-B (frozen) on all `dev_internal` samples once.
    - store the resulting 14-dimensional probability vectors and the corresponding ground-truth labels.
    - this is a single forward pass; no backpropagation or re-training is involved.
  - Step 2 — grid search over thresholds:
    - define a search space for `threshold` (e.g., values from 0.1 to 0.9, step 0.05).
    - for each candidate threshold, apply the prompt-injection logic:
      - if at least one of the 13 non-`No Finding` pathologies has probability ≥ `threshold`:
        → treat them as detected and include them in the prompt.
      - otherwise:
        → treat the study as normal and output `"chexpert findings: no significant findings"`.
    - compute the macro-averaged F1 score over the 13 non-`No Finding` pathologies (excluding `No Finding` itself, since it is a logical derivative rather than an independent prediction target).
  - Step 3 — select the final threshold:
    - pick the threshold that yields the highest macro-averaged F1 score on `dev_internal`.
  - Rationale:
    - macro-averaged F1 is chosen instead of micro-averaged F1 to ensure that rare pathologies (e.g., Fracture, Pleural Other) contribute equally to the selection process, preventing the threshold from being dominated by majority classes such as Cardiomegaly or No Finding.
    - restricting tuning to `dev_internal` prevents over-optimization to the test distribution, preserving the validity of the final evaluation.
    - the selected threshold is then fixed and applied consistently during Stage-MEETING inference.
- LoRA configuration:
  - r = 8
  - alpha = 16
  - dropout = 0.05
  - target = ['q_proj', 'v_proj']

## Future Work
- Full-fidelity Stage-A alignment module (`Blip2QFormerModel`, BERT-initialized, joint self-attention between queries and text, + LM loss) — conditional trigger: pursue only if exp_001's Stage-A retrieval accuracy (Recall@1/@5 on dev_internal) plateaus early or shows a clear ceiling that appears to bottleneck downstream RRG quality in stage-MEETING.