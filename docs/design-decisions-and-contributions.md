## Design Decisions
- Adaptation of RaDialog Workflow:
  - Utilizing RaDialog's multi-stage modular framework (Alignment Module + Auxiliary Pathology Classifier + Instruction-Tuned Text Decoder).
  - Streamlining the alignment/fusion module (e.g., evaluating contrastive-only vs. full Q-Former ITM/LM loss) to balance computational efficiency with vision-language alignment quality.
- Dataset Selection & Cross-Dataset Setup:
  - Training and developing on the official CheXpert Plus train/val splits instead of standard MIMIC-CXR.
  - Evaluating out-of-domain performance on the official MIMIC-CXR test set to rigorously test cross-dataset generalization.
- SLM Exploration for Text Generation:
  - Replacing heavy 7B parameter LLMs (e.g., Vicuna-7B) with Small Language Models (SLMs) such as BioGPT and Qwen-2.5 (0.5B / 1.5B / 3B).
  - Rationale: Investigating resource-efficient generation that lowers hardware requirements for training and inference while maintaining clinical accuracy and semantic coherence.
- Single-View Input Focus:
  - Exclusively focusing on single-view frontal CXR images as a deliberate design boundary, establishing a lightweight baseline without requiring multi-view or longitudinal inputs.

## What to Contribute
- Empirical Benchmark of SLMs in Radiology Report Generation:
  - Providing a systematic analysis comparing lightweight SLMs (BioGPT, Qwen-2.5) against larger 7B LLMs within a multi-stage VLM framework, quantifying trade-offs in GPU memory, training throughput, and generation quality.
- Comparative Analysis of Report Generation Paradigms (Option-A vs. Option-B):
  - Experimentally validating whether a Unified Autoregressive Strategy (Option-A) preserves clinical context consistency between Findings and Impression better than a Dual Separated Generator (Option-B).
- Cross-Dataset Generalization Insights:
  - Evaluating how well an SLM-based VLM trained on CheXpert Plus generalizes to unseen clinical distributions in MIMIC-CXR, measured via clinical efficacy (CheXbert F1) and advanced semantic metrics (RadGraph, GREEN, RaTEscore).
- Architectural Trade-Off Study for Constrained Hardware:
  - Documenting practical design choices and loss function simplifications (e.g., contrastive-only alignment) to serve as a reference for deploying medical VLMs under limited hardware conditions.
- CheXpert Plus Pre-processing Methodology Decision:
  - under consideration.

## Future Works
- Localization and Multilingual Fine-Tuning:
  - Translating and adapting CXR report datasets into Bahasa Indonesia to fine-tune localized SLMs, supporting clinical workflow automation in Indonesian healthcare institutions.
- Radiologist-in-the-Loop Clinical Validation:
  - Conducting qualitative human evaluations with professional radiologists to analyze clinical safety, hallucination rates, and practical utility beyond automated NLP and clinical metrics.