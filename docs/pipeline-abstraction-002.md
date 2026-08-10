alignment module rely on `Blip2QFormerModel` for stage-A
- full-fidelity Stage-A alignment module (`Blip2QFormerModel`, BERT-initialized,
    joint self-attention between queries and text, + LM loss) — conditional
    trigger: pursue this only if exp_001's Stage-A retrieval accuracy
    (Recall@1/@5 on dev_internal) plateaus early or shows a clear ceiling
    that appears to bottleneck downstream RRG quality in stage-MEETING.

    - RaDialog Style: BERT-based, Q-Former/BLIP-2 style — learnable query embeddings (i.e. 32 queries) which interact with visual features map through cross-attention that this module has to produce soft visual tokens that will be aligned to language space of BioGPT