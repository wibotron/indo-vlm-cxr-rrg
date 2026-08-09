# CheXpert Plus Preprocessing Workflow

This document describes the preprocessing methodology implemented in eda_and_preprocessing_chexpert-plus.ipynb, from raw data ingestion to the final train/dev/test parquet files used for model development. It is written to be reusable as-is (or lightly adapted) for a methodology section in a paper/thesis.

## Data Sources

- df_chexpert_plus_240401.csv — official CheXpert Plus metadata (image paths, view labels, patient/study identifiers, demographics, report sections).
- label/findings_fixed.json — CheXpert-style multilabel pathology labels derived from the findings section of each report.
- label/impression_fixed.json — same, derived from the impression section.

## Loading and Merging

The three sources are merged on path_to_image using an inner join. Integrity is verified explicitly rather than assumed:

```python
assert df.shape[0] == df_full.shape[0]
```

This confirms no rows were silently dropped due to unmatched keys between the metadata CSV and either label file. Merging produces two label columns per pathology — a _x suffix (findings-derived) and a _y suffix (impression-derived) — both retained through the rest of the pipeline until the auxiliary classifier label step (Section 8), where they are combined.

## Image Path Synchronization

The official CSV references .jpg paths that do not match the actual on-disk PNG layout used in this project. A derived column, actual_image_path, reconstructs the real file path (.jpg to .png, relocated under the local preprocessed/PNG directory). All downstream steps read images through this column, never the original path_to_image.

## EDA-Informed Data Quality Findings

Several patterns discovered during EDA directly shaped the preprocessing decisions below, rather than being treated as separate, disconnected observations:

- View label anomaly: 17 rows (16 LL, 1 RL — all in the train split) had a frontal_lateral label of "Frontal" that disagreed with the file path string, which consistently indicated "Lateral". The path was treated as ground truth (see Section 5).
- Findings/Impression asymmetry: ~73% of rows have no section_findings content, while section_impression is present in virtually all rows. This is not a parsing defect — it is a documented characteristic of CheXpert Plus itself (Chambon et al., 2024, Limitations section: data collection prioritized detailed impressions over findings). This asymmetry directly motivated the dynamic target-sequence construction in Section 7, rather than a fixed two-section template.
- Non-clinical boilerplate: attestation/signature lines (e.g. "I have personally reviewed the images... and agree with the report..."), administrative call-logs ("results called to..."), and de-identification artifacts (repeated underscores) are present in raw report text and required explicit, targeted removal (Section 6) — generic whitespace cleanup alone does not remove them.
- Template/duplicate reports: short, generic impressions (e.g. "No acute cardiopulmonary process.") repeat verbatim across many studies. This reflects genuine clinical documentation patterns, not a data error, and was not deduplicated — removing these rows would bias the dataset away from realistic report-length and content distributions.

## Data Quality Correction

The 17 view-label anomalies identified in Section 4 are corrected in-place on the full merged dataframe, before any train/test split, using the file-path-derived label as ground truth:

```python
path_view = df_full['path_to_image'].str.lower().apply(
    lambda x: 'Lateral' if 'lateral' in x else ('Frontal' if 'frontal' in x else 'Unknown')
)
mismatch_mask = df_full['frontal_lateral'] != path_view
df_full.loc[mismatch_mask, 'frontal_lateral'] = path_view[mismatch_mask]
```

Applying this correction pre-split ensures both df_train and df_test inherit consistent labels without needing to duplicate the fix per split.

## Radiology-Domain-Aware Text Cleaning

### Guiding principles

Generic NLP preprocessing (stopword removal, stemming/lemmatization) is explicitly not used, because it can silently invert clinical meaning or destroy terminology:

- Whitespace/newline normalization: Applied (Pure formatting noise)
- Boilerplate/attestation removal: Applied (Confirmed present in Section 4; describes reporting process, not image content)
- De-identification/separator artifact normalization: Applied (Confirmed present as repeated-underscore patterns)
- Numbered list marker removal: Toggle (Produces continuous prose; discards explicit ordering — documented trade-off, not a silent default)
- Lowercasing: Toggle (Standard in RRG literature; reduces subword vocabulary sparsity)
- Stopword removal: Never applied (Would strip clinically load-bearing negators like "no", "without")
- Stemming/lemmatization: Never applied (Destroys clinical terminology; redundant with the decoder's own subword tokenizer)

### Implementation notes and fixes applied during development

- Broken-newline handling: an early implementation collapsed mid-sentence newlines by deleting them outright, which fused adjacent words together (e.g. "is\nseen" to "isseen"). This was corrected to replace with a single space instead of an empty string.
- Boilerplate regex coverage: initial attestation-line patterns matched only a narrow phrasing (e.g. requiring "agreed" with a "d", or only "reviewed the images"). Empirical residual checks (searching cleaned text for leftover boilerplate substrings) surfaced additional phrasing variants (present-tense "agree", "reviewed the findings", "addendum ends", synonyms like "concur"/"scans"/"radiographs"). The pattern was broadened iteratively, each time re-validated against the same residual check.
- Greedy-match risk: an administrative call-log pattern ("results called to...") originally used an unbounded greedy character class, risking large spans of legitimate clinical text being deleted if the phrase appeared mid-report. This was fixed with a non-greedy quantifier and an explicit lookahead boundary (sentence-end or newline).
- Degenerate-value filtering: rows where cleaning leaves only a single placeholder word (e.g. "none", "unspecified") are dropped via a whole-string equality check — never a per-word/token removal, preserving the no-stopword-removal principle.

### Validation practice

Every cleaning pattern added or modified during development was validated with two checks before being accepted:
1. Residual check — confirm the targeted noise pattern no longer appears in cleaned text.
2. Content-preservation check — confirm legitimate clinical content adjacent to removed boilerplate is not damaged.

## Two-Stage Row Filtering

Filtering for at least one non-empty section is applied twice:

1. Raw-based (before cleaning): drops rows where both section_findings and section_impression are null/empty in their original form.
2. Cleaned-based (after cleaning): re-applies the same condition on cleaned_findings/cleaned_impression, catching any row that became empty purely as a side effect of cleaning.

Both stages report row counts before/after, so the impact of cleaning on row survival is measurable, not assumed.

## Target Sequence Construction

The autoregressive training target (target_report_text) is built dynamically per row, including a section marker only when that section has content:

```python
def build_target_sequence(row):
    parts = []
    if row['has_findings']:
        parts.append(f"findings: {row['cleaned_findings']}")
    if row['has_impression']:
        parts.append(f"impression: {row['cleaned_impression']}")
    return " ".join(parts)
```

This avoids training the model to hallucinate an empty findings section for the ~73% of rows without one (Section 4). The plain-text marker format matches the convention used by the ReXrank benchmark's official CheXpert Plus evaluation format, and requires no tokenizer vocabulary modification.

## Auxiliary Classifier Label Preparation

Pathology labels for the image-based auxiliary classifier are built by combining the _x (findings-derived) and _y (impression-derived) columns from Section 2, following the CheXpert Plus paper's own convention: uncertain (-1.0) to positive (1.0); not-mentioned (NaN) to negative (0.0).

Priority routing is findings-first (falls back to impression only when findings is unavailable) — reversed from an earlier impression-first design. This is based on CheXpert Plus's own labeler validation (Chambon et al., 2024, Table 4): on a human-annotated, image-based ground truth set (n=154), findings-derived CheXbert labels achieved higher agreement (F1=0.44) than impression-derived labels (F1=0.34). Since the auxiliary classifier predicts pathology from the image, the more image-grounded label source is prioritized. This is treated as directional evidence (small n, no reported confidence interval), not a statistically confirmed effect size.

Revise (09/08/2026):
- Steps on build pathology labels:
  - input: `df_train` and `df_test` with `pathology-label_x` from findings text and `pathology-label_y` from impression text.
  - output: 14 well-prepared multilabel pathology (values = 0.0, 1.0).
  - 13 pathologies excluding `No Finding` are combined.
  - for each pathologies, on `pathology-label_x` and `pathology-label_y`:
    - if -1.0 then convert to 1.0.
    - fallback is decided PER PATHOLOGY, based on whether `pathology-label_x` itself is NaN or not — NOT based on `cleaned_findings_text`/`has_findings`, since these two are produced by different processes at different points in the pipeline (`pathology-label_x` comes from Stanford's official labeler run on the RAW `section_findings`, while `cleaned_findings_text` is the result of our own cleaning pipeline applied afterward). A row can have `cleaned_findings_text` empty while `pathology-label_x` still holds real (non-NaN) values, or vice versa — they are not guaranteed to agree.
    - if both `pathology-label_x` and `pathology-label_y` are NaN then convert to 0.0.
  - `No Finding` = 1.0 $\leftrightarrow$ another 13 pathologies = 0.0.
  - `No Finding` = 0.0 if at least there is one of 13 pathologies = 1.0.
- Priority Algorithm:
  - for each pathology independently: if `pathology-label_x` is not NaN, prioritize it; otherwise fallback to `pathology-label_y`; if both are NaN, fallback to 0.0.
  - this fallback happens per-pathology-column, not per-row — `cleaned_findings_text`/`has_findings` play no role in this decision.

## Sequence Length Budget

Word-level length statistics for target_report_text are computed from the train split only (never dev/test, to avoid leaking evaluation-set characteristics into a training-time design decision). The 95th/99th percentiles are reported as a starting max_length recommendation, to be converted to a token-level budget once the actual decoder tokenizer is loaded (word count and subword token count are not 1:1).

## View Filtering Decision

Both frontal and lateral views are retained as independent samples (no filtering by frontal_lateral). This follows the primary reference architecture (RaDialog), which processes single images of either view type without distinguishing them, and reports state-of-the-art clinical efficacy without view-specific mitigation. This deviates from a competing convention in other RRG literature (CCS, PRIOR/FaNe, BiomedJourney) that filters to frontal-only.

A boolean is_frontal column is retained specifically to support stratified evaluation later — to measure whether model performance differs systematically by view type, rather than assuming it does not.

(If this decision is later reversed in favor of frontal-only filtering, an empirical check on this dataset found 0% of test-split studies would be lost entirely, since every study retains at least one frontal image — so either decision is executable without losing study coverage.)

## Train / Dev-Internal / Test Split

The official CheXpert Plus test split (234 rows) is reserved untouched for final reporting only — it is too small on its own to support reliable model-selection decisions during development (a two-proportion test on has_findings rate between train and test returned p=0.088, an inconclusive result attributable to low statistical power from the small test size, not evidence the distributions are identical).

An internal 85/15 split is carved out of the official train set for development-time model selection, using patient-level, multilabel-stratified sampling:

1. Aggregate labels to patient level — one row per deid_patient_id, with each pathology label set to 1.0 if positive in any of that patient's studies (union/OR logic), to avoid splitting a patient's studies across both sides of the split.
2. Multilabel stratified split at the patient level (MultilabelStratifiedShuffleSplit, test_size=0.15), preserving per-pathology prevalence balance across the resulting patient groups.
3. Map back to row level — all studies belonging to each patient are assigned to that patient's split.
4. Leakage assertion — explicitly verify zero patient-ID overlap between the resulting train_internal and dev_internal sets before proceeding.
5. Distribution balance report — per-pathology positive rate compared across train_internal / dev_internal / official test, with a max-difference column to flag any pathology (typically the rarest ones, e.g. Fracture, Pleural Other) where balance is harder to guarantee due to low absolute positive counts.

## Final Outputs

Three parquet files (plus CSV copies for manual inspection) are produced:

- train_internal.parquet
- dev_internal.parquet
- test_official.parquet (untouched official CheXpert Plus test split)

Each contains: actual_image_path, frontal_lateral, is_frontal, has_findings, has_impression, cleaned_findings, cleaned_impression, target_report_text, target_words, and one <Pathology>_label column per pathology (findings-priority combined label, Section 9).

## Documented Limitations

- Findings/impression asymmetry (Section 4) is inherited from the source dataset, not corrected — reports without a findings section are trained on impression-only targets by design (Section 8), a deliberate choice among multiple valid strategies documented in the RRG literature.
- "Not mentioned" to negative assumption (Section 9) is a simplifying convention shared with CheXpert Plus's own analysis, not a claim that radiologist silence always means confirmed absence.
- Single-view, non-longitudinal input: each sample is one CXR image; no comparison to prior studies is available to the model, even though some ground-truth reports reference prior imaging. Frequency of such references is a candidate future measurement.
- Small official test set (234 rows) limits the statistical power of any single evaluation run; the internal dev split (Section 12) exists specifically to reduce reliance on it during development.

## References

- Chambon, P. et al. (2024). CheXpert Plus. arXiv:2405.19538.
- Pellegrini, C. et al. (2023). RaDialog. arXiv:2311.18681.
- RadEval (2025) — incomplete-section exclusion convention.
- ReXrank Challenge V1.0 — official CheXpert Plus RRG leaderboard, findings+impression concatenation format reference.