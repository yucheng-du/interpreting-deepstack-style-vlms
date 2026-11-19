# Experimental results

These 12 CSV files contain estimates from three experiments. The `fixed_contrast_*` and `calibrated_*` files give the unrounded estimates used in Appendices F and G. The `coding_*` files give results for removing or compressing visual features in Appendix H, computed from the per-question observations. The tables below list the contents of each file and explain how to read its columns.

## Common conventions

- **Models.** `qwen8b` Qwen3-VL-8B-Instruct, `qwen32b` Qwen3-VL-32B-Instruct, `ibm4b` Granite Vision 4.1-4B, `deepstack7b` DeepStack-L-Vicuna-7B.
- **Identifiers.** `model` names the model as above. `task` is `color`, `stripe_orientation`, `relative_position` or `text_reading` for synthetic scenes, `natural_scene_color` for photographs, and `gqa_attribute`, `gqa_spatial` or `docvqa_document_qa` for benchmarks. Count columns (`n`, `n_r`, `n_i_quality`, `n_templates`, `n_clusters`) give the number of complete templates, photographs, images or documents behind an estimate.
- **Cells.** `cc`, `cd`, `dc`, `dd` give main quality first and later quality second (C clean, D degraded).
- **Recovery.** `g_d = dc - dd` is later recovery with degraded main; `i_rec = g_d - (cc - cd)` compares it with the gain under clean main; `joint_loss = cc - dd`.
- **Conflict.** `target_effect` and `control_effect` are directed effects toward source b on the edited-object and unchanged-object queries; `selectivity_r` is their paired difference R; `i_quality = R(degraded main) - R(normal main)`, with the later features fixed.
- **Kinds.** `generation` values are accuracies or directed answer-frequency changes; `score` values are candidate log odds summed over the candidate tokens.
- **Readouts.** Direct answers and numeric labels in identity or cyclic option order (`direct`, `identity`, `cyclic` in fixed-contrast files; `direct/none`, `label/identity`, `label/cyclic` in calibrated files).
- **Intervals.** `*_ci_low` and `*_ci_high` are descriptive, unadjusted 95% intervals: Student-t over templates or photographs for synthetic and photo panels, normal and clustered by image or document for benchmarks. An empty interval beside a mean means a zero standard error (flagged explicitly as `*_se_zero` in fixed-contrast files); an empty mean means the estimate is not defined for that row.

## Fixed-contrast experiment (Appendix F)

Main is degraded to contrast 0.12 for synthetic tasks (32 templates) and photographs (64 images); benchmarks use Gaussian blur of radius 2.0.

| File | Rows | Content |
|---|---:|---|
| [fixed_contrast_recovery.csv](fixed_contrast_recovery.csv) | 120 | Four-cell accuracy (invalid answers incorrect), `g_d` and `i_rec` for every model, task, kind and readout. `material` is `synthetic` or `natural`. |
| [fixed_contrast_conflict.csv](fixed_contrast_conflict.csv) | 120 | Target, control, R and `i_quality` with degraded main; `n_r` counts complete units. |
| [fixed_contrast_granite_marginal.csv](fixed_contrast_granite_marginal.csv) | 24 | Granite entry-21 marginal given fixed entries 15 and 18: target and control effects, `marginal` (target minus control), and `native_accuracy` from the fixed-contrast panel. |
| [fixed_contrast_format_pairs.csv](fixed_contrast_format_pairs.csv) | 24 | Within-template differences between readouts for synthetic-color recovery (`later_recovery`) and its interaction (`recovery_quality_interaction`): `contrast` names the readout pair (for example `cyclic-direct`), `difference` is the paired difference, and `invalid_policy` is `accuracy_invalid_as_incorrect` (invalid answers count as incorrect). |
| [fixed_contrast_benchmarks.csv](fixed_contrast_benchmarks.csv) | 108 | GQA attribute, GQA spatial and DocVQA four cells, `g_d` and `i_rec` for every metric and invalid-answer policy; `primary_metric` marks strict GQA matching and organizer ANLS. |

## Calibrated experiment (Appendix G)

Three degradation levels per task, chosen on separate calibration units (`level` names the family and setting, `level_value` the setting); evaluation uses 32 new templates per synthetic task, 192 GQA images per question group and 192 DocVQA documents. `in_shared_grid` is false for the four model-specific levels run outside the shared grid. Level codes: `desat-c<x>` keeps a fraction x of chroma with luminance unchanged (color); `blurrel-period<x>` and `blurrel-dotsep<x>` apply Gaussian blur with sigma x relative to the stripe period (stripes) or the dot separation (relative position); `lowres-font<x>` reduces resolution to an equivalent font size of x (text); `lowres-long<x>` reduces resolution to a long side of x pixels (GQA images and DocVQA pages).

| File | Rows | Content |
|---|---:|---|
| [calibrated_synthetic_recovery.csv](calibrated_synthetic_recovery.csv) | 300 | Four cells, `joint_loss`, `g_d` and `i_rec` for every model, task, readout, kind and level; `n_templates` counts templates. |
| [calibrated_synthetic_conflict.csv](calibrated_synthetic_conflict.csv) | 396 | `main_state` is `reference` (normal main) or `label` (degraded main at `level`); target, control and R effects, and `i_quality` with its template count. |
| [calibrated_benchmarks.csv](calibrated_benchmarks.csv) | 342 | GQA and DocVQA cells and contrasts for every metric (`gqa_exact`, `gqa_case_punct_normalized`, `gqa_lower_strip_sensitivity`, `docvqa_anls_organizer_framework`, `docvqa_anls_strict_paper_cutoff`, `docvqa_exact_lower_strip`) and policy (`invalid_as_zero`, `valid_cells_only`, `score_returned_strings`); `n_clusters` counts images or documents. |
| [calibrated_capability_levels.csv](calibrated_capability_levels.csv) | 25 | For each model and task, the calibration level whose retained ability `retained_ability_r` is nearest 0.5. |

## Removing and coding entry features (Appendix H)

Accuracy when an entry's features are replaced, and accuracy against NNCodec bytes when the features are quantized and coded, for the models and tasks of Appendix H. Unlike the quality files above, these estimates are recomputed from the per-question observations with the protocol's paired bootstrap: 95% percentile intervals over images (GQA) or documents (DocVQA), 4,000 resamples, unit weights drawn once per image set and shared by every estimate on that set. Per-question values are averaged within the image or document first.

- **Values.** Scores, differences, margins and their intervals are in percent or percentage points (accuracy x 100 for GQA, ANLS x 100 for DocVQA), unrounded. `bytes_*` are NNCodec bitstream bytes per image or document (the mean over its questions of the coded size of the image they use); zeroed or mean-replaced entries count 0 bytes; for unchanged entries (`C`) the bytes are the raw BF16 tensor size.
- **Configurations.** `config` uses `main|later`: `C` features unchanged, `Z` all zeros, `M` every position set to the entry's mean feature vector over 48 calibration images (DocVQA: the 56 page images of 48 calibration documents), `q<QP>` coded at that QP (shared QP when both sides carry the same QP), `k0` the scale-normalized setting (later entry i at QP q + d_i), `k+n` / `k-n` n QP units coarser / finer than `k0`. `config_paper` gives the paper notation (for example `-2|n+2`). Larger QP is coarser; four QP steps double the quantization step.
- **Identifiers.** `image_set` is `confirm1000` (1,000 confirmation images, 1,838 questions), `confirm474` (all 474 remaining DocVQA validation documents, 907 questions), `granite468` (Granite's first 468 confirmation images, 866 questions), `granite531` (the 531 remaining confirmation images, unseen by Granite, 970 questions), `granite999` (both), or `development192` (the 192 GQA attribute evaluation images of the calibrated experiment of Appendix G, 349 questions, training pilot only). `task` is `gqa_attribute` or `docvqa` in these files. `n_units` and `n_questions` count the images or documents and the questions behind an estimate; `n_questions` is empty for the matched-bytes rows, which use the same questions as the configurations they compare. `status` is `L` (fixed in the protocol before any confirmation image was scored), `F` (follow-up comparison fixed after the first confirmation results and before the follow-up runs), `P` (post hoc) or `E` (exploratory, other images); `E, P` marks a post hoc comparison in the exploratory training pilot.
- **Not included.** The exploratory runs on an Apple-silicon GPU that Appendix H quotes (DeepStack on GQA spatial questions and DocVQA, and one Granite configuration) are not in these files.

| File | Rows | Content |
|---|---:|---|
| [coding_single_entry.csv](coding_single_entry.csv) | 98 | Single-entry scores (`estimate` = `score`), paired differences (`difference`, `config_a` minus `config_b`) and the protocol's single-entry predictions (`endpoint_margin`: margin f x gain minus loss, with `tolerated_fraction` f, `loss_over_gain` and `outcome`). |
| [coding_configurations.csv](coding_configurations.csv) | 235 | Every configuration of every image set with `family` (`shared`, `scale_normalized`, `later_coarser`, `later_finer`, `main_only`, `single_entry`, or `training_<arm>`), bytes (`bytes_total`, `bytes_main`, `bytes_later`) and score; training rows add `arm` (`base` no fine-tuning; `sft` plain fine-tuning on the full input; `generic`, `role` and `anti` fine-tuning on coded features with the later entries offset from n by a random sign, the role sign or the opposite sign, called anti-role in Appendix H) and `checkpoint` (`long` 500 steps, `short` 150 steps, `none`). |
| [coding_comparisons.csv](coding_comparisons.csv) | 119 | Paired comparisons `config_a` minus `config_b` with both byte sizes, `bytes_change_pct` (bytes of a relative to b, in percent), both scores and the difference; `group` is `deepstack_pairs`, `granite_pair`, `qwen_main_only`, `role_direction`, `matched_bytes_shared_curve` (post hoc: score minus the shared-QP curve interpolated linearly in log bytes at the same bytes, curve recomputed in each resample; `config_b` names the bracketing shared configurations) or `training` (`arm_a` minus `arm_b` at the same configuration). |
