# Experiments

Run the commands from the repository root. The [experiment runners](cloud_prep/), [shared intervention code](deepstack/) and [analysis scripts](scripts/) use separate environments for each model family.

```sh
export DS_DATA_ROOT="$PWD/work"
export HF_HOME="$DS_DATA_ROOT/huggingface"
export PYTHONPATH="$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
```

## Environments and data

Use a separate Python environment for each model family. The original DeepStack implementation uses Transformers 4.39.3. The Qwen and Granite quality experiments use Transformers 5.8.0 and different PyTorch builds. Their package lists under `environments/quality/` identify the Python versions and PyTorch wheel sources. The separate `bench.txt` records the tools for acquiring benchmark data. The coding environment locks in `environments/coding/` specify the Python version, platform and wheel source. The NNCodec environment runs as a separate process.

The top-level `requirements.lock.txt` records the earlier Python 3.12 Qwen environment used by the shared library and its tests. The shared adapter library also includes Gemma 4 support, which was unused in the reported experiments. The numbered measurement packages contain the checkpoint revisions and analysis definitions for the individual experiments.

Obtain the GQA v1.2 questions, v1.1 scene graphs and photographs from the GQA download page. Use the balanced validation questions for the benchmark analyses. For Single-Document DocVQA, obtain the validation questions and document images through the organizers' challenge portal. The instructions below describe how to reconstruct the selected units from these inputs. [GQA downloads](https://cs.stanford.edu/people/dorarad/gqa/download.html) · [Single-Document DocVQA](https://www.docvqa.org/datasets/docvqa)

For a model-free check of the shared library with its small synthetic fixtures, use the Python 3.12 environment described by the top-level lock and run `python -m pytest -q`. Run the commands for the numbered experiments in their specified environments.

## Photographs, source conflicts and entry composition

Run the following examples with Python 3.10 or later. The input JPEGs and scene graphs come from the official GQA 2019 release. The photograph selection file contains image IDs, checksums, the 320 excluded development IDs and reviewed query descriptions. The original selection did not use model responses.

~~~sh
graph="$DS_DATA_ROOT/datasets/gqa-official"
for step in scene-graphs extract index; do
  python -m cloud_prep.measurement_v7.natobj_gqa "$step" --root "$graph"
done
python -m cloud_prep.measurement_v12.rebuild_photos
python -m cloud_prep.measurement_v12.p2_data build
~~~

The photo command retrieves only the 64 selected JPEGs by byte range from the official image archive. If the JPEGs are already available, use `--images /path/to/jpegs` with names of the form `<image_id>.jpg`. The original manifest SHA-256 is `208e4627a931fde9669ea0151998d5de94ba6343a80990ee16531c0e43729e08`. The paired synthetic panel uses all 64 templates for local and full-source experiments. The dose and composition arms select the balanced 32-template subset.

The following commands produce all three response formats for one model. Set `model` to `ibm4b`, `qwen8b`, or `qwen32b`. The campaign root and cache arguments must be absolute. Use the model's own runtime environment. A single full run is sufficient. To use disjoint shards, pass all of their score or generation files to the same analysis with the full plan.

~~~sh
model=qwen8b
runs="$DS_DATA_ROOT/runs/conflict"
raw="$runs/$model-photo"
python -m cloud_prep.measurement_v12.p1_runner --model "$model" --campaign-root "$runs" --cache-root "$HF_HOME" --plan-output "$runs/$model-photo-plan.json" --output "$raw"
python -m cloud_prep.measurement_v12.p1_analysis --plan "$runs/$model-photo-plan.json" --kind score --records "$raw/results.jsonl" --output "$runs/$model-photo-score"
python -m cloud_prep.measurement_v12.p1_analysis --plan "$runs/$model-photo-plan.json" --kind generation --records "$raw/generations.jsonl" --output "$runs/$model-photo-generation"
for arm in local full dose; do
  raw="$runs/$model-$arm"
  python -m cloud_prep.measurement_v12.p2_runner --model "$model" --arm "$arm" --campaign-root "$runs" --cache-root "$HF_HOME" --plan-output "$runs/$model-$arm-plan.json" --output "$raw"
  python -m cloud_prep.measurement_v12.p2_analysis --plan "$runs/$model-$arm-plan.json" --scores "$raw/results.jsonl" --generations "$raw/generations.jsonl" --output "$runs/$model-$arm-analysis.json"
done
python -m cloud_prep.measurement_v13.p4_runner --model "$model" --campaign-root "$runs" --cache-root "$HF_HOME" --output "$runs/$model-groups"
python -m cloud_prep.measurement_v13.p4_analysis --runs "$runs/$model-groups" --output "$runs/$model-composition"
~~~

For the extension checkpoints, use the same arguments and analysis. The unified arm names are `p1`, `local`, `full`, and `dose`. Run `cloud_prep.measurement_v13.core_runner --arm ARM` for `qwen4b`, `qwen30b_a3b`, `cosmos_nano`, and `cosmos_super`. Run `cloud_prep.measurement_v13.candidate_runner --arm ARM --model qwen2b` for Qwen2B and `cloud_prep.measurement_v13.deepstack_runner --arm ARM --model deepstack7b` for DeepStack-L. For Cosmos3, install the pinned upstream loader in the Qwen environment. The adapter verifies its three source files before loading. For DeepStack-L use a separate environment with Transformers 4.39.3 and the pinned upstream loader. The checkpoint includes the vision weights. Its runner obtains the CLIP configuration from the pinned public repository.

~~~sh
python -m pip install --no-deps 'transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git@cf5d68c00d97ccd2480a2320ed652b92dec63102#subdirectory=packages/transformers-cosmos3'
# In the DeepStack-L environment
python -m pip install --no-deps 'git+https://github.com/MengLcool/DeepStack-VL.git@3aab69bff6d7ce587b839f518dda40d3379e522d'
~~~

For the earlier local-transfer panel, rebuild its images and use the separate local-role manifest. Run each of the three models independently.

~~~sh
python -m cloud_prep.measurement_v8.mechanism_data build
python -m cloud_prep.measurement_v10.local_data build
runs="$DS_DATA_ROOT/runs/local-role"
mkdir -p "$runs"
manifest="$DS_DATA_ROOT/datasets/local-role-v10/manifest.jsonl"
model=qwen8b
python -m cloud_prep.measurement_v10.runner --model "$model" --arm local --manifest "$manifest" --campaign-root "$runs" --cache-root "$HF_HOME" --dry-run > "$runs/$model-plan.json"
python -m cloud_prep.measurement_v10.runner --model "$model" --arm local --manifest "$manifest" --campaign-root "$runs" --cache-root "$HF_HOME" --output "$runs/$model-raw"
python -m cloud_prep.measurement_v10.analysis --plan "$runs/$model-plan.json" --runs "$runs/$model-raw" --output "$runs/$model-analysis"
~~~

The additional-experiment text uses the BLINK preparation and removal scripts, the spatial and natural-crop routines in measurement_v7 and measurement_v8, the three-object routines in measurement_v9, the correspondence and candidate-score analyses for measurement_v10, and the dense strength grid from the opposite-native version of measurement_v11. That last runner is `cloud_prep.measurement_v12.v11_v2_runner --arm dose`. Its frozen calibration and pairing functions are also shared by the later conflict panels. The BLINK response-format and visual-similarity comparisons use measurement_v3.readout and measurement_v3.slots, with scripts.prepare_measurement_v3, scripts.analyze_measurement_v3 and scripts.summarize_slot_effects_v3 for panel preparation and scoring. The single-entry interaction on 135 Visual Similarity items uses measurement_v4.natural_panel, measurement_v4.natural_qwen, and measurement_v4.extension_analysis. The other early shared functions are in measurement_v6 (scene construction and response parsing). The release contains the scientific runners; rerunning inference requires the checkpoint-specific dependencies and GPU memory given in the paper.

## Quality experiments (Appendices F and G)

Run the following commands from the repository root. The model keys are `qwen8b`, `qwen32b`, `ibm4b` (Granite), and `deepstack7b`. Use separate model environments from `environments/quality`. The DeepStack loader is pinned to commit `3aab69bff6d7ce587b839f518dda40d3379e522d` of MengLcool/DeepStack-VL. The fixed-contrast code comes from the batched runner used for all 21 final jobs. The calibrated runner and scoring modules come from the executed revision with the final shared grid and model-specific configuration files.

Set `FONT` to an accessible copy of the original Arial TrueType file for identical synthetic text images (SHA-256 `525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9`). Set `PHOTO_MANIFEST` to the 64-image natural-selectivity manifest from the earlier photograph experiment. Provide GQA 1.2 balanced validation annotations, original GQA images, Single-Document DocVQA Task 1 validation annotations, and page images as `GQA_VAL`, `GQA_IMAGES`, `DOC_VAL` and `DOC_IMAGES`. Annotations for the fixed-contrast import are unpacked JSON. The exclusions contain IDs only.

```sh
F="$DS_DATA_ROOT/quality/fixed"
G="$DS_DATA_ROOT/quality/calibrated"
FC=experiments/cloud_prep/measurement_v14/configs
GC=experiments/cloud_prep/measurement_v15/configs
mkdir -p "$F/analysis" "$F/scores" "$G/analysis" "$G/scores"
python experiments/scripts/prepare_followup_v14.py --font "$FONT" --output "$F/synthetic"
python experiments/scripts/prepare_followup_v14.py --manifest "$PHOTO_MANIFEST" --output "$F/natural"
python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["image_ids"]))' \
  "$GC/exclusions/gqa-project-photo-ids.json" > "$F/excluded-photo-ids.json"
for group in attribute spatial; do
  python experiments/scripts/import_followup_benchmark_v14.py --benchmark gqa --dataset-version 1.2 \
    --annotations "$GQA_VAL" --images-root "$GQA_IMAGES" --gqa-group "$group" \
    --exclude-image-ids "$F/excluded-photo-ids.json" --max-units 256 \
    --answer-instruction 'Answer with a single lowercase word or short phrase, without punctuation.' \
    --output "$F/gqa-$group"
done
python experiments/scripts/import_followup_benchmark_v14.py --benchmark docvqa --dataset-version 1.0 \
  --annotations "$DOC_VAL" --images-root "$DOC_IMAGES" --max-units 256 --output "$F/docvqa"
```

For each model and material use the runner and analyzer below. Set `MODEL` and `JOB` separately for each run. For synthetic or natural scenes set `MANIFEST` to that directory's `materials.jsonl` and `CONFIG` to `$FC/line1_batched.json`. For the additional Granite synthetic run use `$FC/granite_new_material_marginal_batched.json`. For the three benchmarks use the corresponding `manifest.jsonl`, `$FC/recovery_freeform_batched.json`, and the benchmark scorer instead of the scene analyzer.

```sh
python -m cloud_prep.measurement_v14.runner --model "$MODEL" --manifest "$MANIFEST" \
  --config "$CONFIG" --campaign-root "$F" --cache-root "$HF_HOME" --output "$F/obs/$JOB"
python -m cloud_prep.measurement_v14.analysis "$F/obs/$JOB/observations.jsonl" --output "$F/analysis/$JOB.json"
python experiments/scripts/score_followup_benchmark_v14.py --manifest "$MANIFEST" \
  --observations "$F/obs/$JOB/observations.jsonl" --output "$F/scores/$JOB.json"
```

After nine scene runs and twelve benchmark runs, assemble the three JSON inputs for the fixed-contrast tables on a CPU. This step uses NumPy 2.4.4 and SciPy 1.17.1.

```sh
python3.14 -m venv .venv-fixed
.venv-fixed/bin/python -m pip install -r environments/quality/fixed_contrast_analysis.txt
.venv-fixed/bin/python experiments/scripts/assemble_fixed_contrast.py --analyses "$F"/analysis/*.json --scores "$F"/scores/*.json \
  --output "$DS_DATA_ROOT/presentation_inputs/fixed"
```

For the calibrated synthetic data the fixed seeds give 16 calibration templates and 32 evaluation templates. The extension script needs Pillow 12.2.0. It adds the severity levels used in the executed analyses.

```sh
for split in calibration evaluation; do
  python experiments/scripts/prepare_followup_v15.py --split "$split" --font "$FONT" --output "$G/$split-base"
  python experiments/scripts/extend_synthetic_levels_v15.py --manifest "$G/$split-base/materials.jsonl" \
    --output-dir "$G/$split" --values text_reading=4,3,2.5,2,1.5,1 stripe_orientation=0.34,0.37,0.43,0.46
done
for group in gqa-attribute gqa-spatial docvqa; do
  python experiments/scripts/import_followup_benchmark_v15.py --from-v14-manifest "$F/$group/manifest.jsonl" \
    --severities candidates --max-units 48 --output "$G/calibration-$group"
done
for group in attribute spatial; do
  python experiments/scripts/import_followup_benchmark_v15.py --benchmark gqa --dataset-version 1.2 \
    --annotations "$GQA_VAL" --images-root "$GQA_IMAGES" --split evaluation --gqa-group "$group" \
    --exclude-unit-ids "$GC/exclusions/v14-gqa-lc-union.json" \
    --exclude-image-ids "$GC/exclusions/gqa-project-photo-ids.json" --severities candidates \
    --max-units 192 --output "$G/evaluation-gqa-$group"
  for split in calibration evaluation; do
    python experiments/scripts/extend_synthetic_levels_v15.py --manifest "$G/$split-gqa-$group/manifest.jsonl" \
      --output-dir "$G/$split-gqa-$group-ext" --values "gqa_$group=28,20,14,10,7,5"
  done
done
python experiments/scripts/import_followup_benchmark_v15.py --benchmark docvqa --annotations "$DOC_VAL" \
  --images-root "$DOC_IMAGES" --unit-ids-file "$GC/selection/docvqa-evaluation-fresh.json" \
  --severities candidates --output "$G/evaluation-docvqa"
```

Run the same model loop with `cloud_prep.measurement_v15.runner`, followed by `cloud_prep.measurement_v15.analysis` for synthetic scenes or `experiments/scripts/score_followup_benchmark_v15.py` for the benchmarks. Use `calibration_native.json` and `benchmark_calibration_native.json` for their respective calibration sets. Evaluation uses `line1_severity_batched.json` and `benchmark_recovery_severity_batched.json`. Substitute the supplied Granite synthetic and Qwen8/Qwen32 benchmark capability configs for those cells to include the four additional model-specific levels. For the two supplemental paired analyses, put evaluation observations at `$G/observations/<model>-synthetic-evaluation/observations.jsonl` or `$G/observations/<model>-<benchmark>-evaluation-fresh/observations.jsonl` (`<benchmark>` is `gqa-attribute`, `gqa-spatial`, or `docvqa`). Use names containing `calibration` and `evaluation` for the corresponding analysis and score JSON files.

```sh
python -m cloud_prep.measurement_v15.calibrate --analysis "$G"/analysis/*calibration*.json \
  --benchmark "$G"/scores/*calibration*.json --output "$G/calibration.json"
python -m cloud_prep.measurement_v15.curves --analysis "$G"/analysis/*evaluation*.json \
  --benchmark "$G"/scores/*evaluation*.json --calibration "$G/calibration.json" --output "$G/curves"
python experiments/scripts/final_tables_v15.py --curves "$G/curves" --calibration "$G/calibration.json" \
  --import-metadata "$G"/*/import_metadata.json --benchmark-scores "$G"/scores/*.json \
  --output "$DS_DATA_ROOT/runs/measurement-v15-cloud-20261029/results/analysis-final.run-1/final/tables"
python experiments/scripts/audit_v15/four_cell_posthoc.py
python experiments/scripts/audit_v15/docvqa_format_sensitivity.py
mkdir -p "$DS_DATA_ROOT/presentation_inputs/posthoc"
cp "$DS_DATA_ROOT/analysis/calibrated/four_cell_posthoc.json" \
   "$DS_DATA_ROOT/analysis/calibrated/docvqa_format_sensitivity.json" \
   "$DS_DATA_ROOT/presentation_inputs/posthoc/"
```

## Coding and removal experiments (Appendix H)

Use the separate `environments/coding` locks for each model family, the NNCodec CPU process and analysis. Each model lock lists the Python version and package index required for its PyTorch build. Install the original DeepStack loader from the revision above with `--no-deps`. Fixed model and vision-encoder snapshots live under `HF_HOME/hub`. The optional DeepStack MPS loader takes the upstream checkout through `DEEPSTACK_LOADER_DIR`. Use CUDA BF16 for the main coding results.

The Appendix H code and coding CSVs cover the CUDA confirmation, follow-up and exploratory training results. The earlier Apple-silicon removal estimates quoted for GQA spatial and DocVQA are outside the scope of these CSVs and analysis scripts.

The confirmation sample can be regenerated from the official annotations and original images. Supply GQA training annotations to `pools` using `--gqa-train` when generating the training pool. The reference unit lists preserve the original membership of 1,000 GQA attribute images, 1,000 GQA spatial images, and 474 DocVQA documents. The experimental conditions come from the archived confirmation protocol. The portable lock records checksums of the unit-ID lists so that paths and manifest serialization can differ across machines. It also records the digests of the original protocol and lock.

```sh
H="$DS_DATA_ROOT/quality/coding"
M=cloud_prep.measurement_v19
X=experiments/cloud_prep/measurement_v19/data/exclusions
mkdir -p "$H"
python -m "$M.data.pools" --gqa-val "$GQA_VAL" --docvqa-val "$DOC_VAL" \
  --gqa-exclusions "$X/gqa_image_exclusions.tsv" --docvqa-exclusions "$X/docvqa_document_exclusions.tsv" \
  --out "$H/pools"
python -m "$M.data.materials" --pool "$H/pools/confirm-gqa-attribute.jsonl" --n 1000 \
  --annotations "$GQA_VAL" --images-root "$GQA_IMAGES" --out "$H/attr"
python -m "$M.data.materials" --pool "$H/pools/confirm-docvqa.jsonl" --n 474 \
  --annotations "$DOC_VAL" --images-root "$DOC_IMAGES" --out "$H/docvqa"
```

For each model, compute the later-entry QP offsets once from the 48 GQA attribute calibration images. Compute means on the 48 calibration units of each task (`TASKCAL`). `MANIFEST` is the material split being coded. `TASK` uses `gqa-attribute`, `gqa-spatial` or `docvqa`. Give each model, split and task a distinct `RUN` directory under `DS_DATA_ROOT`. For development-budget selection on the old units, combine the scores of the 48 calibration and 192 evaluation units. The confirmation CLI can select configurations using these calibration observations. For following a published comparison directly, pass its configurations to the same coding and sweep commands.

```sh
python -m "$M.runner" --model "$MODEL" --mode stats --device cuda --manifest "$G/calibration-gqa-attribute/manifest.jsonl" \
  --out "$H/$MODEL-offsets"
python -m "$M.runner" --model "$MODEL" --mode means --device cuda --manifest "$TASKCAL" --out "$RUN/means" --means "$RUN/means.npz"
python -m "$M.confirm.report" configs --model "$MODEL" --task "$TASK" > "$RUN/configs.txt"
python -m "$M.runner" --model "$MODEL" --mode extract --device cuda --manifest "$MANIFEST" --out "$RUN/features"
python -m "$M.rate_audit" code --features "$RUN/features" --out "$RUN/rate" --configs-file "$RUN/configs.txt" --offsets "$H/$MODEL-offsets/offsets.json"
python -m "$M.rate_audit" finalize --out "$RUN/rate" --features "$RUN/features"
python -m "$M.runner" --model "$MODEL" --mode sweep --device cuda --manifest "$MANIFEST" --out "$RUN/scores" \
  --configs-file "$RUN/configs.txt" --means "$RUN/means.npz" --offsets "$H/$MODEL-offsets/offsets.json" --bytes "$RUN/rate/bytes.json"
python -m "$M.confirm.report" select --mode confirm --obs "$CAL_OBS" --out "$H/selection"
python -m "$M.confirm.report" report --mode confirm --obs "$CAL_OBS" "$TEST_OBS" --out "$H/confirmation"
```

For the exploratory training rows use `cloud_prep.measurement_v19.train.rate_table` to prepare the rate table for the ordered training manifest. Then use `python -m cloud_prep.measurement_v19.train.train --model "$MODEL" --arm "$ARM" --manifest "$TRAIN" --rate-table "$RATES" --offsets "$OFFSETS" --out "$TRAIN_OUT"` for Qwen8 and Granite. The arms are `sft`, `generic`, `role` and `anti`. Evaluate the 150- and 500-step adapters with the same sweep and `--lora` pointing to the saved adapter. The unadapted model is the base arm.
