# Figures and tables

Run these commands from the repository root. Figure and table numbers match the paper.

## Reproduce Figure 11 from the CSV files

This example plots accuracy against bytes per image on a CPU using the three coding CSV files in [data/](../data/).

```sh
export DS_DATA_ROOT="$PWD/work"
export MPLCONFIGDIR="$DS_DATA_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
python3.14 -m venv .venv-figures
. .venv-figures/bin/activate
python -m pip install -r environments/figures.lock.txt
python -m paper.figure_11
```

The output is `$DS_DATA_ROOT/rendered/supp_coding_rate.pdf`. The same CSVs supply the estimates for Table 3 and Tables 19–26. Tables 1, 4 and 5 can be typeset directly from the bundled templates. These two commands build their TeX files on the CPU.

```sh
python -m paper.tables_1_to_10 1 3 4 5
python -m paper.tables_19_to_26
```

Each table is written to `$DS_DATA_ROOT/rendered/table_<n>.tex`, where `<n>` is the table number.

The `fixed_contrast_*.csv` files let you check the numerical results in Tables 11–14. The builders for those tables and the other displays read intermediate analysis files listed below. If you have saved these files, you can use them directly. To produce them from scratch, follow the [experiment guide](../experiments/README.md) for the relevant datasets and checkpoints. Figure 1 also uses the source photograph.

## Rebuild the figures and tables

The figure scripts use Matplotlib, NumPy and Pillow. The original plots use Arial, including its bold and italic faces. Install those fonts to reproduce the original lettering and spacing. Direct the plotting cache to the work directory.

~~~sh
export DS_DATA_ROOT="${DS_DATA_ROOT:-$PWD/work}"
export MPLCONFIGDIR="$DS_DATA_ROOT/rendered/mpl-cache"
mkdir -p "$MPLCONFIGDIR" "$DS_DATA_ROOT/presentation_inputs"
~~~

The renderers read per-model analysis outputs under `$DS_DATA_ROOT/presentation_inputs`, abbreviated to `I` below. Copy, link or write the named files there. Use the model keys `qwen8b`, `qwen32b`, `ibm4b` (Granite) and `deepstack7b`. The nine-checkpoint photo table also uses `qwen2b`, `qwen4b`, `qwen30b_a3b`, `cosmos_nano` and `cosmos_super`.

| Input under I | Producer |
|---|---|
| `photographs/<model>/generation.json` or `score.json` | Photo intervention analysis, `measurement_v12.p1_analysis` with the corresponding `--kind`. Use its `summary.json` from the main panel. |
| `paired_endpoints/<model>/p1-paired_endpoints.jsonl` | The same photo analysis, generation run, file `paired_endpoints.jsonl`. Needed for the four principal models. |
| `paired/<model>/local.json` and `dose.json` | `measurement_v12.p2_analysis` on local and matched-strength panels for Qwen8, Qwen32 and Granite. Also save the Cosmos3-Super dose analysis as `cosmos_super_dose.json` directly under I. |
| `spatial/<model>/summary.json` | `measurement_v10.analysis` on the local-role runs of Qwen8, Qwen32 and Granite. This is the local color experiment in Figure 2. |
| `decomposition/<model>/score.json` and `generation.json` | `measurement_v13.p4_analysis` for Qwen8 and Qwen32. Use the full `score-analysis.json` and `generation-analysis.json`, including per-edit records for Qwen32. |
| `fixed/` | `summary.json`, `paper_numbers.json` and `benchmark_numbers.json` from the fixed-contrast analysis. |
| `calibrated/` | `main_table.csv`, `sensitivities.csv`, `grid.csv`, `capability_calibrated.csv` and `sample_sizes.csv` from the calibrated experiment. Figure 3 needs the cell intervals in these analysis outputs. |
| `posthoc/` | `four_cell_posthoc.json` and `docvqa_format_sensitivity.json` from the paired four-cell diagnostics and the DocVQA format sensitivity analysis. |

The coding plots and tables read the three CSV files in `data/` on a fresh clone. To rebuild the CSVs from per-question outputs, run `PYTHONPATH="$PWD/experiments" python -m analysis.write_coding_csv` from the repository root. It reads the confirmation sweeps, follow-up sweeps, split manifests and development-panel training outputs under `I/coding/`. The exact per-model filenames are listed in `analysis/coding_observations.py` (`MEMBERS` and `load_observations`). The estimator pairs scores within questions, averages within images or documents and uses 4,000 paired bootstrap resamples. It writes to `$DS_DATA_ROOT/supplementary_data`, which the renderers use when that directory exists. Set `DS_SUPPLEMENTARY_DATA` to read from a different directory.

Compute the shared photo and stripe summaries once before Figures 1 and 8 and Tables 6–9. Compute the matched-strength and local stripe summaries before Figures 4–7 and 9.

~~~sh
python -m analysis.prepare_components
python -m analysis.prepare_matched
~~~

| Figure | Analysis and command |
|---|---|
| 1 | Photo edit, visual entries and answer transitions. Run `python -m paper.figure_1` with the prepared `datasets/natural-selectivity-v12` panel and the photo counts. The illustration uses edit `vg-2390681-edit-0` from “River Brent at Brentford” by Maxwell Hamilton (Flickr, CC BY 2.0). It crops and recolors locally acquired inputs. |
| 2 | Local color transfer by layout and destination. Run `python -m paper.figure_2`. |
| 3 | Four quality cells and stripe recovery under blur. Run `python -m paper.figure_3`. |
| 4 | Stripe spatial interactions. Run `python -m paper.figures_4_to_7`. |
| 5 | Photograph generation components and paired entry differences. The same command writes `supp_natural_generation.pdf`. |
| 6 | Photograph score components and paired entry differences. The same command writes `supp_natural_score.pdf`. |
| 7 | Paired entry differences across the five magnitude settings. The same command writes `supp_dose_reference.pdf`. |
| 8 | Photograph answer changes and Qwen32 stripe composition. Run `python -m paper.figure_8`. |
| 9 | Qwen32 and Cosmos3-Super stripe contrasts at matched magnitudes. Run `python -m paper.figure_9`. |
| 10 | Recovery against joint loss. Run `python -m paper.figure_10`. |
| 11 | Accuracy against coded bytes. Run `python -m paper.figure_11`. |

All outputs go under `$DS_DATA_ROOT/rendered`. For Tables 1–10 the command takes one or more table numbers. It writes standalone TeX, for example `tectonic "$DS_DATA_ROOT/rendered/table_4.tex"` typesets Table 4. The remaining table commands produce TeX fragments for inclusion in a paper. Numeric rows retain the original estimate and rounding conventions. Captions and final manuscript typography received additional editorial edits.

| Table | Content | Command |
|---|---|---|
| 1 | Related studies, manually typeset | `python -m paper.tables_1_to_10 1` |
| 2 | Nine-checkpoint photograph selectivity | `python -m paper.tables_1_to_10 2` |
| 3 | GQA attribute accuracy at seven single-entry configurations | `python -m paper.tables_1_to_10 3` |
| 4 | Checkpoint identifiers and revisions, manually typeset | `python -m paper.tables_1_to_10 4` |
| 5 | Intervention specification, manually typeset | `python -m paper.tables_1_to_10 5` |
| 6 | Photo answer transitions and photographs with a target change | `python -m paper.tables_1_to_10 6` |
| 7 | Transitions with room to move toward blue | `python -m paper.tables_1_to_10 7` |
| 8 | Stripe composition residuals within each layout | `python -m paper.tables_1_to_10 8` |
| 9 | Paired differences between direction-aligned residuals | `python -m paper.tables_1_to_10 9` |
| 10 | Qwen entry decomposition at raw and common-low magnitudes | `python -m paper.tables_1_to_10 10` |
| 11 | Four fixed-contrast color accuracy cells | `python -m paper.tables_11_to_14` |
| 12 | Fixed-contrast quality interactions | `python -m paper.tables_11_to_14` |
| 13 | Granite entry-21 marginals | `python -m paper.tables_11_to_14` |
| 14 | Fixed-contrast standard-task cells | `python -m paper.tables_11_to_14` |
| 15 | Shared degradation grid | `python -m paper.tables_15_and_18` |
| 16 | Post hoc single-entry losses | `python -m paper.tables_16_and_17` |
| 17 | Paired recovery contrasts and DocVQA format sensitivity | `python -m paper.tables_16_and_17` |
| 18 | Calibrated color conflict scores | `python -m paper.tables_15_and_18` |
| 19 | Single-entry accuracy and intervals | `python -m paper.tables_19_to_26` |
| 20 | Paired single-entry differences | `python -m paper.tables_19_to_26` |
| 21 | Single-entry protocol predictions | `python -m paper.tables_19_to_26` |
| 22 | DeepStack configuration scores and coded sizes | `python -m paper.tables_19_to_26` |
| 23 | DeepStack paired comparisons | `python -m paper.tables_19_to_26` |
| 24 | Granite held-out configurations and paired comparisons | `python -m paper.tables_19_to_26` |
| 25 | Qwen main-only and scale-normalized coding | `python -m paper.tables_19_to_26` |
| 26 | Role-direction predictions | `python -m paper.tables_19_to_26` |
