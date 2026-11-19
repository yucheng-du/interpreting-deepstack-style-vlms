# Analysis

These scripts turn per-question results into the estimates used by the plots and tables. To work with the published estimates directly, use the [CSV files](../data/).

| Command | What it computes |
| --- | --- |
| `python -m analysis.prepare_components` | Answer changes on edited photographs and the effect of combining visual features on stripe images. |
| `python -m analysis.prepare_matched` | Comparisons between the object being edited and a control object under matched conditions. |
| `PYTHONPATH="$PWD/experiments" python -m analysis.write_coding_csv` | Single-entry scores, coding results, and paired comparisons from the per-question observations. |

Run from the repository root. Place the outputs from the [model experiments](../experiments/README.md) under `$DS_DATA_ROOT/presentation_inputs/` (`work/presentation_inputs/` by default). The [figure and table guide](../paper/README.md#rebuild-the-figures-and-tables) lists the expected files and the commands that use them.
