# Interpreting DeepStack-Style Vision-Language Models

Code and experimental results for **Interpreting DeepStack-Style Vision-Language Models: Do Input or Deeper-Layer Visual Tokens Drive Answers?**

**[Data](data/README.md) · [Experiment guide](experiments/README.md) · [Analysis scripts](analysis/) · [Figures and tables](paper/README.md)**

Yucheng Du¹, Yi Nian¹, Ruixue Lian², Yue Zhao¹, and Xiyang Hu³<br>
¹ University of Southern California · ² University of Wisconsin–Madison · ³ Arizona State University<br>
The first three authors contributed equally. Yue Zhao and Xiyang Hu are the corresponding authors.

## Reproduce the results

The 12 CSV files in `data/` contain the estimates for Appendices F–H. The `fixed_contrast_*` and `calibrated_*` files cover image degradation and recovery. The `coding_*` files report results for removing or compressing visual features, including task scores and bytes per image.

To redraw Figure 11 from these CSV files on a CPU, run the following from the repository root.

```sh
python3.14 -m venv .venv-figures
. .venv-figures/bin/activate
python -m pip install -r environments/figures.lock.txt
python -m paper.figure_11
```

The plot is saved to `work/rendered/supp_coding_rate.pdf`. The figure and table guide lists the inputs needed for each of the other figures and tables.

To rerun the model experiments, download the model weights and the original datasets for the tasks you want to run, such as GQA or DocVQA. The experiment guide gives the commands and the Python environment for each model family.

<details>
<summary>Cite this work</summary>

```bibtex
@misc{du2026deepstack,
  author = {Du, Yucheng and Nian, Yi and Lian, Ruixue and Zhao, Yue and Hu, Xiyang},
  title  = {Interpreting {DeepStack}-Style Vision-Language Models: Do Input or Deeper-Layer Visual Tokens Drive Answers?},
  year   = {2026}
}
```

</details>

The authors’ contributions are released under the [MIT License](LICENSE). Third-party materials retain their original licenses.
