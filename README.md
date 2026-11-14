# PSA-CMA-ES

PSA-CMA-ES: CMA-ES with population size adaptation. The population size is adapted based on the estimation accuracy of the natural gradient, so that multimodal and noisy problems can be solved without tuning the population size.

## Requirements

Python 3 with numpy, scipy, pandas, matplotlib (optional: quadprog).

## Usage

```bash
python psacmaes.py
```

Runs an example. The comments in the `__main__` block describe the requirements for the objective function.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{nishida2018psa-cma-es,
  author    = {Nishida, Kouhei and Akimoto, Youhei},
  title     = {PSA-CMA-ES: CMA-ES with Population Size Adaptation},
  booktitle = {Proceedings of Genetic and Evolutionary Computation Conference, GECCO 2018, Kyoto, Japan, July 15-19, 2018},
  pages     = {865--872},
  year      = {2018},
  doi       = {10.1145/3205455.3205467}
}
```

```bibtex
@inproceedings{nishida2016population,
  author    = {Nishida, Kouhei and Akimoto, Youhei},
  title     = {Population Size Adaptation for the CMA-ES Based On the Estimation Accuracy of the Natural Gradient},
  booktitle = {GECCO '16: Proceedings of the Genetic and Evolutionary Computation Conference 2016},
  pages     = {237--244},
  year      = {2016},
  doi       = {10.1145/2908812.2908864}
}
```

## History

This code was first published as a GitHub Gist (https://gist.github.com/youheiakimoto/f342a139a6dfcfd55dbc5e345dbf4f9f). The gist remains available, and its history is included in this repository.

## License

MIT License. See [LICENSE](LICENSE).

---

Black-Box Optimization Lab (BBOLab), University of Tsukuba — https://www.bbo.cs.tsukuba.ac.jp/
