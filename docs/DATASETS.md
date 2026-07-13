# Dataset adapters

## Time-series forecasting

### TIME (2026)

TIME is the newest supported benchmark. It contains 50 fresh datasets and 98
task-aligned forecasting problems. Clone the official repository into
`data/TIME`, then use `configs/time_2026_direct.yaml`.

```bash
git clone https://github.com/zqiao11/TIME data/TIME
qrecon --config configs/time_2026_direct.yaml
```

The generic adapter reads sufficiently long numeric columns from CSV files. A
future benchmark-specific adapter can preserve TIME's complete task metadata.

### GIFT-Eval (2024)

GIFT-Eval contains 144,000 time series and 177 million observations across
multiple domains and frequencies. Q-RECON uses Hugging Face streaming and only
loads the requested number of series.

```bash
qrecon --config configs/time_gifteval_quantum.yaml
```

## Images

### Community Forensics Small (CVPR 2025)

The default image experiment uses the redistributable small split of Community
Forensics, a recent real-vs-generated image benchmark. NSFW-flagged samples are
excluded by the loader.

```bash
qrecon --config configs/image_community_forensics.yaml
```

### CLOFAI (2025)

CLOFAI is supported through the generic `image_folder` adapter after following
the official dataset preparation procedure. Point `dataset.root` at a directory
whose subdirectories correspond to class labels.

## Reproducibility and ethics

- Dataset artifacts are not committed to this repository.
- Experiments are intended for public benchmarks and models trained by the
  researcher.
- Do not use the reconstruction pipeline against third-party private systems or
  data without explicit authorization.
- Record dataset revisions and licenses in every published experiment.

