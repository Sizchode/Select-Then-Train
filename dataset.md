# Datasets

Most datasets used by this codebase are available through HuggingFace `datasets`
and will be downloaded automatically the first time the corresponding loader is
used.

## HuggingFace Datasets

### LLM Tasks

| Task | HuggingFace dataset | Loader |
| --- | --- | --- |
| `boolq` | `super_glue`, config `boolq` | `stt.dataset.genloader_2.BoolQ` |
| `arc-e` | `ai2_arc`, config `ARC-Easy` | `stt.dataset.genloader_2.ARC` |
| `arc-c` | `ai2_arc`, config `ARC-Challenge` | `stt.dataset.genloader_2.ARC` |
| `obqa` | `openbookqa`, config `main` | `stt.dataset.genloader_2.OBQA` |

### Text Classification Tasks

These are loaded with `datasets.load_dataset(...)` in
`stt/dataset/classification/get_loader.py`.

| Task | HuggingFace dataset |
| --- | --- |
| `mrpc` | `glue`, config `mrpc` |
| `qnli` | `glue`, config `qnli` |
| `sst2` | `glue`, config `sst2` |

### Image Classification Tasks

| Task | Source |
| --- | --- |
| `clevr_count` | `clip-benchmark/wds_vtab-clevr_count_all` |

## Local Datasets

### CLUTRR

`clutrr` is loaded from the local repository path configured in `META.py`:

```text
datasets/clutrr/
  train.json
  val.json
  test.json
```

### GTSRB

`gtsrb` is not downloaded automatically by the current loader. Download it from:

https://drive.google.com/file/d/1MYxDcaJnMUsPgfz66khDHrWf6uhy49Qq/view?usp=drive_link

Unzip it under the repository root so the extracted directory is:

```text
gtrsb/
  GTSRB/
    Training/
    Final_Test/
      Images/
  GT-final_test.csv
```

If you use the default image classification loader path in
`stt/dataset/classification/get_loader.py`, place or symlink the same contents at:

```text
datasets/gtsrb/
  GTSRB/
    Training/
    Final_Test/
      Images/
  GT-final_test.csv
```

### EuroSAT

`eurosat` is not downloaded automatically by the current loader. Download it from:

https://drive.google.com/file/d/12mYljRTUPWHwHarcTu8aztDoc5EuZGjO/view

Unzip it under `datasets/` so the extracted directory is:

```text
datasets/eurosat/
  EuroSAT_RGB/
    AnnualCrop/
    Forest/
    HerbaceousVegetation/
    Highway/
    Industrial/
    Pasture/
    PermanentCrop/
    Residential/
    River/
    SeaLake/
```
