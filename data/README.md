# Data

Not committed to git.

## Carinthia-S (target dataset)
- Zenodo record 16895427, DOI 10.5281/zenodo.16895427, licence CC-BY-4.0. Authors: Kofler, C.; Hasić, V. (2025).
- Files: `data.zip` (139.2 MB, md5 `3a41242574635cf1302fab15accb7730`), `carinthia-s_dataset.html` (description).
- From the record: "4,591 images, each paired with a segmentation mask, unevenly distributed across six defect classes"; "expert-validated binary segmentation masks".
- **Not verified during development**: class names, archive folder layout, mask file naming and image resolution. Zenodo returned HTTP 403 from our network. `scripts/prepare_dataset.py` therefore takes labels from the folder names and prints the structure it finds. Check that output before training.
- Severity labels: the record does not describe any. The severity in this project is a rule-based heuristic.

Setup: `python scripts/download_dataset.py` (or `--zip PATH`), then `python scripts/prepare_dataset.py [--mask-token mask]`.

## Synthetic demo set (used for the current prototype)
`python scripts/prepare_dataset.py --demo` generates 5 × 80 images (256×256): a periodic line/space pattern with Gaussian noise and one drawn defect each (`demo_clean`, `demo_particle`, `demo_scratch`, `demo_bridge`, `demo_void`), each with an exact mask. **These are synthetic and are not real SEM images.** They exist only so the full pipeline can be demonstrated.
