# MvLKIF Python Application Demo

This demo reproduces the **application calling patterns** in `Multi_LKIF/` for the Python MvLKIF package.

## MATLAB Applications Mapped
- `Multi_LKIF/multiLK_toymodel1.m`
- `Multi_LKIF/multiLK_toymodel2.m`
- `Multi_LKIF/multiLK_toymodel3.m`
- `Multi_LKIF/multiLK_huabeirealtest.m`
- `Multi_LKIF/multiLK_huananrealtest.m`
- `Multi_LKIF/MultiLK_with_negatives.m`

## Python Demo Script
- `lkif/demos/mvlkif_applications_demo.py`

The script preserves each application's:
- parameter sets (`MAlen`, `NN`, `MA`, `np`)
- variable ordering in `kal_lkif(...)`
- post-processing trims used by MATLAB (e.g., `97:end`, `276:end`)

## Run Demo

```bash
conda run -n NASAOpenscapes python -m lkif.demos.mvlkif_applications_demo
```

Run only selected applications or larger settings:

```bash
conda run -n NASAOpenscapes python -m lkif.demos.mvlkif_applications_demo --apps toy1 huabei --length 1200 --realizations 8
```

## Note on Data

The MATLAB scripts load `.mat` files (for example `toymodel1.mat`, `huabei.mat`, `Amazon.mat`).
These files are not currently in this repository, so the demo runs on synthetic realizations while maintaining the same workflow structure.
