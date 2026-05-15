# Central Bank Communication and Market Reactions

This repository is the public artifact archive for the thesis:

**Central Bank Communication and Market Reactions: Text-Based Hawk/Dove Classification and Event-Study Evidence from the FOMC and ECB**

It contains the raw thesis data and the final pipeline code used to construct the model outputs for the project.

Authors: Adrian Piotr Pulchny and Kajetan Moravec Cvelbar  
Institution: Copenhagen Business School, MSc in Advanced Economics and Finance  
Artifact set assembled on 2026-05-14

## Overview

The thesis studies whether the hawkish or dovish tone of FOMC and ECB communication helps explain announcement-window financial market reactions and predict near-term policy outcomes.

The empirical pipeline combines:

- raw FOMC and ECB communication text
- USMPD and EA-MPD monetary policy surprise data
- BERT sentence-level prediction files
- dictionary, Random Forest, and BERT-based tone measures
- event-study and expanding-window forecast specifications

## Repository Contents

```text
code/pipeline_v31_final.R   Final thesis model and estimation pipeline
data/raw/                   Raw FOMC, ECB, USMPD, EA-MPD, recovery, and BERT sentence-prediction files
```

The manuscript text, processed panels, saved model objects, generated tables, generated figures, and diagnostic result files are maintained outside this repository. This archive is intentionally limited to the raw input data and the final pipeline code.

## Raw Data Included

The raw folder from the thesis workspace is included under `data/raw/`, including:

- FOMC statement and press-conference econometric text files
- ECB decision and statement or press-conference files
- USMPD and EA-MPD Excel workbooks
- BERT sentence-level prediction files
- Manual text-recovery file

Only the two zero-byte Excel lock files, `~$EA-MPD.xlsx` and `~$USMPD.xlsx`, were excluded.

## License

Code is released under the MIT License. See `LICENSE-CODE`.

Data and repository documentation are released under Creative Commons Attribution 4.0 International. See `LICENSE-DATA`.
