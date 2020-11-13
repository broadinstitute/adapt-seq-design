# Data overview

## Cas13a library and dataset

`CCF-curated/` contains our Cas13a library (`library.guides.tsv` and `library.targets.csv` in that directory).
It also contains the dataset of Cas13a activity measurements.
`CCF_merged_pairs_annotated.curated.tsv` contains the full dataset.
`CCF_merged_pairs_annotated.curated.resampled.tsv.gz` contains resampled replicate values for each guide-target pair, and was the input for training/testing of models predicting Cas13a activity.

## Scripts

`scripts/` contains various scripts for processing the data.
Notably, `resample_ccf_data.py` produces `CCF_merged_pairs_annotated.curated.resampled.tsv.gz`.

## Miscellaneous

`orig/` contains data separated by experiment.
This data was merged in `CCF-curated/`.

`cas9/` contains an initial playground dataset, which was not used in the ADAPT paper.
