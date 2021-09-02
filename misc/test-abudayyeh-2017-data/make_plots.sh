#!/bin/bash

# Allow activating conda environments
source ~/anaconda3/etc/profile.d/conda.sh

conda activate data-analysis

# Knockdown luciferase comparison
Rscript plot_knockdown_comparison.R out/knockdown-with-adapt-predictions.tsv out/plots/knockdown.pdf &> out/plots/knockdown.out

