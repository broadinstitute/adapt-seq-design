#!/bin/bash

# Allow activating conda environments
source ~/anaconda3/etc/profile.d/conda.sh

conda activate data-analysis

# Cleavage rate comparison
Rscript plot_cleavage_rate_comparison.R out/cleavage-rates-with-adapt-predictions.tsv out/plots/cleavage-rates.pdf out/plots/cleavage-rates.decrease.pdf &> out/plots/cleavage-rates.out

# Binding specificity comparison
Rscript plot_binding_specificity_comparison.R out/binding-specificities-with-adapt-predictions.tsv.gz out/plots/binding.scatter.pdf out/plots/binding.quartiles.pdf out/plots/binding.quartiles-rev.pdf out/plots/binding.scatter.decrease.pdf out/plots/binding.scatter.1mm.pdf &> out/plots/binding.out

