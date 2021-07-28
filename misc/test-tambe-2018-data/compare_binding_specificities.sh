#!/bin/bash


# Allow activating conda environments
source ~/anaconda3/etc/profile.d/conda.sh

# Set a seed for random nucleotides on padding
SEED=1

# Prepare Tambe et al. data for input to ADAPT's model
ADAPT_CALL_IN=$(mktemp)
python prepare_data_for_adapt_model.py binding data/binding-foldchange-specificities.tsv.gz 19 data/binding-foldchange-specificities.for-adapt.tsv.gz --seed $SEED
zcat data/binding-foldchange-specificities.for-adapt.tsv.gz | tail -n +2 | awk '{print $1"\t"$2}' > $ADAPT_CALL_IN

# Run ADAPT's predictive model
ADAPT_PREDICTIONS=$(mktemp)
conda activate adapt
python ../../predictor_call.py ~/adapt/adapt/models/classify/cas13a/v1_0 ~/adapt/adapt/models/regress/cas13a/v1_0 $ADAPT_CALL_IN $ADAPT_PREDICTIONS

# Merge predictions with Tambe et al. data
ADAPT_PREDICTIONS_COL=$(mktemp)
echo "adapt_prediction" > $ADAPT_PREDICTIONS_COL
cat $ADAPT_PREDICTIONS | awk '{print $3}' >> $ADAPT_PREDICTIONS_COL
paste <(zcat data/binding-foldchange-specificities.for-adapt.tsv.gz) $ADAPT_PREDICTIONS_COL | gzip > out/binding-specificities-with-adapt-predictions.tsv.gz

rm $ADAPT_CALL_IN
rm $ADAPT_PREDICTIONS
rm $ADAPT_PREDICTIONS_COL
