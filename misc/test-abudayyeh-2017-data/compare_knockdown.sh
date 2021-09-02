#!/bin/bash


# Allow activating conda environments
source ~/anaconda3/etc/profile.d/conda.sh

# Set a seed for random nucleotides on padding
SEED=10

# Prepare Abudayyeh et al. data for input to ADAPT's model
ADAPT_CALL_IN=$(mktemp)
python prepare_data_for_adapt_model.py data/knockdown.tsv data/knockdown.for-adapt.tsv --seed $SEED
cat data/knockdown.for-adapt.tsv | tail -n +2 | awk '{print $1"\t"$2}' > $ADAPT_CALL_IN

# Run ADAPT's predictive model
ADAPT_PREDICTIONS=$(mktemp)
conda activate adapt
python ../../predictor_call.py ~/adapt/adapt/models/classify/cas13a/v1_0 ~/adapt/adapt/models/regress/cas13a/v1_0 $ADAPT_CALL_IN $ADAPT_PREDICTIONS

# Merge predictions with Abudayyeh et al. data
ADAPT_PREDICTIONS_COL=$(mktemp)
echo "adapt_prediction" > $ADAPT_PREDICTIONS_COL
cat $ADAPT_PREDICTIONS | awk '{print $3}' >> $ADAPT_PREDICTIONS_COL
paste data/knockdown.for-adapt.tsv $ADAPT_PREDICTIONS_COL > out/knockdown-with-adapt-predictions.tsv

rm $ADAPT_CALL_IN
rm $ADAPT_PREDICTIONS
rm $ADAPT_PREDICTIONS_COL