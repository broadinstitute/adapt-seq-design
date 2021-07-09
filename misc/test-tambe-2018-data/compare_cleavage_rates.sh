#!/bin/bash


# Allow activating conda environments
source ~/anaconda3/etc/profile.d/conda.sh

# Set a seed for random nucleotides on padding
SEED=1

# Prepare Tambe et al. data for input to ADAPT's model
ADAPT_CALL_IN=$(mktemp)
python prepare_data_for_adapt_model.py data/cleavage-rates.tsv 11 data/cleavage-rates.for-adapt.tsv --seed $SEED
cat data/cleavage-rates.for-adapt.tsv | tail -n +2 | awk '{print $1"\t"$2}' > $ADAPT_CALL_IN

# Run ADAPT's predictive model
ADAPT_PREDICTIONS=$(mktemp)
conda activate adapt
python ../../predictor_call.py ~/adapt/models/classify/model-51373185 ~/adapt/models/regress/model-f8b6fd5d $ADAPT_CALL_IN $ADAPT_PREDICTIONS

# Merge predictions with Tambe et al. data
ADAPT_PREDICTIONS_COL=$(mktemp)
echo "adapt_prediction" > $ADAPT_PREDICTIONS_COL
cat $ADAPT_PREDICTIONS | awk '{print $3}' >> $ADAPT_PREDICTIONS_COL
paste data/cleavage-rates.for-adapt.tsv $ADAPT_PREDICTIONS_COL > out/cleavage-rates-with-adapt-predictions.tsv

rm $ADAPT_CALL_IN
rm $ADAPT_PREDICTIONS
rm $ADAPT_PREDICTIONS_COL
