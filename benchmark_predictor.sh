#!/bin/bash
# Benchmark predictor on different inputs against baseline.

echo -e "data\tcontext_nt\tmodel\tAUC-ROC\tAUC-PR" > out/predictor-benchmark.tsv

for subset in "all" "guide-mismatch-and-good-pam" "guide-match"; do
    if [[ "$subset" == "all" ]]; then
        subset_args=""
    else
        subset_args="--subset $subset"
    fi

    for context_nt in 0 3 10 20; do
        # Run the CNN predictor
        predictor_epochs=50
        python predictor.py $subset_args --context-nt $context_nt --conv-num-filters 20 --conv-filter-width 2 --max-pool-window-width 2 --fully-connected-dim 20 --dropout-rate 0.25 --epochs $predictor_epochs > /tmp/predictor.out

        # Pull out the validation data metrics from the predictor output
        predictor_auc_roc=$(cat /tmp/predictor.out | grep -A 10 "EPOCH $predictor_epochs" | grep -A 4 'Validate metrics' | grep 'AUC-ROC' | awk '{print $2}')
        predictor_auc_pr=$(cat /tmp/predictor.out | grep -A 10 "EPOCH $predictor_epochs" | grep -A 4 'Validate metrics' | grep 'AUC-PR' | awk '{print $2}')
        rm /tmp/predictor.out

        # Run the CNN predictor with more convolutional layers
        # Note that this is hard-coded for context_nt==20, so results may not
        # be good on other values
        # It also seems to do better (less overfitting) with fewer epochs
        predictor_epochs=30
        python predictor.py $subset_args --deeper-conv-model --context-nt $context_nt --epochs $predictor_epochs > /tmp/predictor.out

        # Pull out the validation data metrics from the predictor output
        deeper_conv_predictor_auc_roc=$(cat /tmp/predictor.out | grep -A 10 "EPOCH $predictor_epochs" | grep -A 4 'Validate metrics' | grep 'AUC-ROC' | awk '{print $2}')
        deeper_conv_predictor_auc_pr=$(cat /tmp/predictor.out | grep -A 10 "EPOCH $predictor_epochs" | grep -A 4 'Validate metrics' | grep 'AUC-PR' | awk '{print $2}')
        rm /tmp/predictor.out

        # Run the baseline predictor
        python predictor_baseline.py $subset_args --context-nt $context_nt > /tmp/baseline.out

        # Pull out the validation data metrics from the baseline output
        baseline_auc_roc=$(cat /tmp/baseline.out | grep -A 1 "Epoch 50/50" | tr '-' '\n' | tail -n 2 | grep 'val_auc_roc' | awk '{print $2}')
        baseline_auc_pr=$(cat /tmp/baseline.out | grep -A 1 "Epoch 50/50" | tr '-' '\n' | tail -n 2 | grep 'val_auc_pr' | awk '{print $2}')
        rm /tmp/baseline.out

        # Write rows to tsv file
        echo -e "$subset\t$context_nt\tcnn\t$predictor_auc_roc\t$predictor_auc_pr" >> out/predictor-benchmark.tsv
        echo -e "$subset\t$context_nt\tcnn-deeper-conv\t$deeper_conv_predictor_auc_roc\t$deeper_conv_predictor_auc_pr" >> out/predictor-benchmark.tsv
        echo -e "$subset\t$context_nt\tlogistic-regression\t$baseline_auc_roc\t$baseline_auc_pr" >> out/predictor-benchmark.tsv
    done
done
