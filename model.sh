#!/bin/bash

# Wrapper for training/evaluating models.

# Args:
#   1: 'baseline'
#       2: 'classify' or 'regress'
#           3: number of jobs to run in parallel
#   1: 'cnn'
#       2: 'classify' or 'regress'
#           3: 'large-search'
#               4: GPU to run on (0-based)
#               5: seed to use
#           3: 'nested-cross-val'
#               4: GPU to run on (0-based)
#               5: outer split to run (0-based)


# Set common arguments
DEFAULT_SEED="1"
CONTEXT_NT="10"
DATASET_ARGS="--dataset cas13 --cas13-subset exp-and-pos"
COMMON_ARGS="$DATASET_ARGS --context-nt $CONTEXT_NT"


if [[ $1 == "baseline" ]]; then
    mkdir -p out/cas13/baseline

    # Perform nested cross-validation; reserve test set
    if [[ $2 == "classify" ]]; then
        outdir="out/cas13/baseline/classify/runs"
        method_arg="--cas13-classify"
        models=("logit" "l1_logit" "l2_logit" "l1l2_logit" "gbt" "rf" "svm" "mlp" "lstm")
    elif [[ $2 == "regress" ]]; then
        outdir="out/cas13/baseline/regress/runs"
        method_arg="--cas13-regress-only-on-active"
        models=("lr" "l1_lr" "l2_lr" "l1l2_lr" "gbt" "rf" "mlp" "lstm")
    else
        echo "FATAL: #2 must be 'classify' or 'regress'"
        exit 1
    fi

    mkdir -p $outdir
    num_outer_splits="5"

    # Generate commands to run
    echo -n "" > /tmp/baseline-cmds.txt
    for outer_split in $(seq 0 $((num_outer_splits - 1))); do
        for model in "${models[@]}"; do
            if [ ! -f $outdir/nested-cross-val.${model}.split-${outer_split}.metrics.tsv ]; then
                echo "python -u predictor_baseline.py $COMMON_ARGS $method_arg --seed $DEFAULT_SEED --test-split-frac 0.3 --models-to-use $model --nested-cross-val --nested-cross-val-outer-num-splits $num_outer_splits --nested-cross-val-run-for $outer_split --nested-cross-val-out-tsv $outdir/nested-cross-val.${model}.split-${outer_split}.metrics.tsv --nested-cross-val-feat-coeffs-out-tsv $outdir/nested-cross-val.${model}.split-${outer_split}.feature-coeffs.tsv &> $outdir/nested-cross-val.${model}.split-${outer_split}.out" >> /tmp/baseline-cmds.txt
            fi
        done
    done

    njobs="$3"
    parallel --jobs $njobs --no-notice --progress < /tmp/baseline-cmds.txt

    # Note that this runs each model and outer split separately; results will still
    # need to be manually combined
elif [[ $1 == "cnn" ]]; then
    mkdir -p out/cas13/cnn

    # Perform nested cross-validation; reserve test set
    if [[ $2 == "classify" ]]; then
        outdir="out/cas13/cnn/classify"
        modeloutdir="models/cas13/classify"
        method_arg="--cas13-classify"
    elif [[ $2 == "regress" ]]; then
        outdir="out/cas13/cnn/regress"
        modeloutdir="models/cas13/regress"
        method_arg="--cas13-regress-only-on-active"
    else
        echo "FATAL: #2 must be 'classify' or 'regress'"
        exit 1
    fi

    mkdir -p $outdir
    mkdir -p $modeloutdir

    # Set the GPU to use
    gpu="$4"
    export CUDA_VISIBLE_DEVICES="$gpu"

    if [[ $3 == "large-search" ]]; then
        # Perform a large hyperparameter search; reserve test set
        # Run on different seeds (so it can be in parallel), but results
        # must be manually concatenated
        seed="$5"
        outdirwithseed="$outdir/large-search/seed-${seed}"
        mkdir -p $outdirwithseed

        python -u predictor_hyperparam_search.py $COMMON_ARGS $method_arg --seed $seed --test-split-frac 0.3 --command hyperparam-search --hyperparam-search-cross-val-num-splits 5 --search-type random --num-random-samples 50 --params-mean-val-loss-out-tsv $outdirwithseed/search.tsv --save-models $modeloutdir &> $outdirwithseed/search.out
        gzip -f $outdirwithseed/search.tsv
        gzip -f $outdirwithseed/search.out
    elif [[ $3 == "nested-cross-val" ]]; then
        # Perform a large nested cross-validation
        # Run on different outer splits (so it can be in parallel), but
        # results must be manually concatenated
        outer_split="$5"
        outdirwithsplit="$outdir/nested-cross-val/split-${outer_split}"
        mkdir -p $outdirwithsplit

        python -u predictor_hyperparam_search.py $COMMON_ARGS $method_arg --seed $DEFAULT_SEED --command nested-cross-val --hyperparam-search-cross-val-num-splits 5 --nested-cross-val-outer-num-splits 5 --search-type random --num-random-samples 25 --params-mean-val-loss-out-tsv $outdirwithsplit/nested-cross-val.models.tsv --nested-cross-val-out-tsv $outdirwithsplit/nested-cross-val.folds.tsv --save-models $modeloutdir --nested-cross-val-run-for $outer_split &> $outdirwithsplit/nested-cross-val.out
        gzip -f $outdirwithsplit/nested-cross-val.models.tsv
        gzip -f $outdirwithsplit/nested-cross-val.folds.tsv
        gzip -f $outdirwithsplit/nested-cross-val.out
    else
        echo "FATAL: #3 must be 'large-search' or 'nested-cross-val'"
        exit 1
    fi

    unset CUDA_VISIBLE_DEVICES

else
    echo "FATAL: Unknown argument '$1'"
    exit 1
fi
