#!/bin/bash

# Wrapper for training/evaluating models.

# Args:
#   1: 'baseline'
#       2: 'classify' or 'regress'
#   1: 'cnn'
#       2: 'classify' or 'regress'
#           3: 'large-search' or 'nested-cross-val'


# Set common arguments
SEED="1"
CONTEXT_NT="10"
DATASET_ARGS="--dataset cas13 --cas13-subset exp-and-pos"
COMMON_ARGS="--seed $SEED $DATASET_ARGS --context-nt $CONTEXT_NT"


if [[ $1 == "baseline" ]]; then
    mkdir -p out/cas13/baseline

    # Perform nested cross-validation; reserve test set
    if [[ $2 == "classify" ]]; then
        outdir="out/cas13/baseline/classify"
        method_arg="--cas13-classify"
    elif [[ $2 == "regress" ]]; then
        outdir="out/cas13/baseline/regress"
        method_arg="--cas13-regress-only-on-active"
    else
        echo "FATAL: #2 must be 'classify' or 'regress'"
        exit 1
    fi

    mkdir -p $outdir
    python -u predictor_baseline.py $COMMON_ARGS $method_arg --test-split-frac 0.3 --nested-cross-val --nested-cross-val-out-tsv $outdir/nested-cross-val.metrics.tsv --nested-cross-val-feat-coeffs-out-tsv $outdir/nested-cross-val.feature-coeffs.tsv &> $outdir/nested-cross-val.out
    gzip $outdir/nested-cross-val.metrics.tsv
    gzip $outdir/nested-cross-val.feature-coeffs.tsv
    gzip $outdir/nested-cross-val.out
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

    if [[ $3 == "large-search" ]]; then
        # Perform a large hyperparameter search; reserve test set
        python -u predictor_hyperparam_search.py $COMMON_ARGS $method_arg --test-split-frac 0.3 --command hyperparam-search --hyperparam-search-cross-val-num-splits 5 --search-type random --num-random-samples 50 --params-mean-val-loss-out-tsv $outdir/search.tsv --save-models $modeloutdir &> $outdir/search.out
        gzip $outdir/search.tsv
        gzip $outdir/search.out
    elif [[ $3 == "nested-cross-val" ]]; then
        # TODO
        echo "Not implemented"
    else
        echo "FATAL: #3 must be 'large-search' or 'nested-cross-val'"
        exit 1
    fi

else
    echo "FATAL: Unknown argument '$1'"
    exit 1
fi
