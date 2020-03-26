#!/bin/bash

# Wrapper for training/evaluating models.

# Args:
#   1: 'baseline'
#       2: 'classify' or 'regress'
#           3: number of jobs to run in parallel
#   1: 'cnn'
#       2: 'classify' or 'regress'
#           3: 'large-search'
#               4: seed to use
#               5: GPU to run on (0-based)
#           3: 'nested-cross-val'
#               4: outer split to run (0-based)
#               5: GPU to run on (0-based)
#           3: 'test'
#               4: params id of model to test


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
        models=("lstm" "mlp" "svm" "logit" "l1_logit" "l2_logit" "l1l2_logit" "gbt" "rf")
    elif [[ $2 == "regress" ]]; then
        outdir="out/cas13/baseline/regress/runs"
        method_arg="--cas13-regress-only-on-active"
        models=("lstm" "mlp" "lr" "l1_lr" "l2_lr" "l1l2_lr" "gbt" "rf")
    else
        echo "FATAL: #2 must be 'classify' or 'regress'"
        exit 1
    fi

    mkdir -p $outdir
    num_outer_splits="5"

    # Generate commands to run
    cmds="/tmp/baseline-cmds.${2}.txt"
    echo -n "" > $cmds
    for outer_split in $(seq 0 $((num_outer_splits - 1))); do
        for model in "${models[@]}"; do
            fn_prefix="$outdir/nested-cross-val.${model}.split-${outer_split}"
            if [ ! -f ${fn_prefix}.metrics.tsv.gz ]; then
                echo "python -u predictor_baseline.py $COMMON_ARGS $method_arg --seed $DEFAULT_SEED --test-split-frac 0.3 --models-to-use $model --nested-cross-val --nested-cross-val-outer-num-splits $num_outer_splits --nested-cross-val-run-for $outer_split --nested-cross-val-out-tsv ${fn_prefix}.metrics.tsv --nested-cross-val-feat-coeffs-out-tsv ${fn_prefix}.feature-coeffs.tsv &> ${fn_prefix}.out; gzip -f ${fn_prefix}.metrics.tsv; gzip -f ${fn_prefix}.feature-coeffs.tsv; gzip -f ${fn_prefix}.out" >> $cmds
            fi
        done
    done

    njobs="$3"
    parallel --jobs $njobs --no-notice --progress < $cmds

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

    if [[ $3 == "large-search" ]]; then
        # Perform a large hyperparameter search; reserve test set
        # Run on different seeds (so it can be in parallel), but results
        # must be manually concatenated
        seed="$4"
        outdirwithseed="$outdir/large-search/seed-${seed}"
        mkdir -p $outdirwithseed

        # Set the GPU to use
        gpu="$5"
        export CUDA_VISIBLE_DEVICES="$gpu"

        python -u predictor_hyperparam_search.py $COMMON_ARGS $method_arg --seed $seed --test-split-frac 0.3 --command hyperparam-search --hyperparam-search-cross-val-num-splits 5 --search-type random --num-random-samples 50 --params-mean-val-loss-out-tsv $outdirwithseed/search.tsv --save-models $modeloutdir &> $outdirwithseed/search.out
        gzip -f $outdirwithseed/search.tsv
        gzip -f $outdirwithseed/search.out
    elif [[ $3 == "nested-cross-val" ]]; then
        # Perform a large nested cross-validation
        # Run on different outer splits (so it can be in parallel), but
        # results must be manually concatenated
        outer_split="$4"
        outdirwithsplit="$outdir/nested-cross-val/split-${outer_split}"
        mkdir -p $outdirwithsplit

        # Set the GPU to use
        gpu="$5"
        export CUDA_VISIBLE_DEVICES="$gpu"

        python -u predictor_hyperparam_search.py $COMMON_ARGS $method_arg --seed $DEFAULT_SEED --command nested-cross-val --hyperparam-search-cross-val-num-splits 5 --nested-cross-val-outer-num-splits 5 --search-type random --num-random-samples 50 --params-mean-val-loss-out-tsv $outdirwithsplit/nested-cross-val.models.tsv --nested-cross-val-out-tsv $outdirwithsplit/nested-cross-val.folds.tsv --nested-cross-val-run-for $outer_split &> $outdirwithsplit/nested-cross-val.out
        gzip -f $outdirwithsplit/nested-cross-val.models.tsv
        gzip -f $outdirwithsplit/nested-cross-val.folds.tsv
        gzip -f $outdirwithsplit/nested-cross-val.out
    elif [[ $3 == "test" ]]; then
        # Run model on the test set and save results
        model_params_id="$4"
        outdirformodel="$outdir/test/model-${model_params_id}"
        mkdir -p $outdirformodel

        python -u predictor.py $COMMON_ARGS $method_arg --seed $DEFAULT_SEED --test-split-frac 0.3 --load-model models/cas13/${2}/model-${model_params_id} --write-test-tsv $outdirformodel/test.tsv.gz &> $outdirformodel/test.out
        gzip -f $outdirformodel/test.out
    else
        echo "FATAL: #3 must be 'large-search' or 'nested-cross-val' or 'test'"
        exit 1
    fi

    unset CUDA_VISIBLE_DEVICES

else
    echo "FATAL: Unknown argument '$1'"
    exit 1
fi
