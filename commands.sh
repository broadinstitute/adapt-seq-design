# Commands used to generate data & models.

###########################################################
# Cas13 data
# exp-and-pos subset, regression only on active data points

# Perform hyperparameter search for the best model
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --command hyperparam-search --hyperparam-search-cross-val-num-splits 5 --search-type random --num-random-samples 1000 --params-mean-val-loss-out-tsv out/cas13-hyperparam-search.exp-and-pos.regress-on-active.tsv --save-models models/predictor_exp-and-pos_regress-on-active --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.regress-on-active.out

# Select model 8f534a8c from the above, which has the lowest MSE
# Its 1_minus_rho is the 7th lowest (one downside is that it has
#   a high SEM for 1_minus_rho, but most models do)

# Create test results
python predictor.py --load-model models/predictor_exp-and-pos_regress-on-active/model-8f534a8c --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --write-test-tsv out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-8f534a8c.test.tsv.gz --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-8f534a8c.test.out

# Plot test results
Rscript plotting_scripts/plot_predictor_test_results.R out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-8f534a8c.test.tsv.gz out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-8f534a8c.test.pdf

# Perform nested cross-validation on the predictor
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --command nested-cross-val --hyperparam-search-cross-val-num-splits 5 --nested-cross-val-outer-num-splits 5 --search-type random --num-random-samples 50 --max-sem 0.05 --test-split-frac 0 --params-mean-val-loss-out-tsv out/cas13-nested-cross-val.exp-and-pos.regress-on-active.models.tsv --nested-cross-val-out-tsv out/cas13-nested-cross-val.exp-and-pos.regress-on-active.folds.tsv --seed 1 &> out/cas13-nested-cross-val.exp-and-pos.regress-on-active.out

# Run the baseline predictors to select models and evaluate
# on the test set
python -u predictor_baseline.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --regression-scoring-method mse --test-split-frac 0.3 --seed 1 &> out/cas13-baseline.exp-and-pos.regress-on-active.out

# Perform nested cross-validation on the baseline models
python -u predictor_baseline.py --seed 1 --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --nested-cross-val --nested-cross-val-outer-num-splits 5 --nested-cross-val-out-tsv out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.folds.tsv --test-split-frac 0 &> out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.out

# Produce plot of nested cross-validation results on
# predictor and baseline mdoels
Rscript plotting_scripts/plot_nested_crossval_results.R out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.folds.tsv.gz out/cas13-nested-cross-val.exp-and-pos.regress-on-active.folds.tsv.gz out/cas13-nested-cross-val.exp-and-pos.regress-on-active.pdf
###########################################################

###########################################################
# Cas13 data
# exp-and-pos subset, regression only on active data points
# Normalize each crRNA across its targets, so that we predict
# the effects due to mismatches

# Perform hyperparameter search for the best model
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --command hyperparam-search --hyperparam-search-cross-val-num-splits 5 --search-type random --num-random-samples 200 --cas13-normalize-crrna-activity --params-mean-val-loss-out-tsv out/cas13-hyperparam-search.exp-and-pos.regress-on-active.normalized-crrnas.tsv --save-models models/predictor_exp-and-pos_regress-on-active_normalized-crrnas --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.regress-on-active.normalized-crrnas.out

# Select model f08075e6 from the above

# Create test results
python predictor.py --load-model models/predictor_exp-and-pos_regress-on-active_normalized-crrnas/model-f08075e6 --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --cas13-normalize-crrna-activity --write-test-tsv out/cas13-hyperparam-search.exp-and-pos.regress-on-active.normalized-crrnas.model-f08075e6.test.tsv.gz --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.regress-on-active.normalized-crrnas.model-f08075e6.test.out

# Plot test results
Rscript plotting_scripts/plot_predictor_test_results.R out/cas13-hyperparam-search.exp-and-pos.regress-on-active.normalized-crrnas.model-f08075e6.test.tsv.gz out/cas13-hyperparam-search.exp-and-pos.regress-on-active.normalized-crrnas.model-f08075e6.test.pdf

# Perform nested cross-validation on the predictor
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --command nested-cross-val --hyperparam-search-cross-val-num-splits 5 --nested-cross-val-outer-num-splits 5 --search-type random --num-random-samples 100 --max-sem 0.05 --test-split-frac 0 --cas13-normalize-crrna-activity --params-mean-val-loss-out-tsv out/cas13-nested-cross-val.exp-and-pos.regress-on-active.normalized-crrnas.models.tsv --nested-cross-val-out-tsv out/cas13-nested-cross-val.exp-and-pos.regress-on-active.normalized-crrnas.folds.tsv --seed 1 &> out/cas13-nested-cross-val.exp-and-pos.regress-on-active.normalized-crrnas.out

# Perform nested cross-validation on the baseline models
python -u predictor_baseline.py --seed 1 --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --nested-cross-val --nested-cross-val-outer-num-splits 5 --nested-cross-val-out-tsv out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.normalized-crrnas.folds.tsv --test-split-frac 0 --regression-scoring-method mse --cas13-normalize-crrna-activity &> out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.normalized-crrnas.out

# Produce plot of nested cross-validation results on
# predictor and baseline models
Rscript plotting_scripts/plot_nested_crossval_results.R out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.normalized-crrnas.folds.tsv.gz out/cas13-nested-cross-val.exp-and-pos.regress-on-active.normalized-crrnas.folds.tsv.gz out/cas13-nested-cross-val.exp-and-pos.regress-on-active.normalized-crrnas.pdf
###########################################################

###########################################################
# Cas13 data
# exp-and-pos subset, regression only on active data points
# Predict the difference of each guide-target pair's activity from the
# wildtype activity for the guide (i.e., matching target)

# Perform nested cross-validation on the predictor
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --command nested-cross-val --hyperparam-search-cross-val-num-splits 5 --nested-cross-val-outer-num-splits 5 --search-type random --num-random-samples 50 --max-sem 0.05 --test-split-frac 0 --cas13-use-difference-from-wildtype-activity --params-mean-val-loss-out-tsv out/cas13-nested-cross-val.exp-and-pos.regress-on-active.diff-from-wildtype.models.tsv --nested-cross-val-out-tsv out/cas13-nested-cross-val.exp-and-pos.regress-on-active.diff-from-wildtype.folds.tsv --seed 1 &> out/cas13-nested-cross-val.exp-and-pos.regress-on-active.diff-from-wildtype.out

# Perform nested cross-validation on the baseline models
python -u predictor_baseline.py --seed 1 --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --nested-cross-val --nested-cross-val-outer-num-splits 5 --nested-cross-val-out-tsv out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.diff-from-wildtype.folds.tsv --test-split-frac 0 --regression-scoring-method mse --cas13-use-difference-from-wildtype-activity &> out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.diff-from-wildtype.out

# Produce plot of nested cross-validation results on
# predictor and baseline models
Rscript plotting_scripts/plot_nested_crossval_results.R out/cas13-baseline.nested-cross-val.exp-and-pos.regress-on-active.diff-from-wildtype.folds.tsv.gz out/cas13-nested-cross-val.exp-and-pos.regress-on-active.diff-from-wildtype.folds.tsv.gz out/cas13-nested-cross-val.exp-and-pos.regress-on-active.diff-from-wildtype.pdf
###########################################################
