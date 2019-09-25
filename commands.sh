# Commands used to generate data & models.

###########################################################
# Cas13 data
# exp-and-pos subset, regression only on active data points

# Perform hyperparameter search for the best model
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --command hyperparam-search --hyperparam-search-cross-val-num-splits 5 --search-type random --num-random-samples 500 --params-mean-val-loss-out-tsv out/cas13-hyperparam-search.exp-and-pos.regress-on-active.tsv --save-models models/predictor_exp-and-pos_regress-on-active --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.regress-on-active.out

# Select model 524b9795 from the above, which has the lowest
#   MSE with SEM <0.1 for both MSE and 1_minus_rho
# Another reason model 524b9795 looks good: in the ranking of
#   MSE it is ranked second; in the ranking of 1_minus_rho
#   it is ranked first. If we average the rankings of MSE and
#   1_minus_rho for all models, model 524b9795 ranks highest.

# Create test results
python predictor.py --load-model models/predictor_exp-and-pos_regress-on-active/model-524b9795 --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --write-test-tsv out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-524b9795.test.tsv.gz --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-524b9795.test.out

# Plot test results
Rscript plotting_scripts/plot_predictor_test_results.R out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-524b9795.test.tsv.gz out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-524b9795.test.pdf

# Perform nested cross-validation on the predictor
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --command nested-cross-val --hyperparam-search-cross-val-num-splits 5 --nested-cross-val-outer-num-splits 5 --search-type random --num-random-samples 200 --max-sem 0.05 --test-split-frac 0 --params-mean-val-loss-out-tsv out/cas13-nested-cross-val.exp-and-pos.regress-on-active.models.tsv --nested-cross-val-out-tsv out/cas13-nested-cross-val.exp-and-pos.regress-on-active.folds.tsv --seed 1 &> out/cas13-nested-cross-val.exp-and-pos.regress-on-active.out

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
python predictor.py --load-model models/predictor_exp-and-pos_regress-on-active_normalized-crrnas/model-f08075e6 --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --cas13-normalize-crrna-activity --seed
###########################################################
