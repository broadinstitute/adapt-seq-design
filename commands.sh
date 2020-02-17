# Commands used to generate data & models.

###########################################################
# Cas13 data
# exp-and-pos subset, regression only on active data points

# Perform hyperparameter search for the best model
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --command hyperparam-search --hyperparam-search-cross-val-num-splits 5 --search-type random --num-random-samples 50 --params-mean-val-loss-out-tsv out/cas13-hyperparam-search.exp-and-pos.regress-on-active.tsv --save-models models/predictor_exp-and-pos_regress-on-active --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.regress-on-active.out

# Select model 8f534a8c from the above, which has the lowest MSE
# Its 1_minus_rho is the 7th lowest (one downside is that it has
#   a high SEM for 1_minus_rho, but most models do)

# Create test results
python predictor.py --load-model models/predictor_exp-and-pos_regress-on-active/model-c15f787d --dataset cas13 --cas13-subset exp-and-pos --cas13-regress-only-on-active --context-nt 10 --write-test-tsv out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-c15f787d.test.tsv.gz --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-c15f787d.test.out

# Plot test results
Rscript plotting_scripts/plot_predictor_test_results.R out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-c15f787d.test.tsv.gz out/cas13-hyperparam-search.exp-and-pos.regress-on-active.model-c15f787d.test.pdf

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
# exp-and-pos subset, classify active vs. inactive

# Perform hyperparameter search for the best model
python -u predictor_hyperparam_search.py --dataset cas13 --cas13-subset exp-and-pos --cas13-classify --context-nt 10 --command hyperparam-search --hyperparam-search-cross-val-num-splits 5 --search-type random --num-random-samples 50 --params-mean-val-loss-out-tsv out/cas13-hyperparam-search.exp-and-pos.classify.tsv --save-models models/predictor_exp-and-pos_classify --seed 1 &> out/cas13-hyperparam-search.exp-and-pos.classify.out
###########################################################
