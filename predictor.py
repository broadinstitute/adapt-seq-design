"""CNN for predicting activity of a guide sequence: classification and
regression.
"""

import argparse
import gzip
import os
import pickle

import fnn
import parse_data

import numpy as np
import scipy
import sklearn
import sklearn.metrics
import tensorflow as tf


__author__ = 'Hayden Metsky <hayden@mit.edu>'


def parse_args():
    """Parse arguments.

    Returns:
        argument namespace
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model',
            help=("Path from which to load parameters and model weights "
                  "for model found by hyperparameter search; if set, "
                  "any other arguments provided about the model "
                  "architecture or hyperparameters will be overridden and "
                  "this will skip training and only test the model"))
    parser.add_argument('--dataset',
            choices=['cas13'],
            default='cas13',
            help=("Dataset to use."))
    parser.add_argument('--cas13-subset',
            choices=['exp', 'pos', 'neg', 'exp-and-pos'],
            help=("Use a subset of the Cas13 data. See parse_data module "
                  "for descriptions of the subsets. To use all data, do not "
                  "set."))
    parser.add_argument('--cas13-classify',
            action='store_true',
            help=("If set, only classify Cas13 activity into inactive/active"))
    parser.add_argument('--cas13-regress-on-all',
            action='store_true',
            help=("If set, perform regression for Cas13 data on all data "
                  "(this can be reduced using --cas13-subset)"))
    parser.add_argument('--cas13-regress-only-on-active',
            action='store_true',
            help=("If set, perform regression for Cas13 data only on the "
                  "active class"))
    parser.add_argument('--cas13-normalize-crrna-activity',
            action='store_true',
            help=("If set, normalize the activity of each crRNA (guide) "
                  "across its targets to have mean 0 and stdev 1; this means "
                  "prediction is performed based on target differences (e.g., "
                  "mismatches) rather than inherent sequence of the crRNA"))
    parser.add_argument('--cas13-use-difference-from-wildtype-activity',
            action='store_true',
            help=("If set, use the activity value of a guide g and target t "
                  "pair to be the difference between the measured activity of "
                  "g-t and the mean activity between g and all wildtype "
                  "(matching) targets of g; this means prediction is "
                  "performed based on targeted differences (e.g., mismatches) "
                  "rather than inherent sequence of the crRNA"))
    parser.add_argument('--context-nt',
            type=int,
            default=10,
            help=("nt of target sequence context to include alongside each "
                  "guide"))
    parser.add_argument('--conv-filter-width',
            type=int,
            nargs='+',
            help=("Width of the convolutional filter (nt) (or multiple widths "
                  "to perform parallel convolutions). If not set, do not "
                  "use convolutional layers (or the batch norm or pooling "
                  "that follow it)."))
    parser.add_argument('--conv-num-filters',
            type=int,
            default=20,
            help=("Number of convolutional filters (i.e., output channels) "
                  "in the first layer"))
    parser.add_argument('--pool-window-width',
            type=int,
            default=2,
            help=("Width of the pooling window"))
    parser.add_argument('--fully-connected-dim',
            type=int,
            nargs='+',
            default=[20],
            help=("Dimension of each fully connected layer (i.e., of its "
                  "output space); specify multiple dimensions for multiple "
                  "fully connected layers"))
    parser.add_argument('--pool-strategy',
            choices=['max', 'avg', 'max-and-avg'],
            default='max',
            help=("For pooling, 'max' only does max pooling; 'avg' only does "
                  "average pooling; 'max-and-avg' does both and concatenates."))
    parser.add_argument('--locally-connected-width',
            type=int,
            nargs='+',
            help=("If set, width (kernel size) of the locally connected layer. "
                  "Use multiple widths to have parallel locally connected layers "
                  "that get concatenated. If not set, do not use have a locally "
                  "connected layer."))
    parser.add_argument('--locally-connected-dim',
            type=int,
            default=1,
            help=("Dimension of each locally connected layer (i.e., of its "
                  "output space)"))
    parser.add_argument('--skip-batch-norm',
            action='store_true',
            help=("If set, skip batch normalization layer"))
    parser.add_argument('--add-gc-content',
            action='store_true',
            help=("If set, add GC content of a guide explicitly into the "
                  "first fully connected layer of the predictor"))
    parser.add_argument('--activation-fn',
            choices=['relu', 'elu'],
            default='relu',
            help=("Activation function to use on hidden layers"))
    parser.add_argument('--dropout-rate',
            type=float,
            default=0.25,
            help=("Rate of dropout in between the 2 fully connected layers"))
    parser.add_argument('--l2-factor',
            type=float,
            default=0,
            help=("L2 regularization factor. This is applied to weights "
                  "(kernal_regularizer). Note that this does not regularize "
                  "bias of activity."))
    parser.add_argument('--sample-weight-scaling-factor',
            type=float,
            default=0,
            help=("Hyperparameter p where sample weight is (1 + p*["
                  "difference in activity from mean wildtype activity]); "
                  "p must be >= 0. Note that p=0 means that all samples are "
                  "weighted the same; higher p means that guide-target pairs "
                  "whose activity deviates from the wildtype from the guide "
                  "are treated as more important. This is only used for "
                  "regression."))
    parser.add_argument('--batch-size',
            type=int,
            default=32,
            help=("Batch size"))
    parser.add_argument('--learning-rate',
            type=float,
            default=0.00001,
            help=("Learning rate for Adam optimizer"))
    parser.add_argument('--max-num-epochs',
            type=int,
            default=1000,
            help=("Maximum number of training epochs (this employs early "
                  "stopping)"))
    parser.add_argument('--test-split-frac',
            type=float,
            default=0.3,
            help=("Fraction of the dataset to use for testing the final "
                  "model"))
    parser.add_argument('--seed',
            type=int,
            default=1,
            help=("Random seed"))
    parser.add_argument('--plot-roc-curve',
            help=("If set, path to PDF at which to save plot of ROC curve"))
    parser.add_argument('--plot-predictions',
            help=("If set, path to PDF at which to save plot of predictions "
                  "vs. true values"))
    parser.add_argument('--write-test-tsv',
            help=("If set, path to .tsv.gz at which to write test results, "
                  "including sequences in the test set and predictions "
                  "(one row per test data point)"))
    parser.add_argument('--determine-classifier-threshold-for-precision',
            type=float,
            default=0.975,
            help=("If set, determine thresholds (across folds) that "
                  "achieve this precision; does not use test data"))
    args = parser.parse_args()

    # Print the arguments provided
    print(args)

    return args


def set_seed(seed):
    """Set tensorflow and numpy seed.

    Args:
        seed: random seed
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


def read_data(args, split_frac=None, make_feats_for_baseline=None):
    """Read input/output data.

    Args:
        args: argument namespace
        split_frac: if set, (train, validate, test) fractions (must sum
            to 1); if None, use 0.3 for the test set, 0.7*(2/3) for the
            train set, and 0.7*(1/3) for the validate set
        use_validation: if True, have the validation set be 1/3 of what would
            be the training set (and the training set be the other 2/3); if
            False, do not have a validation set
        make_feats_for_baseline: if set, make feature vector for baseline
            models; see parse_data module for description of values

    Returns:
        data parser object from parse_data
    """
    if make_feats_for_baseline is not None and args.dataset != 'cas13':
        raise Exception("make_feats_for_baseline only works with Cas13 data")

    # Read data
    if args.dataset == 'cas13':
        parser_class = parse_data.Cas13ActivityParser
        subset = args.cas13_subset
        if args.cas13_classify:
            regression = False
        else:
            regression = True
    if split_frac is None:
        test_frac = 0.3
        train_frac = (1.0 - test_frac) * (2.0/3.0)
        validation_frac = (1.0 - test_frac) * (1.0/3.0)
    else:
        train_frac, validation_frac, test_frac = split_frac
    data_parser = parser_class(
            subset=subset,
            context_nt=args.context_nt,
            split=(train_frac, validation_frac, test_frac),
            shuffle_seed=args.seed,
            stratify_by_pos=True)
    if args.dataset == 'cas13':
        classify_activity = args.cas13_classify
        regress_on_all = args.cas13_regress_on_all
        regress_only_on_active = args.cas13_regress_only_on_active
        data_parser.set_activity_mode(
                classify_activity, regress_on_all, regress_only_on_active)
        if make_feats_for_baseline is not None:
            data_parser.set_make_feats_for_baseline(make_feats_for_baseline)
        if args.cas13_normalize_crrna_activity:
            data_parser.set_normalize_crrna_activity()
        if args.cas13_use_difference_from_wildtype_activity:
            data_parser.set_use_difference_from_wildtype_activity()
    data_parser.read()

    x_train, y_train = data_parser.train_set()
    x_validate, y_validate = data_parser.validate_set()
    x_test, y_test = data_parser.test_set()

    # Print the size of each data set
    data_sizes = 'DATA SIZES - Train: {}, Validate: {}, Test: {}'
    print(data_sizes.format(len(x_train), len(x_validate), len(x_test)))

    if regression:
        # Print the mean outputs and its variance
        print('Mean train output: {}'.format(np.mean(y_train)))
        print('Variance of train output: {}'.format(np.var(y_train)))
    else:
        # Print the fraction of the training data points that are in each class
        classes = set(tuple(y) for y in y_train)
        for c in classes:
            num_c = sum(1 for y in y_train if tuple(y) == c)
            frac_c = float(num_c) / len(y_train)
            frac_c_msg = 'Fraction of train data in class {}: {}'
            print(frac_c_msg.format(c, frac_c))
        if len(x_validate) == 0:
            print('No validation data')
        else:
            for c in classes:
                num_c = sum(1 for y in y_validate if tuple(y) == c)
                frac_c = float(num_c) / len(y_validate)
                frac_c_msg = 'Fraction of validate data in class {}: {}'
                print(frac_c_msg.format(c, frac_c))
        for c in classes:
            num_c = sum(1 for y in y_test if tuple(y) == c)
            frac_c = float(num_c) / len(y_test)
            frac_c_msg = 'Fraction of test data in class {}: {}'
            print(frac_c_msg.format(c, frac_c))
        if args.dataset == 'cas13' and args.cas13_classify:
            print('Note that inactive=1 and active=0')

    return data_parser


def make_dataset_and_batch(x, y, batch_size=32):
    """Make tensorflow dataset and batch.

    Args:
        x: input data
        y: outputs (labels if classification)
        batch_size: batch size

    Returns:
        batched tf.data.Dataset object
    """
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)


def load_model(load_path, params, x_train, y_train, x_validate, y_validate,
        data_parser):
    """Construct model and load weights according to hyperparameter search.

    Args:
        load_path: path containing model weights
        params: dict of parameters
        x_train, y_train, x_validate, y_validate: train and validate data
            (only needed to initialize variables)
        data_parser: data parser object from parse_data

    Returns:..
        fnn.CasCNNWithParallelFilters object
    """
    # First construct the model
    model = construct_model(params, x_train.shape,
            regression=params['regression'],
            y_train=y_train,
            compile_for_keras=True)

    # Note: Previoulsly, this would have to train the model on one
    # data point (reason below); however, this is no longer needed with Keras
    # See https://www.tensorflow.org/beta/guide/keras/saving_and_serializing
    # for details on loading a serialized subclassed model
    # To initialize variables used by the optimizers and any stateful metric
    # variables, we need to train it on some data before calling `load_weights`;
    # note that it appears this is necessary (otherwise, there are no variables
    # in the model, and nothing gets loaded)
    # Only train the models on one data point, and for 1 epoch

    def copy_weights(model):
        # Copy weights, so we can verify that they changed after loading
        return [tf.Variable(w) for w in model.weights]

    def weights_are_eq(weights1, weights2):
        # Determine whether weights1 == weights2
        for w1, w2 in zip(weights1, weights2):
            # 'w1' and 'w2' are each collections of weights (e.g., the kernel
            # for some layer); they are tf.Variable objects (effectively,
            # tensors)
            # Make a tensor containing element-wise boolean comparisons (it
            # is a 1D tensor with True/False)
            elwise_eq = tf.equal(w1, w2)
            # Check if all elements in 'elwise_eq' are True (this will make a
            # Tensor with one element, True or False)
            all_are_eq_tensor = tf.reduce_all(elwise_eq)
            # Convert the tensor 'all_are_eq_tensor' to a boolean
            all_are_eq = all_are_eq_tensor.numpy()
            if not all_are_eq:
                return False
        return True

    def load_weights(model, fn):
        # Load weights
        # There are some concerns about whether weights are actually being
        # loaded (e.g., https://github.com/tensorflow/tensorflow/issues/27937),
        # so check that they have changed after calling `load_weights`
        # Use expect_partial() to silence warnings because this will not
        # load optimizer parameters, which are loaded in construct_model()
        w_before = copy_weights(model)
        w_before2 = copy_weights(model)
        model.load_weights(os.path.join(load_path, fn)).expect_partial()
        w_after = copy_weights(model)
        w_after2 = copy_weights(model)

        assert (weights_are_eq(w_before, w_before2) is True)
        assert (weights_are_eq(w_before, w_after) is False)
        assert (weights_are_eq(w_after, w_after2) is True)

    load_weights(model, 'model.weights')

    return model


def construct_model(params, shape, regression=False, compile_for_keras=True,
        y_train=None, parallelize_over_gpus=False):
    """Construct model.

    This uses the fnn module.

    This can also compile the model for Keras, to use multiple GPUs if
    available.

    Args:
        params: dict of hyperparameters
        shape: shape of input data; only used for printing model summary
        regression: if True, perform regression; if False, classification
        compile_for_keras: if set, compile for keras
        y_train: training data to use for computing class weights; only needed
            if compile_for_keras is True and regression is False
        parallelize_over_gpus: if True, parallelize over all available GPUs

    Returns:
        fnn.CasCNNWithParallelFilters object
    """
    if not compile_for_keras:
        # Just return a model
        return fnn.construct_model(params, shape, regression=regression)

    def make():
        model = fnn.construct_model(params, shape, regression=regression)

        # Define an optimizer, loss, metrics, etc.
        if model.regression:
            # When doing regression, sometimes the output would always be the
            # same value regardless of input; decreasing the learning rate fixed this
            optimizer = tf.keras.optimizers.Adam(lr=model.learning_rate)
            loss = 'mse'

            # Note that using other custom metrics like R^2, Pearson, etc. (as
            # implemented above) seems to raise errors; they are really only
            # needed during testing
            metrics = ['mse', 'mae']

            model.class_weight = None
        else:
            optimizer = tf.keras.optimizers.Adam(lr=model.learning_rate)
            loss = 'binary_crossentropy'    # using class_weight should weight

            # Note that using other custom metrics like auROC (as implemented
            # above) seems to raise errors; they are really only needed during
            # testing
            metrics = ['bce', 'accuracy']

            assert y_train is not None
            y_train_labels = [y_train[i][0] for i in range(len(y_train))]
            class_weight = sklearn.utils.class_weight.compute_class_weight(
                    'balanced', sorted(np.unique(y_train_labels)), y_train_labels)
            model.class_weight = {i: weight for i, weight in enumerate(class_weight)}

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    if parallelize_over_gpus:
        # Use a MirroredStrategy to take advantage of multiple GPUs, if there are
        # multiple
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = make()
    else:
        model = make()

    return model


def pred_from_nt(model, pairs):
    """Predict activity from nucleotide sequence.

    Args:
        model: model object with call() function
        pairs: list of tuples (target with context, guide)

    Returns:
        output of model.call()
    """
    FASTA_CODES = {'A': set(('A')),
                   'T': set(('T')),
                   'C': set(('C')),
                   'G': set(('G')),
                   'K': set(('G', 'T')),
                   'M': set(('A', 'C')),
                   'R': set(('A', 'G')),
                   'Y': set(('C', 'T')),
                   'S': set(('C', 'G')),
                   'W': set(('A', 'T')),
                   'B': set(('C', 'G', 'T')),
                   'V': set(('A', 'C', 'G')),
                   'H': set(('A', 'C', 'T')),
                   'D': set(('A', 'G', 'T')),
                   'N': set(('A', 'T', 'C', 'G'))}
    onehot_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    def onehot(b):
        # One-hot encoding of base b
        real_bases = FASTA_CODES[b]
        v = [0, 0, 0, 0]
        for b_real in real_bases:
            assert b_real in onehot_idx.keys()
            v[onehot_idx[b_real]] = 1.0 / len(real_bases)
        return v

    context_nt = model.context_nt

    l = 2*context_nt + len(pairs[0][1])
    x = np.empty((len(pairs), l, 8), dtype='f')
    for i, (target_with_context, guide) in enumerate(pairs):
        assert len(target_with_context) == 2*context_nt + len(guide)

        # Determine one-hot encodings -- i.e., an input vector
        input_vec = []
        for pos in range(context_nt):
            v_target = onehot(target_with_context[pos])
            v_guide = [0, 0, 0, 0]
            input_vec += [v_target + v_guide]
        for pos in range(len(guide)):
            v_target = onehot(target_with_context[context_nt + pos])
            v_guide = onehot(guide[pos])
            input_vec += [v_target + v_guide]
        for pos in range(context_nt):
            v_target = onehot(target_with_context[context_nt + len(guide) + pos])
            v_guide = [0, 0, 0, 0]
            input_vec += [v_target + v_guide]
        input_vec = np.array(input_vec, dtype='f')
        x[i] = input_vec

    pred_activity = model.call(x, training=False)
    pred_activity = [p[0] for p in pred_activity.numpy()]
    return pred_activity


def load_model_for_cas13_regression_on_active(load_path):
    """Construct model and load parameters and weights.

    This wraps load_model(), without the need to specify x_train, etc. for
    initializing variables.

    Args:
        load_path: path containing model weights

    Returns:..
        fnn.CasCNNWithParallelFilters object
    """
    # Load parameters
    load_path_params = os.path.join(load_path,
            'model.params.pkl')
    with open(load_path_params, 'rb') as f:
        saved_params = pickle.load(f)
    params = {'dataset': 'cas13', 'cas13_subset': 'exp-and-pos',
            'cas13_regress_only_on_active': True}
    for k, v in saved_params.items():
        params[k] = v

    # Load data; we only need 1 data point, which is used to initialize
    # variables
    parser_class = parse_data.Cas13ActivityParser
    subset = 'exp-and-pos'
    regression = True
    test_frac = 0.3
    train_frac = (1.0 - test_frac) * (2.0/3.0)
    validation_frac = (1.0 - test_frac) * (1.0/3.0)
    context_nt = params['context_nt']
    data_parser = parser_class(
            subset=subset,
            context_nt=context_nt,
            split=(train_frac, validation_frac, test_frac),
            shuffle_seed=1,
            stratify_by_pos=True)
    data_parser.set_activity_mode(False, False, True)
    data_parser.read()
    x_train, y_train = data_parser.train_set()
    x_validate, y_validate = data_parser.validate_set()

    # Load the model
    return load_model(load_path, params, x_train, y_train, x_validate,
            y_validate, data_parser)



def determine_classifier_threshold_for_precision(params, x, y,
        num_splits, data_parser, precision_threshold):
    """Find a threshold, via cross-valiation, to achieve a desired precision.

    This focuses on precision because it is an important metric for
    deploying assays.

    It finds the smallest threshold that achieves a desired precision.
    It does this across multiple splits of the training data.

    Args:
        params: model parameters (model should *not* be pre-trained)
        x, y: data to perform cross-validation with
        num_splits: number of folds to compute threshold
        data_parser: object to parse data from parse_data
        precision_threshold: desired threshold on precision

    Returns:
        list of thresholds, one per split
    """
    # Construct a function that the test function will callback
    best_thresholds = []
    def find_threshold(y_true, y_pred):
        # Compute threshold
        pr_curve = sklearn.metrics.precision_recall_curve(y_true, y_pred)
        precision, recall, thresholds = pr_curve

        # Find the smallest threshold (highest i) where precision is
        # >= precision_threshold
        for i, prec in enumerate(precision):
            if prec >= precision_threshold:
                thres = float(thresholds[i])
                break
        best_thresholds.append(thres)

    import predictor_hyperparam_search as phs
    phs.cross_validate(params, x, y, num_splits, False,
            callback=find_threshold, dp=data_parser)

    return best_thresholds


#####################################################################
#####################################################################
# Custom functions for training and testing
#####################################################################
#####################################################################

# For classification, use cross-entropy as the loss function
# This expects sigmoids (values in [0,1]) as the output; it will
# transform back to logits (not bounded betweem 0 and 1) before
# calling tf.nn.sigmoid_cross_entropy_with_logits
bce_per_sample = tf.keras.losses.BinaryCrossentropy()

# For regression, use mean squared error as the loss function
mse_per_sample = tf.keras.losses.MeanSquaredError()

# When outputting loss, take the mean across the samples from each batch
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
validate_loss_metric = tf.keras.metrics.Mean(name='validate_loss')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

# Define metrics for regression
# tf.keras.metrics does not have Pearson correlation or Spearman's correlation,
# so we have to define these; note that it becomes much easier to use these
# outside of the tf.function functions rather than inside of them (like the
# other metrics are used)
# This also defines a metric for R^2 (below, R2Score)
# Note that R2Score does not necessarily equal r^2 here, where r is
# pearson_corr. The value R2Score is computed by definition of R^2 (1 minus
# (residual sum of squares)/(total sum of squares)) from the true vs. predicted
# values. This is why R2Score can be negative: it can do an even worse job with
# prediction than just predicting the mean. r is computed by simple least
# squares regression between y_true and y_pred, and finding the Pearson's r of
# this curve (since this is simple linear regression, r^2 should be
# nonnegative). The value R2 measures the goodness-of-fit of the specific
# linear correlation y_pred = y_true, whereas r measures the correlation from
# the regression (y_pred = m*y_true + b).
def pearson_corr(y_true, y_pred):
    if len(y_true) < 2:
        # Avoid exception
        r = np.nan
    else:
        r, _ = scipy.stats.pearsonr(y_true, y_pred)
    return r
def spearman_corr(y_true, y_pred):
    if len(y_true) < 2:
        # Avoid exception
        rho = np.nan
    else:
        rho, _ = scipy.stats.spearmanr(y_true, y_pred)
    return rho
class CustomMetric:
    def __init__(self, name):
        self.__name__ = name
        self.y_true = []
        self.y_pred = []
    def __call__(self, y_true, y_pred):
        # Save y_true and y_pred (tensors) into a list
        self.y_true += [y_true]
        self.y_pred += [y_pred]
    def to_np_array(self):
        # Concat tensors and convert to numpy arrays
        y_true_np = tf.reshape(tf.concat(self.y_true, 0), [-1]).numpy()
        y_pred_np = tf.reshape(tf.concat(self.y_pred, 0), [-1]).numpy()
        return y_true_np, y_pred_np
    def result(self):
        raise NotImplementedError("result() must be implemented in a subclass")
    def reset_states(self):
        self.y_true = []
        self.y_pred = []
class Correlation(CustomMetric):
    def __init__(self, corrtype, name='correlation'):
        assert corrtype in ('pearson_corr', 'spearman_corr')
        if corrtype == 'pearson_corr':
            self.corr_fn = pearson_corr
        if corrtype == 'spearman_corr':
            self.corr_fn = spearman_corr
        super().__init__(name)
    def result(self):
        y_true_np, y_pred_np = super(Correlation, self).to_np_array()
        return self.corr_fn(y_true_np, y_pred_np)
class R2Score(CustomMetric):
    def __init__(self, name='r2_score'):
        super().__init__(name)
    def result(self):
        y_true_np, y_pred_np = super(R2Score, self).to_np_array()
        return sklearn.metrics.r2_score(y_true_np, y_pred_np)
train_mse_metric = tf.keras.metrics.MeanSquaredError(name='train_mse')
train_mse_weighted_metric = tf.keras.metrics.MeanSquaredError(name='train_mse_weighted')
train_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
train_mape_metric = tf.keras.metrics.MeanAbsolutePercentageError(name='train_mape')
train_r2_score_metric = R2Score(name='train_r2_score')
train_pearson_corr_metric = Correlation('pearson_corr', name='train_pearson_corr')
train_spearman_corr_metric = Correlation('spearman_corr', name='train_spearman_corr')
validate_mse_metric = tf.keras.metrics.MeanSquaredError(name='validate_mse')
validate_mse_weighted_metric = tf.keras.metrics.MeanSquaredError(name='validate_mse_weighted')
validate_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='validate_mae')
validate_mape_metric = tf.keras.metrics.MeanAbsolutePercentageError(name='validate_mape')
validate_r2_score_metric = R2Score(name='validate_r2_score')
validate_pearson_corr_metric = Correlation('pearson_corr', name='validate_pearson_corr')
validate_spearman_corr_metric = Correlation('spearman_corr', name='validate_spearman_corr')
test_mse_metric = tf.keras.metrics.MeanSquaredError(name='test_mse')
test_mse_weighted_metric = tf.keras.metrics.MeanSquaredError(name='test_mse_weighted')
test_mae_metric = tf.keras.metrics.MeanAbsoluteError(name='test_mae')
test_mape_metric = tf.keras.metrics.MeanAbsolutePercentageError(name='test_mape')
test_r2_score_metric = R2Score(name='test_r2_score')
test_pearson_corr_metric = Correlation('pearson_corr', name='test_pearson_corr')
test_spearman_corr_metric = Correlation('spearman_corr', name='test_spearman_corr')

# Define metrics for classification
# Report on the accuracy and AUC for each epoch (each metric is updated
# with data from each batch, and computed using data from all batches)
train_bce_metric = tf.keras.metrics.BinaryCrossentropy(name='train_bce')
train_bce_weighted_metric = tf.keras.metrics.BinaryCrossentropy(name='train_bce_weighted')
train_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
train_auc_roc_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', name='train_auc_roc')
train_auc_pr_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='PR', name='train_auc_pr')
validate_bce_metric = tf.keras.metrics.BinaryCrossentropy(name='validate_bce')
validate_bce_weighted_metric = tf.keras.metrics.BinaryCrossentropy(name='validate_bce_weighted')
validate_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='validate_accuracy')
validate_auc_roc_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', name='validate_auc_roc')
validate_auc_pr_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='PR', name='validate_auc_pr')
test_bce_metric = tf.keras.metrics.BinaryCrossentropy(name='test_bce')
test_bce_weighted_metric = tf.keras.metrics.BinaryCrossentropy(name='test_bce_weighted')
test_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
test_auc_roc_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', name='test_auc_roc')
test_auc_pr_metric = tf.keras.metrics.AUC(
        num_thresholds=200, curve='PR', name='test_auc_pr')


# Store the model and optimizer as global (module-wide) variables
# If passing them directly to train_step(), validate_step(), and test_step(),
# TensorFlow complains about having to do tf.function retracing, which is
# expensive and due to passing Python objects instead of tensors
_model = None
_optimizer = None


# Train the model using GradientTape; this is called on each batch
def train_step(seqs, outputs, sample_weight=None):
    if _model.regression:
        loss_fn = mse_per_sample
    else:
        loss_fn = bce_per_sample
    with tf.GradientTape() as tape:
        # Compute predictions and loss
        # Pass along `training=True` so that this can be given to
        # the batchnorm and dropout layers; an alternative to passing
        # it along would be to use `tf.keras.backend.set_learning_phase(1)`
        # to set the training phase
        predictions = _model(seqs, training=True)
        prediction_loss = loss_fn(outputs, predictions,
                sample_weight=sample_weight)
        # Add the regularization losses
        regularization_loss = tf.add_n(_model.losses)
        loss = prediction_loss + regularization_loss
    # Compute gradients and opitmize parameters
    gradients = tape.gradient(loss, _model.trainable_variables)
    _optimizer.apply_gradients(zip(gradients, _model.trainable_variables))

    # Record metrics
    train_loss_metric(loss)
    if _model.regression:
        train_mse_metric(outputs, predictions)
        train_mse_weighted_metric(outputs, predictions, sample_weight=sample_weight)
        train_mae_metric(outputs, predictions)
        train_mape_metric(outputs, predictions)
    else:
        train_bce_metric(outputs, predictions)
        train_bce_weighted_metric(outputs, predictions, sample_weight=sample_weight)
        train_accuracy_metric(outputs, predictions)
        train_auc_roc_metric(outputs, predictions)
        train_auc_pr_metric(outputs, predictions)

    return outputs, predictions

# Define functions for computing validation and test metrics; these are
# called on each batch
def validate_step(seqs, outputs, sample_weight=None):
    # Compute predictions and loss
    predictions = _model(seqs, training=False)
    if _model.regression:
        loss_fn = mse_per_sample
    else:
        loss_fn = bce_per_sample
    prediction_loss = loss_fn(outputs, predictions,
                sample_weight=sample_weight)
    regularization_loss = tf.add_n(_model.losses)
    loss = prediction_loss + regularization_loss
    # Record metrics
    validate_loss_metric(loss)
    if _model.regression:
        validate_mse_metric(outputs, predictions)
        validate_mse_weighted_metric(outputs, predictions, sample_weight=sample_weight)
        validate_mae_metric(outputs, predictions)
        validate_mape_metric(outputs, predictions)
    else:
        validate_bce_metric(outputs, predictions)
        validate_bce_weighted_metric(outputs, predictions, sample_weight=sample_weight)
        validate_accuracy_metric(outputs, predictions)
        validate_auc_roc_metric(outputs, predictions)
        validate_auc_pr_metric(outputs, predictions)
    return outputs, predictions

def test_step(seqs, outputs, sample_weight=None):
    # Compute predictions
    predictions = _model(seqs, training=False)

    if _model.regression:
        loss_fn = mse_per_sample
    else:
        loss_fn = bce_per_sample
    prediction_loss = loss_fn(outputs, predictions,
                sample_weight=sample_weight)
    regularization_loss = tf.add_n(_model.losses)
    loss = prediction_loss + regularization_loss

    # Record metrics
    test_loss_metric(loss)
    if _model.regression:
        test_mse_metric(outputs, predictions)
        test_mse_weighted_metric(outputs, predictions, sample_weight=sample_weight)
        test_mae_metric(outputs, predictions)
        test_mape_metric(outputs, predictions)
    else:
        test_bce_metric(outputs, predictions)
        test_bce_weighted_metric(outputs, predictions, sample_weight=sample_weight)
        test_accuracy_metric(outputs, predictions)
        test_auc_roc_metric(outputs, predictions)
        test_auc_pr_metric(outputs, predictions)
    return outputs, predictions


# Here we will effectively implement tf.keras.callbacks.EarlyStopping() to
# decide when to stop training; because we are not using model.fit(..) we
# cannot use this callback out-of-the-box
# Set the number of epochs that must pass with no improvement in the
# validation loss, after which we will stop training
STOPPING_PATIENCE = 2

def train_and_validate(model, x_train, y_train, x_validate, y_validate,
        max_num_epochs, data_parser):
    """Train the model and also validate on each epoch.

    Args:
        model: model object
        x_train, y_train: training input and outputs (labels, if
            classification)
        x_validate, y_validate: validation input and outputs (labels, if
            classification)
        max_num_epochs: maximum number of epochs to train for
        data_parser: data parser object from parse_data

    Returns:
        tuple (dict with training metrics at the end, dict with validation
        metrics at the end); keys in the dicts are 'loss' and
        ('bce' or 'mse') and ('auc-roc' or 'r-spearman')
    """
    # Define an optimizer
    if model.regression:
        # When doing regression, sometimes the output would always be the
        # same value regardless of input; decreasing the learning rate fixed this
        optimizer = tf.keras.optimizers.Adam(lr=model.learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(lr=model.learning_rate)

    # Set the global module-level variables _model and _optimizer, needed by
    # train_step() and validate_step()
    global _model
    global _optimizer
    _model = model
    _optimizer = optimizer

    # model may be new, and calling train_step on a new model will yield
    # an error; tf.function was designed such that a new one is needed
    # whenever there is a new model
    # (see
    # https://github.com/tensorflow/tensorflow/issues/27525#issuecomment-481025914)
    tf_train_step = tf.function(train_step)
    tf_validate_step = tf.function(validate_step)

    train_ds = make_dataset_and_batch(x_train, y_train,
            batch_size=model.batch_size)
    validate_ds = make_dataset_and_batch(x_validate, y_validate,
            batch_size=model.batch_size)

    # For classification, determine class weights
    if not model.regression:
        y_train_labels = [y_train[i][0] for i in range(len(y_train))]
        class_weights = sklearn.utils.class_weight.compute_class_weight(
                'balanced', sorted(np.unique(y_train_labels)), y_train_labels)
        class_weights = list(class_weights)
        model.class_weights = class_weights
        print('Using class weights: {}'.format(class_weights))

    # For regression, determine mean sample weight so that we can
    # normalize to have mean=1
    if model.regression:
        train_weights = []
        for seqs, outputs in train_ds:
            train_weights += [data_parser.sample_regression_weight(xi, yi,
                    p=model.sample_weight_scaling_factor)
                    for xi, yi in zip(seqs, outputs)]
        train_weight_mean = np.mean(train_weights)
        validate_weights = []
        for seqs, outputs in validate_ds:
            validate_weights += [data_parser.sample_regression_weight(xi, yi,
                    p=model.sample_weight_scaling_factor)
                    for xi, yi in zip(seqs, outputs)]
        validate_weight_mean = np.mean(validate_weights)
    else:
        train_weight_mean = None
        validate_weight_mean = None

    def determine_sample_weights(seqs, outputs, norm_factor=None):
        if not model.regression:
            # Classification; weight by class
            labels = [int(o.numpy()[0]) for o in outputs]
            return [class_weights[label] for label in labels]
        else:
            # Regression; weight by variance
            weights = [data_parser.sample_regression_weight(xi, yi,
                    p=model.sample_weight_scaling_factor)
                    for xi, yi in zip(seqs, outputs)]
            if norm_factor is not None:
                weights = [w / norm_factor for w in weights]
            return weights

    # Compute weights for all samples once, rather than having to do so in
    # every epoch
    train_ds_weights = []
    for seqs, outputs in train_ds:
        sample_weight = determine_sample_weights(seqs, outputs,
                norm_factor=train_weight_mean)
        train_ds_weights += [sample_weight]
    validate_ds_weights = []
    for seqs, outputs in validate_ds:
        sample_weight = determine_sample_weights(seqs, outputs,
                norm_factor=validate_weight_mean)
        validate_ds_weights += [sample_weight]

    best_val_loss = None
    num_epochs_past_best_loss = 0

    for epoch in range(max_num_epochs):
        # Train on each batch
        for i, (seqs, outputs) in enumerate(train_ds):
            sample_weight = tf.constant(train_ds_weights[i])
            y_true, y_pred = tf_train_step(seqs, outputs,
                    sample_weight=sample_weight)
            if model.regression:
                train_r2_score_metric(y_true, y_pred)
                train_pearson_corr_metric(y_true, y_pred)
                train_spearman_corr_metric(y_true, y_pred)

        # Validate on each batch
        # Note that we could run the validation data through the model all
        # at once (not batched), but batching may help with memory usage by
        # reducing how much data is run through the network at once
        for i, (seqs, outputs) in enumerate(validate_ds):
            sample_weight = tf.constant(validate_ds_weights[i])
            y_true, y_pred = tf_validate_step(seqs, outputs,
                    sample_weight=sample_weight)
            if model.regression:
                validate_r2_score_metric(y_true, y_pred)
                validate_pearson_corr_metric(y_true, y_pred)
                validate_spearman_corr_metric(y_true, y_pred)

        # A note on one easy source of confusion (written here for
        # classification, but it may apply to weighted MSE with regression as
        # well):
        # We might think the prediction_loss as calculated by train_step() and
        # validate_step() should be equal to the weighted BCE because the loss
        # function is binary cross-entropy. However, it appears that the loss
        # funcion calculation multiples the binary cross-entropy for each
        # sample i by sample_weight[i], directly as it is given.  On the other
        # hand, it seems that {train,validate}_bce_weighted_metric normalize
        # the input sample weights (sample_weight) before the multiplication.
        # As a result, the prediction_loss (and thus the train or validate loss
        # value) can be quite different than weighted BCE; this might be
        # especially true for validation, as the class weights are computed
        # over the train data so, for validation, the sample weights might not
        # have a mean of 1. One way to see this is that, when multiplying the
        # sample_weight input to {train,validate}_step() by a scalar, the loss
        # value is multipled by that scalar but the weighted BCE is unchanged.
        # As a result, the loss value on different validation/test sets might
        # not be comparable, but the weighted BCE might be. The normalization
        # by {train,validate}_bce_weighted_metric seems to happen *per batch*
        # (i.e., on the call to update state), rather than on the calculation
        # at the end -- likely because it computes the updated mean every time
        # the state is updated. One way to see this is to change, above:
        # ```
        #    y_true, y_pred = tf_train_step(seqs, outputs,
        #            sample_weight=sample_weight)
        # ```
        # to
        # ```
        #    y_true, y_pred = tf_train_step(seqs, outputs,
        #            sample_weight=(1.0/np.mean(sample_weight))*sample_weight)
        # ```
        # so that a normalized sample_weight is passed for each batch; the loss
        # function value (namely, predicted_loss) should adjust whereas the
        # weighted BCE metric should not change, and the two will now equal
        # each other. As a result of this per batch normalization, I would lean
        # toward ignoring the weighted BCE metric (and likely weighted MSE
        # too); variation across batches will likely make it an unreliable
        # metric.

        # Log the metrics from this epoch
        print('EPOCH {}'.format(epoch+1))
        print('  Train metrics:')
        print('    Loss: {}'.format(train_loss_metric.result()))
        if model.regression:
            print('    MSE: {}'.format(train_mse_metric.result()))
            print('    Weighted MSE: {}'.format(train_mse_weighted_metric.result()))
            print('    MAE: {}'.format(train_mae_metric.result()))
            print('    MAPE: {}'.format(train_mape_metric.result()))
            print('    R^2 score: {}'.format(train_r2_score_metric.result()))
            print('    r-Pearson: {}'.format(train_pearson_corr_metric.result()))
            print('    r-Spearman: {}'.format(train_spearman_corr_metric.result()))
        else:
            print('    BCE: {}'.format(train_bce_metric.result()))
            print('    Weighted BCE: {}'.format(train_bce_weighted_metric.result()))
            print('    Accuracy: {}'.format(train_accuracy_metric.result()))
            print('    AUC-ROC: {}'.format(train_auc_roc_metric.result()))
            print('    AUC-PR: {}'.format(train_auc_pr_metric.result()))
        print('  Validate metrics:')
        print('    Loss: {}'.format(validate_loss_metric.result()))
        if model.regression:
            print('    MSE: {}'.format(validate_mse_metric.result()))
            print('    Weighted MSE: {}'.format(validate_mse_weighted_metric.result()))
            print('    MAE: {}'.format(validate_mae_metric.result()))
            print('    MAPE: {}'.format(validate_mape_metric.result()))
            print('    R^2 score: {}'.format(validate_r2_score_metric.result()))
            print('    r-Pearson: {}'.format(validate_pearson_corr_metric.result()))
            print('    r-Spearman: {}'.format(validate_spearman_corr_metric.result()))
        else:
            print('    BCE: {}'.format(validate_bce_metric.result()))
            print('    Weighted BCE: {}'.format(validate_bce_weighted_metric.result()))
            print('    Accuracy: {}'.format(validate_accuracy_metric.result()))
            print('    AUC-ROC: {}'.format(validate_auc_roc_metric.result()))
            print('    AUC-PR: {}'.format(validate_auc_pr_metric.result()))

        train_loss = train_loss_metric.result()
        val_loss = validate_loss_metric.result()
        if model.regression:
            train_mse = train_mse_metric.result()
            train_pearson_corr = train_pearson_corr_metric.result()
            train_spearman_corr = train_spearman_corr_metric.result()
            val_mse = validate_mse_metric.result()
            val_pearson_corr = validate_pearson_corr_metric.result()
            val_spearman_corr = validate_spearman_corr_metric.result()
        else:
            train_bce = train_bce_metric.result()
            train_bce_weighted = train_bce_weighted_metric.result()
            train_auc_roc = train_auc_roc_metric.result()
            train_auc_pr = train_auc_pr_metric.result()
            val_bce = validate_bce_metric.result()
            val_bce_weighted = validate_bce_weighted_metric.result()
            val_auc_roc = validate_auc_roc_metric.result()
            val_auc_pr = validate_auc_pr_metric.result()

        # Reset metric states so they are not cumulative over epochs
        train_loss_metric.reset_states()
        train_mse_metric.reset_states()
        train_mse_weighted_metric.reset_states()
        train_mae_metric.reset_states()
        train_mape_metric.reset_states()
        train_r2_score_metric.reset_states()
        train_pearson_corr_metric.reset_states()
        train_spearman_corr_metric.reset_states()
        train_bce_metric.reset_states()
        train_bce_weighted_metric.reset_states()
        train_accuracy_metric.reset_states()
        train_auc_roc_metric.reset_states()
        train_auc_pr_metric.reset_states()
        validate_loss_metric.reset_states()
        validate_mse_metric.reset_states()
        validate_mse_weighted_metric.reset_states()
        validate_mae_metric.reset_states()
        validate_mape_metric.reset_states()
        validate_r2_score_metric.reset_states()
        validate_pearson_corr_metric.reset_states()
        validate_spearman_corr_metric.reset_states()
        validate_bce_metric.reset_states()
        validate_bce_weighted_metric.reset_states()
        validate_accuracy_metric.reset_states()
        validate_auc_roc_metric.reset_states()
        validate_auc_pr_metric.reset_states()

        # Decide whether to stop at this epoch
        if best_val_loss is None or val_loss < best_val_loss:
            # Update the best validation loss
            best_val_loss = val_loss
            num_epochs_past_best_loss = 0
        else:
            # This loss is worse than one seen before
            num_epochs_past_best_loss += 1
        if num_epochs_past_best_loss >= STOPPING_PATIENCE:
            # Stop here
            print('  Stopping at EPOCH {}'.format(epoch+1))
            break

    if model.regression:
        train_metrics = {'loss': train_loss.numpy(), 'mse': train_mse.numpy(),
                'r-pearson': train_pearson_corr,
                'r-spearman': train_spearman_corr}
        val_metrics = {'loss': val_loss.numpy(), 'mse': val_mse.numpy(),
                'r-pearson': val_pearson_corr,
                'r-spearman': val_spearman_corr}
    else:
        train_metrics = {'loss': train_loss.numpy(), 'bce': train_bce.numpy(),
                'weighted-bce': train_bce_weighted.numpy(),
                'auc-pr': train_auc_pr.numpy(),
                'auc-roc': train_auc_roc.numpy()}
        val_metrics = {'loss': val_loss.numpy(), 'bce': val_bce.numpy(),
                'weighted-bce': val_bce_weighted.numpy(),
                'auc-pr': val_auc_pr.numpy(),
                'auc-roc': val_auc_roc.numpy()}
    return train_metrics, val_metrics


def test(model, x_test, y_test, data_parser, plot_roc_curve=None,
        plot_predictions=None, write_test_tsv=None,
        y_train=None):
    """Test a model.

    This prints metrics.

    Args:
        model: model object
        x_test, y_test: testing input and outputs (labels, if
            classification)
        data_parser: data parser object from parse_data
        plot_roc_curve: if set, path to PDF at which to save plot of ROC curve
        plot_predictions: if set, path to PDF at which to save plot of
            predictions vs. true values
        write_test_tsv: if set, path to TSV at which to write data on predictions
            as well as the testing sequences (one row per data point)
        y_train: (optional) if set, print metrics if the predictor simply
            predicts the mean of the training data

    Returns:
        dict with test metrics at the end (keys are 'loss'
        and ('bce' or 'mse') and ('auc-roc' or 'r-spearman'))
    """
    # Set the global module-level variables _model, needed by test_step()
    global _model
    _model = model

    tf_test_step = tf.function(test_step)

    test_ds = make_dataset_and_batch(x_test, y_test,
            batch_size=model.batch_size)

    # For regression, determine mean sample weight so that we can
    # normalize to have mean=1
    if model.regression:
        test_weights = []
        for seqs, outputs in test_ds:
            test_weights += [data_parser.sample_regression_weight(xi, yi,
                    p=model.sample_weight_scaling_factor)
                    for xi, yi in zip(seqs, outputs)]
        test_weight_mean = np.mean(test_weights)
    else:
        test_weight_mean = None

    def determine_sample_weights(seqs, outputs, norm_factor=None):
        if not model.regression:
            # Classification; weight by class
            labels = [int(o.numpy()[0]) for o in outputs]
            return [model.class_weights[label] for label in labels]
        else:
            # Regression; weight by variance
            weights = [data_parser.sample_regression_weight(xi, yi,
                    p=model.sample_weight_scaling_factor)
                    for xi, yi in zip(seqs, outputs)]
            if norm_factor is not None:
                weights = [w / norm_factor for w in weights]
            return weights

    all_true = []
    all_predictions = []
    for seqs, outputs in test_ds:
        sample_weight = determine_sample_weights(seqs, outputs,
                norm_factor=test_weight_mean)
        sample_weight = tf.constant(sample_weight)
        y_true, y_pred = tf_test_step(seqs, outputs,
                sample_weight=sample_weight)
        if model.regression:
            test_r2_score_metric(y_true, y_pred)
            test_pearson_corr_metric(y_true, y_pred)
            test_spearman_corr_metric(y_true, y_pred)
        all_true += list(tf.reshape(y_true, [-1]).numpy())
        all_predictions += list(tf.reshape(y_pred, [-1]).numpy())

    # Check the ordering: y_test should be the same as all_true
    # (y_test is a 2d array: [[value], [value], ..., [value]] so it must
    # be flattened prior to comparison)
    assert np.allclose(y_test.flatten(), all_true)

    # See note in train_and_validate() for discrepancy between loss and
    # weighted metric values.

    print('TEST DONE')
    print('  Test metrics:')
    print('    Loss: {}'.format(test_loss_metric.result()))
    if model.regression:
        print('    MSE: {}'.format(test_mse_metric.result()))
        print('    Weighted MSE: {}'.format(test_mse_weighted_metric.result()))
        print('    MAE: {}'.format(test_mae_metric.result()))
        print('    MAPE: {}'.format(test_mape_metric.result()))
        print('    R^2 score: {}'.format(test_r2_score_metric.result()))
        print('    r-Pearson: {}'.format(test_pearson_corr_metric.result()))
        print('    r-Spearman: {}'.format(test_spearman_corr_metric.result()))
    else:
        print('    BCE: {}'.format(test_bce_metric.result()))
        print('    Weighted BCE: {}'.format(test_bce_weighted_metric.result()))
        print('    Accuracy: {}'.format(test_accuracy_metric.result()))
        print('    AUC-ROC: {}'.format(test_auc_roc_metric.result()))
        print('    AUC-PR: {}'.format(test_auc_pr_metric.result()))

    test_loss = test_loss_metric.result()
    if model.regression:
        test_mse = test_mse_metric.result()
        test_pearson_corr = test_pearson_corr_metric.result()
        test_spearman_corr = test_spearman_corr_metric.result()
    else:
        test_bce = test_bce_metric.result()
        test_bce_weighted = test_bce_weighted_metric.result()
        test_auc_roc = test_auc_roc_metric.result()
        test_auc_pr = test_auc_pr_metric.result()

    test_loss_metric.reset_states()
    test_mse_metric.reset_states()
    test_mse_weighted_metric.reset_states()
    test_mae_metric.reset_states()
    test_mape_metric.reset_states()
    test_r2_score_metric.reset_states()
    test_pearson_corr_metric.reset_states()
    test_spearman_corr_metric.reset_states()
    test_bce_metric.reset_states()
    test_bce_weighted_metric.reset_states()
    test_accuracy_metric.reset_states()
    test_auc_roc_metric.reset_states()
    test_auc_pr_metric.reset_states()

    if model.regression and y_train is not None:
        # Print what the MSE would be if only predicting the mean of
        # the training data
        print('  MSE on test data if predicting mean of train data:',
                np.mean(np.square(np.mean(y_train) - np.array(all_true))))

    x_test_pos = [data_parser.pos_for_input(xi) for xi in x_test]

    if write_test_tsv:
        # Determine features for all input sequences
        seq_feats = []
        for i in range(len(x_test)):
            seq_feats += [data_parser.seq_features_from_encoding(x_test[i])]

        cols = ['target', 'target_without_context', 'guide',
                'hamming_dist', 'cas13a_pfs', 'crrna_pos', 'true_activity',
                'predicted_activity']
        with gzip.open(write_test_tsv, 'wt') as fw:
            def write_row(row):
                fw.write('\t'.join(str(x) for x in row) + '\n')

            # Write header
            write_row(cols)

            # Write row for each data point
            for i in range(len(x_test)):
                def val(k):
                    if k == 'true_activity':
                        return all_true[i]
                    elif k == 'predicted_activity':
                        return all_predictions[i]
                    elif k == 'crrna_pos':
                        if x_test_pos is None:
                            # Use -1 if position is unknown
                            return -1
                        else:
                            # x_test_pos[i] gives position of x_test[i]
                            return x_test_pos[i]
                    else:
                        return seq_feats[i][k]
                write_row([val(k) for k in cols])


    if plot_roc_curve:
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        fpr, tpr, thresholds = roc_curve(all_true, all_predictions)
        plt.figure(1)
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.show()
        plt.savefig(plot_roc_curve)
    if plot_predictions:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.scatter(all_true, all_predictions, c=x_test_pos)
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.title('True vs. predicted values')
        plt.show()
        plt.savefig(plot_predictions)

    if model.regression:
        test_metrics = {'loss': test_loss.numpy(), 'mse': test_mse.numpy(),
                'r-pearson': test_pearson_corr,
                'r-spearman': test_spearman_corr}
    else:
        test_metrics = {'loss': test_loss.numpy(), 'bce': test_bce.numpy(),
                'weighted-bce': test_bce_weighted.numpy(),
                'auc-pr': test_auc_pr.numpy(),
                'auc-roc': test_auc_roc.numpy()}
    return test_metrics

#####################################################################
#####################################################################
#####################################################################
#####################################################################


#####################################################################
#####################################################################
# Functions to train and test using Keras
#
# Compared to the custom functions above, this provides less
# flexibility but makes it simpler and possible to train across
# multiple GPUs.
#####################################################################
#####################################################################
def train_with_keras(model, x_train, y_train, x_validate, y_validate,
        max_num_epochs=1000):
    """Fit a model using Keras.

    The model must have already been compiled (e.g., with construct_model()
    above).

    Args:
        model: compiled model, e.g., output by construct_model()
        x_train/y_train: training data
        x_validate/y_validate: validation data; also used for early stopping
        max_num_epochs: maximum number of epochs to train for; note that
            the number it is trained for should be less due to early stopping
    """
    # Setup early stopping
    # The validation data is only used for early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
            mode='min', patience=1)

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_validate, y_validate),
            batch_size=model.batch_size, callbacks=[es],
            class_weight=model.class_weight,
            epochs=max_num_epochs,
            verbose=2)


def test_with_keras(model, x_test, y_test, data_parser, write_test_tsv=None,
        callback=None):
    """Test a model.

    This prints metrics.

    Args:
        model: model object
        x_test, y_test: testing input and outputs (labels, if
            classification)
        data_parser: data parser object from parse_data
        write_test_tsv: if set, path to TSV at which to write data on predictions
            as well as the testing sequences (one row per data point)
        callback: if set, a function to call that accepts the true and
            predicted test values -- called like callback(y_true, f_pred)

    Returns:
        dict with test metrics at the end (keys are 'loss'
        and ('bce' or 'mse') and ('auc-roc' or 'r-spearman'))
    """
    # Evaluate on test data
    test_metrics = model.evaluate(x_test, y_test,
            batch_size=model.batch_size)

    # Turn test_metrics from list into dict
    test_metrics = dict(zip(model.metrics_names, test_metrics))

    y_true = y_test
    y_pred = model.predict(x_test, batch_size=model.batch_size)

    if write_test_tsv:
        x_test_pos = [data_parser.pos_for_input(xi) for xi in x_test]

        # Determine features for all input sequences
        seq_feats = []
        for i in range(len(x_test)):
            seq_feats += [data_parser.seq_features_from_encoding(x_test[i])]

        cols = ['target', 'target_without_context', 'guide',
                'hamming_dist', 'cas13a_pfs', 'crrna_pos', 'true_activity',
                'predicted_activity']
        with gzip.open(write_test_tsv, 'wt') as fw:
            def write_row(row):
                fw.write('\t'.join(str(x) for x in row) + '\n')

            # Write header
            write_row(cols)

            # Write row for each data point
            for i in range(len(x_test)):
                def val(k):
                    if k == 'true_activity':
                        yt = y_true[i]
                        assert len(yt) == 1
                        return yt[0]
                    elif k == 'predicted_activity':
                        yp = y_pred[i]
                        assert len(yp) == 1
                        return yp[0]
                    elif k == 'crrna_pos':
                        if x_test_pos is None:
                            # Use -1 if position is unknown
                            return -1
                        else:
                            # x_test_pos[i] gives position of x_test[i]
                            return x_test_pos[i]
                    else:
                        return seq_feats[i][k]
                write_row([val(k) for k in cols])

    if model.regression:
        mse_metric = tf.keras.metrics.MeanSquaredError()
        mse_metric(y_true, y_pred)
        mse = mse_metric.result().numpy()
        pearson_corr_metric = Correlation('pearson_corr')
        pearson_corr_metric(y_true, y_pred)
        pearson_corr = pearson_corr_metric.result()
        spearman_corr_metric = Correlation('spearman_corr')
        spearman_corr_metric(y_true, y_pred)
        spearman_corr = spearman_corr_metric.result()

        test_metrics = {'loss': test_metrics['loss'],
                'mse': mse,
                'r-pearson': pearson_corr,
                'r-spearman': spearman_corr}
    else:
        bce_metric = tf.keras.metrics.BinaryCrossentropy()
        bce_metric(y_true, y_pred)
        bce = bce_metric.result().numpy()
        auc_pr_metric = tf.keras.metrics.AUC(num_thresholds=500, curve='PR')
        auc_pr_metric(y_true, y_pred)
        auc_pr = auc_pr_metric.result().numpy()
        auc_roc_metric = tf.keras.metrics.AUC(num_thresholds=500, curve='ROC')
        auc_roc_metric(y_true, y_pred)
        auc_roc = auc_roc_metric.result().numpy()

        test_metrics = {'loss': test_metrics['loss'],
                'bce': bce,
                'auc-pr': auc_pr,
                'auc-roc': auc_roc}

    print('TEST METRICS:', test_metrics)

    if callback is not None:
        callback(y_true, y_pred)

    return test_metrics


#####################################################################
#####################################################################
#####################################################################
#####################################################################


def main():
    # Read arguments and data
    args = parse_args()

    if args.load_model:
        # Read saved parameters and load them into the args namespace
        print('Loading parameters for model..')
        load_path_params = os.path.join(args.load_model,
                'model.params.pkl')
        with open(load_path_params, 'rb') as f:
            saved_params = pickle.load(f)
        params = vars(args)
        for k, v in saved_params.items():
            print("Setting argument '{}'={}".format(k, v))
            params[k] = v

    if args.test_split_frac:
        train_and_validate_split_frac = 1.0 - args.test_split_frac
        if not args.load_model:
            # Since this will be training a model, reserve validation
            # data (25%) for early stopping
            validate_frac = 0.2*train_and_validate_split_frac
            train_frac = train_and_validate_split_frac - validate_frac
        else:
            train_frac = train_and_validate_split_frac
            validate_frac = 0.0
        split_frac = (train_frac, validate_frac, args.test_split_frac)
    else:
        split_frac = None

    # Set seed and read data
    set_seed(args.seed)
    data_parser = read_data(args, split_frac=split_frac)
    x_train, y_train = data_parser.train_set()
    x_validate, y_validate = data_parser.validate_set()
    x_test, y_test = data_parser.test_set()

    # Determine, based on the dataset, whether to do regression or
    # classification
    if args.dataset == 'cas13':
        if args.cas13_classify:
            regression = False
        else:
            regression = True

    if regression and args.plot_roc_curve:
        raise Exception(("Can only use --plot-roc-curve when doing "
            "classification"))
    if not regression and args.plot_predictions:
        raise Exception(("Can only use --plot-predictions when doing "
            "regression"))

    if args.load_model:
        # Load the model
        print('Loading model weights..')
        model = load_model(args.load_model, params,
                x_train, y_train, x_validate, y_validate,
                data_parser)
        print('Done loading model.')
    else:
        # Construct model
        params = vars(args)
        model = construct_model(params, x_train.shape, regression,
                compile_for_keras=True, y_train=y_train)

        # Train the model, with validation
        train_with_keras(model, x_train, y_train, x_validate, y_validate)

    # Test the model
    test_with_keras(model, x_test, y_test, data_parser,
            write_test_tsv=args.write_test_tsv)

    if not regression:
        # Determine threshold for classifier
        # Note that this should only use params; it should *not* use
        # a pre-trained model with loaded weights
        # This does not use test data
        print('Determining classifier threshold via cross-validation')
        num_splits = 5
        thresholds = determine_classifier_threshold_for_precision(
                params, x_train, y_train, num_splits, data_parser,
                args.determine_classifier_threshold_for_precision)
        print(('Mean threshold across folds to achieve precision of %f = %f') %
                (args.determine_classifier_threshold_for_precision,
                    np.mean(thresholds)))
        print(('Median threshold across folds to achieve precision of %f = %f') %
                (args.determine_classifier_threshold_for_precision,
                    np.median(thresholds)))
        print('  Thresholds are:', thresholds)


if __name__ == "__main__":
    main()
