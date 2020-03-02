"""Classes and methods to parse data for input to ML tools.
"""

from collections import defaultdict
import gzip
import os
import random

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

__author__ = 'Hayden Metsky <hayden@mit.edu>'


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


_onehot_idx = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
def oh_to_nt(xi):
    """Convert one-hot encoded nucleotide xi to a nucleotide in sequence space

    Args:
        xi: one-hot encoded nucleotide

    Returns:
        nucleotide sequence of xi
    """
    assert len(xi) == 4
    if sum(xi) == 0:
        # All values are 0, use '-'
        return '-'
    else:
        assert np.isclose(sum(xi), 1.0) # either one-hot encoded or softmax
        return _onehot_idx[np.argmax(xi)]


class Cas13ActivityParser:
    """Parse data from paired crRNA/target Cas13 data tested with CARMEN.

    The output is numeric values (for regression) rather than labels.
    """
    INPUT_TSV = os.path.join(SCRIPT_PATH,
            'data/CCF-curated/CCF_merged_pairs_annotated.curated.resampled.tsv.gz')

    # Define crRNA (guide) length; used for determining range of crRNA
    # in nucleotide space
    CRRNA_LEN = 28

    # Define the seed region; for Cas13a, the middle ~third of the spacer
    SEED_START = int(CRRNA_LEN * 1/3) # 0-based, inclusive
    SEED_END = int(CRRNA_LEN * 2/3) + 1 # 0-based, exclusive

    # Define threshold on activity for inactive/active data points
    ACTIVITY_THRESHOLD = -4.0

    def __init__(self, subset=None, context_nt=20, split=(0.8, 0.1, 0.1),
            shuffle_seed=1, stratify_randomly=False, stratify_by_pos=False):
        """
        Args:
            subset: either 'exp' (use only experimental data points, which
                generally have a mismatch between guide/target), 'pos' (use
                only data points corresponding to a positive guide/target
                match with no mismatches (i.e., the wildtype target)),
                'neg' (use only data points corresponding to negative guide/
                target (i.e., high divergence between the two)), or
                'exp-and-pos;; if 'None', use all data points
            context_nt: nt of target sequence context to include alongside
                each guide
            split: (train, validation, test) split; must sum to 1.0
            shuffle_seed: seed to use for the random module to shuffle
            stratify_randomly: if set, shuffle rows before splitting into
                train/validate/test
            stratify_by_pos: if set, consider the position along the target
                and split based on this
        """
        assert subset in (None, 'exp', 'pos', 'neg', 'exp-and-pos')
        self.subset = subset

        self.context_nt = context_nt

        assert sum(split) == 1.0
        self.split_train, self.split_validate, self.split_test = split

        if stratify_randomly and stratify_by_pos:
            raise ValueError("Cannot set by stratify_randomly and stratify_by_pos")
        self.stratify_randomly = stratify_randomly
        self.stratify_by_pos = stratify_by_pos

        random.seed(shuffle_seed)

        self.classify_activity = False
        self.regress_on_all = False
        self.regress_only_on_active = False

        self.make_feats_for_baseline = None

        self.normalize_crrna_activity = False
        self.use_difference_from_wildtype_activity = False

        self.was_read = False

    def set_activity_mode(self, classify_activity, regress_on_all,
            regress_only_on_active):
        """Set mode for which points to read regarding their activity.

        Args:
            classify_activity: if True, have the output variable be a label
                (False/True) regarding activity of a guide/target pair
            regress_on_all: if True, output all guide/target pairs
            regress_only_on_active: if True, only output guide/target pairs
                corresponding to high activity
        """
        num_set = (int(classify_activity) + int(regress_on_all) +
                int(regress_only_on_active))
        if num_set != 1:
            raise Exception(("Exactly one of 'classify_activity' and "
                "'regress_on_all' and 'regress_only_on_active' can be set"))
        self.classify_activity = classify_activity
        self.regress_on_all = regress_on_all
        self.regress_only_on_active = regress_only_on_active

    def set_make_feats_for_baseline(self, feats):
        """Generate input features specifically for the baseline model.

        Args:
            feats: one of 'onehot-flat' (one-hot encoding, flattened to 1D);
                'onehot-simple' (one-hot encoding that encodes the target
                sequence and mismatches between the guide and it, leaving the
                guide encoding as all 0 when matching); 'handcrafted'
                (nucleotide frequency, dinucleotide frequency, etc.); or
                'combined' (concatenated 'onehot-simple' and 'handcrafted')
        """
        self.make_feats_for_baseline = feats

    def set_normalize_crrna_activity(self):
        """Normalize activity of each crRNA, across targets, to have mean 0 and
        stdev 1.

        We can only set one of 'normalize_crrna_activity' and
        'use_difference_from_wildtype_activity'.
        """
        assert self.use_difference_from_wildtype_activity is False
        self.normalize_crrna_activity = True

    def set_use_difference_from_wildtype_activity(self):
        """Use, as the activity value for a pair of guide g and target t, the
        difference between the g-t activity and the mean activity between g and
        its wildtype (matching) targets.

        We can only set one of 'normalize_crrna_activity' and
        'use_difference_from_wildtype_activity'.
        """
        assert self.normalize_crrna_activity is False
        self.use_difference_from_wildtype_activity = True

    def _gen_input_and_output(self, row):
        """Generate input features and output for each row.

        This generates a one-hot encoding for each sequence. Because we have
        the target ('target_at_guide') and guide sequence ('guide_seq'),
        we must encode how they compare to each other. Here, for each nucleotide
        position, we use an 8-bit vector (4 to encode the target sequence and
        4 for the guide sequence). For example, 'A' in the target and 'G' in the
        guide will be [1,0,0,0,0,0,1,0] for [A,C,G,T] one-hot encoding. There are
        other ways to do this as well: e.g., a 4-bit vector that represents an OR
        between the one-hot encoding of the target and guide (e.g., 'A' in the
        target and 'G' in the guide would be [1,0,1,0]), but this does not
        distinguish between bases in the target and guide (i.e., the encoding
        is the same for 'G' in the target and 'A' in the guide) which might
        be important here.

        Note that when self.make_feats_for_baseline is set, the input feature
        vector is different, depending on its value.

        Args:
            row: dict representing row of data (key'd by column
                name)

        Returns:
            tuple (i, out) where i is a one-hot encoding of the input
            and out is an output value (or out is in the format specified by
            self.make_feats_for_baseline)
        """
        # Check self.make_feats_for_baseline
        assert self.make_feats_for_baseline in [None, 'onehot-flat',
            'onehot-simple', 'handcrafted', 'combined']

        onehot_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        def onehot(b):
            # One-hot encoding of base b
            assert b in onehot_idx.keys()
            v = [0, 0, 0, 0]
            v[onehot_idx[b]] = 1
            return v

        # Create the input features for target sequence context on
        # the end before the guide
        input_feats_context_before = []
        context_before = row['target_before']
        assert self.context_nt <= len(context_before)
        start = len(context_before) - self.context_nt
        for pos in range(start, len(context_before)):
            # Make a one-hot encoding for this position of the target sequence
            v = onehot(context_before[pos])
            if self.make_feats_for_baseline is None:
                # For the 4 bits of guide sequence, use [0,0,0,0] (there is
                # no guide at this position)
                v += [0, 0, 0, 0]
                input_feats_context_before += [v]
            elif self.make_feats_for_baseline in ['onehot-flat',
                    'onehot-simple', 'combined']:
                # For the baseline, only use a one-hot encoding of the
                # target (and in a 1D array)
                input_feats_context_before += v
            elif self.make_feats_for_baseline == 'handcrafted':
                # No feature for the baseline here
                pass
            else:
                raise ValueError("Unknown choice of make_feats_for_baseline")

        # Create the input features for target and guide sequence
        input_feats_guide = []
        target = row['target_at_guide']
        guide = row['guide_seq']
        baseline_mismatches_pos = []
        assert len(target) == len(guide)
        for pos in range(len(guide)):
            # Make a one-hot encoding (4 bits) for each of the target
            # and the guide
            v_target = onehot(target[pos])
            v_guide = onehot(guide[pos])
            if self.make_feats_for_baseline is None:
                # Combine them into an 8-bit vector
                v = v_target + v_guide
                input_feats_guide += [v]
            elif self.make_feats_for_baseline == 'onehot-flat':
                # For the baseline, use an 8-bit vector (flattened - i.e.,
                # concatenated with the other positions)
                v = v_target + v_guide
                input_feats_guide += v
            elif self.make_feats_for_baseline in ['onehot-simple', 'combined']:
                # For the baseline, use a one-hot encoding of the target
                # (in a 1D array) and then use a one-hot encoding that gives
                # whether there is a mismatch and, if so, what the guide
                # base is
                if target[pos] == guide[pos]:
                    # No mismatch; use 0,0,0,0 for the guide
                    input_feats_guide += v_target + [0, 0, 0, 0]
                else:
                    # Mismatch; have the guide indicate which base there is
                    input_feats_guide += v_target + v_guide
                    baseline_mismatches_pos += [pos]
            elif self.make_feats_for_baseline == 'handcrafted':
                # Mark number of mismatches, but do not use input_feats_guide
                if target[pos] != guide[pos]:
                    baseline_mismatches_pos += [pos]
            else:
                raise ValueError("Unknown choice of make_feats_for_baseline")

        # Create the input features for target sequence context on
        # the end after the guide
        input_feats_context_after = []
        context_after = row['target_after']
        assert self.context_nt <= len(context_after)
        for pos in range(self.context_nt):
            # Make a one-hot encoding for this position of the target sequence
            v = onehot(context_after[pos])
            if self.make_feats_for_baseline is None:
                # For the 4 bits of guide sequence, use [0,0,0,0] (there is
                # no guide at this position)
                v += [0, 0, 0, 0]
                input_feats_context_after += [v]
            elif self.make_feats_for_baseline in ['onehot-flat',
                    'onehot-simple', 'combined']:
                # For the baseline, only use a one-hot encoding of the
                # target (and in a 1D array)
                input_feats_context_after += v
            elif self.make_feats_for_baseline == 'handcrafted':
                # No feature for the baseline here
                pass
            else:
                raise ValueError("Unknown choice of make_feats_for_baseline")

        # Combine the input features
        input_feats = input_feats_context_before + input_feats_guide + input_feats_context_after
        if self.make_feats_for_baseline == 'handcrafted':
            # The features directly from sequence should not have been set
            assert len(input_feats) == 0
        if self.make_feats_for_baseline in ['handcrafted', 'combined']:
            # Have the feature vector for the baseline include additional
            # features: position-independent nucleotide frequency, dinucleotide
            # frequency, and GC content (all in the guide); and number of
            # mismatches between guide and target
            bases = ('A', 'C', 'G', 'T')
            for b in bases:
                # Add a feature giving nucleotide frequency (count) of b in
                # the guide
                input_feats += [guide.count(b)]
            for b1 in bases:
                for b2 in bases:
                    # Add a feature giving dinucleotide frequency (count) of
                    # b1+b2 in the guide
                    input_feats += [guide.count(b1 + b2)]
            # Add a feature giving GC count in the guide
            input_feats += [guide.count('G') + guide.count('C')]
            # Add a feature giving number of mismatches outside the seed region
            # and in the seed
            seed_num_mismatches = len([p for p in baseline_mismatches_pos
                if p >= self.SEED_START and p < self.SEED_END])
            nonseed_num_mismatches = (len(baseline_mismatches_pos) -
                seed_num_mismatches)
            input_feats += [seed_num_mismatches]
            input_feats += [nonseed_num_mismatches]
        input_feats = np.array(input_feats)

        # Determine an output for this row
        activity = float(row['out_logk_measurement'])
        if self.classify_activity:
            # Make the output be a 1/0 label

            # Let 0 be inactive labels, and 1 be active ones
            if activity <= self.ACTIVITY_THRESHOLD:
                activity = 0
            else:
                activity = 1
        else:
            pos = int(row['guide_pos_nt'])
            if self.normalize_crrna_activity:
                crrna_mean = self.crrna_activity_mean[pos]
                crrna_stdev = self.crrna_activity_stdev[pos]
                activity = (activity - crrna_mean) / crrna_stdev
            if self.use_difference_from_wildtype_activity:
                wildtype_mean = self.crrna_wildtype_activity_mean[pos]
                activity = activity - wildtype_mean

        return (input_feats, activity)

    def read(self):
        """Read and parse TSV file.
        """
        # Read all rows
        header_idx = {}
        rows = []
        with gzip.open(self.INPUT_TSV, 'rt') as f:
            for i, line in enumerate(f):
                ls = line.rstrip().split('\t')
                if i == 0:
                    # Parse header
                    for j in range(len(ls)):
                        header_idx[ls[j]] = j
                else:
                    rows += [ls]

        # Convert rows to be key'd by column name
        rows_new = []
        for row in rows:
            row_dict = {k: row[header_idx[k]] for k in header_idx.keys()}
            rows_new += [row_dict]
        rows = rows_new

        if self.subset == 'exp':
            # Only keep rows where type is 'exp'
            rows = [row for row in rows if row['type'] == 'exp']
        if self.subset == 'pos':
            # Only keep rows where type is 'pos'
            rows = [row for row in rows if row['type'] == 'pos']
        if self.subset == 'neg':
            # Only keep rows where type is 'neg'
            rows = [row for row in rows if row['type'] == 'neg']
        if self.subset == 'exp-and-pos':
            # Only keep rows where type is 'exp' or 'pos'
            rows = [row for row in rows if row['type'] == 'exp' or
                    row['type'] == 'pos']

        # Shuffle the rows before splitting
        if self.stratify_randomly:
            random.shuffle(rows)

        # Sort by position before splitting
        if self.stratify_by_pos:
            rows = sorted(rows, key=lambda x: float(x['guide_pos_nt']))

        # Remove the inactive points
        if self.regress_only_on_active:
            rows = [row for row in rows if
                    float(row['out_logk_measurement']) >= self.ACTIVITY_THRESHOLD]

        # Calculate the mean and stdev of activity for each crRNA (according
        # to position); note that the input includes multiple measurements
        # (technical replicates) for each, so these statistics are taken across
        # the sampled measurements
        # Note that this is only used for regression, so only add it for the
        # active guide-target pairs
        activity_by_pos = defaultdict(list)
        for row in rows:
            pos = int(row['guide_pos_nt'])
            activity = float(row['out_logk_measurement'])
            activity_by_pos[pos].append(activity)
        self.crrna_activity_mean = {pos: np.mean(activity_by_pos[pos])
                for pos in activity_by_pos.keys()}
        self.crrna_activity_stdev = {pos: np.std(activity_by_pos[pos])
                for pos in activity_by_pos.keys()}

        # For each crRNA (according to positive), calculate the mean activity
        # between it and the wildtype targets
        wildtype_activity_by_pos = defaultdict(list)
        for row in rows:
            if int(row['guide_target_hamming_dist']) == 0:
                # This is a wildtype target
                pos = int(row['guide_pos_nt'])
                activity = float(row['out_logk_measurement'])
                wildtype_activity_by_pos[pos].append(activity)
        self.crrna_wildtype_activity_mean = {pos: np.mean(wildtype_activity_by_pos[pos])
                for pos in wildtype_activity_by_pos.keys()}

        # Generate input and outputs for each row
        inputs_and_outputs = []
        self.input_feats_pos = {}
        row_idx_pos = []
        for row in rows:
            pos = int(row['guide_pos_nt'])

            # Generate an input feature vector and a (list of) output(s)
            input_feats, output = self._gen_input_and_output(row)
            inputs_and_outputs += [(input_feats, output)]
            row_idx_pos += [pos]

            # Store a mapping from the input feature vector to the guide
            # position in the library design
            input_feats_key = np.array(input_feats, dtype='f').tostring()
            if input_feats_key in self.input_feats_pos:
                assert self.input_feats_pos[input_feats_key] == pos
            else:
                self.input_feats_pos[input_feats_key] = pos

        # Split into train, validate, and test sets
        train_end_idx = int(len(inputs_and_outputs) * self.split_train)
        validate_end_idx = int(len(inputs_and_outputs) * (self.split_train + self.split_validate))
        self._train_set = []
        self._validate_set = []
        self._test_set = []
        for i in range(len(inputs_and_outputs)):
            if i <= train_end_idx:
                self._train_set += [inputs_and_outputs[i]]
            elif i <= validate_end_idx:
                self._validate_set += [inputs_and_outputs[i]]
            else:
                if self.stratify_by_pos:
                    # If there are points with the exact same position that
                    # are split across the validate/test sets, then the ones
                    # from the test set will get removed because they overlap
                    # the validate set; instead of letting these ones get
                    # tossed, just put them in the validate set
                    last_validate_pos = row_idx_pos[validate_end_idx]
                    if row_idx_pos[i] == last_validate_pos:
                        if self.split_validate == 0:
                            # There should be no validate set; put it in train
                            # set instead
                            self._train_set += [inputs_and_outputs[i]]
                        else:
                            self._validate_set += [inputs_and_outputs[i]]
                    else:
                        self._test_set += [inputs_and_outputs[i]]
                else:
                    self._test_set += [inputs_and_outputs[i]]

        self.was_read = True

        if self.stratify_by_pos:
            # Make sure there is no overlap (or leakage) between the
            # train/validate and test sets in the split; to do this, toss
            # test data that is too "close" (according to the below function) to
            # the train/validate data
            train_and_validate = self._train_set + self._validate_set
            test_set_nonoverlapping, _ = self.make_nonoverlapping_datasets(
                    train_and_validate, self._test_set)
            assert len(test_set_nonoverlapping) <= len(self._test_set)
            self._test_set = test_set_nonoverlapping

            # The data points should still be shuffled; currently they
            # are sorted within each data set by position in the target
            random.shuffle(self._train_set)
            random.shuffle(self._validate_set)
            random.shuffle(self._test_set)

        # Verify the correctness of self.pos_for_input(); it's key for later
        # steps
        # Include making a numpy array out of input_feats, as done when
        # generating the data sets
        for row in rows:
            pos = int(row['guide_pos_nt'])
            input_feats, _ = self._gen_input_and_output(row)
            input_feats = np.array(input_feats, dtype='f')
            assert self.pos_for_input(input_feats) == pos


    def pos_for_input(self, x):
        """Return position (in nucleotide space) of the crRNA of a data point.

        Args:
            x: data point, namely input features

        Returns:
            x's position in nucleotide space
        """
        if not self.was_read:
            raise Exception("read() must be called first")
        x_key = np.array(x, dtype='f').tostring()
        assert x_key in self.input_feats_pos
        return self.input_feats_pos[x_key]

    def make_nonoverlapping_datasets(self, data1, data2):
        """Make sure there is no overlap (leakage) between two datasets.

        Each data point takes up some region in nucleotide space. This
        makes sure that the crRNAs from two datasets do not overlap in that
        nucleotide space -- i.e., there is not leakage between the two.
        To do this, this modifies data2 by removing points that overlap
        in nucleotide space with points in data1.

        Note that this permits sequence context of points in data2 to overlap
        with crRNAs for points in data1, and vice-versa. Ensuring no overlap
        of sequence context would probably require tossing too many data
        points, if self.context_nt is large.

        Args:
            data1: list of tuples (X, y) where X is input data and y is outputs
            data2: list of tuples (X, y) where X is input data and y is outputs

        Returns:
            tuple (a, b) where a is data2, after having removed any data
            points in data2 that overlap with ones in data1, and b is
            the indices in data2 that were kept
        """
        def range_for_input(x):
            # Compute x's crRNA range in nucleotide space: (start, end) where
            # start is inclusive and end is exclusive
            start_pos_nt = self.pos_for_input(x)
            length_in_nt = self.CRRNA_LEN
            return (start_pos_nt, start_pos_nt + length_in_nt)

        # The nucleotide space in this dataset is small (~1000 nt); just
        # create a set of all nucleotides in data1 and check data2 against
        # this
        # Since many ranges will be the same (i.e., points are at the exact
        # same position), first create a set of ranges and then create
        # a set of nucleotide positions from those
        data1_ranges = set()
        for X, y in data1:
            X_start, X_end = range_for_input(X)
            data1_ranges.add((X_start, X_end))
        data1_nt = set()
        for start, end in data1_ranges:
            for p in range(start, end):
                data1_nt.add(p)

        # Find which points in data2 to remove because they overlap a
        # nucleotide in data1
        # Make a copy of data2, only including points that do not overlap
        data2_nonoverlapping = []
        data2_idx_kept = []
        for i, (X, y) in enumerate(data2):
            X_start, X_end = range_for_input(X)
            include = True
            for p in range(X_start, X_end):
                if p in data1_nt:
                    # Remove this data point from data2
                    include = False
                    break
            if include:
                data2_nonoverlapping += [(X, y)]
                data2_idx_kept += [i]

        return (data2_nonoverlapping, data2_idx_kept)

    def _data_set(self, data):
        """Return data set.

        Args:
            data: one of self._train_set, self._validate_set, or
                self._test_set

        Returns:
            (X, y) where X is input vectors and y is outputs
        """
        if not self.was_read:
            raise Exception("read() must be called first")
        inputs = []
        outputs = []
        for input_feats, output in data:
            inputs += [input_feats]
            outputs += [[output]]
        return np.array(inputs, dtype='f'), np.array(outputs, dtype='f')

    def train_set(self):
        """Return training set.

        Returns:
            (X, y) where X is input vectors and y is outputs
        """
        return self._data_set(self._train_set)

    def validate_set(self):
        """Return validation set.

        Returns:
            (X, y) where X is input vectors and y is outputs
        """
        return self._data_set(self._validate_set)

    def test_set(self):
        """Return test set.

        Returns:
            (X, y) where X is input vectors and y is outputs
        """
        return self._data_set(self._test_set)

    def sample_regression_weight(self, xi, yi, p=0):
        """Compute a sample weight to use during regression while training.

        Args:
            xi: data point, namely input features
            yi: activity of xi
            p: scaling factor for importance weight (p>=0; p=0 does not incorporate
                sample importance, and all samples will have the same weight)

        Returns:
            relative weight of sample
        """
        # xi is a guide-target pair
        # Determine the mean and stdev across all targets of the guide in xi
        guide_pos = self.pos_for_input(xi)
        guide_wildtype_mean = self.crrna_activity_mean[guide_pos]
        guide_wildtype_stdev = self.crrna_activity_stdev[guide_pos]

        # Let the weight be 1 + p*|z|, where z is
        #     ([guide activity for xi] - [mean activity across targets at xi]) /
        #         [stdev of activity across targets at xi]
        # This way guide-target pairs where the activity is much more different
        # than the average for the guide are weighted more heavily during
        # training; intuitively, these are more interesting/important samples so we
        # want to weight them higher, and also the variation within guides (i.e.,
        # for target variants of a guide) may be harder to learn than the variation
        # across guides
        z = (yi - guide_wildtype_mean) / guide_wildtype_stdev
        weight = 1 + p*np.absolute(z)

        return weight

    def split(self, x, y, num_splits, shuffle_and_split=False,
            stratify_by_pos=False, yield_indices=False):
        """Split the data using stratified folds, for k-fold cross validation.

        Args:
            x: input data
            y: labels
            num_splits: number of folds
            shuffle_and_split: if True, shuffle before splitting and stratify based
                on the distribution of output variables (here, classes) to ensure they
                are roughly the same across the different folds
            stratify_by_pos: if True, determine the different folds based on
                the position of each data point (ensuring that the
                validate set is a contiguous region of the target molecule; this is
                similar to leave-one-gene-out cross-validation
            yield_indices: if True, return indices rather than data points (see
                'Iterates:' below for detail)

        Iterates:
            if yield_indices is False:
                (x_train_i, y_train_i, x_validate_i, y_validate_i) where each is
                for a fold of the data; these are actual data points from x, y
            if yield_indices is True:
                (train_i, validate_i) where each is for a fold of the data;
                these are indices referring to data in x, y
        """
        assert len(x) == len(y)

        if ((shuffle_and_split is False and stratify_by_pos is False) or
                (shuffle_and_split is True and stratify_by_pos is True)):
            raise ValueError(("Exactly one of shuffle_and_split or stratify_by_pos "
                "must be set"))

        if shuffle_and_split:
            idx = list(range(len(x)))
            random.shuffle(idx)
            x_shuffled = [x[i] for i in idx]
            y_shuffled = [y[i] for i in idx]
            x = np.array(x_shuffled)
            y = np.array(y_shuffled)
            skf = StratifiedKFold(n_splits=num_splits)
            def split_iter():
                return skf.split(x, y)
        elif stratify_by_pos:
            x_with_pos = [(xi, self.pos_for_input(xi)) for xi in x]
            all_pos = np.array(sorted(set(pos for xi, pos in x_with_pos)))
            kf = KFold(n_splits=num_splits)
            def split_iter():
                for train_pos_idx, test_pos_idx in kf.split(all_pos):
                    train_pos = set(all_pos[train_pos_idx])
                    test_pos = set(all_pos[test_pos_idx])
                    train_index, test_index = [], []
                    for i, (xi, pos) in enumerate(x_with_pos):
                        assert pos in train_pos or pos in test_pos
                        if pos in train_pos:
                            train_index += [i]
                        if pos in test_pos:
                            test_index += [i]

                    # Get rid of test indices for data points that overlap with
                    # ones in the train set
                    x_train = [(x[i], y[i]) for i in train_index]
                    x_test = [(x[i], y[i]) for i in test_index]
                    _, test_index_idx_to_keep = self.make_nonoverlapping_datasets(
                            x_train, x_test)
                    test_index_idx_to_keep = set(test_index_idx_to_keep)
                    test_index_nonoverlapping = []
                    for i in range(len(test_index)):
                        if i in test_index_idx_to_keep:
                            test_index_nonoverlapping += [test_index[i]]
                    test_index = test_index_nonoverlapping

                    # Shuffle order of data points within the train and test sets
                    random.shuffle(train_index)
                    random.shuffle(test_index)
                    yield train_index, test_index

        for train_index, test_index in split_iter():
            if yield_indices:
                yield (train_index, test_index)
            else:
                x_train_i, y_train_i = x[train_index], y[train_index]
                x_validate_i, y_validate_i = x[test_index], y[test_index]
                yield (x_train_i, y_train_i, x_validate_i, y_validate_i)

    def seq_features_from_encoding(self, x):
        """Determine sequence features by parsing the input vector for a data point.

        In some ways this reverses the parsing done above. This converts a one-hot
        encoding of both target and guide back into nucleotide sequence space.

        Args:
            x: input sequence as Lx8 vector where L is the target length; x[i][0:4]
                gives a one-hot encoding of the target at position i and
                x[i][4:8] gives a one-hot encoding ofthe guide at position i

        Returns:
            dict where keys are features (e.g., 'target', 'guide', and 'PFS')
        """
        x = np.array(x)

        target_len = len(x)
        assert x.shape == (target_len, 8)

        # Read the target
        target = ''.join(oh_to_nt(x[i][0:4]) for i in range(target_len))

        # Everything in the target should be a nucleotide
        assert '-' not in target

        # Read the guide
        guide_with_context = ''.join(oh_to_nt(x[i][4:8]) for i in range(target_len))

        # Verify the context of the guide is all '-' (all 0s), and extract just
        # the guide
        guide_start = self.context_nt
        guide_end = target_len - self.context_nt
        assert guide_with_context[:guide_start] == '-'*self.context_nt
        assert guide_with_context[guide_end:] == '-'*self.context_nt
        guide = guide_with_context[guide_start:guide_end]
        assert '-' not in guide
        target_without_context = target[guide_start:guide_end]

        # Compute the Hamming distance
        hd = sum(1 for i in range(len(guide)) if guide[i] != target_without_context[i])

        # Determine the Cas13a PFS (immediately adjacent to guide,
        # 3' end in target space)
        cas13a_pfs = target[guide_end]

        return {'target': target,
                'target_without_context': target_without_context,
                'guide': guide,
                'hamming_dist': hd,
                'cas13a_pfs': cas13a_pfs}


def input_vec_for_embedding(x, context_nt):
    """Create an input vector to use with an embedding layer, from one-hot
    encoded sequence.

    Args:
        x: input sequence as Lx8 vector where L is the target length; x[i][0:4]
            gives a one-hot encoding of the target at position i and
            x[i][4:8] gives a one-hot encoding ofthe guide at position i
        context_nt: amount of context in target

    Returns:
        Lx2 vector where x[i][0] gives an index corresponding to a nucleotide
        of the target at position i and x[i][1] for the guide. Each x[i][j]
        is an index in [0,4] -- 0 for A, 1 for C, 2 for G, 3 for T, and 4
        for no guide sequence (i.e., target context)
    """
    x = np.array(x)

    target_len = len(x)
    assert x.shape == (target_len, 8)

    # Read the target
    target = ''.join(oh_to_nt(x[i][0:4]) for i in range(target_len))

    # Everything in the target should be a nucleotide
    assert '-' not in target

    # Read the guide
    guide_with_context = ''.join(oh_to_nt(x[i][4:8]) for i in range(target_len))

    # Verify the context of the guide is all '-' (all 0s)
    guide_start = context_nt
    guide_end = target_len - context_nt
    assert guide_with_context[:guide_start] == '-'*context_nt
    assert guide_with_context[guide_end:] == '-'*context_nt

    idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
    x_idx = [[idx[target[i]], idx[guide_with_context[i]]]
                for i in range(target_len)]
    return np.array(x_idx)

