"""Classes and methods to parse data for input to ML tools.
"""

import random

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

__author__ = 'Hayden Metsky <hayden@mit.edu>'


class Doench2016Cas9ActivityParser:
    """Parse data from Doench et al. 2016 in NBT (curated from
    Supplementary Table 18).
    """
    INPUT_TSV='data/doench2016-nbt.supp-table-18.curated.with-context.tsv'

    # A threshold is of 1.0 for negative/positive points is reasonable based
    # on the distribution of the output variable (day21_minus_etp) and
    # consistent with Figure 5b of the paper
    ACTIVITY_THRESHOLD = 1.0

    # Set the number of nucleotides represented by each percent in the
    # 'protein_pos_pct' value for a data point
    PROTEIN_POS_PCT_NT = 1462 / 100.0

    def __init__(self, subset=None, context_nt=20, split=(0.8, 0.1, 0.1),
            shuffle_seed=1, stratify_randomly=False, stratify_by_pos=False):
        """
        Args:
            subset: if 'guide-mismatch-and-good-pam', only use data points
                representing guides with the canonical PAM (NGG) but a
                mismatch to the target; if 'guide-match', only use data points
                representing guides that perfectly match the target but may or
                may not have the canonical PAM (NGG); if None, use all data
                points
            context_nt: nt of target sequence context to include alongside
                each guide
            split: (train, validation, test) split; must sum to 1.0
            shuffle_seed: seed to use for the random module to shuffle
            stratify_randomly: if set, shuffle rows before splitting into
                train/validate/test
            stratify_by_pos: if set, sort by position on the genome and
                split based on this
        """
        assert subset in (None, 'guide-mismatch-and-good-pam', 'guide-match')
        self.subset = subset

        self.context_nt = context_nt

        assert sum(split) == 1.0
        self.split_train, self.split_validate, self.split_test = split

        if stratify_randomly and stratify_by_pos:
            raise ValueError("Cannot set by stratify_randomly and stratify_by_pos")
        self.stratify_randomly = stratify_randomly
        self.stratify_by_pos = stratify_by_pos

        random.seed(shuffle_seed)

        self.was_read = False

    def _gen_input_and_label(self, row):
        """Generate input features and label for each row.

        This generates a one-hot encoding for each sequence. Because we have
        the target (wildtype guide, 'guide_wt') and guide sequence ('guide_mutated'),
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

        This determines a label based on the value of 'day21_minus_etp', the
        output variable, compared to self.ACTIVITY_THRESHOLD.

        Args:
            row: dict representing row of data (key'd by column
                name)

        Returns:
            tuple (input, label) where i is a one-hot encoding of the input
            and label is 0 or 1
        """
        onehot_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        def onehot(b):
            # One-hot encoding of base b
            assert b in onehot_idx.keys()
            v = [0, 0, 0, 0]
            v[onehot_idx[b]] = 1
            return v

        # Create the input features for target sequence context on
        # the 5' end
        input_feats_context_5 = []
        context_5 = row['guide_wt_context_5']
        assert self.context_nt <= len(context_5)
        start = len(context_5) - self.context_nt
        for pos in range(start, len(context_5)):
            # Make a one-hot encoding for this position of the target sequence
            v = onehot(context_5[pos])
            # For the 4 bits of guide sequence, use [0,0,0,0] (there is
            # no guide at this position)
            v += [0, 0, 0, 0]
            input_feats_context_5 += [v]

        # Create the input features for target and guide sequence
        input_feats_guide = []
        target = row['guide_wt']
        guide = row['guide_mutated']
        assert len(target) == len(guide)
        for pos in range(len(guide)):
            # Make a one-hot encoding (4 bits) for each of the target
            # and the guide
            v_target = onehot(target[pos])
            v_guide = onehot(guide[pos])
            # Combine them into an 8-bit vector
            v = v_target + v_guide
            input_feats_guide += [v]

        # Create the input features for target sequence context on
        # the 3' end
        input_feats_context_3 = []
        context_3 = row['guide_wt_context_3']
        assert self.context_nt <= len(context_3)
        for pos in range(self.context_nt):
            # Make a one-hot encoding for this position of the target sequence
            v = onehot(context_3[pos])
            # For the 4 bits of guide sequence, use [0,0,0,0] (there is
            # no guide at this position)
            v += [0, 0, 0, 0]
            input_feats_context_3 += [v]

        # Combine the input features
        input_feats = input_feats_context_5 + input_feats_guide + input_feats_context_3
        input_feats = np.array(input_feats)

        # Determine a label for this row
        activity = float(row['day21_minus_etp'])
        if activity >= self.ACTIVITY_THRESHOLD:
            label = 1
        else:
            label = 0

        return (input_feats, label)

    def read(self):
        """Read and parse TSV file.
        """
        # Read all rows
        header_idx = {}
        rows = []
        with open(self.INPUT_TSV) as f:
            for i, line in enumerate(f):
                ls = line.rstrip().split('\t')
                if i == 0:
                    # Parse header
                    for j in range(len(ls)):
                        header_idx[ls[j]] = j
                else:
                    rows += [ls]

        if self.subset == 'guide-mismatch-and-good-pam':
            # Only keep rows where category is 'Mismatch' and the PAM
            # (according to guide_wt_context_3 is NGG)
            rows = [row for row in rows if
                    (row[header_idx['category']] == 'Mismatch' and
                     row[header_idx['guide_wt_context_3']][1:3] == 'GG')]
        if self.subset == 'guide-match':
            # Only keep rows where the category is 'PAM' (these ones
            # vary the PAM); and check that, for these, the guide matches
            # the target (i.e., guide_mutated == guide_wt)
            rows = [row for row in rows if row[header_idx['category']] == 'PAM']
            for row in rows:
                assert row[header_idx['guide_mutated']] == row[header_idx['guide_wt']]

        # Shuffle the rows before splitting
        if self.stratify_randomly:
            random.shuffle(rows)

        # Sort by position before splitting
        if self.stratify_by_pos:
            rows = sorted(rows, key=lambda x: float(x[header_idx['protein_pos_pct']]))

        # Generate input and labels for each row
        inputs_and_labels = []
        self.input_feats_pct_pos = {}
        row_idx_pct_pos = {}
        for i, row in enumerate(rows):
            row_dict = {k: row[header_idx[k]] for k in header_idx.keys()}
            input_feats, label = self._gen_input_and_label(row_dict)
            inputs_and_labels += [(input_feats, label)]

            input_feats_key = np.array(input_feats, dtype='f').tostring()
            self.input_feats_pct_pos[input_feats_key] = float(row[header_idx['protein_pos_pct']])
            row_idx_pct_pos[i] = float(row[header_idx['protein_pos_pct']])

        # Split into train, validate, and test sets
        train_end_idx = int(len(inputs_and_labels) * self.split_train)
        validate_end_idx = int(len(inputs_and_labels) * (self.split_train + self.split_validate))
        self._train_set = []
        self._validate_set = []
        self._test_set = []
        for i in range(len(inputs_and_labels)):
            if i <= train_end_idx:
                self._train_set += [inputs_and_labels[i]]
            elif i <= validate_end_idx:
                self._validate_set += [inputs_and_labels[i]]
            else:
                if self.stratify_by_pos:
                    # If there are points with the exact same position that
                    # are split across the validate/test sets, then the ones
                    # from the test set will get removed because they overlap
                    # the validate set; instead of letting these ones get
                    # tossed, just put them in the validate set
                    last_validate_pct_pos = row_idx_pct_pos[validate_end_idx]
                    if row_idx_pct_pos[i] == last_validate_pct_pos:
                        if self.split_validate == 0:
                            # There should be no validate set; put it in train
                            # set instead
                            self._train_set += [inputs_and_labels[i]]
                        else:
                            self._validate_set += [inputs_and_labels[i]]
                    else:
                        self._test_set += [inputs_and_labels[i]]
                else:
                    self._test_set += [inputs_and_labels[i]]

        self.was_read = True

        if self.stratify_by_pos:
            # Make sure there is no overlap (or leakage) between the
            # train/validate and test sets in the split; to do this, toss
            # test data that is too "close" (within the given parameter) to
            # the train/validate data
            train_and_validate = self._train_set + self._validate_set
            test_set_nonoverlapping, _ = self.make_nonoverlapping_datasets(
                    train_and_validate, self._test_set)
            self._test_set = test_set_nonoverlapping

            # The data points should still be shuffled; currently they
            # are sorted within each data set by protein position
            random.shuffle(self._train_set)
            random.shuffle(self._validate_set)
            random.shuffle(self._test_set)

    def pos_for_input(self, x):
        """Return position (in nucleotide space) of a data point.

        Args:
            x: data point, namely input features

        Returns:
            x's position in nucleotide space
        """
        if not self.was_read:
            raise Exception("read() must be called first")
        x_key = np.array(x, dtype='f').tostring()
        return int(self.input_feats_pct_pos[x_key] * self.PROTEIN_POS_PCT_NT)

    def make_nonoverlapping_datasets(self, data1, data2):
        """Make sure there is no overlap (leakage) between two datasets.

        Each data point takes up some region in nucleotide space. This
        makes sure that points from two datasets to not overlap in that
        nucleotide space -- i.e., there is not leakage between the two.
        To do this, this modifies data2 by removing points that overlap
        in nucleotide space with points in data1.

        Args:
            data1: list of tuples (X, y) where X is input data and y is labels
            data2: list of tuples (X, y) where X is input data and y is labels

        Returns:
            tuple (a, b) where a is data2, after having removed any data
            points in data2 that overlap with ones in data1, and b is
            the indices in data2 that were kept
        """
        def range_for_input(x):
            # Compute x's range in nucleotide space: (start, end) where start
            # is inclusive and end is exclusive
            start_pos_nt = self.pos_for_input(x)
            length_in_nt = len(x)
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
            (X, y) where X is input vectors and y is labels
        """
        if not self.was_read:
            raise Exception("read() must be called first")
        inputs = []
        labels = []
        for input_feats, label in data:
            inputs += [input_feats]
            labels += [[label]]
        return np.array(inputs, dtype='f'), np.array(labels, dtype='f')

    def train_set(self):
        """Return training set.

        Returns:
            (X, y) where X is input vectors and y is labels
        """
        return self._data_set(self._train_set)

    def validate_set(self):
        """Return validation set.

        Returns:
            (X, y) where X is input vectors and y is labels
        """
        return self._data_set(self._validate_set)

    def test_set(self):
        """Return test set.

        Returns:
            (X, y) where X is input vectors and y is labels
        """
        return self._data_set(self._test_set)


class Cas13SimulatedData(Doench2016Cas9ActivityParser):
    """Simulate Cas13 data from the Cas9 data.
    """
    def __init__(self, subset=None, context_nt=20, split=(0.8, 0.1, 0.1),
            shuffle_seed=None, stratify_by_pos=False):
        super(Cas13SimulatedData, self).__init__(
                subset=subset, context_nt=context_nt,
                split=split, shuffle_seed=shuffle_seed,
                stratify_by_pos=stratify_by_pos)

    def _gen_input_and_label(self, row):
        """Modify target and guide sequence to make it resemble Cas13
        before generating vectors.

        Args:
            row: dict representing row of data (key'd by column
                name)

        Returns:
            tuple (input, label) where i is a one-hot encoding of the input
            and label is 0 or 1
        """
        # Make lists out of the target and guide, to modify them
        target_list = list(row['guide_wt'])
        guide_list = list(row['guide_mutated'])
        assert len(target_list) == 20
        assert len(guide_list) == 20

        # The 'seed' region of Cas9 (where mismatches matter a lot) is
        # the ~10 nt proximal to the PAM on the 3' end; for Cas13, the
        # seed region seems to be in the middle (~9-15 nt out of its 28 nt,
        # or ~6-11 out of 20 nt in terms of fractional position)
        # Swap the last 6 nt of the Cas9 guide (in its seed region) with
        # positions 6-11 (inclusive); this way, if there are mismatches
        # at the end of Cas9 (which should hurt performance), they'll
        # move toward the middle of the simulated Cas13 guide
        target_end = target_list[14:20]
        target_list[14:20] = target_list[6:12]
        target_list[6:12] = target_end
        guide_end = guide_list[14:20]
        guide_list[14:20] = guide_list[6:12]
        guide_list[6:12] = guide_end

        # Add 8 random matching nucleotides to the guide and target
        # so that they are 28 nt (Cas13 guides are 28 nt, Cas9 are 20 nt);
        # draw these from the same distribution of bases already in the
        # target, so that the nucleotide composition does not change
        # Add 4 to the start and 4 to the end -- this previously added
        # the bases at random positions, but that adds too much noise
        # (the effect of a particular position becomes less meaningful)
        target_bases = list(target_list)
        for k in range(4):
            b = random.choice(target_bases)
            target_list.insert(0, b)
            guide_list.insert(0, b)
        for k in range(4):
            b = random.choice(target_bases)
            target_list.insert(len(target_list), b)
            guide_list.insert(len(guide_list), b)

        # Randomly change the target and guide to have some G-U base
        # pairing, as Cas13 acts on RNA-RNA binding; this does not affect
        # the output variable
        for i in range(len(target_list)):
            if target_list[i] == guide_list[i]:
                if random.random() < 1.0/3:
                    # Make a change for this position to have G-U pairing
                    # The two possibilities are: (target=G and guide=A)
                    # OR (target=T and guide=C)
                    if target_list[i] == 'A':
                        # Make target=G, guide=A
                        target_list[i] = 'G'
                    elif target_list[i] == 'C':
                        # Make target=T, guide=C
                        target_list[i] = 'T'
                    elif target_list[i] == 'G':
                        # Make target=G, guide=A
                        guide_list[i] = 'A'
                    elif target_list[i] == 'T':
                        # Make target=T, guide=C
                        guide_list[i] = 'C'
            else:
                # This is a mismatch between target and guide; if it
                # is G-U paired, make it not G-U paired because it may
                # have had an effect for Cas9
                if target_list[i] == 'G' and guide_list[i] == 'A':
                    if random.random() < 0.5:
                        # Make the target be C or T
                        target_list[i] = random.choice(['C','T'])
                    else:
                        # Make the guide be C or T
                        guide_list[i] = random.choice(['C','T'])
                elif target_list[i] == 'T' and guide_list[i] == 'C':
                    if random.random() < 0.5:
                        # Make the target be A or G
                        target_list[i] = random.choice(['A','G'])
                    else:
                        # Make the guide be A or G
                        guide_list[i] = random.choice(['A','G'])

        # Update what is stored for the target and guide with the
        # modified versions
        row['guide_wt'] = ''.join(target_list)
        row['guide_mutated'] = ''.join(guide_list)

        # Alter the PAM/PFS: if it was good, make it good; if it was bad,
        # make it bad
        # A 'good' PAM for Cas9 is 'NGG' on the 3' end; a 'good' PFS for
        # Cas13 is 'H' on the 5' end
        context_5 = row['guide_wt_context_5']
        context_3 = row['guide_wt_context_3']
        if context_3[1:3] == 'GG':
            # This guide has the canonical 'NGG' PAM
            # Mutate the PAM to some random trinucleotide
            context_3_list = list(context_3)
            context_3_list[0:3] = random.choices(['A','C','G','T'], k=3)
            context_3 = ''.join(context_3_list)

            # Give the guide a 'good' PFS on the 5' end -- i.e., give it
            # 'H' (not 'G')
            context_5_list = list(context_5)
            context_5_list[-1] = random.choice(['A','C','T'])
            context_5 = ''.join(context_5_list)
        else:
            # This guide does *not* have the canonical 'NGG' PAM
            # Mutate the PAM to some random trinucleotide
            context_3_list = list(context_3)
            context_3_list[0:3] = random.choices(['A','C','G','T'], k=3)
            context_3 = ''.join(context_3_list)

            # Give the guide a 'bad' PFS on the 5' end -- i.e., 'G'
            context_5_list = list(context_5)
            context_5_list[-1] = 'G'
            context_5 = ''.join(context_5_list)
        row['guide_wt_context_5'] = context_5
        row['guide_wt_context_3'] = context_3

        return super(Cas13SimulatedData, self)._gen_input_and_label(row)


_split_parser = None
def split(x, y, num_splits, shuffle_and_split=False, stratify_by_pos=False):
    """Split the data using stratified folds, for k-fold cross validation.

    Args:
        x: input data
        y: labels
        num_splits: number of folds
        shuffle_and_split: if True, shuffle before splitting and stratify based
            on the distribution of output variables (here, classes) to ensure they
            are roughly the same across the different folds
        stratify_by_pos: if True, determine the different folds based on
            the protein position of each data point (ensuring that the
            validate set is a contiguous region of the protein); this is
            similar to leave-one-gene-out cross-validation

    Iterates:
        (x_train_i, y_train_i, x_validate_i, y_validate_i) where each is
        for a fold of the data
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
            return sfk.split(x, y)
    elif stratify_by_pos:
        x_with_pos = [(xi, _split_parser.pos_for_input(xi)) for xi in x]
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
                _, test_index_idx_to_keep = _split_parser.make_nonoverlapping_datasets(
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
        x_train_i, y_train_i = x[train_index], y[train_index]
        x_validate_i, y_validate_i = x[test_index], y[test_index]
        yield (x_train_i, y_train_i, x_validate_i, y_validate_i)
