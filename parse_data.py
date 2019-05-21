"""Classes and methods to parse data for input to ML tools.
"""

import random

import numpy as np

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

    def __init__(self, subset=None, context_nt=20, split=(0.8, 0.1, 0.1),
            shuffle_seed=None):
        """
        Args:
            subset: if 'mismatch', only use data points representing guides
                with the canonical PAM but a mismatch to the target; if None,
                use all data points (including ones with perfect match and
                canonical PAM, and ones with wrong PAM but perfect match)
            context_nt: nt of target sequence context to include alongside
                each guide
            split: (train, validation, test) split; must sum to 1.0
            shuffle_seed: seed to use for the random module (used when
                shuffling input)
        """
        assert subset in (None, 'mismatch')
        self.subset = subset

        self.context_nt = context_nt

        assert sum(split) == 1.0
        self.split_train, self.split_validate, self.split_test = split

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

        if self.subset == 'mismatch':
            # Only keep rows where category is 'Mismatch'
            rows = [row for row in rows if row[header_idx['category']] == 'Mismatch']

        # Shuffle the rows
        random.shuffle(rows)

        # Generate input and labels for each row
        inputs_and_labels = []
        for row in rows:
            row_dict = {k: row[header_idx[k]] for k in header_idx.keys()}
            input_feats, label = self._gen_input_and_label(row_dict)
            inputs_and_labels += [(input_feats, label)]

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
                self._test_set += [inputs_and_labels[i]]

        self.was_read = True

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

