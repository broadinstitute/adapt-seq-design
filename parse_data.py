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
            subset: if 'guide-mismatch-and-good-pam', only use data points
                representing guides with the canonical PAM (NGG) but a
                mismatch to the target; if 'guide-match', only use data points
                representing guides that perfectly match the target but may or
                may not have the canonical PAM (NGG); if None, use all data
                points
            context_nt: nt of target sequence context to include alongside
                each guide
            split: (train, validation, test) split; must sum to 1.0
            shuffle_seed: seed to use for the random module to shuffle rows
                (if None, do not shuffle rows)
        """
        assert subset in (None, 'guide-mismatch-and-good-pam', 'guide-match')
        self.subset = subset

        self.context_nt = context_nt

        assert sum(split) == 1.0
        self.split_train, self.split_validate, self.split_test = split

        if shuffle_seed is None:
            self.shuffle_rows = False
        else:
            self.shuffle_rows = True
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

        # Shuffle the rows
        if self.shuffle_rows:
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


class Cas13SimulatedData(Doench2016Cas9ActivityParser):
    """Simulate Cas13 data from the Cas9 data.
    """
    def __init__(self, subset=None, context_nt=20, split=(0.8, 0.1, 0.1),
            shuffle_seed=None):
        super(Cas13SimulatedData, self).__init__(
                subset=subset, context_nt=context_nt,
                split=split, shuffle_seed=shuffle_seed)

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
        # so that they are 28 nt (Cas13 guides are 28 nt, Cas9 are 20 nt)
        for k in range(8):
            insert_pos = random.randint(0, len(target_list))
            b = random.choice(['A','C','G','T'])
            target_list.insert(insert_pos, b)
            guide_list.insert(insert_pos, b)

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

