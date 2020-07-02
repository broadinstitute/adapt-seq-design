#!/bin/bash

# Combine several files, from two experiments, to generate a file giving all
# guides in the library and another giving all targets in the library.
# Note that this includes *all* that we designed; it does not exclude
# guides that we filtered out.

# Make list of guides (spacer sequences); this is easy because it was
# the same across both experiments and we already have the list.
fn_guides="CCF-curated/library.guides.tsv"
echo -e "name\ttype\tspacer_seq" > $fn_guides
cat CCF024_crrnas.tsv | tail -n +2 | awk -F'\t' '{print $1"\t"$4"\t"$3}' >> $fn_guides

# Make list of targets; here, we have to combine from two experiments
fn_targets="CCF-curated/library.targets.tsv"
echo -e "name\ttype\ttarget_seq" > $fn_targets
t=$(mktemp)
cat CCF005_pairs_annotated.csv | tail -n +2 | awk -F',' '{print $3"\t"$16"\t"$15}' > $t
cat CCF024_targets.tsv | grep '^Target_MMM' | awk -F'\t' '{print $1"\texp\t"$2}' >> $t
cat CCF024_targets.tsv | grep '^Target_WT' | awk -F'\t' '{print $1"\tpos\t"$2}' >> $t
cat CCF024_targets.tsv | grep '^Target_neg' | awk -F'\t' '{print $1"\tneg\t"$2}' >> $t
cat $t | sort | uniq >> $fn_targets
rm $t
