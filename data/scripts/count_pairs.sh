#!/bin/bash

# Count stats on guide-target pairs before curation.
# Note that this will include guides removed for technical reasons -- i.e.,
# it counts based on the library design, not what we used for the model.

# Combine CCF005 and CCF024
# Pull out of the following columns, in order: crRNA name, target name, replicate count
tmpf=$(mktemp)
cat <(tail -n +2 CCF005_pairs_annotated.csv) <(tail -n +2 CCF024_pairs_annotated.csv) | awk -F',' '{print $2"\t"$3"\t"$6}' > $tmpf

# Remove pairs where guide or target is a negative control
# Also remove Target_WT_{2,3,4} since these are equivalent to Target_WT_1
# Alow, where the guide is control{1,2,3,4} only keep Target_WT_1 since these positive control guides
# are identical to all targets
nonnegunique=$(mktemp)
cat $tmpf | grep -v 'neg_control' | grep -v 'Target_neg' | grep -v 'blank' | grep -v 'Target_WT_2\|Target_WT_3\|Target_WT_4' | awk '$1 !~ /^control/ || ($1 ~ /^control/ && $2 == "Target_WT_1") {print}' > $nonnegunique

numguides=$(cat $nonnegunique | cut -f1 | sort | uniq | wc -l)
numtargets=$(cat $nonnegunique | cut -f2 | sort | uniq | wc -l)
numuniquepairs=$(cat $nonnegunique | cut -f1,2 | sort | uniq | wc -l)
avgnumreplicates=$(cat $nonnegunique | cut -f3 | awk '{s+=$1} END {print s/NR}')

echo "Number of guides: $numguides"
echo "Number of targets: $numtargets"
echo "Number of unique pairs: $numuniquepairs"
echo "Average number of replicates: $avgnumreplicates"
echo "The number of expected unique pairs should be: [(87 experimental guides)*(226 targets) + (4 positive control guides)*(1 unique target) = 19,666 pairs; however, only 19,665 is reported above. This is because the pair (block12_guide23, Target_MMM_104) is missing, possibly because it had 0 replicates (by chance) in the data. I think it is simpler to report that we have (87 experimental guides)*(226 targets) = 19,662 experimental guide-target pairs."

rm $tmpf
rm $nonnegunique


# Also count the number of unique guide-target pairs included in the final
# dataset; for this, include the context around the target because this
# is included in the input to the model
numuniquedatasetpairs=$(zcat CCF-curated/CCF_merged_pairs_annotated.curated.resampled.tsv.gz | tail -n +2 | awk '$7!="neg" {print}' | cut -f1,3,4,5 | sort | uniq | wc -l)
numuniquedatasetpairswithoutcontext=$(zcat CCF-curated/CCF_merged_pairs_annotated.curated.resampled.tsv.gz | tail -n +2 | awk '$7!="neg" {print}' | cut -f1,3 | sort | uniq | wc -l)
echo "Number of unique pairs (incl. context) in dataset: $numuniquedatasetpairs"
echo "Number of unique pairs (not incl. context) in dataset: $numuniquedatasetpairswithoutcontext"
