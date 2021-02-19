#!/bin/bash

# Make file of curated CCF data where the only measurement is the median.

# Only keep the first 9 columns. Rename the 9th from 'out_logk_median' to
# 'out_logk_measurement'. Columns 10+ contain information to leave out.

IN="CCF-curated/CCF_merged_pairs_annotated.curated.tsv"
OUT="CCF-curated/CCF_merged_pairs_annotated.curated.median.tsv"

header=$(cat $IN | head -n 1 | cut -f-8)
header=$(echo -e "$header\tout_logk_measurement")

echo "$header" > $OUT

# Copy first 9 columns, leaving out the header
cat $IN | tail -n +2 | cut -f-9 >> $OUT

gzip $OUT
