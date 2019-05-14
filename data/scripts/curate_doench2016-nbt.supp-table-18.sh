#!/bin/bash

# This data is from:
# Doench et al. 2016 (Nature Biotechnology volume 34,
# pages 184–191 (2016); https://doi.org/10.1038/nbt.3437)
#
# It is from Supplementary Table 18, which presents data on
# a Cas9 screen for off-target effects by targeting across the
# human CD33 coding sequence:
# ```
#  We sought to under- stand off-target effects under typical
#  pooled-screening conditions in mammalian cells using a library
#  targeting the coding sequence of human CD33 with all possible sgRNAs,
#  regardless of PAM. For all sites with the canonical NGG PAM, in
#  addition to the perfect-match sgRNAs, we introduced three types of
#  sgRNA mutations: first, all 1-nucleotide deletions; second, all
#  1-nucleotide insertions; third, all 1-nucleotide mismatches to the
#  target DNA, generating a library with 27,897 unique sgRNAs. To these
#  we added 10,618 sgRNAs targeting the mouse Thy1 locus to serve as
#  negative controls.
# ```
#
# Human CD33 targets are annotated with `ENST00000262262` and mouse
# (negative control) targets are annotated with `ENSMUST00000114840`
#
# The paper looked at the activity profile of perfect match sgRNAs with
# the canonical NGG PAM, to find a region in CD33 where targeting leads
# to loss of function. This is shown in Supplementary Figure 19: it
# is where the percentage of the protein (i.e., how far along the
# coding sequence) is about 5% through 59%. To be safe, we'll only
# look where the target site is between 7% and 57%. This percentage is
# given in a column of the data.
#
# Also, we'll only want to look at perfect matches, varied PAMs, and
# mismatches (so leave out insertions and deletions).
#
# In case we rely on order later (e.g., to split train/test), this
# should also randomly shuffle the rows.

IN="doench2016-nbt.supp-table-18.tsv"
OUT="doench2016-nbt.supp-table-18.curated.tsv"

# Filter data for the above characteristics

head -n 1 $IN > $OUT
tail -n +2 $IN | awk '$5=="ENST00000262262" && $6>=7.0 && $6<=57.0 && ($8 == "PAM" || $8=="Mismatch") {print}' | shuf >> $OUT

####################

OUT_WITH_CONTEXT="doench2016-nbt.supp-table-18.curated.with-context.tsv"

# Write a header for the table with sequence context
echo -e "barcode\tguide_mutated\tguide_wt\tguide_wt_context_5\tguide_wt_context_3\tguide_wt_strand\tannotation\ttranscript_id\tprotein_pos_pct\tday21_minus_etp\tcategory" > $OUT_WITH_CONTEXT

# Look up each wildtype guide sequence in the targeted sequence
# (human CD33 coding sequence), which has accession NM_001772.4
#
# First, convert the FASTA for NM_001772.4 into a sequence all on one line
cat CD33-target-sequence.fasta | grep -v '>' | tr '\n' ' ' | sed 's/ //g' | sed '/^$/d' > /tmp/CD33-target-sequence.one-line.txt

# Some guides will target the other strand, so take the reverse
# complement of this sequence as well to search against
cat /tmp/CD33-target-sequence.one-line.txt | rev | tr "ACGT" "TGCA" > /tmp/CD33-target-sequence.one-line.rc.txt

# Lookup each wildtype sequence, and write a new table that includes
# the context of the sequence
# For each grep, also find 20 nt on each side of the result (i.e.,
# to get sequence context); because we are restricting the range of the
# sequence above (7% to 57%), results should not be at the ends and there
# should be at least 20 nt on each side
while read -r row; do
    barcode=$(echo "$row" | awk '{print $1}')
    mutated_guide=$(echo "$row" | awk '{print $2}')
    wildtype_guide=$(echo "$row" | awk '{print $3}')
    annotation=$(echo "$row" | awk '{print $4}')
    transcript_id=$(echo "$row" | awk '{print $5}')
    protein_pos_pct=$(echo "$row" | awk '{print $6}')
    day21_minus_etp=$(echo "$row" | awk '{print $7}')
    category=$(echo "$row" | awk '{print $8}')

    # grep the forward strand; use '.{0,20}' (with -P) for 20 chars before and after
    grep_forward=$(grep -o -P ".{0,20}$wildtype_guide.{0,20}" /tmp/CD33-target-sequence.one-line.txt)
    if [ -z "$grep_forward" ]; then
        # No results on the forward, so check the reverse complement strand
        grep_rc=$(grep -o -P ".{0,20}$wildtype_guide.{0,20}" /tmp/CD33-target-sequence.one-line.rc.txt)
        if [ -z "$grep_rc" ]; then
            # No results on either strand
            # Just skip this row
            continue
        fi
        result="$grep_rc"
        strand="rc"
    else
        result="$grep_forward"
        strand="forward"
    fi

    # Extract the sequence (20 nt) on the 5' and 3' ends
    context_5=${result:0:20}
    context_3=${result:(-20)}

    # Write a new row containing the sequence context and strand
    # that contains the wildtype guide
    echo -e "$barcode\t$mutated_guide\t$wildtype_guide\t$context_5\t$context_3\t$strand\t$annotation\t$transcript_id\t$protein_pos_pct\t$day21_minus_etp\t$category" >> $OUT_WITH_CONTEXT
done < <(tail -n +2 $OUT)

# Remove the one-line tmp sequence
rm /tmp/CD33-target-sequence.one-line.txt
rm /tmp/CD33-target-sequence.one-line.rc.txt
