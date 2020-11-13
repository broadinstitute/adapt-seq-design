CCF005_pairs_annotated.* contains data from our Cas13a library.
CCF005_pairs_droplets.filtered.csv.gz contains droplet-level measurements; here
'k' (actually, log(k)) represents the growth rate of an activity curve fit
(over time), 'B' is the starting value of the curve, 'C' is the saturation
point of the curve, and 'loss' is the loss function of the final fit (droplets
with high loss are already filtered).  See ../scripts/curate_ccf005_pairs.py
for details and curating the data.  CCF005 represents the first experiment for
Cas13/CARMEN: the one tested in Spring, 2019 with 0 or 1 mismatch.

CCF024 is like CCF005, but for the second experiment (Fall, 2019) with 4
CARMEN chips, expanding the number of mismatches. CCF024_targets.tsv had all
the experimental (MMM) targets; I added the WT and neg targets to this file,
taken from CCF005_pairs_annotated.csv. The file CCF024_crrnas.tsv was created
from CCF005_pairs_annotated.csv (since all crRNAs were the same in CCF024 as in
CCF005), and verified some info in curate_ccf024_pairs.py; note that this file
cannot contain the PFS because the PFS varies with the target (not fixed for a
given crRNA).

The ../CCF-curated/ directory contains files output by parsing the more 'raw'
data (e.g., filtering or reformatting or merging) directly in here.
