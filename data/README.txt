doench2016-nbt.supp-table-18.* contains data from Doench et al. 2016
in NBT, and CD33-target-sequence.fasta contains a sequence targeted
in the corresponding experiment; see scripts/curate_doench2016-nbt.supp-table-18.sh for
details on the experiment, data, and curating it.

CCF005_pairs_annotated.* contains data from Nick Haradhvala's library
on Cas13. CCF005_pairs_droplets.filtered.csv.gz contains droplet-level
measurements; here 'k' (actually, log(k)) represents the growth rate of
an activity curve fit (over time), 'B' is the starting value of the curve,
'C' is the saturation point of the curve, and 'loss' is the loss function
of the final fit (droplets with high loss are already filtered).
See scripts/curate_ccf005_pairs.py for details and curating the data.
