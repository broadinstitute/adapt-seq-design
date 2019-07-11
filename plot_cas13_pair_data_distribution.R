# Plot distribution of output variable describing Cas13 activity.
#
# This data is from Nick Haradhvala's library, tested using CARMEN, of
# guide/target pairs.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(reshape2)

IN.TABLE <- "data/CCF005_pairs_annotated.curated.tsv"
OUT.DIST.PDF <- "out/cas13-pair-activity-dist.pdf"
OUT.DIST.BLOCKS.PDF <- "out/cas13-pair-activity-dist.blocks.pdf"

# Read table and replace '_' in column names with '.'
all.data <- read.table(IN.TABLE, header=TRUE, sep="\t")
names(all.data) <- gsub("_", ".", names(all.data))

# Extract subset of data points corresponding to an 'experiment' (generally,
# a mismatch between guide/target, but not always)
guide.target.exp <- subset(all.data, type == "exp")

# Extract subset of data points corresponding to positive guide/target match
# (i.e., the wildtype target)
guide.target.pos <- subset(all.data, type == "pos")

# Extract subset of data points corresponding to negative guide/target match
# (i.e., high divergence between the two)
guide.target.neg <- subset(all.data, type == "neg")

# Melt all data and the different subsets into a single
# data frame
# Leave out guide.target.neg, which has the same value (-4) for
# out.logk.median at every data point; when including this, the density plot
# smooths by too much and doesn't show a strong peak at -4
df <- melt(list(all=all.data,
                guide.target.exp=guide.target.exp,
                guide.target.pos=guide.target.pos),
           id.vars=names(all.data))
names(df)[names(df) == "L1"] <- "dataset"

# Show a density plot for each dataset (all.data, guide.target.exp, etc.)
# In particular, show density of the output variable (out.logk.median)
p <- ggplot(df, aes(x=out.logk.median, fill=dataset, color=dataset))
# Use position='identity' to overlay plots
p <- p + geom_density(alpha=0.1, position='identity')
p + ggsave(OUT.DIST.PDF, width=8, height=8, useDingbats=FALSE)

# Make a separate plot showing a separate facet for each choice of crrna.block
# (which will be used to split data in train/validate/test)
p.faceted <- p + facet_wrap(. ~ crrna.block, scales="free")
p.faceted + ggsave(OUT.DIST.BLOCKS.PDF, width=16, height=16, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
