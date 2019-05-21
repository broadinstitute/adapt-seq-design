# Plot distribution of output variable describing Cas9 activity.
#
# This should be comparable to Figure 5b in Doench et al. 2016, except
# the data here is a curated subset of that data.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(reshape2)

IN.TABLE <- "data/doench2016-nbt.supp-table-18.curated.with-context.tsv"
OUT.PDF <- "out/activity-dist.pdf"

# Read table and replace '_' in column names with '.'
all.data <- read.table(IN.TABLE, header=TRUE, sep="\t")
names(all.data) <- gsub("_", ".", names(all.data))

# Extract the subset of data points with perfect match and
# canonical NGG PAM
canonical.match <- subset(all.data, category == "PAM" & grepl(".GG", annotation))

# Extract subset of data points with perfect match but
# not the canonical NGG PAM
wrong.pam <- subset(all.data, category == "PAM" & !grepl(".GG", annotation))

# Extract subset of data points with mismatch to target
mismatch <- subset(all.data, category == "Mismatch")

# Melt all data and the different subsets into a single
# data frame
df <- melt(list(all=all.data, canonical.match=canonical.match,
                wrong.pam=wrong.pam, mismatch=mismatch),
           id.vars=names(all.data))
names(df)[names(df) == "L1"] <- "dataset"

# Show a density plot for each dataset (all.data, canonical.match, etc.)
# In particular, show density of the output variable (day21.minus.etp)
p <- ggplot(df, aes(x=day21.minus.etp, fill=dataset, color=dataset))
# Use position='identity' to overlay plots
p <- p + geom_density(alpha=0.25, position='identity')

p <- p + ggsave(OUT.PDF, width=8, height=8, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
