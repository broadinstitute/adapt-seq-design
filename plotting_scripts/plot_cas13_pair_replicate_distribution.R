# Plot distributions related to replicates of guide-target pairs.
#
# Unlike plot_cas13_pair_data_distribution.R, the input table to this
# script is *not* resampled data (so that it counts each replicatei value).
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(reshape2)
require(viridis)
require(ggpubr)

IN.TABLE <- "data/CCF-curated/CCF_merged_pairs_annotated.curated.tsv"
OUT.REPLICATE.COUNT.PDF <- "out/cas13/dataset/cas13-pair-replicate-counts.pdf"
OUT.REPLICATE.STDEV.PDF <- "out/cas13/dataset/cas13-pair-replicate-stdev.pdf"


## A helper function from:
##   http://www.cookbook-r.com/Graphs/Plotting_means_and_error_bars_(ggplot2)/#Helper%20functions
## Gives count, mean, standard deviation, standard error of the mean, and
## confidence interval (default 95%).
##   data: a data frame.
##   measurevar: the name of a column that contains the variable to be
##     summarized
##   groupvars: a vector containing names of columns that contain grouping
##     variables
##   na.rm: a boolean that indicates whether to ignore NA's
##   conf.interval: the percent range of the confidence interval (default is
##     95%)
summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
    library(plyr)

    # New version of length which can handle NA's: if na.rm==T, don't count
    # them
    length2 <- function (x, na.rm=FALSE) {
        if (na.rm) sum(!is.na(x))
        else       length(x)
    }

    # This does the summary. For each group's data frame, return a vector with
    # N, mean, and sd
    datac <- ddply(data, groupvars, .drop=.drop,
      .fun = function(xx, col) {
        c(N    = length2(xx[[col]], na.rm=na.rm),
          mean = mean   (xx[[col]], na.rm=na.rm),
          sd   = sd     (xx[[col]], na.rm=na.rm),
          median = median(xx[[col]], na.rm=na.rm),
          pctile.05 = quantile(xx[[col]], 0.05, na.rm=na.rm)[[1]],
          pctile.20 = quantile(xx[[col]], 0.20, na.rm=na.rm)[[1]],
          pctile.80 = quantile(xx[[col]], 0.80, na.rm=na.rm)[[1]],
          pctile.95 = quantile(xx[[col]], 0.95, na.rm=na.rm)[[1]]
        )
      },
      measurevar
    )

    # Add measurevar as another name for the "mean" column
    datac[, measurevar] <- datac$mean

    datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean

    # Confidence interval multiplier for standard error
    # Calculate t-statistic for confidence interval: 
    # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
    ciMult <- qt(conf.interval/2 + .5, datac$N-1)
    datac$ci <- datac$se * ciMult

    return(datac)
}


# Read table and replace '_' in column names with '.'
all.data <- read.table(gzfile(IN.TABLE), header=TRUE, sep="\t")
names(all.data) <- gsub("_", ".", names(all.data))

# Add a column giving whether the data point will get placed into the train
# or test set
# Currently the test set is the guide/targets where the guide positions
# are the 30% highest, which is nt position >= 629
TEST.START.POS <- 629
all.data$train.or.test <- ifelse(all.data$guide.pos.nt >= TEST.START.POS,
                                 'Test',
                                 'Train')
all.data$train.or.test <- factor(all.data$train.or.test)

# Extract subset of data points corresponding to an 'experiment' (generally,
# a mismatch between guide/target, but not always)
guide.target.exp <- subset(all.data, type == "exp")

# Extract subset of data points corresponding to positive guide/target match
# (i.e., the wildtype target)
guide.target.pos <- subset(all.data, type == "pos")

# Extract subset of data points corresponding to negative guide/target match
# (i.e., high divergence between the two)
guide.target.neg <- subset(all.data, type == "neg")

# Extract exp and pos (everything except negatives)
guide.target.expandpos <- subset(all.data, type == "exp" | type == "pos")

# Melt all data and the different subsets into a single
# data frame
# Note that 'All' refers to expandpos (i.e., it does exclude negatives)
df <- melt(list("All"=guide.target.expandpos,
                #guide.target.exp=guide.target.exp,
                "Wildtype"=guide.target.pos),
           id.vars=names(guide.target.expandpos))
names(df)[names(df) == "L1"] <- "dataset"


# Show a density plot of the number of replicates; do this across all data
# (including negatives)
p <- ggplot(all.data, aes(x=out.logk.replicate.count))
# Use position='identity' to overlay plots
p <- p + geom_density(alpha=0.8, color="black", fill="black", position='identity')
p <- p + xlab("Number of replicates") + ylab("Density")
p <- p + xlim(0, 45)    # leave out outliers
p <- p + theme_pubr()
p + ggsave(OUT.REPLICATE.COUNT.PDF, width=8, height=8, useDingbats=FALSE)

# Show a density plot of the standard deviation in measurement across
# replicates for each guide-target pair; do this across all data
# (including negatives)
p <- ggplot(all.data, aes(x=out.logk.stdev))
# Use position='identity' to overlay plots
p <- p + geom_density(alpha=0.8, color="black", fill="black", position='identity')
p <- p + xlab("Standard deviation of replicate measurements") + ylab("Density")
p <- p + theme_pubr()
p + ggsave(OUT.REPLICATE.STDEV.PDF, width=8, height=8, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
