# Plot distribution of output variable describing Cas13 activity.
#
# Note that most of these plots incorporate a data point for each technical
# replicate (i.e., measurement), in order to account for the measurement
# variability; they are generally not showing summary statistics across
# measurements for each guide-target pair.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(reshape2)
require(viridis)
require(ggridges)
require(stringr)
require(ggforce)    # for sina plots
require(ggpubr)

IN.TABLE <- "data/CCF-curated/CCF_merged_pairs_annotated.curated.resampled.tsv.gz"
OUT.DIST.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.pdf"
OUT.DIST.BLOCKS.FACETS.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.blocks.facets.pdf"
OUT.DIST.BLOCKS.RIDGES.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.blocks.ridges.pdf"
OUT.DIST.TRAIN.AND.TEST.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.train-and-test.pdf"
OUT.DIST.VARIATION.BETWEEN.AND.WITHIN.GUIDES.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.between-and-within-guides.pdf"
OUT.DIST.VARIATION.BETWEEN.AND.WITHIN.GUIDE.TARGET.PAIRS.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.between-and-within-guide-target-pairs.pdf"
OUT.DIST.GC.CONTENT.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.gc-content.pdf"
OUT.DIST.DIFF.FROM.WILDTYPE.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.diff-from-wildtype.pdf"
OUT.DIST.HAMMING.DIST.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.hamming-dist.pdf"
OUT.DIST.HAMMING.DIST.VIOLIN.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.hamming-dist.violin.pdf"
OUT.DIST.HAMMING.DIST.RIDGES.PDF <- "out/cas13/dataset/cas13-pair-activity-dist.hamming-dist.ridges.pdf"


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

# Add a column giving the mean wildtype activity for each guide (i.e., for each
# guide g, the mean activity across the wildtype/matching targets); use
# position to distinguish guides and, for each wildtype guide-target pair, use
# a summary statistic across the resampled technical replicates
wildtypes <- all.data[all.data$guide.target.hamming.dist == 0, ]
mean.wildtype.activity <- summarySE(wildtypes,
                                    measurevar="out.logk.measurement",
                                    groupvars=c("guide.pos.nt", "guide.seq"))
all.data$out.logk.wildtype.mean <- with(mean.wildtype.activity,
                                        out.logk.measurement[match(all.data$guide.pos.nt,
                                                              guide.pos.nt)])
all.data$diff.from.wildtype <- all.data$out.logk.measurement - all.data$out.logk.wildtype.mean

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

##############################################################################
# Plot separate distributions of exp, pos, and all data points

# Melt all data and the different subsets into a single
# data frame
# Note that 'All' refers to expandpos (i.e., it does exclude negatives)
df <- melt(list("All"=guide.target.expandpos,
                #guide.target.exp=guide.target.exp,
                "Wildtype"=guide.target.pos),
           id.vars=names(guide.target.expandpos))
names(df)[names(df) == "L1"] <- "dataset"

# Show a density plot for each dataset (all.data, guide.target.exp, etc.)
# In particular, show density of the output variable (out.logk.measurement)
p <- ggplot(df, aes(x=out.logk.measurement, fill=dataset, color=dataset))
# Use position='identity' to overlay plots
p <- p + geom_density(alpha=0.5, position='identity')
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
p <- p + labs(fill="", color="")    # remove legend title
p <- p + xlab("Activity") + ylab("Density")
p <- p + theme_pubr()
p + ggsave(OUT.DIST.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Make a separate plot showing a separate facet for each choice of crrna.block
# (which will be used to split data in train/validate/test)
p.faceted <- p + facet_wrap(. ~ crrna.block, scales="free")
p <- p + theme_pubr()
p.faceted + ggsave(OUT.DIST.BLOCKS.FACETS.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Make a plot showing the distribution of just exp-and-pos for each block,
# all with a common x-axis
guide.target.expandpos$crrna.block.factor <- factor(guide.target.expandpos$crrna.block)
p <- ggplot(guide.target.expandpos, aes(x=out.logk.measurement, y=crrna.block.factor))
p <- p + geom_density_ridges()
p <- p + xlab("Activity") + ylab("Block")
p <- p + theme_pubr()
p + ggsave(OUT.DIST.BLOCKS.RIDGES.PDF, width=8, height=48, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot a separate distribution of train data vs. test data
# This is drawn from the exp-and-pos dataset
p <- ggplot(guide.target.expandpos, aes(x=out.logk.measurement, fill=train.or.test, color=train.or.test))
p <- p + geom_density(alpha=0.5, position='identity')
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
p <- p + labs(fill="", color="")    # remove legend title
p <- p + xlab("Activity") + ylab("Density")
p <- p + theme_pubr()
p + ggsave(OUT.DIST.TRAIN.AND.TEST.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot an ordered dot plot (Cleveland dot plot)-like representation of the
# activity between and within (across targets) guides
# This is meant to show how the variation compares across guides vs. across
# targets for the same guide
# Each row corresponds to a guide (crRNA): we show a dot for its median
# activity across the targets, and also error bars across the targets (showing
# 20/80% quantiles); the rows are sorted such that the smallest median is on
# the bottom and the largest median is on the top

# Summarize activity across targets for each guide (crRNA); use the
# guide position (guide.pos.nt) as an identifier to group by
guide.expandpos.summarized <- summarySE(guide.target.expandpos,
                                        measurevar="out.logk.measurement",
                                        groupvars=c("guide.pos.nt"))

guide.expandpos.summarized$out.logk.wildtype.mean <- with(mean.wildtype.activity,
                                                          out.logk.measurement[match(guide.expandpos.summarized$guide.pos.nt,
                                                                               guide.pos.nt)])

# Add a column, order, giving the order of the guides (rows) sorted by mean
# out.logk.median value (in the column by that name)
guide.expandpos.summarized.ordered <- transform(guide.expandpos.summarized,
                                                order=rank(median, ties.method="first"))

# Add upper/lower bounds according to quantiles
guide.expandpos.summarized.ordered <- transform(guide.expandpos.summarized.ordered,
                                                lower=pctile.20,
                                                upper=pctile.80)

# Produce an ordered dot plot
p <- ggplot(guide.expandpos.summarized.ordered, aes(y=order))
p <- p + geom_errorbarh(aes(xmin=lower, xmax=upper), height=0, size=0.5, color="black", alpha=0.5)
p <- p + geom_point(aes(x=median), size=1)
p <- p + geom_point(aes(x=out.logk.wildtype.mean), color="#42075E", size=1)    # show a purple dot for mean wildtype activity of the guide
p <- p + xlab("Activity with variation is across targets") + ylab("Guide")
p <- p + theme_pubr()
p <- p + theme(axis.text.y=element_blank(), # y-axis text/ticks are meaningless
               axis.ticks.y=element_blank())
p + ggsave(OUT.DIST.VARIATION.BETWEEN.AND.WITHIN.GUIDES.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot an ordered dot plot (Cleveland dot plot)-like representation of the
# variation across technical replicates (measurements) for each guide-target pair
# Each row corresponds to a guide-target pair: we show a dot for its mean
# activity across measurements, and also error bars across (showing
# 95% CI); the rows are sorted such that the smallest mean is on
# the bottom and the largest median is on the top

# Summarize activity across guide-target pairs
# Include Hamming distance in the groupvars so that it remains in the
# summarized df
guide.target.expandpos.summarized <- summarySE(guide.target.expandpos,
                                               measurevar="out.logk.measurement",
                                               groupvars=c("guide.seq", "guide.pos.nt", "target.at.guide", "target.before", "target.after", "guide.target.hamming.dist"))


# Add a column, order, giving the order of the guides (rows) sorted by mean
# out.logk.measurement value (in the column by that name)
guide.target.expandpos.summarized.ordered <- transform(guide.target.expandpos.summarized,
                                                       order=rank(mean, ties.method="first"))

# Add upper/lower bounds according to quantiles
guide.target.expandpos.summarized.ordered <- transform(guide.target.expandpos.summarized.ordered,
                                                       lower=mean-ci,
                                                       upper=mean+ci)

# Produce an ordered dot plot
p <- ggplot(guide.target.expandpos.summarized.ordered, aes(y=order))
p <- p + geom_errorbarh(aes(xmin=lower, xmax=upper), height=0, size=0.5, color="black", alpha=0.5)
p <- p + geom_point(aes(x=mean), color="#42075E", size=1)
p <- p + xlab("Activity with 95% CI across replicate measurements") + ylab("Guide-target pair")
p <- p + theme_pubr()
p <- p + theme(axis.text.y=element_blank(), # y-axis text/ticks are meaningless
               axis.ticks.y=element_blank())
p + ggsave(OUT.DIST.VARIATION.BETWEEN.AND.WITHIN.GUIDE.TARGET.PAIRS.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot GC content of each guide vs. its mean activity across wildtype targets

# Compute GC content of each guide
mean.wildtype.activity <- transform(mean.wildtype.activity,
    gc.content=(str_count(guide.seq, "G") + str_count(guide.seq, "C")) / nchar(as.character(guide.seq)))

# Compute Spearman's rho for the mean values
spearman.rho <- cor(mean.wildtype.activity$gc.content,
           mean.wildtype.activity$mean,
           method="spearman")

# Produce a scatter plot
p <- ggplot(mean.wildtype.activity, aes(x=gc.content, y=mean))
p <- p + geom_point(aes(color=guide.pos.nt))
p <- p + scale_color_viridis() # adjust color gradient
p <- p + xlab("GC content") + ylab("Mean activity for guide")
p <- p + labs(color="Guide position on target")
p <- p + theme_pubr()
# Include text with the rho value
p <- p + annotate(geom='text', x=Inf, y=Inf, hjust=1, vjust=1, size=5,
                  label=as.character(as.expression(substitute(
                      rho~"="~spearman.rho, list(spearman.rho=format(spearman.rho, digits=3))))),
                  parse=TRUE)
p + ggsave(OUT.DIST.GC.CONTENT.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot density of (activity - [mean wildtype activity]) over different data
# subsets
# That is, for each (guide g, target t) pair with activity A, consider the mean
# wildtype activity for g (i.e., mean activity over all the targets matching
# g). This plots A - [mean wildtype activity for g]. This gives a measure of
# how different all the guide-target pair activities are from the wildtypes

p <- ggplot(df, aes(x=diff.from.wildtype, fill=dataset, color=dataset))
# Use position='identity' to overlay plots
p <- p + geom_density(alpha=0.5, position='identity')
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
p <- p + xlab("Activity minus guide's wildtype activity") + ylab("Density")
p <- p + labs(fill="", color="")    # remove legend title
p <- p + theme_pubr()
p + ggsave(OUT.DIST.DIFF.FROM.WILDTYPE.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot activity by guide-target Hamming distance
# Only do this for the exp-and-pos (non-negative control) data points.

# Only show up to 7 mismatches; beyond that there are few data points
guide.target.expandpos.trimmed <- guide.target.expandpos[guide.target.expandpos$guide.target.hamming.dist <= 7, ]

# Make factor out of Hamming distance
guide.target.expandpos.trimmed$guide.target.hamming.dist.factor <- factor(guide.target.expandpos.trimmed$guide.target.hamming.dist)

p <- ggplot(guide.target.expandpos.trimmed, aes(x=guide.target.hamming.dist.factor, y=out.logk.measurement))
p <- p + geom_sina(aes(group=guide.target.hamming.dist.factor), size=0.1)
p <- p + xlab("Guide-target distance") + ylab("Activity")
p <- p + theme_pubr()
p + ggsave(OUT.DIST.HAMMING.DIST.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot activity by guide-target Hamming distance with violin plots instead
# of sina
# Only do this for the exp-and-pos (non-negative control) data points.

p <- ggplot(guide.target.expandpos.trimmed, aes(x=guide.target.hamming.dist.factor, y=out.logk.measurement))
p <- p + geom_violin(aes(group=guide.target.hamming.dist.factor), fill="gray")
p <- p + xlab("Guide-target distance") + ylab("Activity")
p <- p + theme_pubr()
p + ggsave(OUT.DIST.HAMMING.DIST.VIOLIN.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot activity by guide-target Hamming distance with ridges (one per Hamming
# distance factor)
# Only do this for the exp-and-pos (non-negative control) data points.

p <- ggplot(guide.target.expandpos.trimmed, aes(y=guide.target.hamming.dist.factor, x=out.logk.measurement))
p <- p + geom_density_ridges()
p <- p + xlab("Activity") + ylab("Guide-target distance")
p <- p + theme_pubr()
p + ggsave(OUT.DIST.HAMMING.DIST.RIDGES.PDF, width=8, height=4, useDingbats=FALSE)
##############################################################################

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
