# Plot distribution of output variable describing Cas13 activity.
#
# This data is from Nick Haradhvala's library, tested using CARMEN, of
# guide/target pairs.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(reshape2)
require(viridis)
require(ggridges)

IN.TABLE <- "data/CCF005_pairs_annotated.curated.tsv"
OUT.DIST.PDF <- "out/cas13-pair-activity-dist.pdf"
OUT.DIST.BLOCKS.FACETS.PDF <- "out/cas13-pair-activity-dist.blocks.facets.pdf"
OUT.DIST.BLOCKS.RIDGES.PDF <- "out/cas13-pair-activity-dist.blocks.ridges.pdf"
OUT.DIST.TRAIN.AND.TEST.PDF <- "out/cas13-pair-activity-dist.train-and-test.pdf"
OUT.DIST.VARIATION.BETWEEN.AND.WITHIN.GUIDES.PDF <- "out/cas13-pair-activity-dist.between-and-within-guides.pdf"


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
          pctile.95 = quantile(xx[[col]], 0.95, na.rm=na.rm)[[1]]
        )
      },
      measurevar
    )

    # Rename the "mean" column    
    datac <- rename(datac, c("mean" = measurevar))

    datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean

    # Confidence interval multiplier for standard error
    # Calculate t-statistic for confidence interval: 
    # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
    ciMult <- qt(conf.interval/2 + .5, datac$N-1)
    datac$ci <- datac$se * ciMult

    return(datac)
}


# Read table and replace '_' in column names with '.'
all.data <- read.table(IN.TABLE, header=TRUE, sep="\t")
names(all.data) <- gsub("_", ".", names(all.data))

# Add a column giving whether the data point will get placed into the train
# or test set
# Currently the test set is the guide/targets where the guide positions
# are the 30% highest, which is nt position >= 629
TEST.START.POS <- 629
all.data$train.or.test <- ifelse(all.data$guide.pos.nt >= TEST.START.POS,
                                 'test',
                                 'train')
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

##############################################################################
# Plot separate distributions of exp, pos, and all data points

# Melt all data and the different subsets into a single
# data frame
df <- melt(list(guide.target.expandpos=guide.target.expandpos,
                guide.target.exp=guide.target.exp,
                guide.target.pos=guide.target.pos),
           id.vars=names(guide.target.expandpos))
names(df)[names(df) == "L1"] <- "dataset"

# Show a density plot for each dataset (all.data, guide.target.exp, etc.)
# In particular, show density of the output variable (out.logk.median)
p <- ggplot(df, aes(x=out.logk.median, fill=dataset, color=dataset))
# Use position='identity' to overlay plots
p <- p + geom_density(alpha=0.5, position='identity')
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
p <- p + xlab("Activity") + ylab("Density")
p + ggsave(OUT.DIST.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Make a separate plot showing a separate facet for each choice of crrna.block
# (which will be used to split data in train/validate/test)
p.faceted <- p + facet_wrap(. ~ crrna.block, scales="free")
p.faceted + ggsave(OUT.DIST.BLOCKS.FACETS.PDF, width=16, height=16, useDingbats=FALSE)
##############################################################################

##############################################################################
# Make a plot showing the distribution of just exp-and-pos for each block,
# all with a common x-axis
guide.target.expandpos$crrna.block.factor <- factor(guide.target.expandpos$crrna.block)
p <- ggplot(guide.target.expandpos, aes(x=out.logk.median, y=crrna.block.factor))
p <- p + geom_density_ridges()
p <- p + xlab("Activity") + ylab("Block")
p + ggsave(OUT.DIST.BLOCKS.RIDGES.PDF, width=8, height=48, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot a separate distribution of train data vs. test data
# This is drawn from the exp-and-pos dataset
p <- ggplot(guide.target.expandpos, aes(x=out.logk.median, fill=train.or.test, color=train.or.test))
p <- p + geom_density(alpha=0.5, position='identity')
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
p <- p + xlab("Activity") + ylab("Density")
p + ggsave(OUT.DIST.TRAIN.AND.TEST.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

##############################################################################
# Plot an ordered dot plot (Cleveland dot plot)-like representation of the
# activity between and within (across targets) guides
# This is meant to show how the variation compares across guides vs. across
# targets for the same guide
# Each row corresponds to a guide (crRNA): we show a dot for its mean
# activity across the targets, and also error bars across the targets (showing
# standard deviation); the rows are sorted such that the smallest mean is on
# the bottom and the largest mean is on the top

# Summarize activity across targets for each guide (crRNA); use the
# guide position (guide.pos.nt) as an identifier to group by
guide.target.expandpos.summarized <- summarySE(guide.target.expandpos,
                                               measurevar="out.logk.median",
                                               groupvars=c("guide.pos.nt"))

# Add a column, order, giving the order of the guides (rows) sorted by mean
# out.logk.median value (in the column by that name)
guide.target.expandpos.summarized.ordered <- transform(guide.target.expandpos.summarized,
                                                       order=rank(out.logk.median, ties.method="first"))

# Add upper/lower bounds according to standard deviation
guide.target.expandpos.summarized.ordered <- transform(guide.target.expandpos.summarized.ordered,
                                                       lower=out.logk.median-sd,
                                                       upper=out.logk.median+sd)

# Produce an ordered dot plot
p <- ggplot(guide.target.expandpos.summarized.ordered, aes(x=out.logk.median, y=order))
p <- p + geom_errorbarh(aes(xmin=lower, xmax=upper), height=0, size=0.5, color="black", alpha=0.5)
p <- p + geom_point(size=1)
p <- p + xlab("Activity (variation is across targets)") + ylab("crRNA")
p <- p + theme(axis.text.y=element_blank(), # y-axis text/ticks are meaningless
               axis.ticks.y=element_blank())
p + ggsave(OUT.DIST.VARIATION.BETWEEN.AND.WITHIN.GUIDES.PDF, width=8, height=8, useDingbats=FALSE)
##############################################################################

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
