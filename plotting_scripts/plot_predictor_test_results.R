# Plot results of testing predictor.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(dplyr)
require(ggridges)
require(viridis)

args <- commandArgs(trailingOnly=TRUE)
IN.TSV <- args[1]
OUT.PDF <- args[2]


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
          sd   = sd     (xx[[col]], na.rm=na.rm)
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


# Read TSV of results and replace '_' in column names with '.'
test.results <- read.table(IN.TSV, header=TRUE, sep="\t")
names(test.results) <- gsub("_", ".", names(test.results))

# Melt across true/predicted values
test.results.melted <- melt(test.results,
                            measure.vars=c("true.activity", "predicted.activity"),
                            variable.name="activity.type",
                            value.name="activity")

#####################################################################
# Plot distribution of true activity and predicted activity
# Show as density plots

p <- ggplot(test.results.melted, aes(x=activity, fill=activity.type))
p <- p + geom_density(alpha=0.5, position='identity')
p <- p + xlab("Activity") + ylab("Density")
p <- p + theme_bw(base_size=18) # bw & larger font sizes
p <- p + theme(legend.justification=c(0,1), # place legend in upper-left
               legend.position=c(0.1,0.9))
p.output.dist <- p
#####################################################################

#####################################################################
# Plot true activity value vs. predicted activity value
# This is a scatter plot, where each dot represents a target/crRNA
# pair (test data point)
# The points are colored by crRNA position along the target: ones with
# the same color are the same crRNA, ones with similar colors likely
# overlap

p <- ggplot(test.results, aes(x=true.activity, y=predicted.activity))
p <- p + geom_point(aes(color=crrna.pos))
p <- p + scale_color_viridis() # adjust color gradient
#p <- p + xlim(-2.5, 0) + ylim(-2.5, 0)  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_bw(base_size=18) # bw & larger font sizes
p <- p + theme(legend.justification=c(0,1), # place legend in upper-left
               legend.position=c(0.05,0.95))
p.true.vs.predicted <- p
#####################################################################

#####################################################################
# Plot true activity value for different predicted quantiles
# y-axis shows the quantile (grouped) for the predicted activity and
# x-axis shows the distribution of true activity values in that
# quantile
# Use quartiles here (4 groupings)

# Create a separate data frame with a quantile factor that also
# includes 'all' for all data points
test.results.with.quantile <- data.frame(test.results)
test.results.with.quantile$predicted.quantile <- ntile(test.results.with.quantile$predicted.activity, 4)
test.results.all <- data.frame(test.results)
test.results.all$predicted.quantile <- rep("all", n=nrow(test.results.all))
test.results.with.quantile <- rbind(test.results.with.quantile,
                                    test.results.all)
test.results.with.quantile$predicted.quantile <- factor(test.results.with.quantile$predicted.quantile,
                                                        levels=c("all", "1", "2", "3", "4"))

p <- ggplot(test.results.with.quantile, aes(x=true.activity, y=predicted.quantile))
p <- p + xlab("Activity") + ylab("Quartile")
p <- p + geom_density_ridges()
p <- p + theme_bw(base_size=18)    # bw & larger font sizes
p.by.predicted.quantile <- p
#####################################################################

#####################################################################
# Plot the prediction within each crRNA (where each is decided by its
# position in the target)
# Plot a separate facet for each crRNA (this is basically like making
# a separate plot for each color in the scatter plot above)
# Also, make a density plot of the distribution of Spearman rho
# rank correlation coefficients across the crRNAs

facet.ncol <- floor(sqrt(length(unique(test.results$crrna.pos)))) + 1
p <- ggplot(test.results, aes(x=true.activity, y=predicted.activity))
p <- p + geom_point()
p <- p + facet_wrap( ~ crrna.pos, ncol=facet.ncol)
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_bw(base_size=18) # bw & larger font sizes
p <- p + theme(aspect.ratio=1)  # square facets
p.true.vs.predicted.faceted.by.crrna <- p

# Compute a rank correlation coefficient for each crRNA
rho.for.crrna <- do.call(rbind, lapply(unique(test.results$crrna.pos),
    function(crrna.pos) {
        # Compute rho (Spearman rank correlation) for crrna.pos
        test.results.for.crrna <- test.results[test.results$crrna.pos == crrna.pos, ]
        rho <- cor(test.results.for.crrna$true.activity,
                   test.results.for.crrna$predicted.activity,
                   method="spearman")
        return(data.frame(crrna.pos=crrna.pos,
                          rho=rho))
    }
))
p <- ggplot(rho.for.crrna, aes(x=rho))
p <- p + geom_density(fill="gray")
p <- p + xlab("Spearman rho") + ylab("Density")
p <- p + theme_bw(base_size=18) # bw & larger font sizes
p.rho.across.crrnas <- p
#####################################################################

#####################################################################
# Plot the mean true activity vs. mean predicted value, where the mean
# is computed for each crRNA (taken across the targets)
# This is a scatter plot with one dot for each crRNA

test.results.summarized.over.targets <- summarySE(test.results.melted,
                                                  measurevar="activity",
                                                  groupvars=c("crrna.pos", "activity.type"))
test.results.summarized.over.targets.true <- test.results.summarized.over.targets[test.results.summarized.over.targets$activity.type == "true.activity", ]
test.results.summarized.over.targets.predicted <- test.results.summarized.over.targets[test.results.summarized.over.targets$activity.type == "predicted.activity", ]
test.results.summarized.over.targets <- merge(test.results.summarized.over.targets.true,
                                              test.results.summarized.over.targets.predicted,
                                              by="crrna.pos")
names(test.results.summarized.over.targets) <- gsub("\\.x", ".true", names(test.results.summarized.over.targets))
names(test.results.summarized.over.targets) <- gsub("\\.y", ".predicted", names(test.results.summarized.over.targets))

# Compute Spearman's rho for the mean values
spearman.rho <- cor(test.results.summarized.over.targets$activity.true,
           test.results.summarized.over.targets$activity.predicted,
           method="spearman")

p <- ggplot(test.results.summarized.over.targets, aes(x=activity.true, y=activity.predicted))
p <- p + geom_point(aes(color=crrna.pos), size=5)
p <- p + geom_errorbarh(aes(xmin=activity.true-sd.true, xmax=activity.true+sd.true), alpha=0.5)
p <- p + geom_errorbar(aes(ymin=activity.predicted-sd.predicted, ymax=activity.predicted+sd.predicted), alpha=0.5)
p <- p + scale_color_viridis() # adjust color gradient
#p <- p + xlim(-2.5, 0) + ylim(-2.5, 0)  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_bw(base_size=18) # bw & larger font sizes
p <- p + theme(legend.justification=c(0,1), # place legend in upper-left
               legend.position=c(0.05,0.95))
# Include text with the rho value
p <- p + annotate(geom='text', x=Inf, y=Inf, hjust=1, vjust=1, size=5,
                  label=as.character(as.expression(substitute(
                      rho~"="~spearman.rho, list(spearman.rho=format(spearman.rho, digits=3))))),
                  parse=TRUE)
p.true.vs.predicted.summarized <- p
#####################################################################

#####################################################################
# Produce PDF
g <- arrangeGrob(p.output.dist,
                 p.true.vs.predicted,
                 p.by.predicted.quantile,
                 p.true.vs.predicted.faceted.by.crrna,
                 p.rho.across.crrnas,
                 p.true.vs.predicted.summarized,
                 ncol=1)
ggsave(OUT.PDF, g, width=8, height=56, useDingbats=FALSE, limitsize=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
#####################################################################
