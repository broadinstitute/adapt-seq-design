# Plot results of testing predictor.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(ggridges)
require(viridis)
require(ggsignif)
require(ggpubr)
require(ggstance)

args <- commandArgs(trailingOnly=TRUE)
IN.TSV <- args[1]
OUT.DIR <- args[2]

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

# Find test results summarized for each crRNA (across targets) and
# activity type
test.results.summarized.over.targets <- summarySE(test.results.melted,
                                                  measurevar="activity",
                                                  groupvars=c("crrna.pos", "activity.type"))

# Average across technical replicates (differences in the true activity); there
# may also be minor differences in the predicted activities due to numerical
# calculations
require(dplyr)
test.results.without.error <- test.results %>% group_by(target, guide) %>%
    mutate(true.activity=mean(true.activity), predicted.activity=mean(predicted.activity))

# Because it is hard to visualize, print regression results
# for different choices of guide-target Hamming distance and PFS
# And save them
metrics <- function(x, y) {
    r <- cor(x, y, method="pearson")
    rho <- cor(x, y, method="spearman")
    return(list(r=r, rho=rho, str=paste0("r=", r, "; rho=", rho)))
}
print("Metrics:")
print("  All data points:")
all.metrics <- metrics(test.results$true.activity, test.results$predicted.activity)
print(paste0("    ", all.metrics$str))
print("  Each value of Hamming distance:")
hd.rho <- data.frame(hamming.dist=c(), rho=c())
for (hd in sort(unique(test.results$hamming.dist))) {
    tr.hd <- test.results[test.results$hamming.dist == hd, ]
    hd.metrics <- metrics(tr.hd$true.activity, tr.hd$predicted.activity)
    print(paste0("    Dist=", hd, ": ", hd.metrics$str))
    hd.rho <- rbind(hd.rho, data.frame(hamming.dist=c(hd), rho=c(hd.metrics$rho)))
}
print("  Each PFS:")
pfs.rho <- data.frame(cas13a.pfs=c(), rho=c())
for (pfs in sort(unique(test.results$cas13a.pfs))) {
    tr.pfs <- test.results[test.results$cas13a.pfs == pfs, ]
    pfs.metrics <- metrics(tr.pfs$true.activity, tr.pfs$predicted.activity)
    print(paste0("    PFS=", pfs, ": ", pfs.metrics$str))
    pfs.rho <- rbind(pfs.rho, data.frame(cas13a.pfs=c(pfs), rho=c(pfs.metrics$rho)))
}
all.metrics.without.error <- metrics(test.results.without.error$true.activity,
                                     test.results.without.error$predicted.activity)

# Determine activity range for plots, so axes have the same range
# Round to the nearest 0.5
lo <- floor(min(test.results$true.activity, test.results$predicted.activity) * 2) / 2
hi <- ceiling(max(test.results$true.activity, test.results$predicted.activity) * 2) / 2
ACTIVITY.RANGE <- c(lo, hi)

# Also have an option to fix the range -- this may cut out a few points
# when they are predicted to have a value outside the range of the true values
ACTIVITY.RANGE.SIMPLE <- c(-4, 0)

#####################################################################
# Plot distribution of true activity and predicted activity
# Show as density plots

test.results.melted$activity.type.formatted <- NA
test.results.melted$activity.type.formatted[test.results.melted$activity.type == "predicted.activity"] <- "Predicted activity"
test.results.melted$activity.type.formatted[test.results.melted$activity.type == "true.activity"] <- "True activity"

p <- ggplot(test.results.melted, aes(x=activity, fill=activity.type.formatted))
p <- p + geom_density(alpha=0.5, position='identity')
p <- p + xlab("Activity") + ylab("Density")
p <- p + scale_fill_viridis(name="", discrete=TRUE) # name="" to not label name of legend
p <- p + theme_pubr()
p.output.dist <- p
#####################################################################

#####################################################################
# Plot true activity value vs. predicted activity value
# This is a scatter plot, where each dot represents a target/crRNA
# pair (test data point)
# The points are colored by crRNA position along the target: ones with
# the same color are the same crRNA, ones with similar colors likely
# overlap

all.rho.val.str <- format(all.metrics$rho, digits=3)
all.rho.expr <- as.expression(bquote(rho~"="~.(all.rho.val.str)))

p <- ggplot(test.results, aes(x=true.activity, y=predicted.activity))
p <- p + geom_point(aes(color=crrna.pos), alpha=0.5, stroke=0)
p <- p + scale_color_viridis(name="Position along target") # adjust color gradient
p <- p + xlim(ACTIVITY.RANGE) + ylim(ACTIVITY.RANGE)  # make ranges be the same
p <- p + coord_fixed()  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_pubr()
# Include text with the rho value
p <- p + annotate(geom="text", label=all.rho.expr,
                  x=Inf, y=Inf, hjust=1, vjust=1, size=3)
p.true.vs.predicted <- p
#####################################################################

#####################################################################
# Plot true activity value vs. predicted activity value with density
# contours
# This is useful because on the scatter plot many points overlap
# and it is hard to read

all.rho.val.str <- format(all.metrics$rho, digits=3)
all.rho.expr <- as.expression(bquote(rho~"="~.(all.rho.val.str)))

p <- ggplot(test.results, aes(x=true.activity, y=predicted.activity))
#p <- p + geom_point(shape='.', color='black', size=0.0005, stroke=0, alpha=0.1) # show the points as small black dots, behind the density map
#p <- p + stat_density_2d(aes(fill=stat(level)), geom='polygon', contour=TRUE, n=c(1000, 1000), h=NULL, adjust=1.75)    # h=c(0.6, 0.6) works too
p <- p + stat_density_2d(aes(fill=stat(level)), geom="polygon", contour=TRUE, n=c(1000, 1000), h=NULL, adjust=1.75)    # h=c(0.6, 0.6) works too
p <- p + stat_density_2d(aes(fill=stat(level)), color="#5E5E5E", alpha=0.2, size=0.05, contour=TRUE, n=c(1000, 1000), h=NULL, adjust=1.75)    # outline around contours; h=c(0.6, 0.6) works too
#p <- p + stat_density_2d(aes(fill=stat(density)), geom='raster', contour=FALSE) # density heatmap
#p <- p + scale_fill_viridis(name="Level", breaks=c(0.2, 0.5)) # viridis colors (purple=low, yellow=high); specify tick labels on legend bar
p <- p + scale_fill_gradient(name="Level", breaks=c(0.2, 0.5), low="#FDF5FF", high="#470E55") # customize colors; specify tick labels on legend bar
p <- p + xlim(ACTIVITY.RANGE.SIMPLE) + ylim(ACTIVITY.RANGE.SIMPLE)  # make ranges be the same
p <- p + coord_fixed()  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_pubr()
# Include text with the rho value
p <- p + annotate(geom="text", label=all.rho.expr,
                  x=Inf, y=Inf, hjust=1, vjust=1, size=3)
p.true.vs.predicted.density.contours <- p
#####################################################################

#####################################################################
# Plot true activity value vs. predicted activity value with density
# contours, without including measurement error in true activities

all.rho.val.str.without.error <- format(all.metrics.without.error$rho, digits=3)
all.rho.expr.without.error <- as.expression(bquote(rho~"="~.(all.rho.val.str.without.error)))

p <- ggplot(test.results.without.error, aes(x=true.activity, y=predicted.activity))
p <- p + stat_density_2d(aes(fill=stat(level)), geom="polygon", contour=TRUE, n=c(1000, 1000), h=NULL, adjust=1.75)    # h=c(0.6, 0.6) works too
p <- p + stat_density_2d(aes(fill=stat(level)), color="#5E5E5E", alpha=0.2, size=0.05, contour=TRUE, n=c(1000, 1000), h=NULL, adjust=1.75)    # outline around contours; h=c(0.6, 0.6) works too
p <- p + scale_fill_gradient(name="Level", breaks=c(0.2, 0.5), low="#FDF5FF", high="#470E55") # customize colors; specify tick labels on legend bar
p <- p + xlim(ACTIVITY.RANGE.SIMPLE) + ylim(ACTIVITY.RANGE.SIMPLE)  # make ranges be the same
p <- p + coord_fixed()  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_pubr()
# Include text with the rho value
p <- p + annotate(geom="text", label=all.rho.expr.without.error,
                  x=Inf, y=Inf, hjust=1, vjust=1, size=3)
p.true.vs.predicted.density.contours.without.error <- p
#####################################################################

#####################################################################
# Plot true activity value vs. predicted activity value
# This is a scatter plot, where each dot represents a target/crRNA
# pair (test data point)
# The points are colored by guide-target Hamming distance

p <- ggplot(test.results, aes(x=true.activity, y=predicted.activity))
p <- p + geom_point(aes(color=hamming.dist), alpha=0.5, stroke=0)
p <- p + scale_color_viridis(name="Hamming distance") # adjust color gradient
p <- p + xlim(ACTIVITY.RANGE) + ylim(ACTIVITY.RANGE)  # make ranges be the same
p <- p + coord_fixed()  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_pubr()
p.true.vs.predicted.colored.by.hamming.dist <- p
#####################################################################

#####################################################################
# Plot true activity value vs. predicted activity value
# This is a scatter plot, where each dot represents a target/crRNA
# pair (test data point)
# There is a facet for each guide-target Hamming distance
# Only show up to Hamming distance of 5; there are few points beyond that

hd.rho$rho.val.str <- format(hd.rho$rho, digits=3)

test.results.hd.limit <- test.results[test.results$hamming.dist <= 5,]
hd.rho.hd.limit <- hd.rho[hd.rho$hamming.dist <= 5,]

#facet.ncol <- floor(sqrt(length(unique(test.results$hamming.dist)))) + 1
facet.ncol <- 3
p <- ggplot(test.results.hd.limit, aes(x=true.activity, y=predicted.activity))
p <- p + geom_point(alpha=0.5, stroke=0, size=0.2)
p <- p + xlim(ACTIVITY.RANGE) + ylim(ACTIVITY.RANGE)  # make ranges be the same
p <- p + coord_fixed()  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + facet_wrap(~ hamming.dist, ncol=facet.ncol)
p <- p + theme_pubr()
p <- p + theme(strip.background=element_blank(),    # remove background on facet label
               panel.border=element_rect(color="gray", fill=NA, size=0.5)) # border facet
# Include text with the rho value
p <- p + geom_text(data=hd.rho.hd.limit, aes(label=paste("rho==", rho.val.str)), parse=TRUE,
                   x=Inf, y=Inf, hjust=1, vjust=1, size=3)
p.true.vs.predicted.facet.by.hamming.dist <- p
#####################################################################

#####################################################################
# Plot true activity value vs. predicted activity value
# This is a scatter plot, where each dot represents a target/crRNA
# pair (test data point)
# The points are colored by Cas13 PFS

p <- ggplot(test.results, aes(x=true.activity, y=predicted.activity))
p <- p + geom_point(aes(color=cas13a.pfs), alpha=0.5, stroke=0)
p <- p + scale_color_viridis(name="PFS", discrete=TRUE) # adjust color gradient
p <- p + xlim(ACTIVITY.RANGE) + ylim(ACTIVITY.RANGE)  # make ranges be the same
p <- p + coord_fixed()  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_pubr()
p.true.vs.predicted.colored.by.pfs <- p
#####################################################################

#####################################################################
# Plot true activity value vs. predicted activity value
# This is a scatter plot, where each dot represents a target/crRNA
# pair (test data point)
# There is a facet for each Cas13a PFS

pfs.rho$rho.val.str <- format(pfs.rho$rho, digits=3)

#facet.ncol <- floor(sqrt(length(unique(test.results$cas13a.pfs)))) + 1
facet.ncol <- 4
p <- ggplot(test.results, aes(x=true.activity, y=predicted.activity))
p <- p + geom_point(alpha=0.5, stroke=0, size=0.2)
p <- p + xlim(ACTIVITY.RANGE) + ylim(ACTIVITY.RANGE)  # make ranges be the same
p <- p + coord_fixed()  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + facet_wrap(~ cas13a.pfs, ncol=facet.ncol)
p <- p + theme_pubr()
p <- p + theme(strip.background=element_blank(),    # remove background on facet label
               panel.border=element_rect(color="gray", fill=NA, size=0.5)) # border facet
# Include text with the rho value
p <- p + geom_text(data=pfs.rho, aes(label=paste("rho==", rho.val.str)), parse=TRUE,
                   x=Inf, y=Inf, hjust=1, vjust=1, size=3)
p.true.vs.predicted.facet.by.pfs <- p
#####################################################################

#####################################################################
# Plot true activity value for different predicted quantile groupings
# y-axis shows the quantile (grouped) for the predicted activity and
# x-axis shows the distribution of true activity values in that
# quantile
# Use quartiles here (4 groupings)

# Create a separate data frame with a quantile factor that also
# includes 'All' for all data points
require(dplyr)
test.results.with.quantile.group <- data.frame(test.results)
test.results.with.quantile.group$predicted.quantile <- ntile(test.results.with.quantile.group$predicted.activity, 4)
test.results.all <- data.frame(test.results)
test.results.all$predicted.quantile <- rep("All", n=nrow(test.results.all))
test.results.with.quantile.group <- rbind(test.results.with.quantile.group,
                                          test.results.all)
test.results.with.quantile.group$predicted.quantile <- factor(test.results.with.quantile.group$predicted.quantile,
                                                              levels=c("All", "1", "2", "3", "4"))

# Add a 'color' factor that is "quantile" for quantile groups and "all" for All
test.results.with.quantile.group$color <- ifelse(test.results.with.quantile.group$predicted.quantile == "All", "all", "quantile")
test.results.with.quantile.group$color <- factor(test.results.with.quantile.group$color, levels=c("all", "quantile"))

p <- ggplot(test.results.with.quantile.group, aes(x=true.activity, y=predicted.quantile))
p <- p + xlab("True activity") + ylab("Quartile of prediction")
p <- p + geom_density_ridges()
p <- p + theme_pubr()
p.by.predicted.quantile.group <- p
#####################################################################

#####################################################################
# Print p-values comparing quantile groupings -- i.e., whether
# the best quartile of predictions has higher true activity than
# the second best quartile
# Use Wilcoxon rank sum (Mann Whitney U) test

compare.quartile <- function(a, b) {
    a.vals <- test.results.with.quantile.group$true.activity[test.results.with.quantile.group$predicted.quantile == a]
    b.vals <- test.results.with.quantile.group$true.activity[test.results.with.quantile.group$predicted.quantile == b]
    return(wilcox.test(a.vals, b.vals, paired=FALSE, alternative="greater")$p.value)
}

print("p-values for quartile comparisons (quartile 4 is best predictions, 1 is worst):")
print(paste0("  4 > 3: p=", compare.quartile("4", "3")))
print(paste0("  3 > 2: p=", compare.quartile("3", "2")))
print(paste0("  2 > 1: p=", compare.quartile("2", "1")))
#####################################################################

#####################################################################
# Plot true activity value for different predicted quantile groupings
# y-axis shows the quantile (grouped) for the predicted activity and
# x-axis shows the true activity as a boxplot
# Use quartiles here (4 groupings)

box.n <- function(x) {
    upper.whisker <- min(max(x), quantile(x, c(0.75))[[1]] + 1.5 * IQR(x))
    # vjust=0.5, hjust=0 to center on whisker; vjust=1.5, hjust=1 to move left
    # and underneath
    return(data.frame(y=upper.whisker + 0.05, vjust=1.5, hjust=1.0, label=length(x)))
}

# For geom_boxplot(), we need the variable of interest to be y, so use
# this and then do coord_flip()
p <- ggplot(test.results.with.quantile.group, aes(y=true.activity, x=predicted.quantile))
p <- p + ylab("True activity") + xlab("Quartile of prediction")
p <- p + geom_boxplot(aes(color=color),
                      #outlier.size=0.25,
                      #outlier.stroke=0,
                      #outlier.color="gray46",
                      outlier.shape=NA) # do not show outliers, which are hard to distinguish from whiskers
# Add p-values, using Wilcoxon rank sum (aka, Mann Whitney U) test to
# compare whether predictions in each quartile are 'better' (have
# higher true activity) than predictions from another quartile
p <- p + geom_signif(comparisons=list(c("4","3"), c("3","2"), c("2","1")),
                     test="wilcox.test", test.args=list(paired=FALSE, alternative="greater"),
                     step_increase=0.1, size=0.1)
p <- p + stat_summary(fun.data=box.n, geom="text", size=2) # show number of observations
p <- p + scale_color_manual(values=c("gray", "black"), guide=FALSE)  # gray for 'all'; black for 'quantile's; guide=FALSE to skip legend
p <- p + coord_flip()
p <- p + theme_pubr()
p.by.predicted.quantile.group.boxplot <- p
#####################################################################

#####################################################################
# Combine the ridges and boxplot plots above

# Plot true activity value for different predicted quantile groupings
# y-axis shows the quantile (grouped) for the predicted activity and
# x-axis shows the distribution of true activity values in that
# quantile, along with a boxplot
# Use quartiles here (4 groupings)

# This is tricky because geom_density_ridges() is usually shown horizontally
# and boxplots are usually shown vertically. To have horizontal boxplots
# in ggplot2, we would use coord_flip(), but this would not work with
# the geom_density_ridges(). Thus, we use the ggstance package to have
# horizontal boxplots through geom_boxploth()

# Note that adding in p-values with geom_signif() doesn't work here
# because it assumes horizontally-separated groupings, the groupings
# are vertically-separated here; however, they are equivalent to the ones
# shown in the boxplot-only plot (p.by.predicted.quantile.group.boxplot)

p <- ggplot(test.results.with.quantile.group, aes(x=true.activity, y=predicted.quantile))
p <- p + xlab("True activity") + ylab("Quartile of prediction")
p <- p + geom_density_ridges(aes(fill=color),
                             #color="white", # white outline
                             scale=1.2, # 1 indicates the maximum of a ridge touches the base of the one above; >1 overlaps
                             alpha=0.7) # some transparency in overlap
p <- p + geom_boxploth(aes(color=color),
                       width=0.15,   # really the height of the boxplot
                       size=1,  # thickness of lines
                       position=position_nudge(y=+0.2), # shift up
                       #fill=NA, # leave empty to see ridges
                       outlier.shape=NA, # do not show outliers, which are hard to distinguish from whiskers
                       coef=0)  # do not show whiskers
p <- p + scale_color_manual(values=c("gray", "black"), guide=FALSE)  # gray for 'all'; black for 'quantile's; guide=FALSE to skip legend
p <- p + scale_fill_manual(values=c("gray", "black"), guide=FALSE)  # gray for 'all'; black for 'quantile's; guide=FALSE to skip legend
p <- p + xlim(-4.25, 0.1)  # a few points with true activity >0 extend the plot to the right; cut it off
p <- p + theme_pubr()
p.by.predicted.quantile.group.ridges.and.boxplot <- p
#####################################################################

#####################################################################
# Repeat the ridges and boxplot plots above, except using the data without
# error in the true activities

# Create a separate data frame with a quantile factor that also
# includes 'All' for all data points
require(dplyr)
test.results.without.error.with.quantile.group <- data.frame(test.results.without.error)
test.results.without.error.with.quantile.group$predicted.quantile <- ntile(test.results.without.error.with.quantile.group$predicted.activity, 4)
test.results.without.error.all <- data.frame(test.results.without.error)
test.results.without.error.all$predicted.quantile <- rep("All", n=nrow(test.results.without.error.all))
test.results.without.error.with.quantile.group <- rbind(test.results.without.error.with.quantile.group,
                                          test.results.without.error.all)
test.results.without.error.with.quantile.group$predicted.quantile <- factor(test.results.without.error.with.quantile.group$predicted.quantile,
                                                              levels=c("All", "1", "2", "3", "4"))

# Add a 'color' factor that is "quantile" for quantile groups and "all" for All
test.results.without.error.with.quantile.group$color <- ifelse(test.results.without.error.with.quantile.group$predicted.quantile == "All", "all", "quantile")
test.results.without.error.with.quantile.group$color <- factor(test.results.without.error.with.quantile.group$color, levels=c("all", "quantile"))

# For geom_boxplot(), we need the variable of interest to be y, so use
# this and then do coord_flip()
p <- ggplot(test.results.without.error.with.quantile.group, aes(y=true.activity, x=predicted.quantile))
p <- p + ylab("True activity") + xlab("Quartile of prediction")
p <- p + geom_boxplot(aes(color=color),
                      #outlier.size=0.25,
                      #outlier.stroke=0,
                      #outlier.color="gray46",
                      outlier.shape=NA) # do not show outliers, which are hard to distinguish from whiskers
# Add p-values, using Wilcoxon rank sum (aka, Mann Whitney U) test to
# compare whether predictions in each quartile are 'better' (have
# higher true activity) than predictions from another quartile
p <- p + geom_signif(comparisons=list(c("4","3"), c("3","2"), c("2","1")),
                     test="wilcox.test", test.args=list(paired=FALSE, alternative="greater"),
                     step_increase=0.1, size=0.1)
p <- p + stat_summary(fun.data=box.n, geom="text", size=2) # show number of observations
p <- p + scale_color_manual(values=c("gray", "black"), guide=FALSE)  # gray for 'all'; black for 'quantile's; guide=FALSE to skip legend
p <- p + coord_flip()
p <- p + theme_pubr()
p.by.predicted.quantile.group.boxplot.without.error <- p

p <- ggplot(test.results.without.error.with.quantile.group, aes(x=true.activity, y=predicted.quantile))
p <- p + xlab("True activity") + ylab("Quartile of prediction")
p <- p + geom_density_ridges(aes(fill=color),
                             #color="white", # white outline
                             scale=1.2, # 1 indicates the maximum of a ridge touches the base of the one above; >1 overlaps
                             alpha=0.7) # some transparency in overlap
p <- p + geom_boxploth(aes(color=color),
                       width=0.15,   # really the height of the boxplot
                       size=1,  # thickness of lines
                       position=position_nudge(y=+0.2), # shift up
                       #fill=NA, # leave empty to see ridges
                       outlier.shape=NA, # do not show outliers, which are hard to distinguish from whiskers
                       coef=0)  # do not show whiskers
p <- p + scale_color_manual(values=c("gray", "black"), guide=FALSE)  # gray for 'all'; black for 'quantile's; guide=FALSE to skip legend
p <- p + scale_fill_manual(values=c("gray", "black"), guide=FALSE)  # gray for 'all'; black for 'quantile's; guide=FALSE to skip legend
p <- p + xlim(-4.25, 0.1)  # a few points with true activity >0 extend the plot to the right; cut it off
p <- p + theme_pubr()
p.by.predicted.quantile.group.ridges.and.boxplot.without.error <- p
#####################################################################

#####################################################################
# Print p-values comparing quantile groupings -- i.e., whether
# the best quartile of predictions has higher true activity than
# the second best quartile -- for the data without error
# Use Wilcoxon rank sum (Mann Whitney U) test

compare.quartile.without.error <- function(a, b) {
    a.vals <- test.results.without.error.with.quantile.group$true.activity[test.results.without.error.with.quantile.group$predicted.quantile == a]
    b.vals <- test.results.without.error.with.quantile.group$true.activity[test.results.without.error.with.quantile.group$predicted.quantile == b]
    return(wilcox.test(a.vals, b.vals, paired=FALSE, alternative="greater")$p.value)
}

print("p-values for quartile comparisons, without error in true activities (quartile 4 is best predictions, 1 is worst):")
print(paste0("  4 > 3: p=", compare.quartile.without.error("4", "3")))
print(paste0("  3 > 2: p=", compare.quartile.without.error("3", "2")))
print(paste0("  2 > 1: p=", compare.quartile.without.error("2", "1")))
#####################################################################

#####################################################################
# Plot true activity value for different predicted quantile groupings,
# with facet for each choice of Hamming distance
# y-axis shows the quantile (grouped) for the predicted activity and
# x-axis shows the true activity as a boxplot
# Use quartiles here (4 groupings)
# coord_flip() does not work well with facet_wrap(), so we cannot use
# the approach above to coord_flip()
# Only show up to a Hamming distance of 5; there are few points beyond this

test.results.with.quantile.group.hd.limit <- test.results.with.quantile.group[test.results.with.quantile.group$hamming.dist <= 5,]

#facet.ncol <- floor(sqrt(length(unique(test.results.with.quantile.group$hamming.dist)))) + 1
facet.ncol <- 3
p <- ggplot(test.results.with.quantile.group.hd.limit, aes(y=true.activity, x=predicted.quantile))
p <- p + facet_wrap(~ hamming.dist, ncol=facet.ncol, scales="fixed")
p <- p + ylab("True activity") + xlab("Quartile of prediction")
p <- p + geom_boxplot(aes(color=color),
                      #outlier.size=0.25,
                      #outlier.stroke=0,
                      #outlier.color="gray46",
                      outlier.shape=NA) # do not show outliers, which are hard to distinguish from whiskers
p <- p + stat_summary(fun.data=box.n, geom="text", size=2) # show N
p <- p + scale_color_manual(values=c("gray", "black"), guide=FALSE)  # gray for 'all'; black for 'quantile's; guide=FALSE to skip legend
p <- p + coord_flip()
p <- p + theme_pubr()
p <- p + theme(strip.background=element_blank(),    # remove background on facet label
               panel.border=element_rect(color="gray", fill=NA, size=0.5)) # border facet
p.by.predicted.quantile.group.boxplot.hamming.dist <- p
#####################################################################

#####################################################################
# Plot true activity value for different predicted quantile groupings,
# with facet for each choice of PFS
# y-axis shows the quantile (grouped) for the predicted activity and
# x-axis shows the true activity as a boxplot
# Use quartiles here (4 groupings)
# coord_flip() does not work well with facet_wrap(), so we cannot use
# the approach above to coord_flip()

facet.ncol <- floor(sqrt(length(unique(test.results.with.quantile.group$cas13a.pfs))))
p <- ggplot(test.results.with.quantile.group, aes(y=true.activity, x=predicted.quantile))
p <- p + facet_wrap(~ cas13a.pfs, ncol=facet.ncol, scales="fixed")
p <- p + ylab("True activity") + xlab("Quartile of prediction")
p <- p + geom_boxplot(aes(color=color),
                      #outlier.size=0.25,
                      #outlier.stroke=0,
                      #outlier.color="gray46",
                      outlier.shape=NA) # do not show outliers, which are hard to distinguish from whiskers
p <- p + stat_summary(fun.data=box.n, geom="text", size=2) # show N
p <- p + scale_color_manual(values=c("gray", "black"), guide=FALSE)  # gray for 'all'; black for 'quantile's; guide=FALSE to skip legend
p <- p + coord_flip()
p <- p + theme_pubr()
p <- p + theme(strip.background=element_blank(),    # remove background on facet label
               panel.border=element_rect(color="gray", fill=NA, size=0.5)) # border facet
p.by.predicted.quantile.group.boxplot.pfs <- p
#####################################################################

#####################################################################
# Plot true activity quantile vs. predicted activity quantile
# x-axis shows the quantile for the true activity and
# y-axis shows the quantile for the predicted

# Create a separate data frame with a quantile factor that also
# includes 'all' for all data points
test.results.with.quantile <- data.frame(test.results)
test.results.with.quantile$predicted.quantile <- ecdf(test.results.with.quantile$predicted.activity)(test.results.with.quantile$predicted.activity)
test.results.with.quantile$true.quantile <- ecdf(test.results.with.quantile$true.activity)(test.results.with.quantile$true.activity)

p <- ggplot(test.results.with.quantile, aes(x=true.quantile, y=predicted.quantile))
p <- p + geom_point(aes(color=crrna.pos))
p <- p + geom_smooth(method=lm)
p <- p + scale_color_viridis() # adjust color gradient
p <- p + xlab("True activity quantile") + ylab("Predicted activity quantile")
p <- p + theme_pubr()
p.true.vs.predicted.quantiles <- p
#####################################################################

#####################################################################
# Plot the prediction within each crRNA (where each is decided by its
# position in the target)
# Plot a separate facet for each crRNA (this is basically like making
# a separate plot for each color in the scatter plot above)
# Also, make a density plot of the distribution of Spearman rho
# rank correlation coefficients across the crRNAs

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
rho.for.crrna$rho.val.str <- format(rho.for.crrna$rho, digits=3)

facet.ncol <- floor(sqrt(length(unique(test.results$crrna.pos)))) + 1
p <- ggplot(test.results, aes(x=true.activity, y=predicted.activity))
p <- p + geom_point(alpha=0.5, stroke=0, size=0.2)
p <- p + facet_wrap( ~ crrna.pos, ncol=facet.ncol)
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_pubr()
p <- p + theme(strip.background=element_blank(),    # remove background on facet label
               panel.border=element_rect(color="gray", fill=NA, size=0.5), # border facet
               aspect.ratio=1)  # square facets
# Include text with the rho value
p <- p + geom_text(data=rho.for.crrna, aes(label=paste("rho==", rho.val.str)), parse=TRUE,
                   x=Inf, y=Inf, hjust=1, vjust=1, size=3)
p.true.vs.predicted.faceted.by.crrna <- p

p <- ggplot(rho.for.crrna, aes(x=rho))
p <- p + geom_density(fill="gray")
p <- p + xlab("Spearman correlation") + ylab("Density")
p <- p + theme_pubr()
p.rho.across.crrnas <- p
#####################################################################

#####################################################################
# Plot the mean true activity vs. mean predicted value, where the mean
# is computed for each crRNA (taken across the targets)
# This is a scatter plot with one dot for each crRNA

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
p <- p + xlim(ACTIVITY.RANGE) + ylim(ACTIVITY.RANGE)  # make ranges be the same
p <- p + coord_fixed()  # make plot be square
p <- p + xlab("True activity") + ylab("Predicted activity")
p <- p + theme_pubr()
# Include text with the rho value
p <- p + annotate(geom='text', x=Inf, y=Inf, hjust=1, vjust=1, size=5,
                  label=as.character(as.expression(substitute(
                      rho~"="~spearman.rho, list(spearman.rho=format(spearman.rho, digits=3))))),
                  parse=TRUE)
p.true.vs.predicted.summarized <- p
#####################################################################

#####################################################################
# Produce PDFs

save <- function(p, filename, width, height) {
    ggsave(file.path(OUT.DIR, paste0(filename, ".pdf")),
           p,
           width=width,
           height=height,
           useDingbats=FALSE)
}

save(p.output.dist, "output-dist", 6, 6)
save(p.true.vs.predicted, "true-vs-predicted", 6, 6)
save(p.true.vs.predicted.density.contours, "true-vs-predicted-density-contours", 4.5, 4.5)
save(p.true.vs.predicted.density.contours.without.error, "true-vs-predicted-density-contours-without-error", 4.5, 4.5)
save(p.true.vs.predicted.colored.by.hamming.dist, "true-vs-predicted-colored-by-hamming-dist", 6, 6)
save(p.true.vs.predicted.facet.by.hamming.dist, "true-vs-predicted-facet-by-hamming-dist", 5, 4)
save(p.true.vs.predicted.colored.by.pfs, "true-vs-predicted-colored-by-pfs", 6, 6)
save(p.true.vs.predicted.facet.by.pfs, "true-vs-predicted-facet-by-pfs", 6, 2.5)
save(p.by.predicted.quantile.group, "by-predicted-quantile-group", 6, 6)
save(p.by.predicted.quantile.group.boxplot, "by-predicted-quantile-group-boxplot", 6.5, 6)
save(p.by.predicted.quantile.group.boxplot.without.error, "by-predicted-quantile-group-boxplot-without-error", 6.5, 6)
save(p.by.predicted.quantile.group.ridges.and.boxplot, "by-predicted-quantile-group-ridges-and-boxplot", 6.25, 4)
save(p.by.predicted.quantile.group.ridges.and.boxplot.without.error, "by-predicted-quantile-group-ridges-and-boxplot-without-error", 6.25, 4)
save(p.by.predicted.quantile.group.boxplot.hamming.dist, "by-predicted-quantile-group-boxplot-hamming-dist", 6, 6)
save(p.by.predicted.quantile.group.boxplot.pfs, "by-predicted-quantiled-group-boxplot-pfs", 6, 6)
save(p.true.vs.predicted.quantiles, "true-vs-predicted-quantiles", 6, 6)
save(p.true.vs.predicted.faceted.by.crrna, "true-vs-predicted-faceted-by-crrna", 7.5, 9)
save(p.rho.across.crrnas, "rho-across-crrnas", 6, 6)
save(p.true.vs.predicted.summarized, "true-vs-predicted-summarized", 6, 6)
#####################################################################
