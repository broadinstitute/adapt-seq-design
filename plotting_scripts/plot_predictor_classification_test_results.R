# Plot results of testing predictor, for classification.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(dplyr)
require(viridis)
require(PRROC)
require(egg)
require(caret)
require(ggpubr)

args <- commandArgs(trailingOnly=TRUE)
IN.TSV <- args[1]
DECISION.THRESHOLD <- as.numeric(args[2])
OUT.DIR <- args[3]


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

# Compute ROC and PR curves
pos <- test.results$predicted.activity[test.results$true.activity == 1]
neg <- test.results$predicted.activity[test.results$true.activity == 0]
print(paste("Number true positive:", length(pos)))
print(paste("Number true negative:", length(neg)))
roc <- roc.curve(scores.class0=pos, scores.class1=neg, curve=TRUE,
                 max.compute=TRUE, min.compute=TRUE, rand.compute=TRUE)
pr <- pr.curve(scores.class0=pos, scores.class1=neg, curve=TRUE,
               max.compute=TRUE, min.compute=TRUE, rand.compute=TRUE)
print(roc)
print(pr)

# For a baseline to compare to, let's imagine choosing positive points
# to be non-G PFS with <=4 Hamming distance (similarly, <=3, <=2, <=1, and 0)
# Compute sensitivity, FPR, and precision for these choices
test.results$baseline.predicted.activity.hd4 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist <= 4, 1, 0)
test.results$baseline.predicted.activity.hd3 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist <= 3, 1, 0)
test.results$baseline.predicted.activity.hd2 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist <= 2, 1, 0)
test.results$baseline.predicted.activity.hd1 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist <= 1, 1, 0)
test.results$baseline.predicted.activity.hd0 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist == 0, 1, 0)
baseline.conf.matrix.hd4 <- table(test.results$baseline.predicted.activity.hd4, test.results$true.activity)
baseline.conf.matrix.hd3 <- table(test.results$baseline.predicted.activity.hd3, test.results$true.activity)
baseline.conf.matrix.hd2 <- table(test.results$baseline.predicted.activity.hd2, test.results$true.activity)
baseline.conf.matrix.hd1 <- table(test.results$baseline.predicted.activity.hd1, test.results$true.activity)
baseline.conf.matrix.hd0 <- table(test.results$baseline.predicted.activity.hd0, test.results$true.activity)
baseline.sensitivity.hd4 <- sensitivity(baseline.conf.matrix.hd4, positive="1", negative="0")
baseline.sensitivity.hd3 <- sensitivity(baseline.conf.matrix.hd3, positive="1", negative="0")
baseline.sensitivity.hd2 <- sensitivity(baseline.conf.matrix.hd2, positive="1", negative="0")
baseline.sensitivity.hd1 <- sensitivity(baseline.conf.matrix.hd1, positive="1", negative="0")
baseline.sensitivity.hd0 <- sensitivity(baseline.conf.matrix.hd0, positive="1", negative="0")
baseline.fpr.hd4 <- 1.0 - specificity(baseline.conf.matrix.hd4, positive="1", negative="0")
baseline.fpr.hd3 <- 1.0 - specificity(baseline.conf.matrix.hd3, positive="1", negative="0")
baseline.fpr.hd2 <- 1.0 - specificity(baseline.conf.matrix.hd2, positive="1", negative="0")
baseline.fpr.hd1 <- 1.0 - specificity(baseline.conf.matrix.hd1, positive="1", negative="0")
baseline.fpr.hd0 <- 1.0 - specificity(baseline.conf.matrix.hd0, positive="1", negative="0")
baseline.precision.hd4 <- posPredValue(baseline.conf.matrix.hd4, positive="1", negative="0")
baseline.precision.hd3 <- posPredValue(baseline.conf.matrix.hd3, positive="1", negative="0")
baseline.precision.hd2 <- posPredValue(baseline.conf.matrix.hd2, positive="1", negative="0")
baseline.precision.hd1 <- posPredValue(baseline.conf.matrix.hd1, positive="1", negative="0")
baseline.precision.hd0 <- posPredValue(baseline.conf.matrix.hd0, positive="1", negative="0")
baseline.results.hd.levels <- c("=0", "<=1", "<=2", "<=3", "<=4")
baseline.results <- data.frame(hd=baseline.results.hd.levels,
                               sensitivity=c(baseline.sensitivity.hd0, baseline.sensitivity.hd1, baseline.sensitivity.hd2, baseline.sensitivity.hd3, baseline.sensitivity.hd4),
                               fpr=c(baseline.fpr.hd0, baseline.fpr.hd1, baseline.fpr.hd2, baseline.fpr.hd3, baseline.fpr.hd4),
                               precision=c(baseline.precision.hd0, baseline.precision.hd1, baseline.precision.hd2, baseline.precision.hd3, baseline.precision.hd4))
baseline.results$hd <- factor(baseline.results$hd, levels=baseline.results.hd.levels)

# Make a data frame comparing the baseline and 'main' predictor
# For each Hamming distance, find the baseline's FPR and the 'main' predictor's
# FPR at equivalent sensitivity (recall), as given in roc$curve; do the
# same for precision, as given in pr$curve
roc.curve.df <- data.frame(roc$curve)
pr.curve.df <- data.frame(pr$curve)
compare.to.baseline <- do.call(rbind, lapply(baseline.results.hd.levels,
    function(hd) {
        # Find the sensitivity, FPR, and precision of the baseline at this hd
        baseline.sensitivity <- baseline.results[baseline.results$hd == hd,]$sensitivity
        baseline.fpr <- baseline.results[baseline.results$hd == hd,]$fpr
        baseline.precision <- baseline.results[baseline.results$hd == hd,]$precision

        # Find the FPR, in roc.curve.df, at equivalent sensitivity (or closest)
        predictor.roc.idx <- which(abs(roc.curve.df$X2 - baseline.sensitivity) == min(abs(roc.curve.df$X2 - baseline.sensitivity)))[[1]]
        predictor.fpr <- roc.curve.df[predictor.roc.idx,]$X1

        # Find the precision, in pr$curve, at equivalent sensitivity (or
        # closest)
        predictor.pr.idx <- which(abs(pr.curve.df$X1 - baseline.sensitivity) == min(abs(pr.curve.df$X1 - baseline.sensitivity)))[[1]]
        predictor.precision <- pr.curve.df[predictor.pr.idx,]$X2

        rows.to.add <- data.frame(hd=c(hd, hd, hd, hd),
                                  model=c("Baseline", "Baseline", "ADAPT", "ADAPT"),
                                  metric=c("fpr", "precision", "fpr", "precision"),
                                  value=c(baseline.fpr, baseline.precision, predictor.fpr, predictor.precision),
                                  color=c(hd, hd, "ADAPT", "ADAPT"))
        return(rows.to.add)
    }
))
compare.to.baseline$color <- factor(compare.to.baseline$color,
                                    levels=c("ADAPT", baseline.results.hd.levels))

# Find sensitivity, FPR, precision, recall at the given classifier threshold
# Find the closest threshold in roc$curve and $pr$curve to the given one
decision.threshold.roc.idx <- which(abs(roc.curve.df$X3 - DECISION.THRESHOLD) == min(abs(roc.curve.df$X3 - DECISION.THRESHOLD)))[[1]]
decision.threshold.roc.fpr <- roc.curve.df[decision.threshold.roc.idx,]$X1
decision.threshold.roc.sensitivity <- roc.curve.df[decision.threshold.roc.idx,]$X2
decision.threshold.pr.idx <- which(abs(pr.curve.df$X3 - DECISION.THRESHOLD) == min(abs(pr.curve.df$X3 - DECISION.THRESHOLD)))[[1]]
decision.threshold.pr.sensitivity <- pr.curve.df[decision.threshold.pr.idx,]$X1
decision.threshold.pr.precision <- pr.curve.df[decision.threshold.pr.idx,]$X2
print(paste0("At provided decision threshold = ", DECISION.THRESHOLD))
print(paste0("  ROC curve: sensitivity = ", decision.threshold.roc.sensitivity,
             " ; FPR = ", decision.threshold.roc.fpr))
print(paste0("  PR curve: sensitivity (recall) = ", decision.threshold.pr.sensitivity,
             " ;  precision = ", decision.threshold.pr.precision))

#####################################################################
# Compute ROC and PR curves for different choices of Hamming distance,
# and separately for different choices of PFS

roc.df.all <- data.frame(roc$curve)
roc.df.all$hamming.dist <- rep("All", nrow(roc.df.all))
roc.df.all$cas13a.pfs <- rep("All", nrow(roc.df.all))
pr.df.all <- data.frame(pr$curve)
pr.df.all$hamming.dist <- rep("All", nrow(pr.df.all))
pr.df.all$cas13a.pfs <- rep("All", nrow(pr.df.all))

# ROC - Hamming distance
hamming.dist.roc <- do.call(rbind, lapply(unique(test.results$hamming.dist),
    function(hamming.dist) {
        if (hamming.dist > 5) {
            # Only use hamming.dist <= 5; higher values have too little data
            return(data.frame())
        }
        vals <- test.results[test.results$hamming.dist == hamming.dist, ]
        pos <- vals$predicted.activity[vals$true.activity == 1]
        neg <- vals$predicted.activity[vals$true.activity == 0]
        roc <- roc.curve(scores.class0=pos, scores.class1=neg, curve=TRUE)
        roc.df <- data.frame(roc$curve)
        roc.df$hamming.dist <- rep(hamming.dist, nrow(roc.df))
        roc.df$cas13a.pfs <- rep(NA, nrow(roc.df))
        return(roc.df)
    }
))
hamming.dist.roc$hamming.dist <- factor(hamming.dist.roc$hamming.dist)
hamming.dist.roc <- rbind(hamming.dist.roc, roc.df.all)

# PR - Hamming distance
hamming.dist.pr <- do.call(rbind, lapply(unique(test.results$hamming.dist),
    function(hamming.dist) {
        if (hamming.dist > 5) {
            # Only use hamming.dist <= 5; higher values have too little data
            return(data.frame())
        }
        vals <- test.results[test.results$hamming.dist == hamming.dist, ]
        pos <- vals$predicted.activity[vals$true.activity == 1]
        neg <- vals$predicted.activity[vals$true.activity == 0]
        pr <- pr.curve(scores.class0=pos, scores.class1=neg, curve=TRUE)
        pr.df <- data.frame(pr$curve)
        pr.df$hamming.dist <- rep(hamming.dist, nrow(pr.df))
        pr.df$cas13a.pfs <- rep(NA, nrow(pr.df))
        return(pr.df)
    }
))
hamming.dist.pr$hamming.dist <- factor(hamming.dist.pr$hamming.dist)
hamming.dist.pr <- rbind(hamming.dist.pr, pr.df.all)

# ROC - Cas13a PFS
cas13a.pfs.roc <- do.call(rbind, lapply(unique(test.results$cas13a.pfs),
    function(cas13a.pfs) {
        vals <- test.results[test.results$cas13a.pfs == cas13a.pfs, ]
        pos <- vals$predicted.activity[vals$true.activity == 1]
        neg <- vals$predicted.activity[vals$true.activity == 0]
        roc <- roc.curve(scores.class0=pos, scores.class1=neg, curve=TRUE)
        roc.df <- data.frame(roc$curve)
        roc.df$cas13a.pfs <- rep(cas13a.pfs, nrow(roc.df))
        roc.df$hamming.dist <- rep(NA, nrow(roc.df))
        return(roc.df)
    }
))
cas13a.pfs.roc$cas13a.pfs <- factor(cas13a.pfs.roc$cas13a.pfs)
cas13a.pfs.roc <- rbind(cas13a.pfs.roc, roc.df.all)

# PR - Cas13a PFS
cas13a.pfs.pr <- do.call(rbind, lapply(unique(test.results$cas13a.pfs),
    function(cas13a.pfs) {
        vals <- test.results[test.results$cas13a.pfs == cas13a.pfs, ]
        pos <- vals$predicted.activity[vals$true.activity == 1]
        neg <- vals$predicted.activity[vals$true.activity == 0]
        pr <- pr.curve(scores.class0=pos, scores.class1=neg, curve=TRUE)
        pr.df <- data.frame(pr$curve)
        pr.df$cas13a.pfs <- rep(cas13a.pfs, nrow(pr.df))
        pr.df$hamming.dist <- rep(NA, nrow(pr.df))
        return(pr.df)
    }
))
cas13a.pfs.pr$cas13a.pfs <- factor(cas13a.pfs.pr$cas13a.pfs)
cas13a.pfs.pr <- rbind(cas13a.pfs.pr, pr.df.all)
#####################################################################

#####################################################################
# Plot ROC curve
# Color represents classifer threshold

p <- ggplot(data.frame(roc$curve))
p <- p + geom_line(aes(x=X1, y=X2, color=X3))
p <- p + geom_abline(slope=1, intercept=0, linetype="dotted")  # diagonal for random classifier
p <- p + geom_point(data=baseline.results, aes(x=fpr, y=sensitivity, shape=hd)) # dots for baseline
p <- p + geom_point(aes(x=decision.threshold.roc.fpr, y=decision.threshold.roc.sensitivity), shape=3, size=4, color="#ED406B", stroke=1.0)  # '+' at given decision threshold
p <- p + xlab("FPR") + ylab("Sensitivity")
p <- p + scale_color_viridis(name="Decision threshold") # adjust color gradient
p <- p + scale_shape_discrete(name="Distance",
                              labels=c(expression(phantom()==0),
                                       expression(phantom()<=1),
                                       expression(phantom()<=2),
                                       expression(phantom()<=3),
                                       expression(phantom()<=4)))
p <- p + theme_pubr()
p <- p + theme(aspect.ratio=1)  # square plot
p.roc <- p
#####################################################################

#####################################################################
# Plot ROC curve
# Color points represent baseline decision thresholds

p <- ggplot(data.frame(roc$curve))
p <- p + geom_line(aes(x=X1, y=X2), size=1.25)
p <- p + geom_abline(slope=1, intercept=0, linetype="dotted")  # diagonal for random classifier
p <- p + geom_point(data=baseline.results, aes(x=fpr, y=sensitivity, fill=hd), size=4, shape=21, color="white", stroke=0.5) # dots for baseline; stroke/outline in white
p <- p + geom_point(aes(x=decision.threshold.roc.fpr, y=decision.threshold.roc.sensitivity), shape=3, size=4, color="#ED406B", stroke=1.0)  # '+' at given decision threshold
p <- p + xlab("FPR") + ylab("Sensitivity")
p <- p + scale_fill_viridis(discrete=TRUE,
                            name="Distance",
                            labels=c(expression(phantom()==0),
                                     expression(phantom()<=1),
                                     expression(phantom()<=2),
                                     expression(phantom()<=3),
                                     expression(phantom()<=4)))
p <- p + theme_pubr()
p <- p + theme(aspect.ratio=1)  # square plot
p.roc.color.baseline <- p
#####################################################################

#####################################################################
# Plot ROC curve
# Color represents choices of Hamming distance

p <- ggplot(hamming.dist.roc, aes(x=X1, y=X2))
p <- p + geom_line(aes(color=hamming.dist))
p <- p + geom_abline(slope=1, intercept=0, linetype="dotted")  # diagonal for random classifier
p <- p + xlab("FPR") + ylab("Sensitivity")
p <- p + scale_color_viridis(discrete=TRUE, # adjust color gradient
                             name="Hamming distance")
p <- p + theme_pubr()
p <- p + theme(aspect.ratio=1)  # square plot
p.hamming.dist.roc <- p
#####################################################################

#####################################################################
# Plot ROC curve
# Color represents choices of Cas13a PFS

p <- ggplot(cas13a.pfs.roc, aes(x=X1, y=X2))
p <- p + geom_line(aes(color=cas13a.pfs))
p <- p + geom_abline(slope=1, intercept=0, linetype="dotted")  # diagonal for random classifier
p <- p + xlab("FPR") + ylab("Sensitivity")
p <- p + scale_color_viridis(discrete=TRUE, # adjust color gradient
                             name="PFS")
p <- p + theme_pubr()
p <- p + theme(aspect.ratio=1)  # square plot
p.cas13a.pfs.roc <- p
#####################################################################

#####################################################################
# Plot PR curve
# Color represents classifier threshold

random.precision <- length(pos) / nrow(test.results)

p <- ggplot(data.frame(pr$curve))
p <- p + geom_line(aes(x=X1, y=X2, color=X3))
p <- p + geom_hline(yintercept=random.precision, linetype="dotted")    # representing random classifier
p <- p + geom_point(data=baseline.results, aes(x=sensitivity, y=precision, shape=hd)) # dots for baseline
p <- p + geom_point(aes(x=decision.threshold.pr.sensitivity, y=decision.threshold.pr.precision), shape=3, size=4, color="#ED406B", stroke=1.0)  # '+' at given decision threshold
p <- p + xlab("Recall") + ylab("Precision")
p <- p + ylim(0.8, 1.0)
p <- p + scale_color_viridis(name="Decision threshold") # adjust color gradient
p <- p + scale_shape_discrete(name="Distance",
                              labels=c(expression(phantom()==0),
                                       expression(phantom()<=1),
                                       expression(phantom()<=2),
                                       expression(phantom()<=3),
                                       expression(phantom()<=4)))
p <- p + theme_pubr()
p <- p + theme(aspect.ratio=1)  # square plot
p.pr <- p
#####################################################################

#####################################################################
# Plot PR curve
# Color represents baseline threshold

random.precision <- length(pos) / nrow(test.results)

p <- ggplot(data.frame(pr$curve))
p <- p + geom_line(aes(x=X1, y=X2), size=1.25)
p <- p + geom_hline(yintercept=random.precision, linetype="dotted")    # representing random classifier
p <- p + geom_point(data=baseline.results, aes(x=sensitivity, y=precision, fill=hd), size=4, shape=21, color="white", stroke=0.5) # dots for baseline; stroke/outline in white
p <- p + geom_point(aes(x=decision.threshold.pr.sensitivity, y=decision.threshold.pr.precision), shape=3, size=4, color="#ED406B", stroke=1.0)  # '+' at given decision threshold
p <- p + xlab("Recall") + ylab("Precision")
p <- p + ylim(0.8, 1.0)
p <- p + scale_fill_viridis(discrete=TRUE,
                            name="Distance",
                            labels=c(expression(phantom()==0),
                                     expression(phantom()<=1),
                                     expression(phantom()<=2),
                                     expression(phantom()<=3),
                                     expression(phantom()<=4)))
p <- p + theme_pubr()
p <- p + theme(aspect.ratio=1)  # square plot
p.pr.color.baseline <- p
#####################################################################

#####################################################################
# Plot PR curve
# Color represents choices of Hamming distance

# Find the precision of a random classifier for each choice of
# Hamming distance
random.precision <- do.call(rbind, lapply(unique(test.results$hamming.dist),
    function(hamming.dist) {
        if (hamming.dist > 5) {
            # Only use hamming.dist <= 5; higher values have too little data
            return(data.frame())
        }
        vals <- test.results[test.results$hamming.dist == hamming.dist, ]
        pos <- vals$predicted.activity[vals$true.activity == 1]
        neg <- vals$predicted.activity[vals$true.activity == 0]
        df <- data.frame(hamming.dist=c(hamming.dist),
                         precision=c(length(pos) / nrow(vals)))
        return(df)
    }
))
random.precision$hamming.dist <- factor(random.precision$hamming.dist)
random.precision <- rbind(random.precision,
                          data.frame(hamming.dist=c("All"),
                                     precision=c(length(pos) / nrow(test.results))))

p <- ggplot(hamming.dist.pr, aes(x=X1, y=X2))
p <- p + geom_line(aes(color=hamming.dist))
p <- p + geom_hline(data=random.precision, aes(yintercept=precision, color=hamming.dist), linetype="dotted")    # representing random classifier
p <- p + xlab("Recall") + ylab("Precision") + labs(color="Hamming distance")
p <- p + ylim(0.5, 1.0)
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + theme_pubr()
p <- p + theme(aspect.ratio=1)  # square plot
p.hamming.dist.pr <- p
#####################################################################

#####################################################################
# Plot PR curve
# Color represents choices of PFS

# Find the precision of a random classifier for each choice of Cas13a PFS
random.precision <- do.call(rbind, lapply(unique(test.results$cas13a.pfs),
    function(cas13a.pfs) {
        vals <- test.results[test.results$cas13a.pfs == cas13a.pfs, ]
        pos <- vals$predicted.activity[vals$true.activity == 1]
        neg <- vals$predicted.activity[vals$true.activity == 0]
        df <- data.frame(cas13a.pfs=c(as.character(cas13a.pfs)),
                         precision=c(length(pos) / nrow(vals)))
        return(df)
    }
))
random.precision <- rbind(random.precision,
                          data.frame(cas13a.pfs=c("All"),
                                     precision=c(length(pos) / nrow(test.results))))

p <- ggplot(cas13a.pfs.pr, aes(x=X1, y=X2))
p <- p + geom_line(aes(color=cas13a.pfs))
p <- p + geom_hline(data=random.precision, aes(yintercept=precision, color=cas13a.pfs), linetype="dotted")    # representing random classifier
p <- p + xlab("Recall") + ylab("Precision") + labs(color="PFS")
p <- p + ylim(0.5, 1.0)
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + theme_pubr()
p <- p + theme(aspect.ratio=1)  # square plot
p.cas13a.pfs.pr <- p
#####################################################################

#####################################################################
# Plot comparison of FPR between baseline and predictor
# Show FPR on the horizontal axis because that is how it is shown on
# a ROC curve

# Make 'ADAPT' be black and the baselines be colored by Hamming distance
# threshold
pal <- c("black", viridis::viridis(n=length(baseline.results.hd.levels)))

compare.to.baseline.fpr <- compare.to.baseline[compare.to.baseline$metric == "fpr",]

p <- ggplot(compare.to.baseline.fpr, aes(x=hd, y=value, fill=color))
p <- p + geom_bar(stat="identity", position=position_dodge(-0.7), width=0.7)    # make position dodge be negative to avoid coord_flip from reversing order of bars within groups
p <- p + scale_fill_manual(values=pal,
                           guide=FALSE, # do not show legend
                           labels=c("ADAPT",
                                    expression(phantom()==0),
                                    expression(phantom()<=1),
                                    expression(phantom()<=2),
                                    expression(phantom()<=3),
                                    expression(phantom()<=4)))
p <- p + scale_x_discrete(labels=c( # manually set x-axis labels (flipped, so y-axis)
                                   "=0" = expression(phantom()==0),
                                    "<=1" = expression(phantom()<=1),
                                    "<=2" = expression(phantom()<=2),
                                    "<=3" = expression(phantom()<=3),
                                    "<=4" = expression(phantom()<=4)))
p <- p + xlab("Distance") + ylab("FPR")
p <- p + coord_flip()
p <- p + theme_pubr()
p.compare.to.baseline.thresholds.fpr <- p
#####################################################################

#####################################################################
# Plot comparison of precision between baseline and predictor

# Make 'ADAPT' be black and the baselines be colored by Hamming distance
# threshold
pal <- c("black", viridis::viridis(n=length(baseline.results.hd.levels)))

compare.to.baseline.precision <- compare.to.baseline[compare.to.baseline$metric == "precision",]

p <- ggplot(compare.to.baseline.precision, aes(x=hd, y=value, fill=color))
p <- p + geom_bar(stat="identity", position=position_dodge(0.7), width=0.7)
p <- p + scale_fill_manual(values=pal,
                           guide=FALSE, # do not show legend
                           labels=c("ADAPT",
                                    expression(phantom()==0),
                                    expression(phantom()<=1),
                                    expression(phantom()<=2),
                                    expression(phantom()<=3),
                                    expression(phantom()<=4)))
p <- p + scale_x_discrete(labels=c( # manually set x-axis labels
                                   "=0" = expression(phantom()==0),
                                    "<=1" = expression(phantom()<=1),
                                    "<=2" = expression(phantom()<=2),
                                    "<=3" = expression(phantom()<=3),
                                    "<=4" = expression(phantom()<=4)))
p <- p + xlab("Distance") + ylab("Precision")
p <- p + coord_cartesian(ylim=c(0.8, 1.0))  # this way instead of only ylim() to avoid throwing away the bars, since they extend outside the range
p <- p + theme_pubr()
p.compare.to.baseline.thresholds.precision <- p
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

save(p.roc, "roc", 8, 8)
save(p.pr, "pr", 8, 8)
save(p.roc.color.baseline, "roc-color-baseline", 8, 8)
save(p.pr.color.baseline, "pr-color-baseline", 8, 8)
save(p.hamming.dist.roc, "hamming-dist-roc", 8, 8)
save(p.hamming.dist.pr, "hamming-dist-pr", 8, 8)
save(p.cas13a.pfs.roc, "cas13a-pfs-roc", 8, 8)
save(p.cas13a.pfs.pr, "cas13a-pfs-pr", 8, 8)
save(p.compare.to.baseline.thresholds.fpr, "compare-to-baseline-thresholds-fpr", 6.7, 4)
save(p.compare.to.baseline.thresholds.precision, "compare-to-baseline-thresholds-precision", 6, 8)
#####################################################################
