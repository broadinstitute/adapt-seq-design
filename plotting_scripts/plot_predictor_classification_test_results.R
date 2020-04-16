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
# to be non-G PFS with <=3 Hamming distance (similarly, <=2, <=1, and 0)
# Compute sensitivity, FPR, and precision for these choices
test.results$baseline.predicted.activity.hd3 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist <= 3, 1, 0)
test.results$baseline.predicted.activity.hd2 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist <= 2, 1, 0)
test.results$baseline.predicted.activity.hd1 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist <= 1, 1, 0)
test.results$baseline.predicted.activity.hd0 <- ifelse(test.results$cas13a.pfs != "G" & test.results$hamming.dist == 0, 1, 0)
baseline.conf.matrix.hd3 <- table(test.results$baseline.predicted.activity.hd3, test.results$true.activity)
baseline.conf.matrix.hd2 <- table(test.results$baseline.predicted.activity.hd2, test.results$true.activity)
baseline.conf.matrix.hd1 <- table(test.results$baseline.predicted.activity.hd1, test.results$true.activity)
baseline.conf.matrix.hd0 <- table(test.results$baseline.predicted.activity.hd0, test.results$true.activity)
baseline.sensitivity.hd3 <- sensitivity(baseline.conf.matrix.hd3, positive="1", negative="0")
baseline.sensitivity.hd2 <- sensitivity(baseline.conf.matrix.hd2, positive="1", negative="0")
baseline.sensitivity.hd1 <- sensitivity(baseline.conf.matrix.hd1, positive="1", negative="0")
baseline.sensitivity.hd0 <- sensitivity(baseline.conf.matrix.hd0, positive="1", negative="0")
baseline.fpr.hd3 <- 1.0 - specificity(baseline.conf.matrix.hd3, positive="1", negative="0")
baseline.fpr.hd2 <- 1.0 - specificity(baseline.conf.matrix.hd2, positive="1", negative="0")
baseline.fpr.hd1 <- 1.0 - specificity(baseline.conf.matrix.hd1, positive="1", negative="0")
baseline.fpr.hd0 <- 1.0 - specificity(baseline.conf.matrix.hd0, positive="1", negative="0")
baseline.precision.hd3 <- posPredValue(baseline.conf.matrix.hd3, positive="1", negative="0")
baseline.precision.hd2 <- posPredValue(baseline.conf.matrix.hd2, positive="1", negative="0")
baseline.precision.hd1 <- posPredValue(baseline.conf.matrix.hd1, positive="1", negative="0")
baseline.precision.hd0 <- posPredValue(baseline.conf.matrix.hd0, positive="1", negative="0")
baseline.results.hd.levels <- c("<=3", "<=2", "<=1", "=0")
baseline.results <- data.frame(hd=baseline.results.hd.levels,
                               sensitivity=c(baseline.sensitivity.hd3, baseline.sensitivity.hd2, baseline.sensitivity.hd1, baseline.sensitivity.hd0),
                               fpr=c(baseline.fpr.hd3, baseline.fpr.hd2, baseline.fpr.hd1, baseline.fpr.hd0),
                               precision=c(baseline.precision.hd3, baseline.precision.hd2, baseline.precision.hd1, baseline.precision.hd0))
baseline.results$hd <- factor(baseline.results$hd, levels=baseline.results.hd.levels)

#####################################################################
# Compute ROC and PR curves for different choices of Hamming distance,
# and separately for different choices of PFS

roc.df.all <- data.frame(roc$curve)
roc.df.all$hamming.dist <- rep("all", nrow(roc.df.all))
roc.df.all$cas13a.pfs <- rep("all", nrow(roc.df.all))
pr.df.all <- data.frame(pr$curve)
pr.df.all$hamming.dist <- rep("all", nrow(pr.df.all))
pr.df.all$cas13a.pfs <- rep("all", nrow(pr.df.all))

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
# Color represents threshold

p <- ggplot(data.frame(roc$curve))
p <- p + geom_line(aes(x=X1, y=X2, color=X3))
p <- p + geom_abline(slope=1, intercept=0, linetype="dotted")  # diagonal for random classifier
p <- p + geom_point(data=baseline.results, aes(x=fpr, y=sensitivity, shape=hd)) # dots for baseline
p <- p + xlab("FPR") + ylab("Sensitivity") + labs(color="Threshold", shape="Baseline HD")
p <- p + scale_color_viridis() # adjust color gradient
p <- p + theme_bw(base_size=18) # bw and larger font sizes
p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())  # remove grid lines
p.roc <- p
#####################################################################

#####################################################################
# Plot ROC curve
# Color represents choices of Hamming distance

p <- ggplot(hamming.dist.roc, aes(x=X1, y=X2))
p <- p + geom_line(aes(color=hamming.dist))
p <- p + geom_abline(slope=1, intercept=0, linetype="dotted")  # diagonal for random classifier
p <- p + xlab("FPR") + ylab("Sensitivity") + labs(color="Hamming distance")
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + theme_bw(base_size=18) # bw and larger font sizes
p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())  # remove grid lines
p.hamming.dist.roc <- p
#####################################################################

#####################################################################
# Plot ROC curve
# Color represents choices of Cas13a PFS

p <- ggplot(cas13a.pfs.roc, aes(x=X1, y=X2))
p <- p + geom_line(aes(color=cas13a.pfs))
p <- p + geom_abline(slope=1, intercept=0, linetype="dotted")  # diagonal for random classifier
p <- p + xlab("FPR") + ylab("Sensitivity") + labs(color="PFS")
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + theme_bw(base_size=18) # bw and larger font sizes
p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())  # remove grid lines
p.cas13a.pfs.roc <- p
#####################################################################

#####################################################################
# Plot PR curve
# Color represents threshold

random.precision <- length(pos) / nrow(test.results)

p <- ggplot(data.frame(pr$curve))
p <- p + geom_line(aes(x=X1, y=X2, color=X3))
p <- p + geom_hline(yintercept=random.precision, linetype="dotted")    # representing random classifier
p <- p + geom_point(data=baseline.results, aes(x=sensitivity, y=precision, shape=hd)) # dots for baseline
p <- p + xlab("Recall") + ylab("Precision") + labs(color="Threshold", shape="Baseline HD")
p <- p + ylim(0.8, 1.0)
p <- p + scale_color_viridis() # adjust color gradient
p <- p + theme_bw(base_size=18) # bw and larger font sizes
p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())  # remove grid lines
p.pr <- p
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
                          data.frame(hamming.dist=c("all"),
                                     precision=c(length(pos) / nrow(test.results))))

p <- ggplot(hamming.dist.pr, aes(x=X1, y=X2))
p <- p + geom_line(aes(color=hamming.dist))
p <- p + geom_hline(data=random.precision, aes(yintercept=precision, color=hamming.dist), linetype="dotted")    # representing random classifier
p <- p + xlab("Recall") + ylab("Precision") + labs(color="Hamming distance")
p <- p + ylim(0.5, 1.0)
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + theme_bw(base_size=18) # bw and larger font sizes
p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())  # remove grid lines
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
                          data.frame(cas13a.pfs=c("all"),
                                     precision=c(length(pos) / nrow(test.results))))

p <- ggplot(cas13a.pfs.pr, aes(x=X1, y=X2))
p <- p + geom_line(aes(color=cas13a.pfs))
p <- p + geom_hline(data=random.precision, aes(yintercept=precision, color=cas13a.pfs), linetype="dotted")    # representing random classifier
p <- p + xlab("Recall") + ylab("Precision") + labs(color="PFS")
p <- p + ylim(0.5, 1.0)
p <- p + scale_color_viridis(discrete=TRUE) # adjust color gradient
p <- p + theme_bw(base_size=18) # bw and larger font sizes
p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())  # remove grid lines
p.cas13a.pfs.pr <- p
#####################################################################

#####################################################################
# Produce PDF
g <- ggarrange(p.roc,
               p.pr,
               p.hamming.dist.roc,
               p.hamming.dist.pr,
               p.cas13a.pfs.roc,
               p.cas13a.pfs.pr,
               ncol=1)
ggsave(OUT.PDF, g, width=10, height=36, useDingbats=FALSE, limitsize=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
#####################################################################
