# Plot results of testing predictor, for classification.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(dplyr)
require(viridis)
require(PRROC)

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


#####################################################################
# Plot ROC curve

p <- ggplot(data.frame(roc$curve), aes(x=X1, y=X2, color=X3))
p <- p + geom_line()
p <- p + geom_abline(slope=1, intercept=0, linetype="dotted")  # diagonal for random classifier
p <- p + xlab("FPR") + ylab("Sensitivity") + labs(color="Threshold")
p <- p + scale_color_viridis() # adjust color gradient
p <- p + theme_bw(base_size=18) # bw and larger font sizes
p.roc <- p
#####################################################################

#####################################################################
# Plot PR curve

random.precision <- length(pos) / nrow(test.results)

p <- ggplot(data.frame(pr$curve), aes(x=X1, y=X2, color=X3))
p <- p + geom_line()
p <- p + geom_hline(yintercept=random.precision, linetype="dotted")    # representing random classifier
p <- p + xlab("Recall") + ylab("Precision") + labs(color="Threshold")
p <- p + ylim(0.8, 1.0)
p <- p + scale_color_viridis() # adjust color gradient
p <- p + theme_bw(base_size=18) # bw and larger font sizes
p.pr <- p
#####################################################################

#####################################################################
# Produce PDF
g <- arrangeGrob(p.roc,
                 p.pr,
                 ncol=1)
ggsave(OUT.PDF, g, width=8, height=16, useDingbats=FALSE, limitsize=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
#####################################################################
