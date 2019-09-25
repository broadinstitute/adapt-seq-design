# Plot learning curve for predictor.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(dplyr)
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
learning.curve <- read.table(IN.TSV, header=TRUE, sep="\t")
names(learning.curve) <- gsub("_", ".", names(learning.curve))

# Summarize over the folds (splits) of the training data by grouping
# based on all the other variables
learning.curve.summarized <- summarySE(learning.curve,
                                       measurevar="value",
                                       groupvars=c("sampling.approach", "size", "dataset", "metric"))

# Only pull out a single metric
if ("mse" %in% learning.curve.summarized$metric) {
    # This was regression; use mse
    learning.curve.summarized <- learning.curve.summarized[learning.curve.summarized$metric == "mse", ]
    metric <- "MSE"
} else if ("bce" %in% learning.curve.summarized$metric) {
    # This was classification; use binary cross-entropy
    learning.curve.summarized <- learning.curve.summarized[learning.curve.summarized$metric == "bce", ]
    metric <- "Binary cross-entropy"
} else {
    stop("Unknown metric")
}

# Break apart by sampling approach for two separate plots
learning.curve.summarized.sampling.all <- learning.curve.summarized[learning.curve.summarized$sampling.approach == "sample_all", ]
learning.curve.summarized.sampling.crrnas <- learning.curve.summarized[learning.curve.summarized$sampling.approach == "sample_crrnas", ]

# Plot for sampling from all data points
# Plot mean value (across folds) with 95% confidence interval
p.sampling.all <- ggplot(learning.curve.summarized.sampling.all,
                         aes(x=size))
p.sampling.all <- p.sampling.all + geom_line(aes(y=value, color=dataset))
p.sampling.all <- p.sampling.all + geom_ribbon(aes(ymin=value-ci, ymax=value+ci, fill=dataset), alpha=0.5)
p.sampling.all <- p.sampling.all + xlim(0, max(learning.curve.summarized.sampling.all$size))
p.sampling.all <- p.sampling.all + xlab("Number of data points for training") + ylab(metric)
p.sampling.all <- p.sampling.all + ggtitle("Sampling from all data points")
p.sampling.all <- p.sampling.all + scale_color_viridis(discrete=TRUE)
p.sampling.all <- p.sampling.all + scale_fill_viridis(discrete=TRUE)

# Plot for sampling from crRNAs
# Plot mean value (across folds) with 95% confidence interval
p.sampling.crrnas <- ggplot(learning.curve.summarized.sampling.crrnas,
                         aes(x=size))
p.sampling.crrnas <- p.sampling.crrnas + geom_line(aes(y=value, color=dataset))
p.sampling.crrnas <- p.sampling.crrnas + geom_ribbon(aes(ymin=value-ci, ymax=value+ci, fill=dataset), alpha=0.5)
p.sampling.crrnas <- p.sampling.crrnas + xlim(0, max(learning.curve.summarized.sampling.crrnas$size))
p.sampling.crrnas <- p.sampling.crrnas + xlab("Number of crRNAs for training") + ylab(metric)
p.sampling.crrnas <- p.sampling.crrnas + ggtitle("Sampling from crRNAs")
p.sampling.crrnas <- p.sampling.crrnas + scale_color_viridis(discrete=TRUE)
p.sampling.crrnas <- p.sampling.crrnas + scale_fill_viridis(discrete=TRUE)

# Produce PDF
g <- arrangeGrob(p.sampling.all,
                 p.sampling.crrnas,
                 ncol=2)
ggsave(OUT.PDF, g, width=16, height=8, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
