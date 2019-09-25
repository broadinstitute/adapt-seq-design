# Plot results of nested cross-validation on baselines and the predictor.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(plyr)
require(ggridges)
require(viridis)

args <- commandArgs(trailingOnly=TRUE)
IN.BASELINES.TSV <- args[1]
IN.PREDICTOR.TSV <- args[2]
OUT.PDF <- args[3]


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


# Read TSV of baseline results and replace '_' in column names with '.'
baseline.results <- read.table(IN.BASELINES.TSV, header=TRUE, sep="\t")
names(baseline.results) <- gsub("_", ".", names(baseline.results))

# Read TSV of predictor results and replace '_' in column names with '.'
predictor.results <- read.table(IN.PREDICTOR.TSV, header=TRUE, sep="\t")
names(predictor.results) <- gsub("_", ".", names(predictor.results))

# Add a column giving the 'model' for the main predictor; call it 'adapt'
predictor.results$model <- rep("adapt", nrow(predictor.results))

# Combine models across baselines and predictor; use plyr's rbind.fill() because
# not all columns are present in all inputs
results <- rbind.fill(baseline.results, predictor.results)

# Remove the unregularized linear regression model ('lr'), which has a large error
results <- results[results$model != "lr", ]

# R renames the column '1.minus.rho' to 'X1.minus.rho' when reading; rename it back
colnames(results)[colnames(results) == "X1.minus.rho"] <- "1.minus.rho"

if ('mse' %in% colnames(results)) {
    # Regression

    # Order the models explicitly
    results$model <- factor(results$model,
                            levels=c("lr", "l1_lr", "l2_lr", "l1l2_lr", "gbrt", "adapt"))

    # Pull out mse and make a plot of this
    results.mse <- results[, c("fold", "model", "mse")]
    # Summarize across folds
    results.mse.summarized <- summarySE(results.mse,
                                        measurevar="mse",
                                        groupvars=c("model"))
    p.mse <- ggplot(results.mse.summarized, aes(x=model, y=mse))
    p.mse <- p.mse + geom_bar(aes(fill=model), stat="identity")
    p.mse <- p.mse + geom_errorbar(aes(ymin=mse-ci, ymax=mse+ci), width=0.1, alpha=0.5)   # 95% CI
    p.mse <- p.mse + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
    p.mse <- p.mse + xlab("Model") + ylab("Mean squared error")

    # Pull out rho and make a plot of this
    results$rho <- 1.0 - results$`1.minus.rho`
    results.rho <- results[, c("fold", "model", "rho")]
    # Summarize across folds
    results.rho.summarized <- summarySE(results.rho,
                                        measurevar="rho",
                                        groupvars=c("model"))
    p.rho <- ggplot(results.rho.summarized, aes(x=model, y=rho))
    p.rho <- p.rho + geom_bar(aes(fill=model), stat="identity")
    p.rho <- p.rho + geom_errorbar(aes(ymin=rho-ci, ymax=rho+ci), width=0.1, alpha=0.5)   # 95% CI
    p.rho <- p.rho + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
    p.rho <- p.rho + xlab("Model") + ylab("Spearman's rho")

    # Save to PDF
    g <- arrangeGrob(p.mse,
                     p.rho,
                     ncol=2)
    ggsave(OUT.PDF, g, width=16, height=8, useDingbats=FALSE)
} else {
    # Classification

    # Order the models explicitly
    results$model <- factor(results$model,
                            levels=c("l1", "l2", "adaot"))

    # Pull out roc_auc and make a plot of this
    results.roc_auc <- results[, c("fold", "model", "roc_auc")]
    # Summarize across folds
    results.roc_auc.summarized <- summarySE(results.roc_auc,
                                        measurevar="roc_auc",
                                        groupvars=c("model"))
    p.roc_auc <- ggplot(results.roc_auc.summarized, aes(x=model, y=roc_auc))
    p.roc_auc <- p.roc_auc + geom_bar(aes(fill=model), stat="identity")
    p.roc_auc <- p.roc_auc + geom_errorbar(aes(ymin=roc_auc-ci, ymax=roc_auc+ci), width=0.1, alpha=0.5)   # 95% CI
    p.roc_auc <- p.roc_auc + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
    p.roc_auc <- p.roc_auc + xlab("Model") + ylab("auROC")

    # Save to PDF
    g <- arrangeGrob(p.roc_auc,
                     ncol=1)
    ggsave(OUT.PDF, g, width=8, height=8, useDingbats=FALSE)
}

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
