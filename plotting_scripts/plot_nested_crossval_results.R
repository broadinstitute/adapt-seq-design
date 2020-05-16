# Plot results of nested cross-validation on baselines and the predictor.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(plyr)
require(ggridges)
require(viridis)
require(scales)
require(ggpubr)

args <- commandArgs(trailingOnly=TRUE)
IN.BASELINES.TSV <- args[1]
IN.PREDICTOR.TSV <- args[2]
IN.CLASSIFICATION.TEST.RESULTS.TSV <- args[3]  # classification results on test data set; only used for computing baseline precision
OUT.PDF <- args[4]


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


# Read TSV of baseline results and replace '_' and '-' in column names with '.'
baseline.results <- read.table(gzfile(IN.BASELINES.TSV), header=TRUE, sep="\t")
names(baseline.results) <- gsub("_", ".", names(baseline.results))
names(baseline.results) <- gsub("-", ".", names(baseline.results))

# Read TSV of predictor results and replace '_' and '-' in column names with '.'
predictor.results <- read.table(gzfile(IN.PREDICTOR.TSV), header=TRUE, sep="\t")
names(predictor.results) <- gsub("_", ".", names(predictor.results))
names(predictor.results) <- gsub("-", ".", names(predictor.results))

# Add a column giving the 'model' for the CNN predictor; call it 'cnn'
predictor.results$model <- rep("cnn", nrow(predictor.results))

# Give the feature type (onehot) for the CNN predictor
predictor.results[predictor.results$model == "cnn", "feats.type"] <- "onehot"

# Combine models across baselines and predictor; use plyr's rbind.fill() because
# not all columns are present in all inputs
results <- rbind.fill(baseline.results, predictor.results)

# Remove the unregularized linear regression model ('lr'), which has a large error
results <- results[results$model != "lr", ]

# R renames the column '1.minus.rho' to 'X1.minus.rho' when reading; rename it back
# And similarly for 1.minus.auc.roc and 1.minus.auc.pr
colnames(results)[colnames(results) == "X1.minus.rho"] <- "1.minus.rho"
colnames(results)[colnames(results) == "X1.minus.auc.roc"] <- "1.minus.auc.roc"
colnames(results)[colnames(results) == "X1.minus.auc.pr"] <- "1.minus.auc.pr"

# Make feats.type and model be factor
results$feats.type <- factor(results$feats.type)
results$model <- factor(results$model)

# Compute baseline precision (precision of random classifier) from
# classification test results; note that this only uses/needs the
# true values, not the predicted values
classification.test.results <- read.table(IN.CLASSIFICATION.TEST.RESULTS.TSV, header=TRUE, sep="\t")
names(classification.test.results) <- gsub("_", ".", names(classification.test.results))
classifier.pos <- classification.test.results$predicted.activity[classification.test.results$true.activity == 1]
random.precision <- length(classifier.pos) / nrow(classification.test.results)

# Reorder and rename feature types
levels(results$feats.type) <- list("One-hot (1D)"="onehot-flat",
                                   "One-hot MM"="onehot-simple",
                                   "Handcrafted"="handcrafted",
                                   "One-hot MM + Handcrafted"="combined",
                                   "One-hot (2D)"="onehot")
if ('mse' %in% colnames(results)) {
    # Regression

    # Remove 'lr' model, which has no regularization
    results <- results[results$model != "lr",]

    # Reorder and rename models
    levels(results$model) <- list("L1 LR"="l1_lr", "L2 LR"="l2_lr",
                                  "L1L2 LR"="l1l2_lr", "GBT"="gbt",
                                  "RF"="rf", "MLP"="mlp", "LSTM"="lstm",
                                  "CNN"="cnn")

    # Pull out mse and make a plot of this
    results.mse <- results[, c("fold", "model", "feats.type", "mse")]
    # Summarize across folds
    results.mse.summarized <- summarySE(results.mse,
                                        measurevar="mse",
                                        groupvars=c("model", "feats.type"))
    p.mse <- ggplot(results.mse.summarized, aes(x=model, y=mse, fill=feats.type))
    p.mse <- p.mse + geom_bar(stat="identity", position=position_dodge(0.7), width=0.7)
    p.mse <- p.mse + geom_errorbar(aes(ymin=mse-ci, ymax=mse+ci), width=0.3, alpha=0.8, position=position_dodge(0.7))   # 95% CI
    p.mse <- p.mse + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
    p.mse <- p.mse + xlab("Model") + ylab("Mean squared error")
    p.mse <- p.mse + labs(fill="")	# no legend title
    p.mse <- p.mse + theme_pubr()

    # Pull out rho and make a plot of this
    # (only pull out rho if not already available; otherwise 1.minus.rho
    # may be NA)
    results$rho[is.na(results$rho)] <- 1.0 - results$`1.minus.rho`[is.na(results$rho)]
    results.rho <- results[, c("fold", "model", "feats.type", "rho")]
    # Summarize across folds
    results.rho.summarized <- summarySE(results.rho,
                                        measurevar="rho",
                                        groupvars=c("model", "feats.type"))
    p.rho <- ggplot(results.rho.summarized, aes(x=model, y=rho, fill=feats.type))
    p.rho <- p.rho + geom_bar(stat="identity", position=position_dodge(0.7), width=0.7)
    p.rho <- p.rho + geom_errorbar(aes(ymin=rho-ci, ymax=rho+ci), width=0.3, alpha=0.8, position=position_dodge(0.7))   # 95% CI
    p.rho <- p.rho + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
    p.rho <- p.rho + xlab("Model") + ylab("Spearman's rho")
    p.rho <- p.rho + labs(fill="")	# no legend title
    p.rho <- p.rho + theme_pubr()

    # Save to PDF
    g <- arrangeGrob(p.mse,
                     p.rho,
                     ncol=2)
    ggsave(OUT.PDF, g, width=16, height=4, useDingbats=FALSE)
} else {
    # Classification

    # Remove 'logit' model, which has no regularization
    results <- results[results$model != "logit",]

    # Reorder and rename models
    levels(results$model) <- list("L1 LR"="l1_logit", "L2 LR"="l2_logit",
                                  "L1L2 LR"="l1l2_logit", "GBT"="gbt",
                                  "RF"="rf", "SVM"="svm", "MLP"="mlp", "LSTM"="lstm",
                                  "CNN"="cnn")

    # Pull out auROC and make a plot of this
    # (only pull out auc.roc if not already available; otherwise 1.minus.auc.roc
    # may be NA)
    results$auc.roc[is.na(results$auc.roc)] <- 1.0 - results$`1.minus.auc.roc`[is.na(results$auc.roc)]
    results.auroc <- results[, c("fold", "model", "feats.type", "auc.roc")]
    # Summarize across folds
    results.auroc.summarized <- summarySE(results.auroc,
                                        measurevar="auc.roc",
                                        groupvars=c("model", "feats.type"))
    results.auroc.baseline <- 0.5   # auROC of random classifier
    p.auroc <- ggplot(results.auroc.summarized, aes(x=model, y=auc.roc, fill=feats.type))
    p.auroc <- p.auroc + geom_bar(stat="identity", position=position_dodge(0.7), width=0.7)
    p.auroc <- p.auroc + geom_errorbar(aes(ymin=auc.roc-ci, ymax=auc.roc+ci), width=0.3, alpha=0.8, position=position_dodge(0.7))   # 95% CI
    p.auroc <- p.auroc + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
    p.auroc <- p.auroc + geom_hline(yintercept=results.auroc.baseline, linetype="dotted")    # representing random classifier
    p.auroc <- p.auroc + scale_y_continuous(limits=c(0.4,1.0), oob=rescale_none)    # ylim() does not work, as described at https://stackoverflow.com/a/10365218
    p.auroc <- p.auroc + xlab("Model") + ylab("auROC")
    p.auroc <- p.auroc + labs(fill="")	# no legend title
    p.auroc <- p.auroc + theme_pubr()

    # Pull out auPR and make a plot of this
    # (only pull out auc.pr if not already available; otherwise 1.minus.auc.pr
    # may be NA)
    results$auc.pr[is.na(results$auc.pr)] <- 1.0 - results$`1.minus.auc.pr`[is.na(results$auc.pr)]
    results.aupr <- results[, c("fold", "model", "feats.type", "auc.pr")]
    # Summarize across folds
    results.aupr.summarized <- summarySE(results.aupr,
                                        measurevar="auc.pr",
                                        groupvars=c("model", "feats.type"))
    results.aupr.baseline <- random.precision # auPR of random classifier
    p.aupr <- ggplot(results.aupr.summarized, aes(x=model, y=auc.pr, fill=feats.type))
    p.aupr <- p.aupr + geom_bar(stat="identity", position=position_dodge(0.7), width=0.7)
    p.aupr <- p.aupr + geom_errorbar(aes(ymin=auc.pr-ci, ymax=auc.pr+ci), width=0.3, alpha=0.8, position=position_dodge(0.7))   # 95% CI
    p.aupr <- p.aupr + scale_fill_viridis(discrete=TRUE) # adjust fill gradient
    p.aupr <- p.aupr + geom_hline(yintercept=results.aupr.baseline, linetype="dotted")    # representing random classifier
    p.aupr <- p.aupr + scale_y_continuous(limits=c(0.8,1.0), oob=rescale_none)    # ylim() does not work, as described at https://stackoverflow.com/a/10365218
    p.aupr <- p.aupr + xlab("Model") + ylab("auPR")
    p.aupr <- p.aupr + labs(fill="")	# no legend title
    p.aupr <- p.aupr + theme_pubr()

    # Save to PDF
    g <- arrangeGrob(p.auroc,
                     p.aupr,
                     ncol=2)
    ggsave(OUT.PDF, g, width=16, height=4, useDingbats=FALSE)
}

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
