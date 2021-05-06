# Plot distribution of mean validation loss (across outer folds) and other
# metrics comparing the choice of not allowing or requiring a locally-connected
# (LC) layer.
#
# Unlike plot_hyperparam_loss_distribution.R, this looks at performance across
# outer folds using a nested cross-validation. On each hyperparameter search
# (across inner folds), the model is restricted to either not using or to using
# a locally-connected layer. The resulting values are on the validation data
# for each of the outer folds.
#
# Note that this plots BCE (for classification), but it is mostly meaningless.
# During training, the metrics is weighted in the loss function to correct for
# class imbalance. Because there is such extreme class imbalance, the
# unweighted BCE might not be very meaningful and is different than what the
# training seeks to optimize.
#
# By Hayden Metsky <hmetsky@broadinstitute.org>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(stringr)
require(ggpubr)
require(ggsignif)

args <- commandArgs(trailingOnly=TRUE)
IN.TSV <- args[1]
OUT.PDF <- args[2]

# Read hyperparameter results and replace '_'/'-' with '.' in column names
folds <- read.table(gzfile(IN.TSV), header=TRUE, sep="\t")
names(folds) <- gsub("_", ".", names(folds))

if ("X1.minus.auc.roc" %in% colnames(folds)) {
    # Classification
    # R renames the column '1.minus.auc-roc' to 'X1.minus.auc.roc' when reading; rename it back
    colnames(folds)[colnames(folds) == "X1.minus.auc.roc"] <- "1.minus.auc.roc"
    colnames(folds)[colnames(folds) == "X1.minus.auc.pr"] <- "1.minus.auc.pr"
    runType <- "classify"
} else if ("X1.minus.rho" %in% colnames(folds)) {
    # Regression
    # R renames the column '1.minus.rho' to 'X1.minus.rho' when reading; rename it back
    colnames(folds)[colnames(folds) == "X1.minus.rho"] <- "1.minus.rho"
    runType <- "regress"
} else {
    stop("Unknown whether classification or regression")
}

# Add a column giving whether or not a choice of hyperparameters includes
# a locally connected layer (note the '!' in front of 'grepl')
folds$has.lc.layer <- !grepl("locally_connected_width: None",
                                   folds$best.params,
                                   fixed=TRUE)

if (runType == "classify") {
    # Compute AUC-ROC and AUC-PR
    folds$auc.roc <- 1.0 - folds$`1.minus.auc.roc`
    folds$auc.pr <- 1.0 - folds$`1.minus.auc.pr`
} else if (runType == "regress") {
    # Compute Spearman's rho
    folds$rho <- 1.0 - folds$`1.minus.rho`
}

# Produce plots of loss and a measurement (e.g., MSE and rho, for regression)
# x_string gives column name to plot as a string
if (runType == "classify") {
    loss.col <- "bce"
    loss.name <- "BCE"
    measure.col <- "auc.roc"
    measure.name <- "auROC"
    measure2.col <- "auc.pr"
    measure2.name <- "auPR"
} else if (runType == "regress") {
    loss.col <- "mse"
    loss.name <- "MSE"
    measure.col <- "rho"
    measure.name <- "Spearman correlation"
}

# Produce plot of MSE
p.loss <- ggplot(folds, aes_string(x="has.lc.layer", y=loss.col))
p.loss <- p.loss + geom_point(size=1)
p.loss <- p.loss + geom_line(aes(group=fold))   # draw a line connecting the pair of points for each fold
p.loss <- p.loss + ylab(paste("Validation", loss.name))
p.loss <- p.loss + xlab("Uses LC layer")
p.loss <- p.loss + theme_pubr()
p.loss <- p.loss + theme(plot.margin=margin(t=10,r=10,b=10,l=10)) # increase spacing between plots

# Produce plot of measure
p.measure <- ggplot(folds, aes_string(x="has.lc.layer", y=measure.col))
p.measure <- p.measure + geom_point(size=1)
p.measure <- p.measure + geom_line(aes(group=fold))   # draw a line connecting the pair of points for each fold
p.measure <- p.measure + ylab(paste("Validation", measure.name))
p.measure <- p.measure + xlab("Uses LC layer")
p.measure <- p.measure + theme_pubr()
p.measure <- p.measure + theme(plot.margin=margin(t=10,r=10,b=10,l=10)) # increase spacing between plots

# Do a paired t-test with a 'less' alternative hypothesis (that x has a lower
# mean than y) for measure
print(paste("Paired t-test for", measure.name))
folds.widened.measure <- dcast(folds, formula=fold~has.lc.layer, value.var=measure.col)
measure.lc.no <- folds.widened.measure[[2]]
measure.lc.yes <- folds.widened.measure[[3]]
measure.t.test <- t.test(measure.lc.no, measure.lc.yes, alternative="less", paired=TRUE)
print(measure.t.test)

if (runType == "classify") {
    # Produce plot of measure2
    p.measure2 <- ggplot(folds, aes_string(x="has.lc.layer", y=measure2.col))
    p.measure2 <- p.measure2 + geom_point(size=1)
    p.measure2 <- p.measure2 + geom_line(aes(group=fold))   # draw a line connecting the pair of points for each fold
    p.measure2 <- p.measure2 + ylab(paste("Validation", measure2.name))
    p.measure2 <- p.measure2 + xlab("Uses LC layer")
    p.measure2 <- p.measure2 + theme_pubr()
    p.measure2 <- p.measure2 + theme(plot.margin=margin(t=10,r=10,b=10,l=10)) # increase spacing between plots

    # Do a paired t-test with a 'less' alternative hypothesis (that x has a lower
    # mean than y) for measure2
    print(paste("Paired t-test for", measure2.name))
    folds.widened.measure2 <- dcast(folds, formula=fold~has.lc.layer, value.var=measure2.col)
    measure2.lc.no <- folds.widened.measure2[[2]]
    measure2.lc.yes <- folds.widened.measure2[[3]]
    measure2.t.test <- t.test(measure2.lc.no, measure2.lc.yes, alternative="less", paired=TRUE)
    print(measure2.t.test)
}

# Save plots to PDF
if (runType == "classify") {
    g <- arrangeGrob(p.loss,
                     p.measure,
                     p.measure2,
                     nrow=1)
} else {
    g <- arrangeGrob(p.loss,
                     p.measure,
                     nrow=1)
}
plot.width <- 3
ggsave(OUT.PDF, g, width=plot.width*3, height=plot.width*1, useDingbats=FALSE, limitsize=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
