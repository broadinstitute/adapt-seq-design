# Plot distribution of mean validation loss (over folds) for different
# choices of hyperparameters.
#
# Note that, for regression, there can be warnings about missing values; this
# is because, for some models, the output is constant and Spearman's rho is
# nan.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(stringr)

args <- commandArgs(trailingOnly=TRUE)
IN.TSV <- args[1]
OUT.PDF <- args[2]

# Read hyperparameter results and replace '_' with '.' in column names
hyperparams <- read.table(gzfile(IN.TSV), header=TRUE, sep="\t")
names(hyperparams) <- gsub("_", ".", names(hyperparams))

if ("X1.minus.auc.roc.mean" %in% colnames(hyperparams)) {
    # Classification
    # R renames the column '1.minus.auc-roc.mean' to 'X1.minus.auc.roc.mean' when reading; rename it back
    colnames(hyperparams)[colnames(hyperparams) == "X1.minus.auc.roc.mean"] <- "1.minus.auc.roc.mean"
    runType <- "classify"
} else if ("X1.minus.rho.mean" %in% colnames(hyperparams)) {
    # Regression
    # R renames the column '1.minus.rho.mean' to 'X1.minus.rho.mean' when reading; rename it back
    colnames(hyperparams)[colnames(hyperparams) == "X1.minus.rho.mean"] <- "1.minus.rho.mean"
    runType <- "regress"
} else {
    stop("Unknown whether classification or regression")
}

# Add a column giving whether or not a choice of hyperparameters includes
# a locally connected layer (note the '!' in front of 'grepl')
hyperparams$has.lc.layer <- !grepl("locally_connected_width: None",
                                   hyperparams$params,
                                   fixed=TRUE)

# Add a column giving the particular locally connected width choice
# If has.lc.layer is FALSE, the width will not have been able to have been
# parsed, so set it to None
lc.width.pattern <- regex("locally_connected_width: (\\[.+?\\]),")
hyperparams$lc.width <- str_match(hyperparams$params,
                                  lc.width.pattern)[,2]
hyperparams[hyperparams$has.lc.layer == FALSE, ]$lc.width <- "None"
hyperparams$lc.width <- factor(hyperparams$lc.width)

# Add a column giving the dimension of the locally connected layer; if there
# is no layer, the dimension (a random choice) is meaningless for the model,
# so set it to 0
lc.dim.pattern <- regex("locally_connected_dim: (\\d+),")
hyperparams$lc.dim <- str_match(hyperparams$params,
                                lc.dim.pattern)[,2]
hyperparams[hyperparams$has.lc.layer == FALSE, ]$lc.dim <- 0

# Add a column giving the L2 factor
l2.factor.pattern <- regex("l2_factor: (.+?),")
hyperparams$l2.factor <- str_match(hyperparams$params,
                                   l2.factor.pattern)[,2]
hyperparams$l2.factor <- as.numeric(hyperparams$l2.factor)

# Add a column giving whether GC content was added
add.gc.content.pattern <- regex("add_gc_content: (False|True),")
hyperparams$add.gc.content <- str_match(hyperparams$params,
                                        add.gc.content.pattern)[,2]
hyperparams$add.gc.content <- factor(hyperparams$add.gc.content)

# Add a column giving whether to skip batch normalization
skip.batch.norm.pattern <- regex("skip_batch_norm: (False|True),")
hyperparams$skip.batch.norm <- str_match(hyperparams$params,
                                         skip.batch.norm.pattern)[,2]
hyperparams$skip.batch.norm <- factor(hyperparams$skip.batch.norm)

# Add a column giving the sample weight scaling factor
sample.weight.scaling.factor.pattern <- regex("sample_weight_scaling_factor: (.+?),")
hyperparams$sample.weight.scaling.factor <- str_match(hyperparams$params,
                                                      sample.weight.scaling.factor.pattern)[,2]
hyperparams$sample.weight.scaling.factor <- as.numeric(hyperparams$sample.weight.scaling.factor)

# Add a column giving the batch size
batch.size.pattern <- regex("batch_size: (\\d+),")
hyperparams$batch.size <- str_match(hyperparams$params,
                                    batch.size.pattern)[,2]
hyperparams$batch.size <- as.numeric(hyperparams$batch.size)

# Add a column giving the learning rate
learning.rate.pattern <- regex("learning_rate: (.+?),")
hyperparams$learning.rate <- str_match(hyperparams$params,
                                       learning.rate.pattern)[,2]
hyperparams$learning.rate <- as.numeric(hyperparams$learning.rate)

# Add a column giving whether or not a choice of hyperparameters includes
# a convolutional layer (note the '!' in front of 'grepl')
hyperparams$has.conv.layer <- !grepl("conv_filter_width: None",
                                     hyperparams$params,
                                     fixed=TRUE)

# Add a column giving the particular convolutional width choice
# If has.conv.layer is FALSE, the width will not have been able to have been
# parsed, so set it to None
conv.filter.width.pattern <- regex("conv_filter_width: (\\[.+?\\]),")
hyperparams$conv.filter.width <- str_match(hyperparams$params,
                                           conv.filter.width.pattern)[,2]
hyperparams[hyperparams$has.conv.layer == FALSE, ]$conv.filter.width <- "None"
hyperparams$conv.filter.width <- factor(hyperparams$conv.filter.width)

# Add a column giving the number of convolutional filters (dimension); if there
# is no layer, the dimension (a random choice) is meaningless for the model,
# so set it to 0
conv.num.filters.pattern <- regex("conv_num_filters: (\\d+),")
hyperparams$conv.num.filters <- str_match(hyperparams$params,
                                          conv.num.filters.pattern)[,2]
hyperparams[hyperparams$has.conv.layer == FALSE, ]$conv.num.filters <- 0
hyperparams$conv.num.filters <- as.numeric(hyperparams$conv.num.filters)

if (runType == "classify") {
    # Compute AUC-ROC
    hyperparams$auc.roc.mean <- 1.0 - hyperparams$`1.minus.auc.roc.mean`
} else if (runType == "regress") {
    # Compute Spearman's rho
    hyperparams$rho.mean <- 1.0 - hyperparams$`1.minus.rho.mean`
}

# Produce plots of loss and a measurement (e.g., MSE and rho, for regression)
# x_string gives column name to plot as a string
# boxplot is FALSE (continuous) or TRUE (discrete)
plots <- function(x_string, x_label, title, boxplot, x_log) {
    if (runType == "classify") {
        loss.col <- "bce.mean"
        loss.name <- "BCE"
        measure.col <- "auc.roc.mean"
        measure.name <- "AUC-ROC"
    } else if (runType == "regress") {
        loss.col <- "mse.mean"
        loss.name <- "MSE"
        measure.col <- "rho.mean"
        measure.name <- "Spearman's rho"
    }
    # Produce plot of MSE
    p.loss <- ggplot(hyperparams, aes_string(x=x_string, y=loss.col))
    if (boxplot) {
        p.loss <- p.loss + geom_boxplot(outlier.shape=NA) + geom_jitter(width=0.2)
    } else {
        p.loss <- p.loss + geom_point()
    }
    p.loss <- p.loss + ylim(0, 2.0)   # ignore the outliers >2.0
    p.loss <- p.loss + ylab(paste("Mean validation", loss.name))
    p.loss <- p.loss + xlab(x_label)
    p.loss <- p.loss + ggtitle(title)
    p.loss <- p.loss + theme_bw()
    p.loss <- p.loss + theme(axis.text=element_text(size=14),
                           axis.title=element_text(size=18))
    # Produce plot of measure
    p.measure <- ggplot(hyperparams, aes_string(x=x_string, y=measure.col))
    if (boxplot) {
        p.measure <- p.measure + geom_boxplot(outlier.shape=NA) + geom_jitter(width=0.2)
    } else {
        p.measure <- p.measure + geom_point()
    }
    p.measure <- p.measure + ylab(paste("Mean validation", measure.name))
    p.measure <- p.measure + xlab(x_label)
    p.measure <- p.measure + ggtitle(title)
    p.measure <- p.measure + theme_bw()
    p.measure <- p.measure + theme(axis.text=element_text(size=14),
                           axis.title=element_text(size=18))

    if (x_log) {
        p.loss <- p.loss + scale_x_log10()
        p.measure <- p.measure + scale_x_log10()
    }

    l <- list(loss=p.loss, measure=p.measure)
    return(l)
}

p.has.lc.layer <- plots("has.lc.layer", "Uses LC layer", "LC layer", TRUE, FALSE)
p.lc.width <- plots("lc.width", "LC width", "LC width", TRUE, FALSE)
p.lc.dim <- plots("lc.dim", "LC dimension", "LC dimension", TRUE, FALSE)

p.has.conv.layer <- plots("has.conv.layer", "Uses convolutional layer", "Convolutional layer", TRUE, FALSE)
p.conv.width <- plots("conv.filter.width", "Conv width", "Conv width", TRUE, FALSE)
p.conv.num.filters <- plots("conv.num.filters", "Conv dimension (num filters)", "Conv dimension (num filters)", FALSE, FALSE)

p.l2.factor <- plots("l2.factor", "L2 factor", "L2 factor", FALSE, TRUE)
p.batch.size <- plots("batch.size", "Batch size", "Batch size", FALSE, FALSE)
p.learning.rate <- plots("learning.rate", "Learning rate", "Learning rate", FALSE, TRUE)
p.skip.batch.norm <- plots("skip.batch.norm", "Skipped batch norm", "Skipped batch norm", TRUE, FALSE)

p.add.gc.content <- plots("add.gc.content", "Added in GC content", "Added in GC content", TRUE, FALSE)
p.sample.weight.scaling.factor <- plots("sample.weight.scaling.factor", "Sample weight scaling factor", "Sample weight scaling factor", FALSE, FALSE)

# Save plots to PDF
g <- arrangeGrob(p.has.lc.layer$loss,
                 p.has.lc.layer$measure,
                 p.lc.width$loss,
                 p.lc.width$measure,
                 p.lc.dim$loss,
                 p.lc.dim$measure,
                 p.has.conv.layer$loss,
                 p.has.conv.layer$measure,
                 p.conv.width$loss,
                 p.conv.width$measure,
                 p.conv.num.filters$loss,
                 p.conv.num.filters$measure,
                 p.l2.factor$loss,
                 p.l2.factor$measure,
                 p.learning.rate$loss,
                 p.learning.rate$measure,
                 p.batch.size$loss,
                 p.batch.size$measure,
                 p.add.gc.content$loss,
                 p.add.gc.content$measure,
                 p.skip.batch.norm$loss,
                 p.skip.batch.norm$measure,
                 p.sample.weight.scaling.factor$loss,
                 p.sample.weight.scaling.factor$measure,
                 ncol=2)
ggsave(OUT.PDF, g, width=8, height=48, useDingbats=FALSE, limitsize=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
