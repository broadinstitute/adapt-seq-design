# Plot distribution of mean validation loss (over folds) for different
# choices of hyperparameters.
#
# This currently only looks at the advantage of locally connected layers.
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

# R renames the column '1.minus.rho.mean' to 'X1.minus.rho.mean' when reading; rename it back
colnames(hyperparams)[colnames(hyperparams) == "X1.minus.rho.mean"] <- "1.minus.rho.mean"

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

# Compute Spearman's rho
hyperparams$rho.mean <- 1.0 - hyperparams$`1.minus.rho.mean`

# Produce plots of MSE and rho
# x_string gives column name to plot as a string
# boxplot is FALSE (continuous) or TRUE (discrete)
plots <- function(x_string, x_label, title, boxplot, x_log) {
    # Produce plot of MSE
    p.mse <- ggplot(hyperparams, aes_string(x=x_string, y="mse.mean"))
    if (boxplot) {
        p.mse <- p.mse + geom_boxplot(outlier.shape=NA) + geom_jitter(width=0.2)
    } else {
        p.mse <- p.mse + geom_point()
    }
    p.mse <- p.mse + ylim(0, 2.0)   # ignore the outliers >2.0
    p.mse <- p.mse + ylab("Mean validation MSE")
    p.mse <- p.mse + xlab(x_label)
    p.mse <- p.mse + ggtitle(title)
    p.mse <- p.mse + theme_bw()
    p.mse <- p.mse + theme(axis.text=element_text(size=14),
                           axis.title=element_text(size=18))
    # Produce plot of rho
    p.rho <- ggplot(hyperparams, aes_string(x=x_string, y="rho.mean"))
    if (boxplot) {
        p.rho <- p.rho + geom_boxplot(outlier.shape=NA) + geom_jitter(width=0.2)
    } else {
        p.rho <- p.rho + geom_point()
    }
    p.rho <- p.rho + ylab("Mean validation Spearman's rho")
    p.rho <- p.rho + xlab(x_label)
    p.rho <- p.rho + ggtitle(title)
    p.rho <- p.rho + theme_bw()
    p.rho <- p.rho + theme(axis.text=element_text(size=14),
                           axis.title=element_text(size=18))

    if (x_log) {
        p.mse <- p.mse + scale_x_log10()
        p.rho <- p.rho + scale_x_log10()
    }

    l <- list(mse=p.mse, rho=p.rho)
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
p.sample.weight.scaling.factor <- plots("sample.weight.scaling.factor", "Sample weight scaling factor", "Sample weight scaling factor", FALSE, TRUE)

# Save plots to PDF
g <- arrangeGrob(p.has.lc.layer$mse,
                 p.has.lc.layer$rho,
                 p.lc.width$mse,
                 p.lc.width$rho,
                 p.lc.dim$mse,
                 p.lc.dim$rho,
                 p.has.conv.layer$mse,
                 p.has.conv.layer$rho,
                 p.conv.width$mse,
                 p.conv.width$rho,
                 p.conv.num.filters$mse,
                 p.conv.num.filters$rho,
                 p.l2.factor$mse,
                 p.l2.factor$rho,
                 p.learning.rate$mse,
                 p.learning.rate$rho,
                 p.batch.size$mse,
                 p.batch.size$rho,
                 p.add.gc.content$mse,
                 p.add.gc.content$rho,
                 p.skip.batch.norm$mse,
                 p.skip.batch.norm$rho,
                 p.sample.weight.scaling.factor$mse,
                 p.sample.weight.scaling.factor$rho,
                 ncol=2)
ggsave(OUT.PDF, g, width=8, height=48, useDingbats=FALSE, limitsize=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
