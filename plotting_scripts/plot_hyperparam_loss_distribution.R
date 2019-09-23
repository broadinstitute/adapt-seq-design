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
hyperparams <- read.table(IN.TSV, header=TRUE, sep="\t")
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
lc.width.pattern <- regex("locally_connected_width: (\\[.+\\]),")
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

# Compute Spearman's rho
hyperparams$rho.mean <- 1.0 - hyperparams$`1.minus.rho.mean`

# Plot a box plots for each choice of has.lc.layer
# Produce plot of MSE
p.mse <- ggplot(hyperparams, aes(x=has.lc.layer, y=mse.mean))
p.mse <- p.mse + geom_boxplot(outlier.shape=NA) + geom_jitter(width=0.2)
p.mse <- p.mse + ylim(0, 0.75)   # ignore the outliers >1.0
p.mse <- p.mse + ylab("Mean validation MSE")
p.mse <- p.mse + xlab("Uses LC layer")
p.mse <- p.mse + theme_bw()
p.mse <- p.mse + theme(axis.text=element_text(size=14),
                       axis.title=element_text(size=18))
# Produce plot of rho
p.rho <- ggplot(hyperparams, aes(x=has.lc.layer, y=rho.mean))
p.rho <- p.rho + geom_boxplot(outlier.shape=NA) + geom_jitter(width=0.2)
p.rho <- p.rho + ylab("Mean validation Spearman's rho")
p.rho <- p.rho + xlab("Uses LC layer")
p.rho <- p.rho + theme_bw()
p.rho <- p.rho + theme(axis.text=element_text(size=14),
                       axis.title=element_text(size=18))

# Save plot to PDF
g <- arrangeGrob(p.mse,
                 p.rho,
                 ncol=1)
ggsave(OUT.PDF, g, width=8, height=16, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
