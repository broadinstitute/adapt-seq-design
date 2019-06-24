# Plot distribution of mean validation loss (over folds) for different
# choices of hyperparameters.
#
# This currently only looks at the advantage of locally connected layers.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(reshape2)
require(stringr)

args <- commandArgs(trailingOnly=TRUE)
cas <- args[1]  # 'cas9' or 'cas13'

IN.RESULTS <- paste0("out/", cas, "-hyperparam-search.random.tsv")
OUT.PDF <- paste0("out/", cas, "-hyperparam-search.random.distributions.pdf")

# Read hyperparameter results
hyperparams <- read.table(IN.RESULTS, header=FALSE, sep="\t")
names(hyperparams) <- c("hyperparams", "mean.val.loss")

# Add a column giving whether or not a choice of hyperparameters includes
# a locally connected layer (note the '!' in front of 'grepl')
hyperparams$has.lc.layer <- !grepl("locally_connected_width: None",
                                   hyperparams$hyperparams,
                                   fixed=TRUE)

# Add a column giving the particular locally connected width choice
# If has.lc.layer is FALSE, the width will not have been able to have been
# parsed, so set it to None
lc.width.pattern <- regex("locally_connected_width: (\\[.+\\]),")
hyperparams$lc.width <- str_match(hyperparams$hyperparams,
                                  lc.width.pattern)[,2]
hyperparams[hyperparams$has.lc.layer == FALSE, ]$lc.width <- "None"
hyperparams$lc.width <- factor(hyperparams$lc.width)

# Add a column giving the dimension of the locally connected layer; if there
# is no layer, the dimension (a random choice) is meaningless for the model,
# so set it to 0
lc.dim.pattern <- regex("locally_connected_dim: (\\d+),")
hyperparams$lc.dim <- str_match(hyperparams$hyperparams,
                                lc.dim.pattern)[,2]
hyperparams[hyperparams$has.lc.layer == FALSE, ]$lc.dim <- 0

# Plot a box plots for each choice of has.lc.layer
p <- ggplot(hyperparams, aes(has.lc.layer, mean.val.loss))
p <- p + geom_boxplot(outlier.shape=NA) + geom_jitter(width=0.2)

p <- p + ylab("Mean validation loss")
p <- p + xlab("Uses LC layer")

# Make axis labels bigger
p <- p + theme(axis.text=element_text(size=14),
               axis.title=element_text(size=18))

# Save plot to PDF
p <- p + ggsave(OUT.PDF, width=5, height=6, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
