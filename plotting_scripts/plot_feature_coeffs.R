# Plot results of feature weights measured by baseline models.
#
# For example, with L1 logistic or linear regression the coefficients
# in front of features represent their significance. This plots
# the top N features according to their absolute balue.
#
# By Hayden Metsky <hayden@mit.edu>

require(ggplot2)
require(gridExtra)
require(ggpubr)
require(stringr)

args <- commandArgs(trailingOnly=TRUE)
IN.FEATURE.COEFFS.TSV <- args[1]
OUT.PDF <- args[2]

# Show the best 20 features (ranked by mean of abs. value across splits)
TOP.N <- 20


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


# Read TSV of feature.coeffs results and replace '_' in column names with '.'
feature.coeffs <- read.table(gzfile(IN.FEATURE.COEFFS.TSV), header=TRUE, sep="\t")
names(feature.coeffs) <- gsub("_", ".", names(feature.coeffs))

# Summarize results across splits (i.e., mean for each coefficient for each
# input type, taken across splits)
feature.coeffs.summarized <- summarySE(feature.coeffs,
                                       measurevar="coeff",
                                       groupvars=c("model", "feats.type", "feat.description"))

# Only used the 'combined' feats.type (handcrafted and one-hot features)
feature.coeffs.summarized <- feature.coeffs.summarized[feature.coeffs.summarized$feats.type == "combined", ]

# Sort by abs(mean coeff) and pull out the highest TOP.N coefficients for
# each group of model,feats.type
require(dplyr)
feature.coeffs.summarized$coeff.abs <- abs(feature.coeffs.summarized$coeff)
data <- feature.coeffs.summarized %>%
    group_by(model, feats.type) %>%
    arrange(desc(coeff.abs), .by_group=TRUE) %>%
    top_n(n=TOP.N, wt=coeff.abs)
data <- data.frame(data)

# Rename some feat.description values
feat.replace <- function(from, to) {
    # '<<-' is needed to modify outside scope
    data$feat.description <<- str_replace_all(data$feat.description, from, to)
}
feat.replace("target-after-0-([ACGT])", "PFS = \\1")   # PFS
feat.replace("target-after-1-([ACGT])", "PFS+1 = \\1")   # 1 after PFS
feat.replace("num-mismatches-seed", "Mismatches in seed")
feat.replace("num-mismatches-nonseed", "Mismatches outside seed")
feat.replace("guide-mismatch-allele-([0-9]+)-([ACGT])", "Mismatch at \\1 = \\2")
feat.replace("target-at-guide-([0-9]+)-([ACGT])", "Match at \\1 = \\2")

plot.for.model <- function(model, model.name) {
    # Produce plot for model
    data.for.model <- data.frame(data[data$model == model, ])

    # Make feat.description be a factor, preserving its order
    data.for.model$feat.description <- factor(data.for.model$feat.description, levels=unique(data.for.model$feat.description))

    p <- ggplot(data.for.model, aes(x=feat.description, y=coeff))
    p <- p + geom_bar(stat="identity")
    p <- p + geom_errorbar(aes(ymin=coeff-ci, ymax=coeff+ci), width=0.3)
    p <- p + xlab("Feature") + ylab("Coefficient")
    p <- p + ggtitle(model.name)
    p <- p + theme_pubr()
    p <- p + theme(axis.text.x=element_text(angle=45, hjust=1), # rotate x labels
                   plot.margin=margin(t=10, r=10, b=10, l=50)) # add margin on left so x label is not cutoff
    return(p)
}

if ("lr" %in% data$model) {
    # Linear regression models
    p.no.regularization <- plot.for.model("lr", "Linear regression")
    p.l1 <- plot.for.model("l1_lr", "L1 linear regression")
    p.l2 <- plot.for.model("l2_lr", "L2 linear regression")
    p.l1l2 <- plot.for.model("l1l2_lr", "L1+L2 linear regression")
} else if ("logit" %in% data$model) {
    # Classification models
    p.no.regularization <- plot.for.model("logit", "Logistic regression")
    p.l1 <- plot.for.model("l1_logit", "L1 logistic regression")
    p.l2 <- plot.for.model("l2_logit", "L2 logistic regression")
    p.l1l2 <- plot.for.model("l1l2_logit", "L1+L2 logistic regression")
} else {
    stop("Unknown whether this is regression or classification")
}

g <- arrangeGrob(#p.no.regularization,
                 p.l1,
                 p.l2,
                 p.l1l2,
                 ncol=1)
ggsave(OUT.PDF, g, width=10, height=16, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
