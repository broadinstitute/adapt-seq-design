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
require(viridis)

args <- commandArgs(trailingOnly=TRUE)
IN.FEATURE.COEFFS.TSV <- args[1]
OUT.PDF.DIR <- args[2]

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
data <- feature.coeffs.summarized

# Determine position along the target
# Number these such that the 5' end of the protospacer is position 1
# and the 3' end is position 28; the PFS (nt immediately 3' of the protospacer
# is position 29; the position immediately 5' of the protospacer is position 0;
# and the most 5' end of the target sequence (10 nt upstream of the start of
# the protospacer) is position -9
# Note that target-before-* positions start at 10 and end at 19 because
# their positions are with respect to a full 20 nt of context before the
# protospacer, but as features we only used the 10 nt adjacent to the
# protospacer (so 10 should be -9 and 19 should be 0)
data$feat.str.pos <- str_replace_all(data$feat.description,
                                     ".+-([0-9]+)-([ACGT])", "\\1")
data$feat.str.pos <- as.numeric(data$feat.str.pos)
data$allele <- str_replace_all(data$feat.description,
                               ".+-([0-9]+)-([ACGT])", "\\2")
data$allele <- factor(data$allele, levels=c("A", "C", "G", "T"))
data$target.pos <- NA
data$target.pos <- ifelse(grepl("target-before-", data$feat.description),
                          data$feat.str.pos - 19,
                          data$target.pos)
data$target.pos <- ifelse(grepl("target-at-guide-", data$feat.description),
                          data$feat.str.pos + 1,
                           data$target.pos)
data$target.pos <- ifelse(grepl("target-after-", data$feat.description),
                          data$feat.str.pos + 29,
                          data$target.pos)
data$target.pos <- ifelse(grepl("guide-mismatch-", data$feat.description),
                          data$feat.str.pos + 1,
                          data$target.pos)

# Rename some feat.description values with prettier versions
data$feat.description.pretty <- data$feat.description
feat.pretty <- function(from, to) {
    # '<<-' is needed to modify outside scope
    data$feat.description.pretty <<- str_replace_all(data$feat.description.pretty, from, to)
}
#feat.pretty("target-after-0-([ACGT])", "PFS = \\1")   # PFS
#feat.pretty("target-after-1-([ACGT])", "PFS+1 = \\1")   # 1 after PFS
feat.pretty("pfs-([ACGT][ACGT])", "PFS = \\1")   # 2 nt PFS
feat.pretty("num-mismatches", "Mismatch count")
#feat.pretty("guide-mismatch-allele-([0-9]+)-([ACGT])", "Mismatch at \\1 = \\2")
#feat.pretty("target-at-guide-([0-9]+)-([ACGT])", "Match at \\1 = \\2")
data$feat.description.pretty <- ifelse(grepl("target-", data$feat.description),
                                       paste0("Target @ ", data$target.pos, " = ", data$allele),
                                       data$feat.description.pretty)
data$feat.description.pretty <- ifelse(grepl("guide-mismatch-", data$feat.description),
                                       paste0("MM: pos. ", data$target.pos, " = ", data$allele),
                                       data$feat.description.pretty)

##############################
# Prepare for combined feature plots (handcrafted and one-hot features)
# These show features sorted by importance (coefficient value)

# Only used the 'combined' feats.type
data.combined <- data[data$feats.type == "combined", ]

# Sort by abs(mean coeff) and pull out the highest TOP.N coefficients for
# each group of model,feats.type
require(dplyr)
data.combined$coeff.abs <- abs(data.combined$coeff)
data.combined <- data.combined %>%
    group_by(model, feats.type) %>%
    arrange(desc(coeff.abs), .by_group=TRUE) %>%
    top_n(n=TOP.N, wt=coeff.abs)
data.combined <- data.frame(data.combined)

plot.combined.feats.for.model <- function(model, model.name) {
    # Produce plot for model
    data.for.model <- data.frame(data.combined[data.combined$model == model, ])

    # Make feat.description.pretty be a factor, preserving its order
    data.for.model$feat.description.pretty <- factor(data.for.model$feat.description.pretty, levels=unique(data.for.model$feat.description.pretty))

    p <- ggplot(data.for.model, aes(y=feat.description.pretty, x=coeff))
    #p <- p + geom_bar(stat="identity")
    p <- p + geom_point()
    p <- p + geom_errorbar(aes(xmin=coeff-ci, xmax=coeff+ci), width=0.3)
    p <- p + geom_vline(xintercept=0, linetype="dashed", color="gray")
    p <- p + scale_y_discrete(limits=rev(levels(data.for.model$feat.description.pretty)))  # reverse y-axis order
    p <- p + ylab("Feature") + xlab("Coefficient")
    p <- p + ggtitle(model.name)
    p <- p + theme_pubr()

    file.save.path <- paste0(OUT.PDF.DIR, "/",
                             "nested-cross-val.feature-coeffs.", model, ".pdf")
    ggsave(file.save.path, p, width=3.1, height=4, useDingbats=FALSE)
    return(p)
}
##############################

##############################
# Prepare for onehot-simple plots (just target bases and mismatch alleles)
# These show coefficient values ordered along the guide/target

# Only used the 'onehot-simple' feats.type
data.onehot <- data[data$feats.type == "onehot-simple", ]

# Break up the data frame based on whether the row represents a target
# base or a mismatch allele
data.onehot.target <- data.onehot[grepl("target-", data.onehot$feat.description),]
data.onehot.mismatch <- data.onehot[grepl("guide-mismatch-", data.onehot$feat.description),]

plot.onehot.target.for.model <- function(model, model.name) {
    # Produce plot for model
    data.for.model <- data.frame(data.onehot.target[data.onehot.target$model == model, ])

    p <- ggplot(data.for.model, aes(x=target.pos, y=coeff, fill=allele))
    p <- p + geom_rect(data=subset(data.for.model, target.pos %% 2 == 0), aes(xmin=target.pos-0.55, xmax=target.pos+0.55, ymin=-Inf, ymax=Inf, alpha=target.pos.background.alpha), color="white", fill="black", alpha=0.03) # alternate (striped) backgrounds of gray and white; put gray rectangle for even positions
    p <- p + geom_bar(stat="identity", width=0.8, position=position_dodge())
    p <- p + geom_errorbar(aes(ymin=coeff-ci, ymax=coeff+ci, group=allele), size=0.1, width=0.5, position=position_dodge(width=0.8))
    p <- p + xlab("Position in target") + ylab("Coefficient")
    p <- p + ggtitle(model.name)
    p <- p + scale_fill_viridis(discrete=TRUE, name="") # adjust fill gradient
    p <- p + theme_pubr()

    file.save.path <- paste0(OUT.PDF.DIR, "/",
                             "nested-cross-val.feature-coeffs-onehot-target-bases.", model, ".pdf")
    ggsave(file.save.path, p, width=7, height=2.5, useDingbats=FALSE)
    return(p)
    # Produce plot for model
}

plot.onehot.mismatches.for.model <- function(model, model.name) {
    # Produce plot for model
    data.for.model <- data.frame(data.onehot.mismatch[data.onehot.mismatch$model == model, ])

    p <- ggplot(data.for.model, aes(x=target.pos, y=coeff, fill=allele))
    p <- p + geom_rect(data=subset(data.for.model, target.pos %% 2 == 0), aes(xmin=target.pos-0.55, xmax=target.pos+0.55, ymin=-Inf, ymax=Inf, alpha=target.pos.background.alpha), color="white", fill="black", alpha=0.03) # alternate (striped) backgrounds of gray and white; put gray rectangle for even positions
    p <- p + geom_bar(stat="identity", width=0.8, position=position_dodge())
    p <- p + geom_errorbar(aes(ymin=coeff-ci, ymax=coeff+ci, group=allele), size=0.1, width=0.5, position=position_dodge(width=0.8))
    p <- p + xlab("Position in target") + ylab("Coefficient")
    p <- p + ggtitle(model.name)
    p <- p + scale_fill_viridis(discrete=TRUE, name="") # adjust fill gradient
    p <- p + theme_pubr()

    file.save.path <- paste0(OUT.PDF.DIR, "/",
                             "nested-cross-val.feature-coeffs-onehot-mismatch-alleles.", model, ".pdf")
    ggsave(file.save.path, p, width=7, height=2.5, useDingbats=FALSE)
    return(p)
    # Produce plot for model
}
##############################


if ("lr" %in% data$model) {
    # Linear regression models
    plot.combined.feats.for.model("lr", "Linear regression")
    plot.combined.feats.for.model("l1_lr", "L1 linear regression")
    plot.combined.feats.for.model("l2_lr", "L2 linear regression")
    plot.combined.feats.for.model("l1l2_lr", "L1+L2 linear regression")
    plot.onehot.target.for.model("lr", "Linear regression - target bases")
    plot.onehot.target.for.model("l1_lr", "L1 linear regression - target bases")
    plot.onehot.target.for.model("l2_lr", "L2 linear regression - target bases")
    plot.onehot.target.for.model("l1l2_lr", "L1+L2 linear regression - target bases")
    plot.onehot.mismatches.for.model("lr", "Linear regression - mismatch alleles")
    plot.onehot.mismatches.for.model("l1_lr", "L1 linear regression - mismatch alleles")
    plot.onehot.mismatches.for.model("l2_lr", "L2 linear regression - mismatch alleles")
    plot.onehot.mismatches.for.model("l1l2_lr", "L1+L2 linear regression - mismatch alleles")
} else if ("logit" %in% data$model) {
    # Classification models
    plot.combined.feats.for.model("logit", "Logistic regression")
    plot.combined.feats.for.model("l1_logit", "L1 logistic regression")
    plot.combined.feats.for.model("l2_logit", "L2 logistic regression")
    plot.combined.feats.for.model("l1l2_logit", "L1+L2 logistic regression")
    plot.onehot.target.for.model("logit", "Logistic regression - target bases")
    plot.onehot.target.for.model("l1_logit", "L1 logistic regression - target bases")
    plot.onehot.target.for.model("l2_logit", "L2 logistic regression - target bases")
    plot.onehot.target.for.model("l1l2_logit", "L1+L2 logistic regression - target bases")
    plot.onehot.mismatches.for.model("logit", "Logistic regression - mismatch alleles")
    plot.onehot.mismatches.for.model("l1_logit", "L1 logistic regression - mismatch alleles")
    plot.onehot.mismatches.for.model("l2_logit", "L2 logistic regression - mismatch alleles")
    plot.onehot.mismatches.for.model("l1l2_logit", "L1+L2 logistic regression - mismatch alleles")
} else {
    stop("Unknown whether this is regression or classification")
}

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
