# Plot information about the guide-target library.
#
# As input, this takes the table of data the actual guide-target sequences
# and information in it. However, this script extracts only that information
# and does not make use of any of the measurement/activity information.
# This means that the information on the library is plotted only on the data
# that we use (e.g., does not count guides we removed for technical reasons
# during curation).
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(reshape2)
require(viridis)
require(ggpubr)

IN.TABLE <- "data/CCF-curated/CCF_merged_pairs_annotated.curated.tsv"
OUT.HAMMING.DIST.PDF <- "out/cas13/dataset/library-hamming-dist.pdf"
OUT.PFS.PDF <- "out/cas13/dataset/library-pfs.pdf"
OUT.PFS2NT.PDF <- "out/cas13/dataset/library-pfs2nt.pdf"


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
          sd   = sd     (xx[[col]], na.rm=na.rm),
          median = median(xx[[col]], na.rm=na.rm),
          pctile.05 = quantile(xx[[col]], 0.05, na.rm=na.rm)[[1]],
          pctile.20 = quantile(xx[[col]], 0.20, na.rm=na.rm)[[1]],
          pctile.80 = quantile(xx[[col]], 0.80, na.rm=na.rm)[[1]],
          pctile.95 = quantile(xx[[col]], 0.95, na.rm=na.rm)[[1]]
        )
      },
      measurevar
    )

    # Add measurevar as another name for the "mean" column
    datac[, measurevar] <- datac$mean

    datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean

    # Confidence interval multiplier for standard error
    # Calculate t-statistic for confidence interval: 
    # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
    ciMult <- qt(conf.interval/2 + .5, datac$N-1)
    datac$ci <- datac$se * ciMult

    return(datac)
}


# Read table and replace '_' in column names with '.'
all.data <- read.table(gzfile(IN.TABLE), header=TRUE, sep="\t")
names(all.data) <- gsub("_", ".", names(all.data))

# Extract subset of data points corresponding to an 'experiment' (generally,
# a mismatch between guide/target, but not always)
guide.target.exp <- subset(all.data, type == "exp")

# Extract subset of data points corresponding to positive guide/target match
# (i.e., the wildtype target)
guide.target.pos <- subset(all.data, type == "pos")

# Extract subset of data points corresponding to negative guide/target match
# (i.e., high divergence between the two)
guide.target.neg <- subset(all.data, type == "neg")

# Extract exp and pos (everything except negatives)
guide.target.expandpos <- subset(all.data, type == "exp" | type == "pos")

# Extract just information about the library
library.design <- guide.target.expandpos[,c("guide.seq","target.at.guide","target.before","target.after","type","guide.target.hamming.dist")]

# Add information about PFS, and PFS + 1 nt
library.design$pfs <- substr(library.design$target.after, 1, 1)
library.design$pfs.2nt <- substr(library.design$target.after, 1, 2)

# Remove duplicate rows
library.design <- unique(library.design)

# Show a bar chart of the number of pairs with each Hamming distance
# Stack each bar by the PFS, to see the combined distribution of Hamming
# distance and PFS
p <- ggplot(library.design, aes(x=guide.target.hamming.dist, fill=pfs))
p <- p + geom_bar(position="stack")
p <- p + xlab("Guide-target Hamming distance") + ylab("Number of pairs")
p <- p + scale_x_continuous(breaks=seq(0,9)) # label integers on axis
p <- p + scale_fill_viridis(discrete=TRUE, name="PFS")
p <- p + theme_pubr()
p + ggsave(OUT.HAMMING.DIST.PDF, width=4, height=4, useDingbats=FALSE)

# Show a bar chart of the number of pairs with each PFS
p <- ggplot(library.design, aes(x=pfs))
p <- p + geom_bar()
p <- p + xlab("PFS") + ylab("Number of pairs")
p <- p + theme_pubr()
p + ggsave(OUT.PFS.PDF, width=4, height=4, useDingbats=FALSE)

# Show a bar chart of the number of pairs with each 2 nt flanking motif
p <- ggplot(library.design, aes(x=pfs.2nt))
p <- p + geom_bar()
p <- p + xlab("2 nt flanking motif") + ylab("Number of pairs")
p <- p + theme_pubr()
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1)) # 45 degree text
p + ggsave(OUT.PFS2NT.PDF, width=4, height=4, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
