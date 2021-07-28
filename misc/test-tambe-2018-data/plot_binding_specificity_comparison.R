# Plot results of testing predictor.
#
# By Hayden Metsky <hmetsky@broadinstitute.org>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(ggridges)
require(viridis)
require(ggsignif)
require(ggpubr)
require(ggstance)
require(dplyr)

args <- commandArgs(trailingOnly=TRUE)
IN.TSV <- args[1]
OUT.PDF.SCATTER <- args[2]
OUT.PDF.QUARTILES <- args[3]

# Read TSV of results and replace '_' in column names with '.'
test.results <- read.table(gzfile(IN.TSV), header=TRUE, sep="\t")
names(test.results) <- gsub("_", ".", names(test.results))

# Randomly sample 1,000 data points per choice of mismatches;
# there are fewer than 1,000 for 0 and 1 mismatch so this will
# use all for those. The reason for this is that there are so
# many data points with 3 and 4 mismatches -- overwhemling
# the number with 0, 1, or 2 -- and the plot/metrics may not
# account for those with fewer mismatches or for the model being
# able to predict differences in activity across different
# numbers of mismatches. `min(n(), 1000)` ensures that when
# there are fewer than 1,000 data points for a choice of
# mismatches (group), it can still work
set.seed(1)
test.results <- test.results %>%
    group_by(number.of.mismatches) %>%
    sample_n(min(n(), 1000), replace=FALSE)

# Compute metrics comparing the Tambe et al. measurement with the ADAPT model's
# prediction
metrics <- function(x, y) {
    r <- cor(x, y, method="pearson")
    r.pvalue <- cor.test(x, y, method="pearson")$p.value
    rho <- cor(x, y, method="spearman")
    rho.pvalue <- cor.test(x, y, method="spearman")$p.value
    return(list(r=r, r.pvalue=r.pvalue, rho=rho, rho.pvalue=rho.pvalue, str=paste0("r=", r, "; rho=", rho, "; r.pvalue=", r.pvalue, "; rho.pvalue=", rho.pvalue)))
}
test.results.metrics <- metrics(test.results$tambe.value, test.results$adapt.prediction)
test.results.rho.str <- format(test.results.metrics$rho, digits=3)
test.results.rho.expr <- as.expression(bquote(rho~"="~.(test.results.rho.str)))

print(test.results.metrics$str)

# Since there are only a few choices of number of mismatches, convert it to a factor
test.results$number.of.mismatches <- factor(test.results$number.of.mismatches)

# Compute quartiles
test.results$tambe.value.quartile <- factor(ntile(test.results$tambe.value, 4))
test.results$adapt.prediction.quartile <- factor(ntile(test.results$adapt.prediction, 4))

# Plot scatter plot of measured Tambe val. versus predicted value
# Uncomment `stat_density_2d()` lines to show density plot
p <- ggplot(test.results, aes(x=tambe.value, y=adapt.prediction)) +
        geom_point(aes(color=number.of.mismatches), size=1, stroke=0, alpha=0.8) +
        #stat_density_2d(aes(fill=stat(level)), geom="polygon", contour=TRUE, n=c(1000, 1000), h=NULL, adjust=1.75) +   # h=c(0.6, 0.6) works too
        #stat_density_2d(aes(fill=stat(level)), color="#5E5E5E", alpha=0.2, size=0.05, contour=TRUE, n=c(1000, 1000), h=NULL, adjust=1.75) +   # outline around contours; h=c(0.6, 0.6) works too
        #scale_fill_gradient(name="Level", breaks=c(0.2, 0.5), low="#FDF5FF", high="#470E55") +  # customize colors; specify tick labels on legend bar
        xlab("Measured fold-change for binding (Tambe et al. 2018)") + ylab("Model prediction for detection") +
        scale_color_viridis(discrete=TRUE) + # adjust colors
        theme_pubr() +
        theme(aspect.ratio=1) +  # make plot be square
        annotate(geom="text", label=test.results.rho.expr, # include text with rho value
                 x=Inf, y=Inf, hjust=1, vjust=1, size=3)
ggsave(OUT.PDF.SCATTER, p, width=8, height=8, useDingbats=FALSE)

# Plot stacked bar for each quartile of ADAPT's predicted value, showing
# the Tambe et al. value
# Use `forcats::fct_rev(..)` to flip the order of each stacked bar
p <- ggplot(test.results, aes(x=adapt.prediction.quartile, fill=forcats::fct_rev(tambe.value.quartile))) +
        geom_bar(stat="count") +
        xlab("Quartile of model prediction for detection") +
        theme_pubr() +
        coord_flip() +    # flip axes
        scale_fill_viridis(name="Quartile of measured fold-change for binding (Tambe et al. 2018)", discrete=TRUE) +   # legend label and fill colors
        theme(axis.title.x=element_blank(), # remove x-axis
              axis.text.x=element_blank(),
              axis.ticks.x=element_blank())
ggsave(OUT.PDF.QUARTILES, p, width=8, height=4, useDingbats=FALSE)

