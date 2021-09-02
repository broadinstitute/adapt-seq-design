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
OUT.PDF.QUARTILES.REV <- args[4]
OUT.PDF.SCATTER.HARM <- args[5]
OUT.PDF.SCATTER.1MM <- args[6]

# Read TSV of results and replace '_' in column names with '.'
test.results <- read.table(gzfile(IN.TSV), header=TRUE, sep="\t")
names(test.results) <- gsub("_", ".", names(test.results))

# Randomly sample 150 data points per choice of mismatches;
# there are fewer than 150 for 0 mismatches so this will
# use all for those. The reason for this is that there are so
# many data points with 3 and 4 mismatches -- overwhemling
# the number with 0, 1, or 2 -- and the plot/metrics may not
# account for those with fewer mismatches or for the model being
# able to predict differences in activity across different
# numbers of mismatches. `min(n(), 150)` ensures that when
# there are fewer than 150 data points for a choice of
# mismatches (group), it can still work#
# 150 because there are 156 total data points with 1 mismatch,
# (and more than 150 for 2+ mismatches), so this will use almost
# all of those
set.seed(1)
test.results <- test.results %>%
    group_by(number.of.mismatches) %>%
    sample_n(min(n(), 150), replace=FALSE)

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
test.results.r.str <- format(test.results.metrics$r, digits=3)
test.results.r.expr <- as.expression(bquote(r~"="~.(test.results.r.str)))

print(test.results.metrics$str)

# Since there are only a few choices of number of mismatches, convert it to a factor
test.results$number.of.mismatches <- factor(test.results$number.of.mismatches)

# Also compute and print metrics for the data where mismatches harm activity
guide.x <- 'CGCCTGAACCACCAGGCTAT'
guide.x.wildtype.val <- 0.5478719380654253
guide.y <- 'CCGCACTATCGGAAGTTCAC'
guide.y.wildtype.val <- 0.8092551340703578
test.results.harm <- test.results[(grepl(guide.x, test.results$guide) & test.results$tambe.value <= guide.x.wildtype.val) | (grepl(guide.y, test.results$guide) & test.results$tambe.value <= guide.y.wildtype.val),]
test.results.harm.metrics <- metrics(test.results.harm$tambe.value, test.results.harm$adapt.prediction)
print(paste("Subset where mismatches hurt activity:", test.results.harm.metrics$str))
test.results.harm.rho.str <- format(test.results.harm.metrics$rho, digits=3)
test.results.harm.rho.expr <- as.expression(bquote(rho~"="~.(test.results.harm.rho.str)))
test.results.harm.r.str <- format(test.results.harm.metrics$r, digits=3)
test.results.harm.r.expr <- as.expression(bquote(r~"="~.(test.results.harm.r.str)))

# Also compute and print metrics for only 1 mismatch data (since this is what
# is mostly used for cleavage rate data)
test.results.1mm <- test.results[test.results$number.of.mismatches == 1,]
test.results.1mm.metrics <- metrics(test.results.1mm$tambe.value, test.results.1mm$adapt.prediction)
print(paste("Subset with only 1 mismatch:", test.results.1mm.metrics$str))
test.results.1mm.rho.str <- format(test.results.1mm.metrics$rho, digits=3)
test.results.1mm.rho.expr <- as.expression(bquote(rho~"="~.(test.results.1mm.rho.str)))
test.results.1mm.r.str <- format(test.results.1mm.metrics$r, digits=3)
test.results.1mm.r.expr <- as.expression(bquote(r~"="~.(test.results.1mm.r.str)))

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
        scale_color_viridis(discrete=TRUE, name="Mismatches") + # adjust colors
        theme_pubr() +
        theme(aspect.ratio=1) +  # make plot be square
        annotate(geom="text", label=test.results.rho.expr, # include text with rho value
                 x=Inf, y=Inf, hjust=1, vjust=1, size=3) +
        annotate(geom="text", label=test.results.r.expr, # include text with r value
                 x=Inf, y=Inf, hjust=1, vjust=2, size=3)
ggsave(OUT.PDF.SCATTER, p, width=4.5, height=4.5, useDingbats=FALSE)

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
ggsave(OUT.PDF.QUARTILES, p, width=7, height=4.5, useDingbats=FALSE)

# Plot stacked bar for each quartile of the Tambe et al. value, showing
# ADAPT's predicted value
# Use `forcats::fct_rev(..)` to flip the order of each stacked bar
p <- ggplot(test.results, aes(x=tambe.value.quartile, fill=forcats::fct_rev(adapt.prediction.quartile))) +
        geom_bar(stat="count") +
        xlab("Quartile of binding fold-change (Tambe 2018)") +
        theme_pubr() +
        coord_flip() +    # flip axes
        scale_fill_viridis(name="Quartile of predicted activity for detection", discrete=TRUE) +   # legend label and fill colors
        theme(axis.title.x=element_blank(), # remove x-axis
              axis.text.x=element_blank(),
              axis.ticks.x=element_blank())
ggsave(OUT.PDF.QUARTILES.REV, p, width=7, height=4.5, useDingbats=FALSE)

# Plot scatter plot of measured Tambe val. versus predicted value, only for
# data where the Tambe value decreases against the wildtype (i.e., mismatches
# harms activity)
p <- ggplot(test.results.harm, aes(x=tambe.value, y=adapt.prediction)) +
        geom_point(aes(color=number.of.mismatches), size=1, stroke=0, alpha=0.8) +
        xlab("Measured fold-change for binding (Tambe et al. 2018)") + ylab("Model prediction for detection") +
        scale_color_viridis(discrete=TRUE, name="Mismatches") + # adjust colors
        theme_pubr() +
        theme(aspect.ratio=1) +  # make plot be square
        annotate(geom="text", label=test.results.harm.rho.expr, # include text with rho value
                 x=Inf, y=Inf, hjust=1, vjust=1, size=3) +
        annotate(geom="text", label=test.results.harm.r.expr, # include text with r value
                 x=Inf, y=Inf, hjust=1, vjust=2, size=3)
ggsave(OUT.PDF.SCATTER.HARM, p, width=4.5, height=4.5, useDingbats=FALSE)

# Plot scatter plot of measured Tambe val. versus predicted value, only for
# data with 1 mismatch
p <- ggplot(test.results.1mm, aes(x=tambe.value, y=adapt.prediction)) +
        geom_point(aes(color=number.of.mismatches), size=1, stroke=0, alpha=0.8) +
        xlab("Binding fold-change (Tambe 2018)") + ylab("Predicted activity for detection") +
        scale_color_viridis(discrete=TRUE, name="Mismatches") + # adjust colors
        theme_pubr() +
        theme(aspect.ratio=1) +  # make plot be square
        annotate(geom="text", label=test.results.1mm.rho.expr, # include text with rho value
                 x=Inf, y=Inf, hjust=1, vjust=1, size=3) +
        annotate(geom="text", label=test.results.1mm.r.expr, # include text with r value
                 x=Inf, y=Inf, hjust=1, vjust=2, size=3)
ggsave(OUT.PDF.SCATTER.1MM, p, width=4.5, height=4.5, useDingbats=FALSE)
