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

args <- commandArgs(trailingOnly=TRUE)
IN.TSV <- args[1]
OUT.PDF <- args[2]

# Read TSV of results and replace '_' in column names with '.'
test.results <- read.table(IN.TSV, header=TRUE, sep="\t")
names(test.results) <- gsub("_", ".", names(test.results))

# Compute metrics comparing the Abudayyeh et al.'s knockdown luciferase
# measurement with the ADAPT model's prediction
# We should expect a negative correlation -- higher luciferase values are
# worse, while higher ADAPT predictions are better
metrics <- function(x, y) {
    r <- cor(x, y, method="pearson")
    r.pvalue <- cor.test(x, y, method="pearson")$p.value
    rho <- cor(x, y, method="spearman")
    rho.pvalue <- cor.test(x, y, method="spearman")$p.value
    return(list(r=r, r.pvalue=r.pvalue, rho=rho, rho.pvalue=rho.pvalue, str=paste0("r=", r, "; rho=", rho, "; r.pvalue=", r.pvalue, "; rho.pvalue=", rho.pvalue)))
}
test.results.metrics <- metrics(test.results$abudayyeh.value, test.results$adapt.prediction)
test.results.rho.str <- format(test.results.metrics$rho, digits=3)
test.results.rho.expr <- as.expression(bquote(rho~"="~.(test.results.rho.str)))
test.results.r.str <- format(test.results.metrics$r, digits=3)
test.results.r.expr <- as.expression(bquote(r~"="~.(test.results.r.str)))

print(test.results.metrics$str)

# Since there are only a few choices of number of mismatches, convert it to a factor
test.results$number.of.mismatches <- factor(test.results$number.of.mismatches)

# Scatter plot of all results
p <- ggplot(test.results, aes(x=abudayyeh.value, y=adapt.prediction)) +
        geom_point(aes(color=number.of.mismatches), size=2) +
        xlab("Normalized knockdown level (Abudayyeh et al. 2017)") + ylab("Predicted activity for detection") +
        scale_color_viridis(discrete=TRUE, name="Mismatches") + # adjust colors
        theme_pubr() +
        theme(aspect.ratio=1) +  # make plot be square
        annotate(geom="text", label=test.results.rho.expr, # include text with rho value
                 x=Inf, y=Inf, hjust=1, vjust=1, size=3) +
        annotate(geom="text", label=test.results.r.expr, # include text with r value
                 x=Inf, y=Inf, hjust=1, vjust=2, size=3)
ggsave(OUT.PDF, p, width=4.5, height=4.5, useDingbats=FALSE)

