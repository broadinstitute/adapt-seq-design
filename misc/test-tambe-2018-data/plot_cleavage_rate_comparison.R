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
OUT.PDF.HARM <- args[3]

# Read TSV of results and replace '_' in column names with '.'
test.results <- read.table(IN.TSV, header=TRUE, sep="\t")
names(test.results) <- gsub("_", ".", names(test.results))

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
test.results.harm <- test.results[test.results$tambe.value <= 100,]
test.results.harm.metrics <- metrics(test.results.harm$tambe.value, test.results.harm$adapt.prediction)
print(paste("Subset where mismatches hurt activity:", test.results.harm.metrics$str))
test.results.harm.rho.str <- format(test.results.harm.metrics$rho, digits=3)
test.results.harm.rho.expr <- as.expression(bquote(rho~"="~.(test.results.harm.rho.str)))
test.results.harm.r.str <- format(test.results.harm.metrics$r, digits=3)
test.results.harm.r.expr <- as.expression(bquote(r~"="~.(test.results.harm.r.str)))

# Scatter plot of all results
p <- ggplot(test.results, aes(x=tambe.value, y=adapt.prediction)) +
        geom_point(aes(color=number.of.mismatches), size=3) +
        xlab("Cleavage rate (Tambe et al. 2018)") + ylab("Predicted activity for detection") +
        geom_vline(xintercept=100, linetype="dashed") +   # vertical line at normalized (0-mismatch) Tambe et al. value
        scale_color_viridis(discrete=TRUE, name="Mismatches") + # adjust colors
        theme_pubr() +
        theme(aspect.ratio=1) +  # make plot be square
        annotate(geom="text", label=test.results.rho.expr, # include text with rho value
                 x=Inf, y=Inf, hjust=1, vjust=1, size=3) +
        annotate(geom="text", label=test.results.r.expr, # include text with r value
                 x=Inf, y=Inf, hjust=1, vjust=2, size=3)
ggsave(OUT.PDF, p, width=4.5, height=4.5, useDingbats=FALSE)

# Scatter plot of only data where the Tambe value decreases against the
# wildtype (i.e., mismatches harm activity)
p <- ggplot(test.results.harm, aes(x=tambe.value, y=adapt.prediction)) +
        geom_point(aes(color=number.of.mismatches), size=3) +
        xlab("Cleavage rate (Tambe et al. 2018)") + ylab("Predicted activity for detection") +
        geom_vline(xintercept=100, linetype="dashed") +   # vertical line at normalized (0-mismatch) Tambe et al. value
        scale_color_viridis(discrete=TRUE, name="Mismatches") + # adjust colors
        theme_pubr() +
        theme(aspect.ratio=1) +  # make plot be square
        annotate(geom="text", label=test.results.harm.rho.expr, # include text with rho value
                 x=Inf, y=Inf, hjust=1, vjust=1, size=3) +
        annotate(geom="text", label=test.results.harm.r.expr, # include text with r value
                 x=Inf, y=Inf, hjust=1, vjust=2, size=3)
ggsave(OUT.PDF.HARM, p, width=4.5, height=4.5, useDingbats=FALSE)

