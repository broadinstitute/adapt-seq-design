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

# Compute metrics comparing the Tambe et al. measurement with the ADAPT model's
# prediction
metrics <- function(x, y) {
    r <- cor(x, y, method="pearson")
    rho <- cor(x, y, method="spearman")
    return(list(r=r, rho=rho, str=paste0("r=", r, "; rho=", rho)))
}
test.results.metrics <- metrics(test.results$tambe.value, test.results$adapt.prediction)
test.results.rho.str <- format(test.results.metrics$rho, digits=3)
test.results.rho.expr <- as.expression(bquote(rho~"="~.(test.results.rho.str)))

print(test.results.metrics$str)

# Since there are only a few choices of number of mismatches, convert it to a factor
test.results$number.of.mismatches <- factor(test.results$number.of.mismatches)

p <- ggplot(test.results, aes(x=tambe.value, y=adapt.prediction)) +
        geom_point(aes(color=number.of.mismatches), size=5) +
        xlab("Measured cleavage rate (Tambe et al. 2018)") + ylab("Model prediction") +
        geom_vline(xintercept=100, linetype="dashed") +   # vertical line at normalized (0-mismatch) Tambe et al. value
        scale_color_viridis(discrete=TRUE) + # adjust colors
        theme_pubr() +
        theme(aspect.ratio=1) +  # make plot be square
        annotate(geom="text", label=test.results.rho.expr, # include text with rho value
                 x=Inf, y=Inf, hjust=1, vjust=1, size=3)
ggsave(OUT.PDF, p, width=8, height=8, useDingbats=FALSE)

