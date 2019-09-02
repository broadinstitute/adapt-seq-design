# Plot results of testing predictor.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)

args <- commandArgs(trailingOnly=TRUE)
IN.TSV <- args[1]
OUT.PDF <- args[2]

# Read TSV of results and replace '_' in column names with '.'
test.results <- read.table(IN.TSV, header=TRUE, sep="\t")
names(test.results) <- gsub("_", ".", names(test.results))

# Melt across true/predicted values
test.results.melted <- melt(test.results,
                            measure.vars=c("true.activity", "predicted.activity"),
                            variable.name="activity.type",
                            value.name="activity")

#####################################################################
# Plot distribution of true activity and predicted activity
# Show as density plots

p <- ggplot(test.results.melted, aes(x=activity, fill=activity.type))
p <- p + geom_density(alpha=0.5, position='identity')
p.output.dist <- p
#####################################################################


#####################################################################
# Produce PDF
g <- arrangeGrob(p.output.dist,
                 ncol=1)
ggsave(OUT.PDF, g, width=8, height=8, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
#####################################################################
