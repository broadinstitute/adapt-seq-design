# Plot results of testing predictor.
#
# By Hayden Metsky <hayden@mit.edu>


require(ggplot2)
require(gridExtra)
require(reshape2)
require(dplyr)
require(ggridges)

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
p <- p + xlab("Activity") + ylab("Density")
p <- p + theme_bw(base_size=18) # bw & larger font sizes
p <- p + theme(legend.justification=c(0,1), # place legend in upper-left
               legend.position=c(0.1,0.9))
p.output.dist <- p
#####################################################################


#####################################################################
# Plot predicted quantile vs. true activity value
# y-axis shows the quantile (grouped) for the predicted activity and
# x-axis shows the distribution of true activity values in that
# quantile
# Use quartiles here (4 groupings)

# Create a separate data frame with a quantile factor that also
# includes 'all' for all data points
test.results.with.quantile <- data.frame(test.results)
test.results.with.quantile$predicted.quantile <- ntile(test.results.with.quantile$predicted.activity, 4)
test.results.all <- data.frame(test.results)
test.results.all$predicted.quantile <- rep("all", n=nrow(test.results.all))
test.results.with.quantile <- rbind(test.results.with.quantile,
                                    test.results.all)
test.results.with.quantile$predicted.quantile <- factor(test.results.with.quantile$predicted.quantile,
                                                        levels=c("all", "1", "2", "3", "4"))

p <- ggplot(test.results.with.quantile, aes(x=true.activity, y=predicted.quantile))
p <- p + xlab("Activity") + ylab("Quartile")
p <- p + geom_density_ridges()
p <- p + theme_bw(base_size=18)    # bw & larger font sizes
p.by.predicted.quantile <- p
#####################################################################

#####################################################################
# Produce PDF
g <- arrangeGrob(p.output.dist,
                 p.by.predicted.quantile,
                 ncol=1)
ggsave(OUT.PDF, g, width=8, height=16, useDingbats=FALSE)

# Remove the empty Rplots.pdf created above
file.remove("Rplots.pdf")
#####################################################################
