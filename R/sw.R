setwd("/home/emmet/Documents/bayesian/bayesian_networks/data")
library(bnlearn)
library(parallel)
library(combinat)
library(Rgraphviz)
library(ggplot2)
library(reshape2)
library(corrplot)
library(caret)
assignInNamespace("supported.clusters", 
                  c("cluster", "MPIcluster",  
                    "PVMcluster", "SOCKcluster"), 
                  "bnlearn")
cl = makeCluster(4)

data <- read.csv('sw.csv')
data <- as.data.frame(sapply(data, as.numeric))
bl <- c('X',                                           # Time
        'robs', 'labobs', 'pinfobs',                   # Observed values are redundant
        'dy', 'dc', 'dw', 'dinve',                     # Differences contain information about the past but we'd like to keep it cross-sectional
        'ewma', 'epinfma',                             # Shocks collinear with other shocks
        'zcap', 'zcapf')                               # Remove to eliminate collinearity
data <- data[,!(colnames(data) %in% bl)]

# Find and eliminate colinear variables
# indexes_to_drop should be empty
corrplot(cor(data))
names(data[,findCorrelation(cor(data), cutoff = 0.999)])
names(data[,findCorrelation(cor(data), cutoff = 0.95)])

# Initial structure learning
init_model <- rsmax2(data, restrict = 'pc.stable')
graphviz.plot(init_model)
root.nodes(init_model)
reversible.arcs(init_model)
dev.print(png, "../text/latex/empirical/images/sw_init.png", width=500, height=350)

# Construct an observationally equivallent graph which better represents the DGP
equiv_model <- reverse.arc(init_model, from = "b", to = "eb")
equiv_model <- reverse.arc(equiv_model, from = "m", to = "em")
equiv_model <- reverse.arc(equiv_model, from = "qs", to = "eqs")
equiv_model <- reverse.arc(equiv_model, from = "rkf", to = "kf")
equiv_model <- reverse.arc(equiv_model, from = "pkf", to = "qs")
equiv_model <- reverse.arc(equiv_model, from = "pkf", to = "eqs")
equiv_model <- reverse.arc(equiv_model, from = "pinf", to = "sw")
equiv_model <- reverse.arc(equiv_model, from = "pinf", to = "ms")
equiv_model <- reverse.arc(equiv_model, from = "pinf", to = "em")
graphviz.plot(equiv_model)
root.nodes(equiv_model)
dev.print(png, "../text/latex/empirical/images/sw_equiv.png", width=500, height=350)

# Force the exogenous variables to be root nodes
all_names <- names(data)
exo_names <- c('ea', 'eb', 'eg', 'em', 'eqs', 'ew', 'epinf', # iid shocks
               'a', 'b', 'g', 'ms', 'qs', 'sw', 'spinf',     # shock stocks
               'kf', 'kpf', 'invef', 'cf',                   # flexible economy states
               'k', 'kp', 'inve', 'c', 'pinf', 'w')          # rigid economy states 
endo_names <- all_names[!(all_names %in% exo_names) & !(all_names %in% bl)]
arc_bl <- tiers2blacklist(list(exo_names, endo_names))
for (exo in exo_names) {
  for (exo2 in exo_names[exo_names != exo]) {
    arc_bl <- rbind(arc_bl, c(exo, exo2))
  }
}

bl_model <- rsmax2(data, restrict = 'pc.stable', blacklist = arc_bl)
graphviz.plot(bl_model)
root.nodes(bl_model)
reversible.arcs(bl_model)
dev.print(png, "../text/latex/empirical/images/sw_bl.png", width=500, height=350)

