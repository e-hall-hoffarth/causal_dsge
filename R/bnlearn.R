# Setup
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
tests <- c(
  'cor',
  'mc-cor',
  'smc-cor',
  'zf',
  'mc-zf',
  'smc-zf',
  'mi-g',
  'mc-mi-g',
  'smc-mi-g',
  'mi-g-sh'
)
graph.par(list(nodes=list(fontsize=12)))

### Load and Transform Data
# RBC
# data <- read.csv('rbc.csv')
# data <- as.data.frame(sapply(data, as.numeric))
# bl <- c('y', 'k', 'c', 'l', 'w', 'i', 'X', 'eps_g', 'eps_z')
# data$"log_k-1" <- c(NA, data$log_k[1:nrow(data)-1])
# data <- data[2:nrow(data),]
# wl <- names(data)

# Gali
# data <- read.csv('gali.csv')
# data <- as.data.frame(sapply(data, as.numeric))
# bl <- c('X', 'pi_ann', 'r_nat_ann', 'r_real_ann', 'c', 'm_growth_ann',
#         'y_gap', 'i_ann', 'mu_hat', 'yhat', 'r_real', 'w_real', 'mu',
#         'pi', 'y_nat', 'r_nat', 'p',
#         'eps_nu', 'eps_z', 'eps_a')
# data$"nu-1" <- c(NA, data$nu[1:nrow(data)-1])
# data$"a-1" <- c(NA, data$a[1:nrow(data)-1])
# data$"z-1" <- c(NA, data$z[1:nrow(data)-1])
# data$"p-1" <- c(NA, data$p[1:nrow(data)-1])
# data$"y-1" <- c(NA, data$y[1:nrow(data)-1])
# data$"i-1" <- c(NA, data$i[1:nrow(data)-1])
# data$"m_real-1" <- c(NA, data$m_real[1:nrow(data)-1])
# data <- data[2:nrow(data),]
# wl <- names(data)

# Ireland (2004)
# data <- read.csv('ireland.csv')
# data <- as.data.frame(sapply(data, as.numeric))
# bl <- c('X', 'z', 'gobs', 'robs', 'piobs', 'r_annual', 'pi_annual')
# data$"a-1" <- c(NA, data$a[1:nrow(data)-1])
# data$"e-1" <- c(NA, data$e[1:nrow(data)-1])
# data$"x-1" <- c(NA, data$x[1:nrow(data)-1])
# data$"pihat-1" <- c(NA, data$pihat[1:nrow(data)-1])
# data$"yhat-1" <- c(NA, data$yhat[1:nrow(data)-1])
# data$"rhat-1" <- c(NA, data$rhat[1:nrow(data)-1])
# data <- data[2:nrow(data),]
# wl <- names(data)

data <- data[,colnames(data) %in% wl & !(colnames(data) %in% bl)]

# Find and eliminate colinear variables
# indexes_to_drop should be empty
corrplot(cor(data))
names(data[,findCorrelation(cor(data), cutoff = 0.999)])
names(data[,findCorrelation(cor(data), cutoff = 0.95)])

# Split for holdout
train <- data[1:floor(0.8*nrow(data)),]
test <- data[(floor(0.8*nrow(data))+1):nrow(data),]

# Try all conditional independence tests for structure learning
# for(test in tests){
#   pc_model <- pc.stable(data, cluster = cl, test=test)
#   print(test)
#   print(root.nodes(pc_model))
# }

### Fit models using different structure learning methods
graph.par(list(nodes=list(lty="solid", fontsize=14)))
# Constraint Based
pc_model <- pc.stable(data, cluster = cl)
graphviz.plot(pc_model)
root.nodes(pc_model)
pc_fitted <- bn.fit(pc_model, train)
  
# Score Based
hc_model <- hc(data)
graphviz.plot(hc_model)
root.nodes(hc_model)
hc_fitted <- bn.fit(hc_model, train)

# Hybrid
hybrid_model <- rsmax2(data, restrict = 'pc.stable')
graphviz.plot(hybrid_model)
root.nodes(hybrid_model)
hybrid_fitted <- bn.fit(hybrid_model, train)

dev.print(png, "../text/latex/images/rbc_dag.png", width=500, height=350)

# Generate and IRF implied by the model
# TODO: Convert this into a function
vars <- nodes(hybrid_model)
exo <- root.nodes(hybrid_model)
endo <- vars[!(vars %in% exo)]
std_z = 1
alpha_z = 0.97
irf_length = 120
irf <- data[1:irf_length,]

for (n in vars) {
  irf[,n] <- 0
}

for (n in exo) {
  irf[,n] <- 0
}

z_sim <- c(std_z)
for (i in 1:(irf_length-1)) {
  z_sim <- c(z_sim, alpha_z * z_sim[length(z_sim)])
}
irf[,"z"] <- z_sim

# Problem with prediction: Does not always propogate predictions.
for (n in endo) {
  for (i in 1:irf_length) {
    irf[i,n] <- predict(hybrid_fitted, n, irf[i,])[1]
  }
}

irf$t <- 1:irf_length

irf = melt(irf, id=c("t"))
ggplot(irf) + 
  geom_point(aes(x = t, y = value, color=variable), size = 1)
ggsave("../text/latex/images/rbc_irf.png")

# V struct example
e <- empty.graph(c('x', 'y', 'z'))
collider_arcs <- matrix(c('x', 'y', 'z', 'y'),
                        ncol=2, byrow=T,
                        dimnames = list(NULL, c('from', 'to')))
chain_arcs <- matrix(c('x', 'y', 'y', 'z'),
                     ncol=2, byrow=T,
                     dimnames = list(NULL, c('from', 'to')))
fork_arcs <- matrix(c('y', 'x', 'y', 'z'),
                    ncol=2, byrow=T,
                    dimnames = list(NULL, c('from', 'to')))
collider <- e 
chain <- e
fork <- e

arcs(collider) <- collider_arcs
arcs(chain) <- chain_arcs
arcs(fork) <- fork_arcs

graph.par(list(nodes=list(lty="solid", fontsize=8)))
graphviz.plot(collider)
dev.print(png, "../text/latex/images/collider.png", width=300, height=200)
graphviz.plot(chain)
dev.print(png, "../text/latex/images/chain.png", width=300, height=200)
graphviz.plot(fork)
dev.print(png, "../text/latex/images/fork.png", width=300, height=200)


# Traffic example
e <- empty.graph(c('rush hour','bad weather','accident','traffic jam','sirens'))
# Figure 2: Empty Graph
graph.par(list(nodes=list(lty="solid", fontsize=14)))
graphviz.plot(e)
dev.print(png, "../text/latex/images/trafficjam_unfit.png", width=500, height=350)

traffic_arcs <- matrix(c('rush hour', 'traffic jam', 
                         'bad weather', 'traffic jam',
                         'accident', 'traffic jam',
                         'bad weather', 'accident',
                         'accident', 'sirens'),
                       ncol=2, byrow=T,
                       dimnames = list(NULL, c('from', 'to')))
traffic <- e
arcs(traffic) <- traffic_arcs
# Figure 1: Example BN
graph.par(list(nodes=list(lty="solid", fontsize=14)))
graphviz.plot(traffic)
dev.print(png, "../text/latex/images/trafficjam.png", width=500, height=350)

e <- empty.graph(c('rush hour','bad weather','do(accident = 1)','traffic jam','sirens'))
do_arcs <- matrix(c('rush hour', 'traffic jam', 
                    'bad weather', 'traffic jam',
                    'do(accident = 1)', 'traffic jam',
                    'do(accident = 1)', 'sirens'),
                   ncol=2, byrow=T,
                   dimnames = list(NULL, c('from', 'to')))
do <- e
arcs(do) <- do_arcs
# Figure 5 (right side): Intervention
graph.par(list(nodes=list(lty="solid", fontsize=14)))
graphviz.plot(do)
dev.print(png, "../text/latex/images/trafficjam_intervention.png", width=500, height=350)

# Figure 4: Potential Outcomes
e <- empty.graph(c('w','x','y(1)','y(0)','y'))
arcs <- matrix(c('w', 'x',
                 'w', 'y(1)',
                 'w', 'y(0)',
                 'x', 'y',
                 'y(1)', 'y',
                 'y(0)', 'y'),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
g <- e
arcs(g) <- arcs
graphviz.plot(g)
dev.print(png, "../text/latex/images/potential_outcomes_dag.png", width=500, height=350)

# Figure 6: Simultaneity
# Undirected
e <- empty.graph(c('x(d)','x(s)','q','p'))
arcs <- matrix(c('x(d)', 'q',
                 'x(s)', 'p',
                 'q', 'p',
                 'p', 'q'),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
g <- e
arcs(g) <- arcs
graph.par(list(nodes=list(fontsize=12)))
graphviz.plot(g, layout = "circo")
dev.print(png, "../text/latex/images/simultaneous.png", width=500, height=350)

# Directed
arcs <- matrix(c('x(d)', 'q',
                 'x(s)', 'p',
                 'x(d)', 'p',
                 'x(s)', 'q'),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
g <- e
arcs(g) <- arcs
graph.par(list(nodes=list(fontsize=12)))
graphviz.plot(g, layout = "dot")
dev.print(png, "../text/latex/images/directed.png", width=500, height=350)

# Figure 4: 
# Front-door criterion
e <- empty.graph(c('x','y','z', 'u'))
arcs <- matrix(c('x', 'z',
                 'z', 'y',
                 'u', 'x',
                 'u', 'y'),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
g <- e
arcs(g) <- arcs
graph.par(list(nodes=list(fontsize=12)))
graphviz.plot(g, layout = "circo")
dev.print(png, "../text/latex/images/frontdoor.png", width=500, height=350)

# Instrumental variables
arcs <- matrix(c('z', 'x',
                 'x', 'y',
                 'u', 'x',
                 'u', 'y'),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
g <- e
arcs(g) <- arcs
graph.par(list(nodes=list(fontsize=12)))
graphviz.plot(g, layout = "neato")
dev.print(png, "../text/latex/images/iv.png", width=500, height=350)


# Supply-Demand Example:
e <- empty.graph(c('x(d)','x(s)','d','s', 'p', 'q'))
arcs <- matrix(c('x(d)', 'd',
                 'x(s)', 's',
                 's', 'p',
                 's', 'q',
                 'd', 'p',
                 'd', 'q'),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
g <- e
arcs(g) <- arcs
graph.par(list(nodes=list(fontsize=12)))
graphviz.plot(g, layout = "circo")





