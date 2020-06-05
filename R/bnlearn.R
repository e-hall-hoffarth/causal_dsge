# Setup
setwd("/home/emmet/Documents/bayesian/bayesian_networks/data")
library(bnlearn)
library(parallel)
library(combinat)
library(Rgraphviz)
library(ggplot2)
library(reshape2)
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

### Baseline RBC (iid shocks)
# data <- read.csv('rbc.csv')
data <- read.csv('rbc.csv')
# data <- read.csv('sw.csv')
data <- as.data.frame(sapply(data, as.numeric))

# RBC
bl <- c('y', 'k', 'c', 'l', 'w', 'i', 'X', 'eps_g', 'eps_z')
# data$"log_k-1" <- c(NA, data$log_k[1:nrow(data)-1])
# data <- data[2:nrow(data),]
wl <- names(data)

# Gali
# bl <- c(# 'x_aux_1', 'x_aux_2', 'pi_star', 's',
#         'X', 'w', 'n', 'z', 'y', 'c','p', 'a', 'm', 'm_nominal', 
#         # 'X', 'log_w', 'log_n', 'log_z', 'log_y', 'c','log_p', 'log_a', 'log_m_nominal', 
#         'money_growth', 'i', 'pi', 'r', 'realinterest')
#         # 'money_growth_ann', 'i_ann', 'pi_ann', 'r_real_ann', 'realinterest')
        
# data$"log_p-1" <- c(NA, data$log_p[1:nrow(data)-1])
# data$"log_a-1" <- c(NA, data$log_a[1:nrow(data)-1])
# data$"log_z-1" <- c(NA, data$log_z[1:nrow(data)-1])
# data$"s-1" <- c(NA, data$s[1:nrow(data)-1])
# data$"mg_ann-1" <- c(NA, data$money_growth_ann[1:nrow(data)-1])

# data$"p-1" <- c(NA, data$p[1:nrow(data)-1])
# data$"a-1" <- c(NA, data$a[1:nrow(data)-1])
# data$"z-1" <- c(NA, data$z[1:nrow(data)-1])
# data$"s-1" <- c(NA, data$s[1:nrow(data)-1])
# data$"mg-1" <- c(NA, data$money_growth[1:nrow(data)-1])
# data <- data[2:nrow(data),]
# wl <- names(data)

# Smets and Wouters
# bl <- c('X')
# wl <- names(data)

data <- data[,colnames(data) %in% wl & !(colnames(data) %in% bl)]

# Get short names
names(data) <- sapply(names(data), function(x) substr(x, nchar(x), nchar(x)))

# Split for holdout
train <- data[1:floor(0.8*nrow(data)),]
test <- data[(floor(0.8*nrow(data))+1):nrow(data),]

# for(test in tests){
#   pc_model <- pc.stable(data, cluster = cl, test=test)
#   print(test)
#   print(root.nodes(pc_model))
# }

# Fit models using different structure learning methods
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
hybrid_model <- rsmax2(data)
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
