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
library(sets)
library(gridExtra)
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
graph.par(list(nodes=list(fontsize=16)))

unique.undirected.arcs <- function(model) {
  uas <- undirected.arcs(model)
  colnames(uas) <- NULL
  uas <- as.set(apply(uas, 1, as.set))
  unique_uas <- matrix(uas[0],
                       ncol=2, byrow=T,
                       dimnames = list(NULL, c('from', 'to')))
  for (ua in uas) {
    unique_uas <- rbind(unique_uas, ua)
  }
  rownames(unique_uas) <- NULL
  return(unique_uas)
}

simulate.irf <- function(model, data, shock_var, shock_amt, t = 5) {
  new_data <- data[FALSE,]
  shifts <- data[FALSE,]
  lags <- names(data)[grep('_1', names(data))]
  shocks <- setdiff(root.nodes(model), c(shock_var))
  endo <- setdiff(nodes(model), root.nodes(model))
  new_data[1, shock_var] <- shock_amt
  
  # if (!(shock_var %in% lags) & (shock_var %in% root.nodes(model))) {
  #   new_data[2:t, shock_var] <- 0  
  # }
  
  for (s in shocks) {
    new_data[,s] <- 0
  }
  for (i in 1:t) {
    new_data[i,] <- impute(model, new_data[i,], method="parents")
    if (i == 1) {
      shifts[1,] <- new_data[1,]
    }
    if (length(lags) != 0) {
      for (l in lags) {
        p <- substring(l, 1, nchar(l)-2)
        new_data[i+1,l] <- new_data[i,p]
      }
    }
  }
  new_data$t <- 1:nrow(new_data)
  return(new_data[1:(nrow(new_data)-1),])
}

plot.irf <- function(irf) {
  irf$t <- 1:nrow(irf)
  irf_flat = melt(irf, id=c("t"))
  ggplot(irf_flat) + 
    geom_line(aes(x = t, y = value, color=variable), size = 1)
}

dynare_theme = list(
  theme_classic()+
    theme(plot.title = element_text(hjust = 0.5, face = "bold"), 
          plot.background = element_blank(), 
          plot.margin = unit(c(0.1,0.1,0.1,0.1), "npc"),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.border = element_rect(fill = NA, color="black"),
          axis.line = element_blank()
    ) 
)

plot_irfs <- function(irf, model='rbc') {
  if (model == 'rbc') {
    grid.arrange(ggplot(irf, aes(x=t, y=y)) + geom_line() + xlab("") + ylab("") + ggtitle("y") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=c)) + geom_line() + xlab("") + ylab("") + ggtitle("c") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=k)) + geom_line() + xlab("") + ylab("") + ggtitle("k") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=l)) + geom_line() + xlab("") + ylab("") + ggtitle("l") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=z)) + geom_line() + xlab("") + ylab("") + ggtitle("z") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=r)) + geom_line() + xlab("") + ylab("") + ggtitle("r") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=w)) + geom_line() + xlab("") + ylab("") + ggtitle("w") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=i)) + geom_line() + xlab("") + ylab("") + ggtitle("i") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=g)) + geom_line() + xlab("") + ylab("") + ggtitle("g") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ncol=3)
  } else if (model == 'nk') {
    grid.arrange(ggplot(irf, aes(x=t, y=pi)) + geom_line() + xlab("") + ylab("") + ggtitle("pi") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=p)) + geom_line() + xlab("") + ylab("") + ggtitle("p") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=y)) + geom_line() + xlab("") + ylab("") + ggtitle("y") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=c)) + geom_line() + xlab("") + ylab("") + ggtitle("c") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=w)) + geom_line() + xlab("") + ylab("") + ggtitle("w") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=w_real)) + geom_line() + xlab("") + ylab("") + ggtitle("w_real") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=r_real)) + geom_line() + xlab("") + ylab("") + ggtitle("r_real") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=i)) + geom_line() + xlab("") + ylab("") + ggtitle("i") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=n)) + geom_line() + xlab("") + ylab("") + ggtitle("n") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=m_real)) + geom_line() + xlab("") + ylab("") + ggtitle("m_real") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=m_nominal)) + geom_line() + xlab("") + ylab("") + ggtitle("m_nominal") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=mu)) + geom_line() + xlab("") + ylab("") + ggtitle("mu") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=nu)) + geom_line() + xlab("") + ylab("") + ggtitle("nu") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=a)) + geom_line() + xlab("") + ylab("") + ggtitle("a") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=z)) + geom_line() + xlab("") + ylab("") + ggtitle("z") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ncol=4)
  }
}

# Import and data cleaning
# data <- read.csv('rbc_100k.csv')
# data <- as.data.frame(sapply(data, as.numeric))
# # bl <- c('X', 'log_y', 'log_k', 'log_c', 'log_l', 'log_w', 'log_i', 'eps_z', 'eps_g')
# bl <- c('X', 'eps_z', 'eps_g')
# data <- data[,!(colnames(data) %in% bl)]

data <- read.csv('gali_100k.csv')
data <- as.data.frame(sapply(data, as.numeric))
bl <- c('X', 
        'pi_ann', 'r_nat_ann', 'r_real_ann', 'm_growth_ann', 'i_ann',
        'y_gap', 'mu_hat', 'yhat', 
        'eps_z', 'eps_nu', 'eps_a')
data <- data[,!(colnames(data) %in% bl)]

# data <- read.csv('sw.csv')
# data <- as.data.frame(sapply(data, as.numeric))
# bl <- c('X',                                           # Time
#        'robs', 'labobs', 'pinfobs',                   # Observed values are redundant
#        'dy', 'dc', 'dw', 'dinve',                    # Differences contain information about the past but we'd like to keep it cross-sectional
#        'ewma', 'epinfma', 'spinf', 'sw',               # Shocks collinear with other shocks
#        'zcap', 'zcapf')                               # Remove to eliminate collinearity
# data <- data[,!(colnames(data) %in% bl)]

# Recenter data
for (name in names(data)) {
  data[,name] <- data[,name] - mean(data[,name])
}

# Add lags for all variables
states <- c()
exo_names <- c()
if (length(states) == 0) {
  for (name in names(data)) {
    data[,paste(name, '_1', sep='')] <- c(NA, data[1:nrow(data)-1, name])
    exo_names <- c(exo_names, paste(name, '_1', sep=''))
  }
} else {
  for (name in states) {
    data[,paste(name, '_1', sep='')] <- c(NA, data[1:nrow(data)-1, name])
    exo_names <- c(exo_names, paste(name, '_1', sep=''))
  }
}
data <- data[2:nrow(data),]

# Set lags as roots
all_names <- names(data)
endo_names <- all_names[!(all_names %in% exo_names)]
arc_bl <- tiers2blacklist(list(exo_names, endo_names))
for (exo in exo_names) {
  for (exo2 in exo_names[exo_names != exo]) {
    arc_bl <- rbind(arc_bl, c(exo, exo2))
  }
}

# Find and eliminate colinear variables
# indexes_to_drop should be empty
# png('../text/latex/empirical/images/rbc_correlation.png', width=500, height=500)
corrplot(cor(data))
# dev.off()
names(data[,findCorrelation(cor(data), cutoff = 0.999)])
names(data[,findCorrelation(cor(data), cutoff = 0.95)])

# Hybrid structure learning
hybrid_model <- rsmax2(data,
                       restrict = 'pc.stable', restrict.args = list(NULL, 'cor', 0.05),
                       maximize = 'hc', maximize.args = list('loglik-g'), 
                       blacklist = arc_bl)
graphviz.plot(hybrid_model)
dev.print(png, "../text/latex/empirical/images/rbc_hybrid_dag.png", width=500, height=350)
root.nodes(hybrid_model)
hybrid_fitted <- bn.fit(hybrid_model, data, replace.unidentifiable=T, zero.intercept=T)
hybrid_irf <- simulate.irf(hybrid_fitted, data, "nu_1", 1, 25)
plot_irfs(hybrid_irf, 'nk')

# Constraint based structure learning
pc_model <- pc.stable(data, blacklist = arc_bl, alpha = 0.05)
if (!(directed(pc_model))) {
  pc_model <- pdag2dag(pc_model, ordering=names(data))
}
graphviz.plot(pc_model)
dev.print(png, "../text/latex/empirical/images/rbc_constraint_dag.png", width=500, height=350)
root.nodes(pc_model)
pc_fitted <- bn.fit(pc_model, data, replace.unidentifiable=T, zero.intercept=T)
pc_irf <- simulate.irf(pc_fitted, data, "z_1", 1, 25)
plot_irfs(pc_irf)

# RBC Manual Specification
true_dag <- empty.graph(names(data))
arcs <- matrix(c(# 'eps_z', 'z',
  'z_1', 'z',
  'z', 'y',
  'z', 'c',
  'z', 'i',
  'z', 'w',
  'z', 'r',
  'z', 'l',
  'z', 'k',
  # 'eps_g', 'g',
  'g_1', 'g',
  'g', 'y',
  'g', 'c',
  'g', 'i',
  'g', 'w',
  'g', 'r',
  'g', 'l',
  'g', 'k',
  'k_1', 'y',
  'k_1', 'c',
  'k_1', 'i',
  'k_1', 'w',
  'k_1', 'r',
  'k_1', 'l',
  'k_1', 'k'),
  ncol=2, byrow=T,
  dimnames = list(NULL, c('from', 'to')))
arcs(true_dag) <- arcs
graphviz.plot(true_dag)
dev.print(png, "../text/latex/empirical/images/rbc_true_dag.png", width=500, height=350)
true_fitted <- bn.fit(true_dag, data, replace.unidentifiable=T, zero.intercept=T)
true_irf <- simulate.irf(true_fitted, data, "z", 0.66, 25)
plot_irfs(true_irf)

# NK Manual Specification
true_dag <- empty.graph(names(data))
arcs <- matrix(c('nu_1', 'nu',
                 'nu', 'pi',
                 'nu', 'y_nat',
                 'nu', 'y',
                 'nu', 'r_nat',
                 'nu', 'r_real',
                 'nu', 'i',
                 'nu', 'n',
                 'nu', 'm_real',
                 'nu', 'm_nominal',
                 'nu', 'p',
                 'nu', 'w',
                 'nu', 'c',
                 'nu', 'w_real',
                 'nu', 'mu',
                 'a_1', 'a',
                 'a', 'pi',
                 'a', 'y_nat',
                 'a', 'y',
                 'a', 'r_nat',
                 'a', 'r_real',
                 'a', 'i',
                 'a', 'n',
                 'a', 'm_real',
                 'a', 'm_nominal',
                 'a', 'p',
                 'a', 'w',
                 'a', 'c',
                 'a', 'w_real',
                 'a', 'mu',
                 'z_1', 'z',
                 'z', 'pi',
                 'z', 'y_nat',
                 'z', 'y',
                 'z', 'r_nat',
                 'z', 'r_real',
                 'z', 'i',
                 'z', 'n',
                 'z', 'm_real',
                 'z', 'm_nominal',
                 'z', 'p',
                 'z', 'w',
                 'z', 'c',
                 'z', 'w_real',
                 'z', 'mu',
                 'y_1', 'pi',
                 'y_1', 'y_nat',
                 'y_1', 'y',
                 'y_1', 'r_nat',
                 'y_1', 'r_real',
                 'y_1', 'i',
                 'y_1', 'n',
                 'y_1', 'm_real',
                 'y_1', 'm_nominal',
                 'y_1', 'p',
                 'y_1', 'w',
                 'y_1', 'c',
                 'y_1', 'w_real',
                 'y_1', 'mu',
                 'i_1', 'pi',
                 'i_1', 'y_nat',
                 'i_1', 'y',
                 'i_1', 'r_nat',
                 'i_1', 'r_real',
                 'i_1', 'i',
                 'i_1', 'n',
                 'i_1', 'm_real',
                 'i_1', 'm_nominal',
                 'i_1', 'p',
                 'i_1', 'w',
                 'i_1', 'c',
                 'i_1', 'w_real',
                 'i_1', 'mu',
                 'm_real_1', 'pi',
                 'm_real_1', 'y_nat',
                 'm_real_1', 'y',
                 'm_real_1', 'r_nat',
                 'm_real_1', 'r_real',
                 'm_real_1', 'i',
                 'm_real_1', 'n',
                 'm_real_1', 'm_real',
                 'm_real_1', 'm_nominal',
                 'm_real_1', 'p',
                 'm_real_1', 'w',
                 'm_real_1', 'c',
                 'm_real_1', 'w_real',
                 'm_real_1', 'mu',
                 'p_1', 'pi',
                 'p_1', 'y_nat',
                 'p_1', 'y',
                 'p_1', 'r_nat',
                 'p_1', 'r_real',
                 'p_1', 'i',
                 'p_1', 'n',
                 'p_1', 'm_real',
                 'p_1', 'm_nominal',
                 'p_1', 'p',
                 'p_1', 'w',
                 'p_1', 'c',
                 'p_1', 'w_real',
                 'p_1', 'mu'),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
arcs(true_dag) <- arcs
graphviz.plot(true_dag)
true_fitted <- bn.fit(true_dag, data, replace.unidentifiable=T, zero.intercept=T)
true_irf <- simulate.irf(true_fitted, data, "nu_1", 1, 25)
plot_irfs(true_irf, 'nk')










# Plot partial correlations
lags <- names(data)[grep('_1', names(data))]
data.pcor <- data[,!(names(data) %in% c("z", "g", "k_1")) & !(names(data) %in% lags)]
for (v in names(data)[!(names(data) %in% c("z", "g", "k_1")) & !(names(data) %in% lags)]) {
  data.pcor[,v] <- print(resid(lm(paste(v, " ~ z + g + k_1", sep=""), data=data)))  
}
corrplot(cor(data.pcor))

# Lasso
library(glmnet)
lags <- names(data)[grep('_1', names(data))]
data.x <- data.matrix(data[,!(names(data) %in% c("y","y_1"))])
data.y <- data$y
lambda_seq <- 10^seq(1, -5, by = -.05)
lasso_results = data.frame(matrix(vector(), 0, 4))
colnames(lasso_results) <- c("mse", "df", "lambda", "nzero")
row <- 1
for (lambda in lambda_seq) {
  lasso_model <- glmnet(data.x, data.y, alpha = 1, lambda = lambda, intercept = F)
  lasso_results[row,]$"mse" <- mean((predict(lasso_model, data.x) - data.y)^2)
  lasso_results[row,]$"df" <- lasso_model$df
  lasso_results[row,]$"lambda" <- lasso_model$lambda
  lasso_results[row,]$"nzero" <- list(rownames(coef(lasso_model))[which(coef(lasso_model) != 0)])
  row <- row + 1
}
lasso_results[order(lasso_results$mse),]$nzero



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





