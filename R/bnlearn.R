# Setup
# setwd("/home/emmet/Documents/bayesian/bayesian_networks/data")
setwd("C:/Users/ehall/Documents/thesis/bayesian_networks/data")
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
  } else if (model == 'sw') {
    grid.arrange(ggplot(irf, aes(x=t, y=r)) + geom_line() + xlab("") + ylab("") + ggtitle("r") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=inve)) + geom_line() + xlab("") + ylab("") + ggtitle("i") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=pk)) + geom_line() + xlab("") + ylab("") + ggtitle("q") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=k)) + geom_line() + xlab("") + ylab("") + ggtitle("k") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=pinf)) + geom_line() + xlab("") + ylab("") + ggtitle("pi") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=w)) + geom_line() + xlab("") + ylab("") + ggtitle("w") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=c)) + geom_line() + xlab("") + ylab("") + ggtitle("c") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=y)) + geom_line() + xlab("") + ylab("") + ggtitle("y") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=lab)) + geom_line() + xlab("") + ylab("") + ggtitle("l") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ncol=3)
  } else if (model == 'real') {
    grid.arrange(ggplot(irf, aes(x=t, y=pi)) + geom_line() + xlab("") + ylab("") + ggtitle("pi") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=rm)) + geom_line() + xlab("") + ylab("") + ggtitle("rm") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=rb)) + geom_line() + xlab("") + ylab("") + ggtitle("rb") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=g)) + geom_line() + xlab("") + ylab("") + ggtitle("g") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=y)) + geom_line() + xlab("") + ylab("") + ggtitle("y") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=i)) + geom_line() + xlab("") + ylab("") + ggtitle("i") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=w)) + geom_line() + xlab("") + ylab("") + ggtitle("w") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=n)) + geom_line() + xlab("") + ylab("") + ggtitle("n") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=rk)) + geom_line() + xlab("") + ylab("") + ggtitle("rk") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=z)) + geom_line() + xlab("") + ylab("") + ggtitle("z") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=u)) + geom_line() + xlab("") + ylab("") + ggtitle("u") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=l)) + geom_line() + xlab("") + ylab("") + ggtitle("l") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ggplot(irf, aes(x=t, y=c)) + geom_line() + xlab("") + ylab("") + ggtitle("c") + geom_hline(yintercept = 0, color = "red") + dynare_theme,
                 ncol=4)
  }
}

# Import and data cleaning
data <- read.csv('rbc_100k.csv')
data <- as.data.frame(sapply(data, as.numeric))
# bl <- c('X', 'log_y', 'log_k', 'log_c', 'log_l', 'log_w', 'log_i', 'eps_z', 'eps_g')
bl <- c('X', 'eps_z', 'eps_g')
data <- data[,!(colnames(data) %in% bl)]
data <- data[1:200,]

# data <- read.csv('gali.csv')
# data <- as.data.frame(sapply(data, as.numeric))
# bl <- c('X', 
#         'pi_ann', 'r_nat_ann', 'r_real_ann', 'm_growth_ann', 'i_ann',
#         'y_gap', 'mu_hat', 'yhat', 
#         'eps_z', 'eps_nu', 'eps_a')
# data <- data[,!(colnames(data) %in% bl)]

# data <- read.csv('sw.csv')
# data <- as.data.frame(sapply(data, as.numeric))
# bl <- c('X',                                           # Time
#         'robs', 'labobs', 'pinfobs',                   # Observed values are redundant
#         'dy', 'dc', 'dw', 'dinve',                    # Differences contain information about the past but we'd like to keep it cross-sectional
#         'ea', 'eb', 'eg', 'eqs', 'em', 'epinf', 'ew',
#         'ewma', 'epinfma')                               
# data <- data[,!(colnames(data) %in% bl)]

# data <- read.csv('real_data.csv')
# data <- as.data.frame(sapply(data, as.numeric))
# bl <- c('DATE', 'dk')                                  
# data <- data[,!(colnames(data) %in% bl)]

# Recenter data
for (name in names(data)) {
  data[,name] <- data[,name] - mean(data[,name])
#   data[,name] <- data[,name] + rnorm(1)
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
                       restrict = 'pc.stable', restrict.args = list(NULL, 'cor', 0.25),
                       maximize = 'hc', maximize.args = list('loglik-g'), 
                       blacklist = arc_bl)
for (lag in exo_names) {
  hybrid_model <- set.arc(hybrid_model, lag, substr(lag, 1, nchar(lag)-2))
}
graph.par(list(nodes=list(fontsize=14)))
graphviz.plot(hybrid_model)
# dev.print(png, "../text/latex/empirical/images/rbc_hybrid_dag.png", width=500, height=350)
root.nodes(hybrid_model)
hybrid_fitted <- bn.fit(hybrid_model, data, replace.unidentifiable=T, zero.intercept=T)
hybrid_irf <- simulate.irf(hybrid_fitted, data, 'z', 0.66, 25)
plot_irfs(hybrid_irf, 'rbc')

# Grid search to minimize SHD
results <- data.frame()
alphas <- logspace(-5, 0, 25)
for (a in alphas) {
  hybrid_model <- rsmax2(data,
                         restrict = 'pc.stable', restrict.args = list(NULL, 'cor', a),
                         maximize = 'hc', maximize.args = list('loglik-g'), 
                         blacklist = arc_bl)
  for (lag in exo_names) {
    hybrid_model <- set.arc(hybrid_model, lag, substr(lag, 1, nchar(lag)-2))
  }
  results <- rbind(results, c(a, shd(true_dag, hybrid_model)))
}
names(results) <- c('alpha', 'shd')
results

# Constraint based structure learning
pc_model <- pc.stable(data, blacklist = arc_bl, alpha = 0.1)
for (lag in exo_names) {
  pc_model <- set.arc(pc_model, lag, substr(lag, 1, nchar(lag)-2))
}
if (!(directed(pc_model))) {
  pc_model <- pdag2dag(pc_model, ordering=names(data))
}
graphviz.plot(pc_model)
# dev.print(png, "../text/latex/empirical/images/rbc_constraint_dag.png", width=500, height=350)
pc_fitted <- bn.fit(pc_model, data, replace.unidentifiable=T, zero.intercept=T)
pc_irf <- simulate.irf(pc_fitted, data, "z", 0.66, 250)
plot_irfs(pc_irf, model='rbc')

# RBC Manual Specification
true_dag <- empty.graph(names(data))
arcs <- matrix(c('z_1', 'z',
                 'z', 'y',
                 'z', 'c',
                 'z', 'i',
                 'z', 'w',
                 'z', 'r',
                 'z', 'l',
                 'z', 'k',
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
# dev.print(png, "../text/latex/empirical/images/rbc_true_dag.png", width=500, height=350)
true_fitted <- bn.fit(true_dag, data, replace.unidentifiable=T, zero.intercept=T)
true_irf <- simulate.irf(true_fitted, data, "z", 0.66, 250)
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


# SW Manual Specification
true_dag <- empty.graph(names(data))
arcs <- matrix(c('yf_1', 'zcapf',
                 'yf_1', 'rkf',
                 'yf_1', 'kf',
                 'yf_1', 'pkf',
                 'yf_1', 'cf',
                 'yf_1', 'invef',
                 'yf_1', 'yf',
                 'yf_1', 'labf',
                 'yf_1', 'wf',
                 'yf_1', 'rrf',
                 'yf_1', 'mc',
                 'yf_1', 'zcap',
                 'yf_1', 'rk',
                 'yf_1', 'k',
                 'yf_1', 'pk',
                 'yf_1', 'c',
                 'yf_1', 'inve',
                 'yf_1', 'y',
                 'yf_1', 'lab',
                 'yf_1', 'kpf',
                 'yf_1', 'kp',
                 'yf_1', 'r',
                 'yf_1', 'w',
                 'yf_1', 'pinf',
                 'y_1', 'zcapf',
                 'y_1', 'rkf',
                 'y_1', 'kf',
                 'y_1', 'pkf',
                 'y_1', 'cf',
                 'y_1', 'invef',
                 'y_1', 'yf',
                 'y_1', 'labf',
                 'y_1', 'wf',
                 'y_1', 'rrf',
                 'y_1', 'mc',
                 'y_1', 'zcap',
                 'y_1', 'rk',
                 'y_1', 'k',
                 'y_1', 'pk',
                 'y_1', 'c',
                 'y_1', 'inve',
                 'y_1', 'y',
                 'y_1', 'lab',
                 'y_1', 'kpf',
                 'y_1', 'kp',
                 'y_1', 'r',
                 'y_1', 'w',
                 'y_1', 'pinf',
                 'r_1', 'zcapf',
                 'r_1', 'rkf',
                 'r_1', 'kf',
                 'r_1', 'pkf',
                 'r_1', 'cf',
                 'r_1', 'invef',
                 'r_1', 'yf',
                 'r_1', 'labf',
                 'r_1', 'wf',
                 'r_1', 'rrf',
                 'r_1', 'mc',
                 'r_1', 'zcap',
                 'r_1', 'rk',
                 'r_1', 'k',
                 'r_1', 'pk',
                 'r_1', 'c',
                 'r_1', 'inve',
                 'r_1', 'y',
                 'r_1', 'lab',
                 'r_1', 'kpf',
                 'r_1', 'kp',
                 'r_1', 'r',
                 'r_1', 'w',
                 'r_1', 'pinf',
                 'kpf_1', 'zcapf',
                 'kpf_1', 'rkf',
                 'kpf_1', 'kf',
                 'kpf_1', 'pkf',
                 'kpf_1', 'cf',
                 'kpf_1', 'invef',
                 'kpf_1', 'yf',
                 'kpf_1', 'labf',
                 'kpf_1', 'wf',
                 'kpf_1', 'rrf',
                 'kpf_1', 'mc',
                 'kpf_1', 'zcap',
                 'kpf_1', 'rk',
                 'kpf_1', 'k',
                 'kpf_1', 'pk',
                 'kpf_1', 'c',
                 'kpf_1', 'inve',
                 'kpf_1', 'y',
                 'kpf_1', 'lab',
                 'kpf_1', 'kpf',
                 'kpf_1', 'kp',
                 'kpf_1', 'r',
                 'kpf_1', 'w',
                 'kpf_1', 'pinf',
                 'kp_1', 'zcapf',
                 'kp_1', 'rkf',
                 'kp_1', 'kf',
                 'kp_1', 'pkf',
                 'kp_1', 'cf',
                 'kp_1', 'invef',
                 'kp_1', 'yf',
                 'kp_1', 'labf',
                 'kp_1', 'wf',
                 'kp_1', 'rrf',
                 'kp_1', 'mc',
                 'kp_1', 'zcap',
                 'kp_1', 'rk',
                 'kp_1', 'k',
                 'kp_1', 'pk',
                 'kp_1', 'c',
                 'kp_1', 'inve',
                 'kp_1', 'y',
                 'kp_1', 'lab',
                 'kp_1', 'kpf',
                 'kp_1', 'kp',
                 'kp_1', 'r',
                 'kp_1', 'w',
                 'kp_1', 'pinf',
                 'cf_1', 'zcapf',
                 'cf_1', 'rkf',
                 'cf_1', 'kf',
                 'cf_1', 'pkf',
                 'cf_1', 'cf',
                 'cf_1', 'invef',
                 'cf_1', 'yf',
                 'cf_1', 'labf',
                 'cf_1', 'wf',
                 'cf_1', 'rrf',
                 'cf_1', 'mc',
                 'cf_1', 'zcap',
                 'cf_1', 'rk',
                 'cf_1', 'k',
                 'cf_1', 'pk',
                 'cf_1', 'c',
                 'cf_1', 'inve',
                 'cf_1', 'y',
                 'cf_1', 'lab',
                 'cf_1', 'kpf',
                 'cf_1', 'kp',
                 'cf_1', 'r',
                 'cf_1', 'w',
                 'cf_1', 'pinf',
                 'c_1', 'zcapf',
                 'c_1', 'rkf',
                 'c_1', 'kf',
                 'c_1', 'pkf',
                 'c_1', 'cf',
                 'c_1', 'invef',
                 'c_1', 'yf',
                 'c_1', 'labf',
                 'c_1', 'wf',
                 'c_1', 'rrf',
                 'c_1', 'mc',
                 'c_1', 'zcap',
                 'c_1', 'rk',
                 'c_1', 'k',
                 'c_1', 'pk',
                 'c_1', 'c',
                 'c_1', 'inve',
                 'c_1', 'y',
                 'c_1', 'lab',
                 'c_1', 'kpf',
                 'c_1', 'kp',
                 'c_1', 'r',
                 'c_1', 'w',
                 'c_1', 'pinf',
                 'invef_1', 'zcapf',
                 'invef_1', 'rkf',
                 'invef_1', 'kf',
                 'invef_1', 'pkf',
                 'invef_1', 'cf',
                 'invef_1', 'invef',
                 'invef_1', 'yf',
                 'invef_1', 'labf',
                 'invef_1', 'wf',
                 'invef_1', 'rrf',
                 'invef_1', 'mc',
                 'invef_1', 'zcap',
                 'invef_1', 'rk',
                 'invef_1', 'k',
                 'invef_1', 'pk',
                 'invef_1', 'c',
                 'invef_1', 'inve',
                 'invef_1', 'y',
                 'invef_1', 'lab',
                 'invef_1', 'kpf',
                 'invef_1', 'kp',
                 'invef_1', 'r',
                 'invef_1', 'w',
                 'invef_1', 'pinf',
                 'inve_1', 'zcapf',
                 'inve_1', 'rkf',
                 'inve_1', 'kf',
                 'inve_1', 'pkf',
                 'inve_1', 'cf',
                 'inve_1', 'invef',
                 'inve_1', 'yf',
                 'inve_1', 'labf',
                 'inve_1', 'wf',
                 'inve_1', 'rrf',
                 'inve_1', 'mc',
                 'inve_1', 'zcap',
                 'inve_1', 'rk',
                 'inve_1', 'k',
                 'inve_1', 'pk',
                 'inve_1', 'c',
                 'inve_1', 'inve',
                 'inve_1', 'y',
                 'inve_1', 'lab',
                 'inve_1', 'kpf',
                 'inve_1', 'kp',
                 'inve_1', 'r',
                 'inve_1', 'w',
                 'inve_1', 'pinf',
                 'spinf_1', 'spinf',
                 'spinf', 'zcapf',
                 'spinf', 'rkf',
                 'spinf', 'kf',
                 'spinf', 'pkf',
                 'spinf', 'cf',
                 'spinf', 'invef',
                 'spinf', 'yf',
                 'spinf', 'labf',
                 'spinf', 'wf',
                 'spinf', 'rrf',
                 'spinf', 'mc',
                 'spinf', 'zcap',
                 'spinf', 'rk',
                 'spinf', 'k',
                 'spinf', 'pk',
                 'spinf', 'c',
                 'spinf', 'inve',
                 'spinf', 'y',
                 'spinf', 'lab',
                 'spinf', 'kpf',
                 'spinf', 'kp',
                 'spinf', 'r',
                 'spinf', 'w',
                 'spinf', 'pinf',
                 'sw_1', 'sw',
                 'sw', 'zcapf',
                 'sw', 'rkf',
                 'sw', 'kf',
                 'sw', 'pkf',
                 'sw', 'cf',
                 'sw', 'invef',
                 'sw', 'yf',
                 'sw', 'labf',
                 'sw', 'wf',
                 'sw', 'rrf',
                 'sw', 'mc',
                 'sw', 'zcap',
                 'sw', 'rk',
                 'sw', 'k',
                 'sw', 'pk',
                 'sw', 'c',
                 'sw', 'inve',
                 'sw', 'y',
                 'sw', 'lab',
                 'sw', 'kpf',
                 'sw', 'kp',
                 'sw', 'r',
                 'sw', 'w',
                 'sw', 'pinf',
                 'a_1', 'a',
                 'a', 'zcapf',
                 'a', 'rkf',
                 'a', 'kf',
                 'a', 'pkf',
                 'a', 'cf',
                 'a', 'invef',
                 'a', 'yf',
                 'a', 'labf',
                 'a', 'wf',
                 'a', 'rrf',
                 'a', 'mc',
                 'a', 'zcap',
                 'a', 'rk',
                 'a', 'k',
                 'a', 'pk',
                 'a', 'c',
                 'a', 'inve',
                 'a', 'y',
                 'a', 'lab',
                 'a', 'kpf',
                 'a', 'kp',
                 'a', 'r',
                 'a', 'w',
                 'a', 'pinf',
                 'b_1', 'b',
                 'b', 'zcapf',
                 'b', 'rkf',
                 'b', 'kf',
                 'b', 'pkf',
                 'b', 'cf',
                 'b', 'invef',
                 'b', 'yf',
                 'b', 'labf',
                 'b', 'wf',
                 'b', 'rrf',
                 'b', 'mc',
                 'b', 'zcap',
                 'b', 'rk',
                 'b', 'k',
                 'b', 'pk',
                 'b', 'c',
                 'b', 'inve',
                 'b', 'y',
                 'b', 'lab',
                 'b', 'kpf',
                 'b', 'kp',
                 'b', 'r',
                 'b', 'w',
                 'b', 'pinf',
                 'g_1', 'g',
                 'g', 'zcapf',
                 'g', 'rkf',
                 'g', 'kf',
                 'g', 'pkf',
                 'g', 'cf',
                 'g', 'invef',
                 'g', 'yf',
                 'g', 'labf',
                 'g', 'wf',
                 'g', 'rrf',
                 'g', 'mc',
                 'g', 'zcap',
                 'g', 'rk',
                 'g', 'k',
                 'g', 'pk',
                 'g', 'c',
                 'g', 'inve',
                 'g', 'y',
                 'g', 'lab',
                 'g', 'kpf',
                 'g', 'kp',
                 'g', 'r',
                 'g', 'w',
                 'g', 'pinf',
                 'qs_1', 'qs',
                 'qs', 'zcapf',
                 'qs', 'rkf',
                 'qs', 'kf',
                 'qs', 'pkf',
                 'qs', 'cf',
                 'qs', 'invef',
                 'qs', 'yf',
                 'qs', 'labf',
                 'qs', 'wf',
                 'qs', 'rrf',
                 'qs', 'mc',
                 'qs', 'zcap',
                 'qs', 'rk',
                 'qs', 'k',
                 'qs', 'pk',
                 'qs', 'c',
                 'qs', 'inve',
                 'qs', 'y',
                 'qs', 'lab',
                 'qs', 'kpf',
                 'qs', 'kp',
                 'qs', 'r',
                 'qs', 'w',
                 'qs', 'pinf',
                 'ms_1', 'ms',
                 'ms', 'zcapf',
                 'ms', 'rkf',
                 'ms', 'kf',
                 'ms', 'pkf',
                 'ms', 'cf',
                 'ms', 'invef',
                 'ms', 'yf',
                 'ms', 'labf',
                 'ms', 'wf',
                 'ms', 'rrf',
                 'ms', 'mc',
                 'ms', 'zcap',
                 'ms', 'rk',
                 'ms', 'k',
                 'ms', 'pk',
                 'ms', 'c',
                 'ms', 'inve',
                 'ms', 'y',
                 'ms', 'lab',
                 'ms', 'kpf',
                 'ms', 'kp',
                 'ms', 'r',
                 'ms', 'w',
                 'ms', 'pinf',
                 'w_1', 'zcapf',
                 'w_1', 'rkf',
                 'w_1', 'kf',
                 'w_1', 'pkf',
                 'w_1', 'cf',
                 'w_1', 'invef',
                 'w_1', 'yf',
                 'w_1', 'labf',
                 'w_1', 'wf',
                 'w_1', 'rrf',
                 'w_1', 'mc',
                 'w_1', 'zcap',
                 'w_1', 'rk',
                 'w_1', 'k',
                 'w_1', 'pk',
                 'w_1', 'c',
                 'w_1', 'inve',
                 'w_1', 'y',
                 'w_1', 'lab',
                 'w_1', 'kpf',
                 'w_1', 'kp',
                 'w_1', 'r',
                 'w_1', 'w',
                 'w_1', 'pinf',
                 'pinf_1', 'zcapf',
                 'pinf_1', 'rkf',
                 'pinf_1', 'kf',
                 'pinf_1', 'pkf',
                 'pinf_1', 'cf',
                 'pinf_1', 'invef',
                 'pinf_1', 'yf',
                 'pinf_1', 'labf',
                 'pinf_1', 'wf',
                 'pinf_1', 'rrf',
                 'pinf_1', 'mc',
                 'pinf_1', 'zcap',
                 'pinf_1', 'rk',
                 'pinf_1', 'k',
                 'pinf_1', 'pk',
                 'pinf_1', 'c',
                 'pinf_1', 'inve',
                 'pinf_1', 'y',
                 'pinf_1', 'lab',
                 'pinf_1', 'kpf',
                 'pinf_1', 'kp',
                 'pinf_1', 'r',
                 'pinf_1', 'w',
                 'pinf_1', 'pinf'
                 ),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
arcs(true_dag) <- arcs
graph.par(list(nodes=list(fontsize=18)))
graphviz.plot(true_dag)
true_fitted <- bn.fit(true_dag, data, replace.unidentifiable=T, zero.intercept=T)
true_irf <- simulate.irf(true_fitted, data, "qs", 0.6017, 50)
plot_irfs(true_irf, 'sw')

sim_irf <- read.csv('sw_qs_irf.csv')
sim_irf <- as.data.frame(sapply(sim_irf, as.numeric))
sim_irf$t <- 1:nrow(sim_irf)
plot_irfs(sim_irf, 'sw')

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

# DSGE General Solution
e <- empty.graph(c('exogenous\nstates(t-1)','endogenous\nstates(t-1)', 'controls\n(t-1)',
                   'exogenous\nstates(t)', 'endogenous\nstates(t)', 'controls\n(t)'))
arcs <- matrix(c('exogenous\nstates(t-1)', 'exogenous\nstates(t)',
                 'exogenous\nstates(t)', 'endogenous\nstates(t)',
                 'exogenous\nstates(t)', 'controls\n(t)',
                 'endogenous\nstates(t-1)', 'endogenous\nstates(t)',
                 'endogenous\nstates(t-1)', 'controls\n(t)'),
               ncol=2, byrow=T,
               dimnames = list(NULL, c('from', 'to')))
g <- e
arcs(g) <- arcs
graph.par(list(nodes=list(fontsize=14)))
graphviz.plot(g, layout = "dot", shape = 'ellipse')
dev.print(png, "../text/latex/empirical/images/dsge_dag.png", width=1000, height=700)


