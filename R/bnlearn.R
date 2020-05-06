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

### Baseline RBC (iid shocks)
data <- read.csv('rbc.csv')
# data <- read.csv('gali.csv')
# data <- read.csv('sw.csv')
data <- as.data.frame(sapply(data, as.numeric))

# RBC
bl <- c('y', 'k', 'c', 'l', 'w', 'i', 'X', 'eps_g', 'eps_z')
# data$"log_k-1" <- c(NA, data$log_k[1:nrow(data)-1])
# data <- data[2:nrow(data),]
wl <- names(data)

# Gali
# bl <- c('X', 'log_w', 'log_n', 'log_z', 'log_y', 'log_p', 'log_a', 
#         'log_m_nominal', 's', 'x_aux_1', 'x_aux_2', 'money_growth_ann', 
#         'i_ann', 'pi_ann', 'r_real_ann', 'realinterest', 'pi_star',
#         'eps_z', 'eps_a', 'eps_m')
# data$"z-1" <- c(NA, data$z[1:nrow(data)-1])
# data$"a-1" <- c(NA, data$a[1:nrow(data)-1])
# data$"mg-1" <- c(NA, data$money_growth[1:nrow(data)-1])
# data <- data[2:nrow(data),]
# wl <- names(data)

# Smets and Wouters
# bl <- c('X')
# wl <- names(data)

data <- data[,colnames(data) %in% wl & !(colnames(data) %in% bl)]


# Split for holdout
train <- data[1:floor(0.8*nrow(data)),]
test <- data[(floor(0.8*nrow(data))+1):nrow(data),]

# for(test in tests){
#   pc_model <- pc.stable(data, cluster = cl, test=test)
#   print(test)
#   print(root.nodes(pc_model))
# }

# Fit models using different structure learning methods
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


### Very simple confounded example
# Setup
n <- 10000
z <- rnorm(n)
w <- rnorm(n)
u <- rnorm(n, 0, 10)
v <- rnorm(n, 0, 1)
x <- 2*z + 2*w + u
y <- 2*x + w + v
iv_data <- data.frame(z, w, u, v, x, y)
names(iv_data) <- c("z", "w", "u", "v", "x", "y")
# iv_data <- iv_data[,!(names(iv_data) == "w")]
iv_train <- iv_data[1:floor(0.8*nrow(iv_data)),]
iv_test <- iv_data[(floor(0.8*nrow(iv_data))+1):nrow(iv_data),]

# Structure Learning
iv_model <-pc.stable(iv_train, cluster = cl, alpha=0.05)
graphviz.plot(iv_model)
root.nodes(iv_model)

# Parameter Learning
iv_model <- bn.fit(iv_model, iv_train)
iv_model
