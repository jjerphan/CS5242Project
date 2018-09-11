# Exploration of atoms

ds = read.csv("./nb_atoms.csv", header = TRUE, row.names = 1)

summary(ds)

boxplot(ds, main= "Number of atoms per molecules")
par(mfrow=c(1,2))
plot(density(ds$pro), main = "Estimation of the density of number of atoms for proteins")
hist(ds$pro, xlab = "Number of atoms",freq = FALSE, add=TRUE,breaks =50)

plot(density(ds$lig, from = 0, to = 80),  main = "Estimation of the density of number of atoms for ligans")
hist(ds$lig, xlab = "Number of atoms",freq = FALSE, add=TRUE, breaks = 100)
     