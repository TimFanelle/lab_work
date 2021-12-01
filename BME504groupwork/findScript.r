# source('findScript.r')

library(dplyr)
library(data.table)
setwd("file/path") #this should be changed based on the file system

inputFile <- "in.csv" #this should be changed for each video
outputFile <- "out.csv" # this should be changed for each video
postureSeconds <- list(5, 27, 53, 75, 104, 122, 144, 19) # this should be changed for each video
fRate <- 30

data <- read.csv(inputFile)
ground <- select(data, 2:4)
eps <- select(data, 5:7)
hMax <- select(data, 8:10)
vMax <- select(data, 11:13)

print("Beginning Analysis")

oFlen <- length(postureSeconds)
outFrame <- data.frame(matrix(0, oFlen, 3))

outFrame[1, ] <- data.frame("Ground X", "Ground Y", "Likelihood")
outFrame[3, ] <- data.frame("hMax X", "hMax Y", "Likelihood")
outFrame[5, ] <- data.frame("vMax X", "vMax Y", "Likelihood")
outFrame[7, ] <- data.frame("Endpoint X", "Endpoint Y", "Likelihood")

outFrame[2,] <- data.frame(ground %>% top_n(1))
outFrame[4,] <- data.frame(hMax %>% top_n(1))
outFrame[6,] <- data.frame(vMax %>% top_n(1))

j <- 8
for (i in postureSeconds) {
    hBound <- (i * fRate) + (fRate / 3)
    lBound <- (i * fRate) - (fRate / 3)
    if (lBound < 0) {
        lBound <- 0
    }
    if (hBound > nrow(eps)) {
        hBound <- nrow(eps)
    }
    tuv <- eps[lBound:hBound, c(1:3)]
    epMax <- data.frame(tuv %>% top_n(1))
    outFrame[j,] <- epMax
    j <- j + 1
}
print("Analysis Done")
fwrite(outFrame, outputFile, col.names = FALSE)
print("output file written")