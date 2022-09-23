library(scales)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)

textwidth <- 3.34

read_data <- function(name) {
  data <- read.csv(paste0('analysis/data/', name, '.csv')) %>%
    mutate(solved = status == 0)
}

r001 <- read_data('001')
r002 <- read_data('002')
r003 <- read_data('003')
r004 <- read_data('004')

source('analysis/plots.R')


instance_info_dist(real, cpu, data=r001)