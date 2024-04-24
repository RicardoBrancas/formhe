library(scales)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(stringr)
library(units)
library(forcats)
library(tikzDevice)
library(ggthemes)
library(elementalist)
library(xtable)
library(mldr)
library(reticulate)
library(rlang)

source('analysis/analysis_common.R')
source('analysis/plots.R')

data_correct <- Sys.glob("correct_instances/mooshak/*/*")

data_main <- read_data('221') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))

data_mcs_missing_relaxed <- read_data('214') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))
data_mcs_missing_strict <- read_data('215') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))
data_mcs_extra <- read_data('216') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))
data_lm <- read_data('217') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))
data_llm <- read_data('213') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))
data_mfl <- read_data('218') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))

data_fl_gemma2 <- read_data('1056') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))
data_fl_phi2 <- read_data('1054') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))
data_fl_codellama7b <- read_data('1057') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))
data_fl_starcoder2_3b <- read_data('1055') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))

data_no_pruning <- read_data('220') %>% filter(!str_detect(instance, "_private") & !str_detect(instance, "demo"))

cat("Total real instances:", length(data_correct) + (data_main %>%
  filter(str_detect(instance, "mooshak")) %>%
  tally())[[1]], "\n")
cat("Correct real instances:", length(data_correct), "\n")
cat("Incorrect real instances:", (data_main %>%
  filter(str_detect(instance, "mooshak")) %>%
  tally())[[1]], "\n")
cat("Synthetic instances:", (data_main %>%
  filter(!str_detect(instance, "mooshak")) %>%
  tally())[[1]], "\n")
cat("Total incorrect instances:", (data_main %>% tally())[[1]], "\n")

fl_table_data <- list('Missing AS (relaxed)' = data_mcs_missing_relaxed,
                      'Missing AS (strict)' = data_mcs_missing_strict,
                      'Extra Answer Set' = data_mcs_extra,
                      'Line Matching' = data_lm,
                      'Gemma 2B' = data_fl_gemma2,
                      'Phi 2' = data_fl_phi2,
                      'StarCoder 2 3B' = data_fl_starcoder2_3b,
                      'CodeLlama 7B' = data_fl_codellama7b,
                      'MFL' = data_mfl,
                      'FormHe Default' = data_main)

print(xtable(fault_identified_plot_new(!!!fl_table_data, get_data = T, filter_f = function(x) { !str_starts(x, "mooshak") }),
             caption = "Results for the different fault localization methods using only synthetic instances.", label = "tab:fault-localizer"),
      include.rownames = F, booktabs = TRUE, NA.string = '--', latex.environments = "center",
      sanitize.text.function = identity, hline.after = c(-1, 0, 3, 4, 8, 9), caption.placement = "top",
      table.placement = "tb")

print(xtable(fault_identified_plot_new(!!!fl_table_data, get_data = T, filter_f = function(x) { str_starts(x, "mooshak") }),
             caption = "Results for the different fault localization methods using only real instances.", label = "tab:fault-localizer-real"),
      include.rownames = F, booktabs = TRUE, NA.string = '--', latex.environments = "center",
      sanitize.text.function = identity, hline.after = c(-1, 0, 3, 4, 8, 9), caption.placement = "top",
      table.placement = "tb")

print(xtable(fault_identified_plot_new(!!!fl_table_data, get_data = T),
             caption = "Results for the different fault localization methods using all instances.", label = "tab:fault-localizer-real"),
      include.rownames = F, booktabs = TRUE, NA.string = '--', latex.environments = "center",
      sanitize.text.function = identity, hline.after = c(-1, 0, 3, 4, 8, 9), caption.placement = "top",
      table.placement = "tb")



repair_data <- list(
  'Synthetic Instances' = data_main %>% filter(!str_starts(instance, "mooshak")),
  'Real Instances' = data_main %>% filter(str_starts(instance, "mooshak"))
  # 'Symmetry Breaking\nDisabled' = r192 %>% select(-models.missing),
  # 'Semantic Pruning\nDisabled' = r193 %>% select(-models.missing),
  # 'Pruning\nDisabled' = data_no_pruning
)

textwidth <- 5.5129

plot_pdf('repair_all', 1 * textwidth, .35 * textwidth,
         times_inverse_cactus(!!!repair_data, every_other = 50
                              #filter_f = function(x) { str_starts(x, "mooshak") }, every_other = 1
         ) +
           geom_vline(xintercept = 600, linetype = 2, color = '#575757') +
           guides(color = guide_legend(byrow = TRUE)) +
           theme(legend.position = 'right',
                 legend.spacing.y = unit(.25, 'cm')))

cat("Average time for fault localization:", (data_main %>% summarise(n = mean(fl.llm.time + fl.line.matching.time + fl.mcs.time, na.rm = T)))[[1]])

print("Repair data for 600 seconds:")
times_inverse_cactus(!!!repair_data, get_data = T)

print("Repair data for 30 seconds:")
times_inverse_cactus(!!!repair_data, get_data = T, get_data_cutoff = 60)

feedback_fault_grid_new(data_main)

cat("Total instances with feedback:", (data_main %>%
  filter(fault.identified == "Yes (no incorrect lines)" |
           fault.identified == "Yes (first MCS)" |
           fault.identified == "Subset (first MCS)" |
           fault.identified == "Superset (first MCS)" |
           fault.identified == "Not Disjoint (first MCS)" |
           feedback_type == "Synthesis Success") %>%
  tally() / data_main %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  tally())[[1]])

cat("Total instances with at least one line correctly identified:", (data_main %>%
  filter(fault.identified == "Yes (no incorrect lines)" |
           fault.identified == "Yes (first MCS)" |
           fault.identified == "Subset (first MCS)" |
           fault.identified == "Superset (first MCS)" |
           fault.identified == "Not Disjoint (first MCS)") %>%
  tally() / data_main %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  tally())[[1]])

cat("Total instances with at least one line correctly identified & repair:", (data_main %>%
  filter((fault.identified == "Yes (no incorrect lines)" |
           fault.identified == "Yes (first MCS)" |
           fault.identified == "Subset (first MCS)" |
           fault.identified == "Superset (first MCS)" |
           fault.identified == "Not Disjoint (first MCS)") & feedback_type == "Synthesis Success") %>%
  tally() / data_main %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  tally())[[1]])

cat("Total instances with repair but wrong FL:", (data_main %>%
  filter(!(fault.identified == "Yes (no incorrect lines)" |
           fault.identified == "Yes (first MCS)" |
           fault.identified == "Subset (first MCS)" |
           fault.identified == "Superset (first MCS)" |
           fault.identified == "Not Disjoint (first MCS)") & feedback_type == "Synthesis Success") %>%
  tally() / data_main %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  tally())[[1]])


