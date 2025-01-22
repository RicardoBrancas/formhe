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
library('devEMF')
library(purrr)

source("analysis/analysis_common.R")

r280 <- read_data('280') # codegemma repair no quant
r281 <- read_data('281') # fl only - llm only - gemma
r282 <- read_data('282') # fl only - llm only - codegemma
r283 <- read_data('283') # fl only - llm only - phi3
r284 <- read_data('284') # fl only - llm only - starcoder 2
r285 <- read_data('285') # fl only - MCSICS only
r286 <- read_data('286') # fl only - line matching only
r287 <- read_data('287') # fl only - default w/o llm
r288 <- read_data('288') # fl only - default
r289 <- read_data('289') # fl only - default w/o reference selection
r290 <- read_data('290') # fl only - MFL only
r291 <- read_data('291') # fl only - default w/o MSICS w/ MFL
r292 <- read_data('292') # gemma fl + mutation only repair
r293 <- read_data('293') # gemma fl + llm only repair - codegemma 7b 4bits
r294 <- read_data('294') # default - no iterative llm
r295 <- read_data('295') # default - w/o reference selection
r296 <- read_data('296', load_log = T) # default - codegemma 7b 4bits
r297 <- read_data('297') # default - codqwen1.5 7b 8bits
r298 <- read_data('298') # default - llama3 8b 4bits
r299 <- read_data('299') # default - phi3 mini
r300 <- read_data('300') # default - gemma 2b
r301 <- read_data('301') # default - codegemma 2b
r301 <- read_data('301') # default - codegemma 2b
r303 <- read_data('303') # default - codegemma 2b

source('analysis/plots.R')


r296 %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error' &
           !is.na(reference_cost)) %>%
  ggplot(aes(x = cut_number(reference_cost, 6), fill = solved)) +
  geom_bar(position = "fill") +
  facet_wrap(~synthetic, scales = "free_y", labeller = label_both) +
  labs(x = "Distance to closest reference implementation", y = "Number of instances", fill = "Repaired?")


fl_table_data <- list('MSICS' = r285,
                      'Line Matching' = r286,
                      'Gemma 2B' = r281,
                      'CodeGemma 2B' = r282,
                      'StarCoder2 3B' = r284,
                      'Phi 3 mini' = r283,
                      'FormHe w/o LLM' = r287,
                      'FormHe with Gemma 2B' = r288,
                      'FormHe w/o implementation choosing' = r289,
                      'MFL' = r290)

fault_identified_plot_new(!!!fl_table_data,
                          facet_vars = vars(synthetic))

print(xtable(fault_identified_plot_new(!!!fl_table_data, facet_vars = vars(synthetic), get_data = T),
             caption = "Results for different fault localization methods for real and synthetic instances.", label = "tab:fault-localizer"),
      include.rownames = F, booktabs = TRUE, NA.string = '--', latex.environments = "center",
      sanitize.text.function = identity, hline.after = c(-1, 0, 1, 2, 6, 9, 10), caption.placement = "top",
      table.placement = "tb", floating.environment = 'table*')

repair_data_1 <- list('Mutation Repair' = r292,
                      'LLM Repair' = r303,
                      'Combined Repair' = r296,
                      # 'Mutation+CodeGemma 7B (4bit) w/o Iterative LLM' = r294,
                      'Combined Repair without\nImpl. Choosing' = r295)

repair_data_2 <- list('Mutation+CodeGemma 7B' = r280,
                      'Mutation+CodeGemma 7B (4bit)' = r296,
                      'Mutation+CodeQwen1.5 7B (8bit)' = r297,
                      # 'Mutation+Llama3 8B (4bit)' = r298,
                      'Mutation+Phi3 mini' = r299,
                      'Mutation+Gemma 2B' = r300,
                      'Mutation+CodeGemma 2B' = r301)

textwidth <- 3.31314

repair_abduction_plot <- times_inverse_cactus(!!!repair_data_1, every_other = 3, filter_expr = expr(!synthetic)) +
  geom_vline(xintercept = 600, linetype = 2, color = '#575757') +
  guides(color = guide_legend(byrow = TRUE, nrow = 2)) +
  theme(legend.position = 'bottom',
        legend.key.spacing.y = unit(0.05, 'cm'),
        legend.margin = margin(0, 0, 0, -5))
print(repair_abduction_plot)
plot_pdf('repair_abduction', 1 * textwidth, .65 * textwidth, repair_abduction_plot)

cat("Average time for fault localization:", (r296 %>% summarise(n = mean(fault.localization.time, na.rm = T)))[[1]])

times_inverse_cactus(!!!repair_data_1, every_other = 3, filter_expr = expr(!synthetic), get_data = T)
times_inverse_cactus(!!!repair_data_1, every_other = 3, filter_expr = expr(synthetic), get_data = T)

times_inverse_cactus(!!!repair_data_2, every_other = 3, filter_expr = expr(!synthetic), get_data = T)

print(xtable(feedback_fault_grid_new(r296)), include.rownames = F)

cat("Longest repair answer with impl choosing:", (r296 %>%
  filter(solved) %>%
  summarise(n = max(real)))[[1]])
cat("Longest repair answer without impl choosing:", (r295 %>%
  filter(solved) %>%
  summarise(n = max(real)))[[1]])

cat("Total instances with feedback:", (r296 %>%
  filter(!synthetic) %>%
  filter(fault.identified == "Yes (no incorrect lines)" |
           fault.identified == "Yes (first MCS)" |
           fault.identified == "Subset (first MCS)" |
           fault.identified == "Superset (first MCS)" |
           fault.identified == "Not Disjoint (first MCS)" |
           feedback_type == "Synthesis Success") %>%
  tally() / r296 %>%
  filter(!synthetic) %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  tally())[[1]])

cat("Total instances with at least one line correctly identified:", (r296 %>%
  filter(!synthetic) %>%
  filter(fault.identified == "Yes (no incorrect lines)" |
           fault.identified == "Yes (first MCS)" |
           fault.identified == "Subset (first MCS)" |
           fault.identified == "Superset (first MCS)" |
           fault.identified == "Not Disjoint (first MCS)") %>%
  tally() / r296 %>%
  filter(!synthetic) %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  tally())[[1]])

cat("Total instances with at least one line correctly identified & repair:", (r296 %>%
  filter(!synthetic) %>%
  filter((fault.identified == "Yes (no incorrect lines)" |
    fault.identified == "Yes (first MCS)" |
    fault.identified == "Subset (first MCS)" |
    fault.identified == "Superset (first MCS)" |
    fault.identified == "Not Disjoint (first MCS)") & feedback_type == "Synthesis Success") %>%
  tally() / r296 %>%
  filter(!synthetic) %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  tally())[[1]])

cat("Total instances with repair but wrong FL:", (r296 %>%
  filter(!synthetic) %>%
  filter(!(fault.identified == "Yes (no incorrect lines)" |
    fault.identified == "Yes (first MCS)" |
    fault.identified == "Subset (first MCS)" |
    fault.identified == "Superset (first MCS)" |
    fault.identified == "Not Disjoint (first MCS)") & feedback_type == "Synthesis Success") %>%
  tally() / r296 %>%
  filter(!synthetic) %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  tally())[[1]])




