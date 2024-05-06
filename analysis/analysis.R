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


# notion <- read_csv('analysis/notion.csv') %>% mutate(instance = paste0('mooshak/', instance))
# notion2 <- read_csv('analysis/notion2.csv') %>% mutate(instance = paste0('mooshak/', instance))

source("analysis/analysis_common.R")

r214 <- read_data('214') # mcs missing relaxed
r215 <- read_data('215') # mcs missing strict
r216 <- read_data('216') # mcs extra
r217 <- read_data('217') # line matching
r218 <- read_data('218') # mfl
r221 <- read_data('221') # default
r222 <- read_data('222') # simulated

r1054 <- read_data('1054') # fl-only (phi 2)
r1055 <- read_data('1055') # fl-only (starcoder 2 3b)
r1056 <- read_data('1056') # fl-only (gemma 2b)
r1057 <- read_data('1057') # fl-only (codellama 7b)
r1058 <- read_data('1058') # llm repair-only (gemma, gemma)
r1059 <- read_data('1059') # llm repair-only (gemma, codellama) + DC
r1060 <- read_data('1060') # fl-only gemma + DC
r1061 <- read_data('1061') # simulated fl + gemma repair
r1062 <- read_data('1062') # fl-only (normal fl + llama 3 8B)
r1063 <- read_data('1063') # fl-only (llama 3 8B)
r1064 <- read_data('1064') # fl-only (mistral 7b)

source('analysis/plots.R')

fault_identified_plot_new('Formhe (Gemma FL)' = r221, 'Formhe (Llama 3 8B FL)' = r1062)
fault_identified_plot_new('Formhe (Gemma FL)' = r221, 'Formhe (Llama 3 8B FL)' = r1062, facet_vars = vars(public, synthetic))
fault_identified_plot_new('Phi 2' = r1054, 'StarCoder 2 3B' = r1055, 'Gemma 2B' = r1056, 'CodeLlama 7B' = r1057, 'Llama 3 8B' = r1063, 'Mistral 7B' = r1064)
fault_identified_plot_new('Gemma 2B' = r1056, 'CodeLlama 7B' = r1057, 'Llama 3 8B' = r1063, 'Mistral 7B' = r1064, facet_vars = vars(public, synthetic))

times_inverse_cactus('FormHe' = r221, 'FormHe (sim FL)' = r222, 'Gemma Repair' = r1058, 'Gemma Repair (sim FL)' = r1061, 'CodeLlama Repair' = r1059)
times_inverse_cactus('FormHe' = r221, 'FormHe (sim FL)' = r222, 'Gemma Repair' = r1058, 'Gemma Repair (sim FL)' = r1061, 'CodeLlama Repair' = r1059, filter_f = function(x) { !str_starts(x, "mooshak") })
times_inverse_cactus('FormHe' = r221, 'FormHe (sim FL)' = r222, 'Gemma Repair' = r1058, 'Gemma Repair (sim FL)' = r1061, 'CodeLlama Repair' = r1059, filter_f = function(x) { str_starts(x, "mooshak") })
times_inverse_cactus('FormHe' = r221, 'FormHe (sim FL)' = r222, 'Gemma Repair' = r1058, 'Gemma Repair (sim FL)' = r1061, 'CodeLlama Repair' = r1059, filter_f = function(x) { str_starts(x, "mooshak") })
times_inverse_cactus('FormHe' = r221, 'FormHe (sim FL)' = r222, 'Gemma Repair' = r1058, 'Gemma Repair (sim FL)' = r1061, 'CodeLlama Repair' = r1059, filter_f = function(x) { str_starts(x, "mooshak/") })
times_inverse_cactus('FormHe' = r221, 'FormHe (sim FL)' = r222, 'Gemma Repair' = r1058, 'Gemma Repair (sim FL)' = r1061, 'CodeLlama Repair' = r1059, filter_f = function(x) { str_starts(x, "mooshak_") })

fault_identified_plot_new('Sketch-based enum' = r195, "Mutation-based enum" = r207, "Mutation-based enum w/ LLM FL" = r210, "New Dataset" = r213, filter_f = function(x) { str_starts(x, "mooshak") })

inner_join(r221, r1058, by = 'instance') %>% ggplot(aes(x = real.x, y = real.y)) +
  geom_point() +
  geom_abline()

a <- inner_join(r221, r1058, by = 'instance') %>%
  filter(str_starts(instance, "mooshak")) %>%
  select(instance, feedback_type.x, feedback_type.y)

detailed_times_boxplot("Mutation-based enum w/ LLM FL" = r210, "Mutation-based enum w/ LLM FL + New Perm" = r211, percentage = F)
detailed_times_boxplot("Mutation-based enum w/ LLM FL" = r210 %>% filter(feedback_type != "Synthesis Success"), percentage = F)
detailed_times_boxplot("Mutation-based enum w/ LLM FL" = r210, percentage = T)

inner_join(r204, r205, by = 'instance') %>% ggplot(aes(x = enum.programs.x, y = enum.programs.y)) +
  geom_point() +
  geom_abline()

data_main %>%
  filter(!str_detect(instance, "mooshak/")) %>%
  sample_n(5)

##

fault_identified_plot_new("FormHe" = r1028, "FormHe 2" = r1041, "LLM" = r1033, "LLM Modified Prompt" = r1034, "Finetuned LLM Classifier" = r1035, "Finetuned LLM Classifier  Comb" = r1036, "Finetuned LLM Classifier  Comb+Sort" = r1037, reduced_labels = T)
fault_identified_plot_new("FormHe" = r1041, "LLM" = r1042, "Formhe+lora gemma 2b it datasetv2 promptv1 targetmodules" = r1044, "Formhe+lora gemma 2b it datasetv2 promptv2 targetmodules" = r1047, "Formhe+lora gemma 2b it datasetv3 promptv2 targetmodules" = r1050, "Formhe+lora phi 2 datasetv3 promptv2 targetmodules" = r1051, reduced_labels = T)
fault_identified_plot_new("FormHe" = r1041, "LLM" = r1042, "Formhe+lora gemma 2b it datasetv2 promptv1 targetmodules" = r1044, "Formhe+lora gemma 2b it datasetv2 promptv2 targetmodules" = r1047, "Formhe+lora gemma 2b it datasetv3 promptv2 targetmodules" = r1050, "Formhe+lora phi 2 datasetv3 promptv2 targetmodules" = r1051, reduced_labels = F)
fault_identified_plot_new("FormHe" = r1041, "LLM" = r1042, "Formhe+lora gemma 2b it datasetv2 promptv1 targetmodules" = r1044, "Formhe+lora gemma 2b it datasetv2 promptv2 targetmodules" = r1047, "Formhe+lora gemma 2b it datasetv3 promptv2 targetmodules" = r1050, "Formhe+lora phi 2 datasetv3 promptv2 targetmodules" = r1051, reduced_labels = T, filter_f = function(x) { str_starts(x, "mooshak") })
fault_identified_plot_new("FormHe" = r1041, "LLM" = r1042, "Formhe+lora gemma 2b it datasetv2 promptv1 targetmodules" = r1044, "Formhe+lora gemma 2b it datasetv2 promptv2 targetmodules" = r1047, "Formhe+lora gemma 2b it datasetv3 promptv2 targetmodules" = r1050, "Formhe+lora phi 2 datasetv3 promptv2 targetmodules" = r1051, reduced_labels = T, filter_f = function(x) { str_starts(x, "mooshak/") })
fault_identified_plot_new("FormHe" = r1041, "LLM" = r1042, "Formhe+lora gemma 2b it datasetv2 promptv1 targetmodules" = r1044, "Formhe+lora gemma 2b it datasetv2 promptv2 targetmodules" = r1047, "Formhe+lora gemma 2b it datasetv3 promptv2 targetmodules" = r1050, "Formhe+lora phi 2 datasetv3 promptv2 targetmodules" = r1051, reduced_labels = T, filter_f = function(x) { str_starts(x, "mooshak_") })
fault_identified_plot_new("FormHe" = r1041, "LLM" = r1042, "Formhe+lora gemma 2b it datasetv2 promptv1 targetmodules" = r1044, "Formhe+lora gemma 2b it datasetv2 promptv2 targetmodules" = r1047, "Formhe+lora gemma 2b it datasetv3 promptv2 targetmodules" = r1050, "Formhe+lora phi 2 datasetv3 promptv2 targetmodules" = r1051, reduced_labels = F, filter_f = function(x) { str_starts(x, "mooshak") })
fault_identified_plot_new("FormHe" = r1028, "FormHe 2" = r1041, "LLM" = r1033, "LLM Modified Prompt" = r1034, "Finetuned LLM Classifier" = r1035, reduced_labels = T)
fault_identified_plot_new("FormHe" = r1028, "Finetuned LLM Classifier" = r1035, reduced_labels = T, filter_f = function(x) { str_starts(x, "mooshak") })
fault_identified_plot_new("FormHe" = r1028, "LLM" = r1033, reduced_labels = F)
fault_identified_plot_new("FormHe" = r1028, "LLM" = r1033, reduced_labels = F, filter_f = function(x) { str_starts(x, "mooshak") })

a <- r1028 %>%
  filter(str_starts(instance, "mooshak")) %>%
  filter(fault.identified == "Yes (first MCS)" | fault.identified == "Yes (no incorrect lines)") %>%
  select(instance, fault.identified)
a <- r1035 %>%
  filter(str_starts(instance, "mooshak")) %>%
  filter(fault.identified == "Yes (first MCS)" | fault.identified == "Yes (no incorrect lines)") %>%
  select(instance, fault.identified)
a <- inner_join(r1028, r1035, by = "instance") %>%
  filter(str_starts(instance, "mooshak")) %>%
  filter((fault.identified.x == "Yes (first MCS)" | fault.identified.x == "Yes (no incorrect lines)") | (fault.identified.y == "Yes (first MCS)" | fault.identified.y == "Yes (no incorrect lines)")) %>%
  select(instance, fault.identified.x, mcss, mcss.sorted.x, fault.identified.y, fl.llm, mcss.sorted.y)

#### PAPER PLOTS

plot_pdf('times', 0.8 * linewidth, .29 * linewidth,
         times_inverse_cactus(FormHe = r184, 'Pruning\nDisabled' = r191 %>% select(-models.missing)) +
           geom_vline(xintercept = 600, linetype = 2, color = '#575757') +
           theme(legend.position = 'right'))

print(xtable(fault_identified_plot_new('Missing AS (relaxed)' = r185,
                                       'Missing AS (strict)' = r186,
                                       'Extra Answer Set' = r187,
                                       'Line Matching' = r188,
                                       'SBFL' = r190,
                                       'MFL' = r189,
                                       'FormHe Default' = r184,
                                       get_data = T)), include.rownames = F, booktabs = TRUE, NA.string = '--', latex.environments = "center", sanitize.text.function = identity, hline.after = c(-1, 0, 5, 6, 7))

feedback_fault_grid_new(r184)
feedback_fault_grid_new(r191)

r184 %>%
  filter(startsWith(problem, 'mooshak')) %>%
  sample_n(5)
r184 %>%
  filter(!startsWith(problem, 'mooshak')) %>%
  sample_n(5)

#### POSTER PLOTS

library(extrafont)
font_import()
loadfonts(device = "win")
library(ggpattern)


plot_emf("fl", 5, 2.5, fault_identified_plot_new('FormHe Default' = r184, pattern = T) +
  coord_polar("y") +
  geom_text(stat = 'count', color = "black", aes(x = 1.8, family = "Noto Sans", label = scales::percent(after_stat(count / sum(count)))), position = position_stack(reverse = TRUE, vjust = .5)) +
  scale_fill_brewer(palette = "Dark2") +
  scale_pattern_fill_manual(values = c("#0c4634", "#743301", "#45417c", "#971156")) +
  theme_void(base_family = "Noto Sans") +
  theme(aspect.ratio = 1) +
  scale_pattern_discrete(guide = guide_legend(nrow = 1)))

plot_emf("r", 4, 2, times_inverse_cactus(FormHe = r184) +
  geom_vline(xintercept = 600, linetype = 2, color = '#575757') +
  theme(legend.position = 'none') +
  scale_y_continuous(breaks = extended_breaks(n = 6), labels = label_percent(accuracy = 1, suffix = '%')) +
  labs(y = '% Hints Found', x = 'Time (s)') +
  scale_color_brewer(palette = "Dark2"))

#### THESIS PLOTS

textwidth <- 6.30045

plot_pdf('times_full', 0.9 * textwidth, .35 * textwidth,
         times_inverse_cactus(FormHe = r184,
                              'Symmetry Breaking\nDisabled' = r192 %>% select(-models.missing),
                              'Semantic Pruning\nDisabled' = r193 %>% select(-models.missing),
                              'All Pruning\nDisabled' = r191 %>% select(-models.missing)) +
           geom_vline(xintercept = 600, linetype = 2, color = '#575757') +
           guides(color = guide_legend(byrow = TRUE)) +
           theme(legend.position = 'right',
                 legend.spacing.y = unit(.25, 'cm')))




