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


notion <- read_csv('analysis/notion.csv') %>% mutate(instance = paste0('mooshak/', instance))
notion2 <- read_csv('analysis/notion2.csv') %>% mutate(instance = paste0('mooshak/', instance))

read_data <- function(name) {
  data <- read_csv(paste0('analysis/data/', name, '.csv')) %>%
    mutate(solved = status == 0 & !timeout & !memout,
           feedback_type = ifelse(timeout & feedback_type != 'Synthesis Success', 'Timeout', feedback_type),
           feedback_type = ifelse(memout & feedback_type != 'Synthesis Success', 'Memout', feedback_type),
           ram = ram / 1024,
           # problem = str_match(instance, '(?:.*)/(.).*')[, 2]) %>%
           problem_tmp1 = str_match(instance, '(.*/.)_.*|(.*)/.*')[, 2],
           problem_tmp2 = str_match(instance, '(.*/.)_.*|(.*)/.*')[, 3],
           problem = coalesce(problem_tmp1, problem_tmp2)) %>%
    filter(problem != 'mooshak/F' & problem != 'mooshak/01/F')
  # filter(str_starts(problem, 'mooshak'))
  if ('fault.partial' %in% names(data)) {
    data <- data %>%
      mutate(fault.identified.full = ifelse(!is.na(fault.partial) & fault.partial == 'Yes', paste(fault.identified, '(Partial)'), fault.identified))
  }
  data %>% mutate(selfeval.deleted.lines = as.character(selfeval.deleted.lines),
                  selfeval.changes.generate.n = as.character(selfeval.changes.generate.n),
                  selfeval.changes.generate.n.unique = as.character(selfeval.changes.generate.n.unique),
                  selfeval.changes.test.n = as.character(selfeval.changes.test.n),
                  selfeval.changes.test.n.unique = as.character(selfeval.changes.test.n.unique))
  # data %>% left_join(notion, by = 'instance')
}

r184 <- read_data('184') # default
r195 <- read_data('195') # default
r196 <- read_data('196') # default
r197 <- read_data('197') # default
r198 <- read_data('198') # default
r199 <- read_data('199') # sim
r200 <- read_data('200') # default
r201 <- read_data('201') # default
r202 <- read_data('202') # default
r203 <- read_data('203') # default
r204 <- read_data('204') # default
r205 <- read_data('205') # default
r206 <- read_data('206') # 1 hour
r207 <- read_data('207') # default
r208 <- read_data('208') # default
r210 <- read_data('210') # default
r211 <- read_data('211') # default
r212 <- read_data('212') # default
r213 <- read_data('213') # default

r1001 <- read_data('1001') # default
r1002 <- read_data('1002') # default
r1003 <- read_data('1003') # default
r1012 <- read_data('1012') # default
r1013 <- read_data('1013') # default
r1014 <- read_data('1014') # default
r1015 <- read_data('1015') # fl only
r1016 <- read_data('1016') # fl only
r1017 <- read_data('1017') # fl only
r1018 <- read_data('1018') # fl only
r1019 <- read_data('1019') # fl only
r1020 <- read_data('1020') # fl only
r1021 <- read_data('1021') # none
r1022 <- read_data('1022') # none-smallest
r1023 <- read_data('1023') # hit-count
r1024 <- read_data('1024') # hit-count-normalized
r1025 <- read_data('1025') # hit-count-smallest
r1026 <- read_data('1026') # random
r1027 <- read_data('1027') # random-smallest
r1028 <- read_data('1028') # default
r1029 <- read_data('1029') # default
r1030 <- read_data('1030') # default
r1031 <- read_data('1031') # default
r1032 <- read_data('1032') # default
r1033 <- read_data('1033') # llm
r1034 <- read_data('1034') # llm
r1035 <- read_data('1035') # llm
r1036 <- read_data('1036') # llm
r1037 <- read_data('1037') # llm
r1038 <- read_data('1038') # llm
r1039 <- read_data('1039') # llm
r1040 <- read_data('1040') # default
r1041 <- read_data('1041') # no llm
r1042 <- read_data('1042') # llm
r1043 <- read_data('1043') # default - lm
r1044 <- read_data('1044') # default (model)
r1045 <- read_data('1045') # default (lora_it_full_datasetv2)
r1046 <- read_data('1046') # default (lora_non_it_full_datasetv2)
r1047 <- read_data('1047') # default (lora_it_full_datasetv2_promptv2)
r1048 <- read_data('1048') # default (lora_it_full_datasetv2_promptv2)
r1049 <- read_data('1049') # default (lora-gemma-2b-it-datasetv3-promptv2)
r1050 <- read_data('1050') # default (lora-gemma-2b-it-datasetv3-promptv2-targetmodules)
r1051 <- read_data('1051') # default (lora-phi-2-datasetv3-promptv2-targetmodules)


source('analysis/plots.R')

times_inverse_cactus('Sketch-based enum' = r195, "Mutation-based enum" = r207, "Mutation-based enum w/ LLM FL" = r210)
times_inverse_cactus('Sketch-based enum' = r195, "Mutation-based enum" = r207, "Mutation-based enum w/ LLM FL" = r210, "Mutation-based enum w/ LLM FL + New Perm" = r211)
times_inverse_cactus("New Dataset" = r213)
times_inverse_cactus("New Dataset" = r213, filter_f = function(x) { str_starts(x, "mooshak") })
times_inverse_cactus('Sketch-based enum' = r195, "Mutation-based enum" = r207, "Mutation-based enum w/ LLM FL" = r210, filter_f = function(x) { str_starts(x, "mooshak") })

fault_identified_plot_new('Sketch-based enum' = r195, "Mutation-based enum" = r207, "Mutation-based enum w/ LLM FL" = r210, "New Dataset" = r213)
fault_identified_plot_new('Sketch-based enum' = r195, "Mutation-based enum" = r207, "Mutation-based enum w/ LLM FL" = r210, "New Dataset" = r213, filter_f = function(x) { str_starts(x, "mooshak") })
fault_identified_plot_new('Sketch-based enum' = r195, "Mutation-based enum" = r207, "Mutation-based enum w/ LLM FL" = r210, "New Dataset" = r213, filter_f = function(x) { str_starts(x, "mooshak") }, reduced_labels = F)
fault_identified_plot_new('Sketch-based enum' = r195, "Mutation-based enum" = r207, "Mutation-based enum w/ LLM FL" = r210, "New Dataset" = r213, reduced_labels = F)

inner_join(r210, r211, by = 'instance') %>% ggplot(aes(x = real.x, y = real.y)) +
  geom_point() +
  geom_abline()

detailed_times_boxplot("Mutation-based enum w/ LLM FL" = r210, "Mutation-based enum w/ LLM FL + New Perm" = r211, percentage = F)
detailed_times_boxplot("Mutation-based enum w/ LLM FL" = r210 %>% filter(feedback_type != "Synthesis Success"), percentage = F)
detailed_times_boxplot("Mutation-based enum w/ LLM FL" = r210, percentage = T)

inner_join(r204, r205, by = 'instance') %>% ggplot(aes(x = enum.programs.x, y = enum.programs.y)) +
  geom_point() +
  geom_abline()

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




