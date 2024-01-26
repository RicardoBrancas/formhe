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
    filter(problem != 'mooshak/F')
  if ('fault.partial' %in% names(data)) {
    data <- data %>%
      mutate(fault.identified.full = ifelse(!is.na(fault.partial) & fault.partial == 'Yes', paste(fault.identified, '(Partial)'), fault.identified))
  }
  data %>% left_join(notion, by = 'instance')
}

r184 <- read_data('184') # default
r185 <- read_data('185') # only relaxed
r186 <- read_data('186') # only strict
r187 <- read_data('187') # only extra
r188 <- read_data('188') # only line matching
r189 <- read_data('189') # only mfl
r190 <- read_data('190') # only sbfl
r191 <- read_data('191') # no pruning
r192 <- read_data('192') # no symettry breaking
r193 <- read_data('193') # no semantic pruning

source('analysis/plots.R')

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
  labs(y = '% Hints Found', x = 'Time (s)')+
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

#plot_pdf('fault_identified', linewidth, 0.4 * linewidth,
#          fault_identified_plot_new('Missing A. S.\n(relaxed)' = r177, 'Missing A. S.\n(strict)' = r178,
#                                    'Extra Answer\nSet' = r179, 'Line\nMatching' = r180,
#                                    'SBFL' = r182, 'MFL' = r181, 'FormHe\nDefault' = r176,
#                                    angle = 30))

#plot_pdf('fault_identified_detailed', linewidth, 0.4 * linewidth * 2,
#          fault_identified_plot_new('Missing A. S.\n(relaxed)' = r177, 'Missing A. S.\n(strict)' = r178,
#                                    'Extra Answer\nSet' = r179, 'Line\nMatching' = r180,
#                                    'SBFL' = r182, 'MFL' = r181, 'FormHe\nDefault' = r176,
#                                    angle = 30) + facet_wrap(~case_when(startsWith(problem, 'mooshak') ~ 'A, B, C, D and E',
#                                                                        problem == 'DT_0' ~ 'Y and Z',
#                                                                        problem == 'JFP_0' ~ 'Y and Z',
#                                                                        TRUE ~ 'A, B, C, D and E synt'), scales = 'free_y', ncol = 1))

# plot_pdf('feedback', .6 * linewidth, .6 * linewidth, feedback_type_plot(FormHe = r163,
#                                                                         # 'Fault Localizer\nOracle' = r086,
#                                                                         'Pruning\nDisabled' = r169,
#                                                                         angle = 0))

# plot_pdf('fault_identified', linewidth, .8 * linewidth, fault_identified_plot(Default = r068, 'Disable missing\nrelaxed' = r076, 'Disable missing\nstrict' = r075,
#                                                                               'Enable extra' = r078, 'Disable line pairings' = r077,
#                                                                               angle = 30))


# plot_pdf('fault_identified_detailed', linewidth, 0.4 * linewidth * 3, fault_identified_plot('Missing A. S.\n(relaxed)' = r164, 'Missing A. S.\n(strict)' = r165,
#                                                                                'Extra Answer\nSet' = r166, 'Line\nMatching' = r167,
#                                                                                # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
#                                                                                # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
#                                                                                'SBFL' = r170, 'MFL' = r168, 'FormHe\nDefault' = r163,
#                                                                                angle = 30) + facet_wrap(~case_when(startsWith(problem, 'mooshak') ~ 'A, B, C, D and E submitted',
#                                                                                                                    problem == 'DT_0' ~ 'Y and Z',
#                                                                                                                    problem == 'JFP_0' ~ 'Y and Z',
#                                                                                                                    TRUE ~ 'A, B, C, D and E synthetic'), scales = 'free_y', ncol = 1))

# plot_pdf('fault_identified_combined', linewidth, 0.35 * linewidth, fault_identified_plot('Missing A. S.\n(relaxed)' = r164, 'Missing A. S.\n(strict)' = r165,
#                                                                                          'Line\nMatching' = r167,
#                                                                                          # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
#                                                                                          # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
#                                                                                          'FormHe Fault\nLocalization' = r163,
#                                                                                          angle = 0))


###### OTHER PLOTS

# r163 %>% group_by(problem) %>% count()
#
# fault_identified_plot('MCS HC' = r147, 'MCS HCN' = r148, 'MCS HCS' = r149, 'MCS RS' = r146, 'WMCS1 RS' = r161, 'MFL HC' = r151, 'MFL HCN' = r152, 'MFL HCS' = r153, 'MFL RS' = r150, reduced_labels = F)
# fault_identified_plot('MCS RS' = r146, 'WMCS2 RS' = r162, 'WMCS1 RS' = r161, reduced_labels = F)
# fault_identified_plot('MCS HC' = r147, 'MCS HCN' = r148, 'MCS HCS' = r149, 'MCS RS' = r146, reduced_labels = F)
# fault_identified_plot('MCS HC' = r147, 'MCS RS' = r146, reduced_labels = T)
# fault_identified_plot('MCS HC' = r147, 'MFL HC' = r151, reduced_labels = T)
# fault_identified_plot('MCS RS' = r146, 'MFL RS' = r150, reduced_labels = T)
# fault_identified_plot('MCS RS' = r146, '136' = r136, '137' = r137, '138' = r138, '139' = r139, '140' = r140, reduced_labels = F)
# fault_identified_plot('141' = r150, '142' = r142, reduced_labels = F)
# feedback_type_plot('MCS HC' = r147, 'MCS HCN' = r148, 'MCS HCS' = r149, 'MCS RS' = r146, 'WMCS RS' = r161, 'MFL HC' = r151, 'MFL HCN' = r152, 'MFL HCS' = r153, 'MFL RS' = r150)
# feedback_type_plot('MCS HC' = r147, 'MFL HC' = r151)
# feedback_type_plot('MCS HCN' = r148, 'MFL HCN' = r152)
# feedback_type_plot('MCS HCS' = r149, 'MFL HCS' = r153)
# feedback_type_plot('MCS RS' = r146, 'MFL RS' = r150)
#
# a <- r146 %>% filter(selfeval.deleted.lines == 0 & selfeval.changes.generate.n == 0)
# a <- inner_join(r160, r161, by = 'instance') %>% select(instance, mcss.negative.pre.x, strong.mcss.negative.pre.y)
#
# fault_identified_plot('MCS RS' = r146, 'MFL RS' = r150, reduced_labels = F) + facet_grid(vars(selfeval.deleted.lines), vars(selfeval.changes.generate.n, selfeval.changes.test.n), scales = 'free')
# fault_identified_plot('MCS RS' = r146 %>% filter(!is.na(fault.identified)), reduced_labels = F) + facet_grid(vars(selfeval.changes.generate.n, selfeval.changes.test.n), vars(selfeval.deleted.lines), scales = 'free')
# fault_identified_plot('MCS RS' = r146 %>% filter(!is.na(fault.identified)), reduced_labels = F) + facet_grid(vars(selfeval.changes.generate.n), vars(selfeval.deleted.lines), scales = 'free')
# fault_identified_plot('MCS RS' = r146 %>% filter(!is.na(fault.identified)), reduced_labels = F) + facet_grid(vars(selfeval.changes.test.n), vars(selfeval.deleted.lines), scales = 'free')
# fault_identified_plot('MCS RS' = r146 %>% filter(selfeval.deleted.lines == 0), reduced_labels = F) + facet_grid(vars(selfeval.changes.test.n), vars(selfeval.changes.generate.n), scales = 'free')
# fault_identified_plot('MCS RS' = r146 %>% filter(selfeval.deleted.lines == 1), reduced_labels = F) + facet_grid(vars(selfeval.changes.test.n), vars(selfeval.changes.generate.n), scales = 'free')
# fault_identified_plot('MCS RS' = r146 %>% filter(selfeval.deleted.lines == 2), reduced_labels = F) + facet_grid(vars(selfeval.changes.test.n), vars(selfeval.changes.generate.n), scales = 'free')
# fault_identified_plot('MCS RS' = r146 %>% filter(selfeval.changes.generate.n == 0 & selfeval.deleted.lines == 0), reduced_labels = F)
# fault_identified_plot('MCS RS' = r146 %>% filter(selfeval.changes.generate.n == 0 & selfeval.changes.test.n != 0), reduced_labels = F)
# fault_identified_plot('MCS RS' = r146, 'MFL RS' = r150, reduced_labels = F) + facet_wrap(~selfeval.deleted.lines, scales = 'free_y')
# fault_identified_plot('MCS RS' = r146, 'MFL RS' = r150, reduced_labels = F) + facet_wrap(~problem, scales = 'free_y')
# fault_identified_plot('MCS HC' = r147, 'MCS RS' = r146, reduced_labels = F) + facet_wrap(~problem, scales = 'free_y')
#
# feedback_type_plot('MCS RS' = r146, 'MFL RS' = r150) + facet_wrap(~problem)
#
# inner_join(r146, r150, by = 'instance') %>%
#   filter(feedback_type.x != 'Solution OK') %>%
#   ggplot(aes(x = ifelse(is.na(mcss.x), 600, mcs.time.x), y = ifelse(is.na(mcss.y), 600, mcs.time.y))) +
#   geom_point(alpha = 0.4) +
#   scale_x_log10(breaks = c(0, 0.1, 1, 5, 10, 60, 600)) +
#   scale_y_log10(breaks = c(0, 0.1, 1, 5, 10, 60, 600)) +
#   geom_abline() +
#   labs(x = 'MCS approach', y = 'MFL approach', title = 'Time to enumerate all MCSs/MFLs') +
#   my_theme()
#
# inner_join(r146, r154, by = 'instance') %>%
#   filter(feedback_type.x != 'Solution OK') %>%
#   ggplot(aes(x = ifelse(is.na(mcss.x), 600, mcs.time.x), y = ifelse(is.na(mcss.y), 600, mcs.time.y))) +
#   geom_point(alpha = 0.4) +
#   scale_x_log10(breaks = c(0, 0.1, 1, 5, 10, 60, 600)) +
#   scale_y_log10(breaks = c(0, 0.1, 1, 5, 10, 60, 600)) +
#   geom_abline() +
#   labs(x = 'MCS approach', y = 'SMCS approach', title = 'Time to enumerate all MCSs/SMCSs') +
#   my_theme()
#
# inner_join(r146, r150, by = 'instance') %>%
#   filter(feedback_type.x != 'Solution OK') %>%
#   ggplot(aes(x = real.x, y = real.y)) +
#   geom_point(alpha = 0.4) +
#   scale_x_log10(breaks = c(0, 0.1, 1, 5, 10, 60, 600)) +
#   scale_y_log10(breaks = c(0, 0.1, 1, 5, 10, 60, 600)) +
#   geom_abline() +
#   labs(x = 'Time MCS approach (s)', y = 'Time MFL approach (s)') +
#   my_theme()
#
# bind_rows('MCS RS' = r146, 'MFL RS' = r150, .id = 'run') %>%
#   mutate(time_new = ifelse(is.na(mcss), 600, mcs.time)) %>%
#   ggplot(aes(x = symbolic.atoms.first, y = mcs.time)) +
#   geom_point() +
#   facet_wrap(~run) +
#   my_theme()
#
#
# r147 %>%
#   filter(feedback_type != 'Solution OK' & is.na(mcss)) %>%
#   count()
# r129 %>%
#   filter(feedback_type != 'Solution OK' & is.na(mcss)) %>%
#   count()
#
# r128 %>%
#   filter(feedback_type != 'Solution OK') %>%
#   ggplot(aes(x = ifelse(is.na(mcss), 600, mcs.time))) + geom_histogram()
# r129 %>%
#   filter(feedback_type != 'Solution OK') %>%
#   ggplot(aes(x = ifelse(is.na(mcss), 600, mcs.time))) + geom_histogram()
#
#
# fault_identified_plot('84' = r084, '109' = r109, '110' = r110, '111' = r111, '112' = r112)
# feedback_type_plot('84' = r084, '109' = r109, '110' = r110, '111' = r111, '112' = r112)
#
# fault_identified_plot('84' = r084, '113' = r113, '114' = r114, '115' = r115, '116' = r116)
# feedback_type_plot('84' = r084, '113' = r113, '114' = r114, '115' = r115, '116' = r116)
#
# fault_identified_plot('84' = r084, '117' = r117, '118' = r118, '119' = r119, '120' = r120)
# feedback_type_plot('84' = r084, '117' = r117, '118' = r118, '119' = r119, '120' = r120)
#
#
# notion2 %>% ggplot(aes(x = lines.missing)) + geom_bar()
# notion2 %>% ggplot(aes(x = lines.wrong)) + geom_bar()
#
# notion2 %>% ggplot(aes(x = literals.missing)) + geom_bar()
# notion2 %>% ggplot(aes(x = literals.wrong)) + geom_bar()
# notion2 %>% ggplot(aes(x = abs(literals.missing) + literals.wrong)) + geom_bar()
#
# notion2 %>% ggplot(aes(x = grade.missing)) +
#   geom_bar() +
#   labs(x = 'Missing lines in sketch generator?')
# notion2 %>% ggplot(aes(x = grade.wrong)) +
#   geom_bar() +
#   labs(x = 'Correction position for wrong lines')
# notion2 %>% ggplot(aes(x = Depth)) +
#   geom_bar() +
#   labs(x = 'Depth')
# notion2 %>% ggplot(aes(x = paste0(grade.missing, grade.wrong))) +
#   geom_bar() +
#   labs(x = 'Correction position for wrong lines')
#
#
# feedback_type_plot(Default = r045, 'Block const expr' = r046, 'Allow unsafe vars' = r047,
#                    '1 extra var' = r048, 'No commutative' = r049, 'No distinct args' = r050,
#                    'No classical negation' = r051, 'Skip negative non relaxed' = r052, 'Skip negative relaxed' = r053,
#                    'Skip positive' = r055, 'Skip line pairings' = r054,
#                    angle = 45)
# feedback_type_plot(Default = r084, 'Reduced DSL' = r085, Simulated = r086, 'Simulated + Reduced DSL' = r087, 'No prune' = r089, 'No prune + reduced DSL' = r090,
#                    angle = 45)
# feedback_type_plot(Default = r091, 'Reduced DSL' = r092, Simulated = r093, 'Simulated + Reduced DSL' = r094, 'No prune' = r095, 'No prune + reduced DSL' = r096,
#                    angle = 45)
# feedback_type_plot(Default = r084, "Old Simulated" = r042, Simulated = r086, 'No prune' = r089,
#                    angle = 45)
# feedback_type_plot(Default = r091, Simulated = r093, 'No prune' = r095,
#                    angle = 45)
# times_inverse_cactus(Default = r084, Simulated = r086, 'No prune' = r089,
#                      angle = 45) + geom_vline(xintercept = 600, linetype = 2, color = '#575757')
# times_inverse_cactus(Default = r091, Simulated = r093, 'No prune' = r095,
#                      angle = 45)
# times_boxplot(Default = r069, 'Simulated' = r071, 'New Default' = r070, 'No pruning' = r072,
#               angle = 45)
#
#
# fault_identified_plot(Default = r068, 'Use SBFL' = r097, 'Disable missing\nstrict' = r075, 'Disable missing\nrelaxed' = r076,
#                       'Enable extra' = r078, 'Disable line pairings' = r077,
#                       angle = 0)
#
# fault_identified_plot(Default = r068, 'Default + SBFL' = r097,
#                       angle = 0)
# fault_identified_plot('Only missing\nstrict' = r079, 'Only missing\nrelaxed' = r080,
#                       'Only extra' = r082, 'Only pairings' = r081, 'Only baseline' = b001, Default = r068,
#                       angle = 0)
#
#
# fault_identified_grid(r045)
#
# feedback_fault_grid(r045)
# feedback_fault_grid(r055)
# feedback_fault_grid(r046)
#
# feedback_type_plot(r029)
# feedback_type_plot(r042)
# feedback_fault_grid(r029)
# feedback_fault_grid(r042)
# feedback_fault_grid(r042, full = T)
#
# inner_join(r045, r056, by = "instance") %>% ggplot(aes(x = real.x, y = real.y)) +
#   geom_point() +
#   scale_x_continuous(limits = c(0, 610)) +
#   scale_y_continuous(limits = c(0, 610)) +
#   scale_x_log10() +
#   scale_y_log10() +
#   geom_abline() +
#   labs(x = 'r030', y = 'r041')
#
#
# a <- inner_join(r084, r105, by = "instance") %>%
#   filter(feedback_type.x != feedback_type.y) %>%
#   select(instance, feedback_type.x, feedback_type.y, pairings.x, mcss.x, mcss.sorted.x, selfeval.lines.x)
#
# a <- inner_join(r099, r103, by = "instance") %>%
#   select(instance, real.x, real.y, feedback_type.x, feedback_type.y, pairings.x, mcss.x, mcss.sorted.x, selfeval.lines.x)
#
#
# a <- r033 %>%
#   filter(feedback_type != 'Solution OK' &
#            feedback_type != 'Parsing Error' &
#            feedback_type != 'Grounding Error') %>%
#   select(instance, feedback_type, fault.identified, fault.identified.manual, pairings, mcss, mcss.sorted)
#
# a <- r039 %>%
#   filter(mcss.negative.pre != mcss.both.pre |
#            mcss.both.pre != mcss |
#            feedback_type == 'Synthesis Success') %>%
#   select(instance, mcss.negative.pre, mcss.positive.pre, mcss.both.pre, mcss, selfeval.lines, fault.identified, feedback_type)
#
#
# a <- inner_join(r084, b005, by = 'instance') %>%
#   filter(feedback_type.x != 'Solution OK' &
#            feedback_type.x != 'Parsing Error' &
#            feedback_type.x != 'Grounding Error') %>%
#   select(instance, fault.identified.x, mcss, fault.identified.y, selected.lines, selfeval.lines.x)
#
# fault_identified_plot('Missing A. S.\n(relaxed)' = r164, 'Missing A. S.\n(strict)' = r165,
#                       'Extra Answer\nSet' = r166, 'Line\nMatching' = r167,
#                       # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
#                       # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
#                       'SBFL' = r170, 'MFL' = r168, 'FormHe\nDefault' = r163, 'FormHe\nDefault Random' = r172, 'FormHe\nDefault-strict+MFL' = r173,
#                       angle = 30) + facet_wrap(~case_when(startsWith(problem, 'mooshak') ~ 'A, B, C, D and E submitted',
#                                                           problem == 'DT_0' ~ 'Y and Z',
#                                                           problem == 'JFP_0' ~ 'Y and Z',
#                                                           TRUE ~ 'A, B, C, D and E synthetic'), scales = 'free_y', ncol = 1)
#
# times_inverse_cactus(FormHe = r163,
#                      # 'Fault Localizer\nOracle' = r086,
#                      'Pruning\nDisabled' = r169) +
#   geom_vline(xintercept = 600, linetype = 2, color = '#575757') +
#   facet_wrap(~problem == 'mooshak', scales = 'free_y')
#
# a <- r163_2 %>% filter(fault.identified == "No (no lines identified)" & selfeval.deleted.lines == 0)
#
# times_inverse_cactus(FormHe = r163,
#                      # 'Fault Localizer\nOracle' = r086,
#                      'Pruning\nDisabled' = r169, 'More Pruning\nDisabled' = r171 %>% select(-models.missing)) +
#   geom_vline(xintercept = 600, linetype = 2, color = '#575757')
#
#
# fault_identified_plot('Missing A. S.\n(relaxed)' = r164, 'Missing A. S.\n(strict)' = r165,
#                       'Extra Answer\nSet' = r166, 'Line\nMatching' = r167,
#                       # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
#                       # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
#                       'SBFL' = r170, 'MFL' = r168, 'FormHe\nDefault' = r163, 'No Sort' = r175,
#                       angle = 30)
#
# fault_identified_plot_new('Missing A. S.\n(relaxed)' = r177, 'Missing A. S.\n(strict)' = r178,
#                           'Extra Answer\nSet' = r179, 'Line\nMatching' = r180,
#                           # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
#                           # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
#                           'SBFL' = r182, 'MFL' = r181, 'FormHe\nDefault' = r176,
#                           angle = 30)
#
# fault_identified_plot_new('Missing A. S.\n(relaxed)' = r177, 'Missing A. S.\n(strict)' = r178,
#                           'Extra Answer\nSet' = r179, 'Line\nMatching' = r180,
#                           # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
#                           # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
#                           'SBFL' = r182, 'MFL' = r181, 'FormHe\nDefault' = r176,
#                           angle = 30, reduced_labels = F)






