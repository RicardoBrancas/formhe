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

notion <- read_csv('analysis/notion.csv') %>% mutate(instance = paste0('mooshak/', instance))
notion2 <- read_csv('analysis/notion2.csv') %>% mutate(instance = paste0('mooshak/', instance))

read_data <- function(name) {
  data <- read_csv(paste0('analysis/data/', name, '.csv')) %>%
    mutate(solved = status == 0 & !timeout & !memout,
           feedback_type = ifelse(timeout & feedback_type != 'Synthesis Success', 'Timeout', feedback_type),
           feedback_type = ifelse(memout & feedback_type != 'Synthesis Success', 'Memout', feedback_type),
           ram = ram / 1024,
           problem = str_match(instance, '(?:.*)/(.).*')[, 2]) %>%
    filter(problem != 'F')
  if ('fault.partial' %in% names(data)) {
    data <- data %>%
      mutate(fault.identified.full = ifelse(!is.na(fault.partial) & fault.partial == 'Yes', paste(fault.identified, '(Partial)'), fault.identified))
  }
  data %>% left_join(notion, by = 'instance')
}

b004 <- read_data('b004') # arith
b005 <- read_data('b005') # geom
b006 <- read_data('b006') # max

r084 <- read_data('084')  # default

fl001 <- read_data('fl001')

r105 <- read_data('105')
r106 <- read_data('106')
r107 <- read_data('107')
r108 <- read_data('108')
r109 <- read_data('109')
r110 <- read_data('110')
r111 <- read_data('111')
r112 <- read_data('112')
r113 <- read_data('113')
r114 <- read_data('114')
r115 <- read_data('115')
r116 <- read_data('116')
r117 <- read_data('117')
r118 <- read_data('118')
r119 <- read_data('119')
r120 <- read_data('120')
r121 <- read_data('121')

source('analysis/plots.R')


fault_identified_plot('84' = r084, '105' = r105, '106' = r106, '107' = r107, '108' = r108, '121' = r121)
feedback_type_plot('84' = r084, '105' = r105, '106' = r106, '107' = r107, '108' = r108, '121' = r121)

fault_identified_plot('84' = r084, '109' = r109, '110' = r110, '111' = r111, '112' = r112)
feedback_type_plot('84' = r084, '109' = r109, '110' = r110, '111' = r111, '112' = r112)

fault_identified_plot('84' = r084, '113' = r113, '114' = r114, '115' = r115, '116' = r116)
feedback_type_plot('84' = r084, '113' = r113, '114' = r114, '115' = r115, '116' = r116)

fault_identified_plot('84' = r084, '117' = r117, '118' = r118, '119' = r119, '120' = r120)
feedback_type_plot('84' = r084, '117' = r117, '118' = r118, '119' = r119, '120' = r120)


notion2 %>% ggplot(aes(x=lines.missing)) + geom_bar()
notion2 %>% ggplot(aes(x=lines.wrong)) + geom_bar()

notion2 %>% ggplot(aes(x=literals.missing)) + geom_bar()
notion2 %>% ggplot(aes(x=literals.wrong)) + geom_bar()
notion2 %>% ggplot(aes(x=abs(literals.missing) + literals.wrong)) + geom_bar()

notion2 %>% ggplot(aes(x=grade.missing)) + geom_bar() + labs(x='Missing lines in sketch generator?')
notion2 %>% ggplot(aes(x=grade.wrong)) + geom_bar() + labs(x='Correction position for wrong lines')
notion2 %>% ggplot(aes(x=Depth)) + geom_bar() + labs(x='Depth')
notion2 %>% ggplot(aes(x=paste0(grade.missing, grade.wrong))) + geom_bar() + labs(x='Correction position for wrong lines')





feedback_type_plot(Default = r045, 'Block const expr' = r046, 'Allow unsafe vars' = r047,
                   '1 extra var' = r048, 'No commutative' = r049, 'No distinct args' = r050,
                   'No classical negation' = r051, 'Skip negative non relaxed' = r052, 'Skip negative relaxed' = r053,
                   'Skip positive' = r055, 'Skip line pairings' = r054,
                   angle = 45)
feedback_type_plot(Default = r084, 'Reduced DSL' = r085, Simulated = r086, 'Simulated + Reduced DSL' = r087, 'No prune' = r089, 'No prune + reduced DSL' = r090,
                   angle = 45)
feedback_type_plot(Default = r091, 'Reduced DSL' = r092, Simulated = r093, 'Simulated + Reduced DSL' = r094, 'No prune' = r095, 'No prune + reduced DSL' = r096,
                   angle = 45)
feedback_type_plot(Default = r084, "Old Simulated" = r042, Simulated = r086, 'No prune' = r089,
                   angle = 45)
feedback_type_plot(Default = r091, Simulated = r093, 'No prune' = r095,
                   angle = 45)
times_inverse_cactus(Default = r084, Simulated = r086, 'No prune' = r089,
                     angle = 45) + geom_vline(xintercept = 600, linetype = 2, color = '#575757')
times_inverse_cactus(Default = r091, Simulated = r093, 'No prune' = r095,
                     angle = 45)
times_boxplot(Default = r069, 'Simulated' = r071, 'New Default' = r070, 'No pruning' = r072,
              angle = 45)


fault_identified_plot(Default = r068, 'Use SBFL' = r097, 'Disable missing\nstrict' = r075, 'Disable missing\nrelaxed' = r076,
                      'Enable extra' = r078, 'Disable line pairings' = r077,
                      angle = 0)

fault_identified_plot(Default = r068, 'Default + SBFL' = r097,
                      angle = 0)
fault_identified_plot('Only missing\nstrict' = r079, 'Only missing\nrelaxed' = r080,
                      'Only extra' = r082, 'Only pairings' = r081, 'Only baseline' = b001, Default = r068,
                      angle = 0)


fault_identified_grid(r045)

feedback_fault_grid(r045)
feedback_fault_grid(r055)
feedback_fault_grid(r046)

feedback_type_plot(r029)
feedback_type_plot(r042)
feedback_fault_grid(r029)
feedback_fault_grid(r042)
feedback_fault_grid(r042, full = T)

inner_join(r045, r056, by = "instance") %>% ggplot(aes(x = real.x, y = real.y)) +
  geom_point() +
  scale_x_continuous(limits = c(0, 610)) +
  scale_y_continuous(limits = c(0, 610)) +
  scale_x_log10() +
  scale_y_log10() +
  geom_abline() +
  labs(x = 'r030', y = 'r041')


a <- inner_join(r084, r105, by = "instance") %>%
  filter(feedback_type.x != feedback_type.y) %>%
  select(instance, feedback_type.x, feedback_type.y, pairings.x, mcss.x, mcss.sorted.x, selfeval.lines.x)

a <- inner_join(r099, r103, by = "instance") %>%
  select(instance, real.x, real.y, feedback_type.x, feedback_type.y, pairings.x, mcss.x, mcss.sorted.x, selfeval.lines.x)


a <- r033 %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  select(instance, feedback_type, fault.identified, fault.identified.manual, pairings, mcss, mcss.sorted)

a <- r039 %>%
  filter(mcss.negative.pre != mcss.both.pre |
           mcss.both.pre != mcss |
           feedback_type == 'Synthesis Success') %>%
  select(instance, mcss.negative.pre, mcss.positive.pre, mcss.both.pre, mcss, selfeval.lines, fault.identified, feedback_type)


a <- inner_join(r084, b005, by = 'instance') %>%
  filter(feedback_type.x != 'Solution OK' &
           feedback_type.x != 'Parsing Error' &
           feedback_type.x != 'Grounding Error') %>%
  select(instance, fault.identified.x, mcss, fault.identified.y, selected.lines, selfeval.lines.x)

#### PAPER PLOTS

r069 %>%
  group_by(problem, feedback_type) %>%
  summarise(n = n())

r080 %>% # relaxed
  group_by(fault.identified) %>%
  summarise(n = n())

r079 %>% # strict
  group_by(fault.identified) %>%
  summarise(n = n())

inner_join(r080, r079, by = 'instance') %>%
  filter(fault.identified.x != 'Yes' &
           fault.identified.x != 'Yes (no incorrect lines)' &
           fault.identified.x != 'Yes (not first MCS)' &
           (fault.identified.y == 'Yes' |
             fault.identified.y == 'Yes (no incorrect lines)' |
             fault.identified.y == 'Yes (not first MCS)')) %>%
  select(instance, fault.identified.x, fault.identified.y)

r082 %>% # extra
  group_by(fault.identified) %>%
  summarise(n = n())

inner_join(r080, r082, by = 'instance') %>%
  filter(fault.identified.x != 'Yes' &
           fault.identified.x != 'Yes (no incorrect lines)' &
           fault.identified.x != 'Yes (not first MCS)' &
           (fault.identified.y == 'Yes' |
             fault.identified.y == 'Yes (no incorrect lines)' |
             fault.identified.y == 'Yes (not first MCS)')) %>%
  select(instance, fault.identified.x, fault.identified.y)

r081 %>% # line matching
  group_by(fault.identified) %>%
  summarise(n = n())

r068 %>% # default
  group_by(fault.identified) %>%
  summarise(n = n())

b004 %>%
  filter(selfeval.lines != "set()") %>%
  count()

fl001 %>% summarise(a = mean(real))
fl001 %>%
  filter(feedback_type != 'Solution OK' &
           feedback_type != 'Parsing Error' &
           feedback_type != 'Grounding Error') %>%
  summarise(a = mean(real))

plot_pdf('feedback', .6 * linewidth, .6 * linewidth, feedback_type_plot(FormHe = r084,
                                                                        # 'Fault Localizer\nOracle' = r086,
                                                                        'Pruning\nDisabled' = r089,
                                                                        angle = 0))
plot_pdf('times', linewidth, .65 * linewidth, times_inverse_cactus(FormHe = r084,
                                                                   # 'Fault Localizer\nOracle' = r086,
                                                                   'Pruning\nDisabled' = r089) +
  geom_vline(xintercept = 600, linetype = 2, color = '#575757'))
# plot_pdf('fault_identified', linewidth, .8 * linewidth, fault_identified_plot(Default = r068, 'Disable missing\nrelaxed' = r076, 'Disable missing\nstrict' = r075,
#                                                                               'Enable extra' = r078, 'Disable line pairings' = r077,
#                                                                               angle = 30))
plot_pdf('fault_identified', linewidth, .8 * linewidth, fault_identified_plot('Missing A. S.\n(relaxed)' = r080, 'Missing A. S.\n(strict)' = r079,
                                                                              'Extra Answer\nSet' = r082, 'Line\nMatching' = r081,
                                                                              # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
                                                                              # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
                                                                              'SBFL' = b006,
                                                                              angle = 30))
plot_pdf('fault_identified_combined', linewidth, .7 * linewidth, fault_identified_plot('Missing A. S.\n(relaxed)' = r080, 'Missing A. S.\n(strict)' = r079,
                                                                                       'Line\nMatching' = r081,
                                                                                       # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
                                                                                       # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
                                                                                       'FormHe Fault\nLocalization' = r068,
                                                                                       angle = 0))
plot_pdf('fault_grid', linewidth, .5 * linewidth, feedback_fault_grid(r084))
plot_pdf('fault_grid_partial', linewidth, .7 * linewidth, fault_identified_grid(r084))



