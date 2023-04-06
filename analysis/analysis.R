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

r068 <- read_data('068')  # default
r070 <- read_data('070')  # new default
r091 <- read_data('091')  # new default 2 combined depth
r071 <- read_data('071')  # simulated
r083 <- read_data('083')  # simulated 2
r086 <- read_data('086')  # simulated 3
r093 <- read_data('093')  # simulated 3 combined depth
r087 <- read_data('087')  # simulated 3 reduced DSL
r094 <- read_data('094')  # simulated 3 combined depth reduced DSL
r072 <- read_data('072')  # no pruning
r073 <- read_data('073')  # no classical negation
r074 <- read_data('074')  # only one extra var
r075 <- read_data('075')  # skip negative non relaxed
r076 <- read_data('076')  # skip negative relaxed
r077 <- read_data('077')  # skip line pairings
r078 <- read_data('078')  # use positive
r079 <- read_data('079')  # only strict
r080 <- read_data('080')  # only relaxed
r081 <- read_data('081')  # only pairings
r082 <- read_data('082')  # only extra

r029 <- read_data('036')
r042 <- read_data('042')

b004 <- read_data('b004') # arith
b005 <- read_data('b005') # geom
b006 <- read_data('b006') # max

# r084_ <- r084
r084 <- read_data('084')  # default
r085 <- read_data('085')  # reduced DSL
r086 <- read_data('086')  # simulated
r087 <- read_data('087')  # simulated + reduced DSL
r089 <- read_data('089')  # No prune
r090 <- read_data('090')  # No prune + reduced DSL

r091 <- read_data('091')  # default combined
r092 <- read_data('092')  # reduced DSL combined
r093 <- read_data('093')  # simulated combined
r094 <- read_data('094')  # simulated + reduced DSL combined
r095 <- read_data('095')  # No prune combined
r096 <- read_data('096')  # No prune + reduced DSL combined

r097 <- read_data('097')  # Use SBFL

fl001 <- read_data('fl001')

source('analysis/plots.R')

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


a <- inner_join(r042, r093, by = "instance") %>%
  filter(feedback_type.x != feedback_type.y) %>%
  select(instance, feedback_type.x, feedback_type.y, pairings.x, mcss.x, mcss.sorted.x, selfeval.lines.x)


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
fl001 %>% filter(feedback_type != 'Solution OK' & feedback_type != 'Parsing Error' & feedback_type != 'Grounding Error') %>% summarise(a = mean(real))

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
                                                                               angle = 30) )
plot_pdf('fault_identified_combined', linewidth, .7 * linewidth, fault_identified_plot('Missing A. S.\n(relaxed)' = r080, 'Missing A. S.\n(strict)' = r079,
                                                                                        'Line\nMatching' = r081,
                                                                                        # 'SBFL -- geom.\nmean' = b005 %>% filter(selfeval.lines != 'set()'),
                                                                                        # 'SBFL -- arith.\nmean' = b004 %>% filter(selfeval.lines != 'set()'),
                                                                                        'FormHe Fault\nLocalization' = r068,
                                                                                        angle = 0))
plot_pdf('fault_grid', linewidth, .5 * linewidth, feedback_fault_grid(r084))
plot_pdf('fault_grid_partial', linewidth, .7 * linewidth, fault_identified_grid(r084))



