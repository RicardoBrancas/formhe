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
  data %>% select(-contains("solution="))
  # data %>% left_join(notion, by = 'instance')
}