read_data <- function(name, load_log = F) {
  data <- read_csv(paste0('analysis/data/', name, '.csv')) %>%
    mutate(solved = status == 0 & !timeout & !memout,
           feedback_type = ifelse(timeout & feedback_type != 'Synthesis Success', 'Timeout', feedback_type),
           feedback_type = ifelse(memout & feedback_type != 'Synthesis Success', 'Memout', feedback_type),
           ram = ram / 1024,
           # problem = str_match(instance, '(?:.*)/(.).*')[, 2]) %>%
           problem_tmp1 = str_match(instance, '(.*/.)_.*|(.*)/.*')[, 2],
           problem_tmp2 = str_match(instance, '(.*/.)_.*|(.*)/.*')[, 3],
           problem = coalesce(problem_tmp1, problem_tmp2),
           problem = sapply(str_split(problem_tmp1, "/"), tail, 1)) %>%
    filter(problem != 'mooshak/F' & problem != 'mooshak/01/F')
  # filter(str_starts(problem, 'mooshak'))
  if ('fault.partial' %in% names(data)) {
    data <- data %>%
      mutate(fault.identified.full = ifelse(!is.na(fault.partial) & fault.partial == 'Yes', paste(fault.identified, '(Partial)'), fault.identified))
  }
  # data %>% mutate(selfeval.deleted.lines = as.character(selfeval.deleted.lines),
  #                 selfeval.changes.generate.n = as.character(selfeval.changes.generate.n),
  #                 selfeval.changes.generate.n.unique = as.character(selfeval.changes.generate.n.unique),
  #                 selfeval.changes.test.n = as.character(selfeval.changes.test.n),
  #                 selfeval.changes.test.n.unique = as.character(selfeval.changes.test.n.unique))
  data <- data %>%
    mutate(synthetic = !str_starts(instance, "mooshak")) %>%
    mutate(public = synthetic | str_starts(instance, "mooshak/")) %>%
    filter(public) %>%
    select(-contains("solution="))

  if (load_log) {
    data <- data %>%
      mutate(log = paste0('analysis/data/', name, '/', instance, '.log'),
             log_content = map(log, function(x) { ifelse(file.exists(x), read_file(x), NA) }),
             reference_cost = parse_double(unlist(map(log_content, function(x) { str_match(x, 'Selecting reference .* with cost (.*)')[, 2] })))) %>%
      select(-log, -log_content)
  }

  data
  # data %>% left_join(notion, by = 'instance')
}