library('devEMF')

options(tikzDefaultEngine = 'pdftex',
        tikzLatexPackages = c(
          getOption("tikzLatexPackages"),
          "\\RequirePackage[T1]{fontenc}\n",
          "\\RequirePackage{lmodern}\n"
        ),
        tikzMetricsDictionary = './metrics_cache_thesis',
        standAlone = T)
textwidth <- 4.8041
linewidth <- 4.8041


my_theme <- function(base_size = 9, base_family = "sans") {
  ltgray <- "#cccccc"
  dkgray <- "#111111"
  dkgray2 <- "#000000"
  theme_foundation(base_size = base_size, base_family = base_family) + theme(
    rect = element_rect(colour = "black", fill = "white"),
    line = element_line(colour = "black"),
    text = element_text(colour = dkgray),
    plot.title = element_text(face = "plain", size = rel(20 / 12), hjust = 0, colour = dkgray),
    plot.subtitle = element_text(hjust = 0, size = rel(1), face = "plain", colour = dkgray),
    plot.caption = element_text(hjust = 0, size = rel(1), face = "plain", colour = dkgray),
    panel.background = element_rect(fill = NA, colour = NA),
    panel.border = element_rect(fill = NA, colour = NA),
    strip.text = element_text(hjust = 0, size = rel(1), colour = dkgray2, face = "plain"),
    strip.background = element_rect(colour = NA, fill = NA),
    axis.title = element_text(face = "plain", colour = dkgray2, size = rel(1)),
    axis.text = element_text(face = "plain", colour = dkgray, size = rel(.9)),
    axis.line = element_line(colour = "black"), axis.line.y = element_blank(),
    axis.ticks = element_blank(), panel.grid.major = element_line(colour = ltgray),
    panel.grid.minor = element_blank(), legend.background = element_rect(colour = NA),
    legend.text = element_text(size = rel(1), colour = dkgray),
    # legend.title = element_text(size = rel(1), colour = dkgray2, face = "plain"),
    legend.key = element_rect_round(color = NA, radius = unit(0.4, "snpc")),
    # legend.position = "right",
    # legend.direction = "vertical",
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.background = element_rect(colour = NA))
}

plot_pdf <- function(filename, width, height, plot) {
  tikz(file = paste0('analysis/plots/', filename, ".tex"), width = width, height = height, standAlone = T)
  print(plot)
  dev.off()
  setwd('analysis/plots/')
  system(paste0('pdflatex ', filename, ".tex"))
  setwd('../..')
}

plot_emf <- function(filename, width, height, plot) {
  emf(file = paste0('analysis/plots/', filename, ".emf"), width = width, height = height)
  print(plot)
  dev.off()
}

ibm_colors <- c("#dc267f", "#fe6100", "#785ef0", "#648fff", "#ffb000")

ibm_palette <- function(n) {
  if (missing(n)) {
    n <- length(ibm_colors)
  }
  if (n <= length(ibm_colors)) {
    colors <- ibm_colors[1:n]
  } else {
    colors <- grDevices::colorRampPalette(ibm_colors)(n)
  }

  structure(colors, name = "ibm_colors", class = "palette")
}

scale_fill_ibm <- function(...) {
  discrete_scale(aesthetics = 'fill', scale_name = 'ibm_scale', palette = ibm_palette, na.value = '#000000')
}

scale_color_ibm <- function(...) {
  discrete_scale(aesthetics = 'color', scale_name = 'ibm_scale', palette = ibm_palette, na.value = '#000000')
}


feedback_type_plot <- function(..., all = F, angle = 0) {
  runs <- list(...)
  data <- bind_rows(runs, .id = 'run')
  if (!all) {
    data <- data %>%
      filter(feedback_type != 'Solution OK' &
               feedback_type != 'Parsing Error' &
               feedback_type != 'Grounding Error')
  }
  data %>%
    mutate(feedback_type = case_when(feedback_type == 'Other' ~ 'Failed',
                                     TRUE ~ feedback_type),
           feedback_type = factor(feedback_type),
           feedback_type = fct_relevel(feedback_type, c('Synthesis Success', 'Timeout', 'Failed'))) %>%
    filter(feedback_type == 'Synthesis Success') %>%
    ggplot(aes(x = factor(run, levels = names(runs)), fill = feedback_type)) +
    geom_bar(position = position_stack(reverse = TRUE)) +
    scale_x_discrete(guide = guide_axis(angle = angle)) +
    scale_y_continuous(n.breaks = 8, expand = expansion(mul = c(0.05, 0.05), add = c(0, 1))) +
    scale_fill_ibm() +
    labs(y = '# Instances Repaired', x = NULL) +
    # coord_polar("y", start = 0) +
    my_theme() +
    theme(legend.position = 'none')
}

times_boxplot <- function(..., all = F, angle = 0) {
  runs <- list(...)
  data <- bind_rows(runs, .id = 'run')
  if (!all) {
    data <- data %>%
      filter(feedback_type != 'Solution OK' &
               feedback_type != 'Parsing Error' &
               feedback_type != 'Grounding Error')
  }
  data %>%
    mutate(feedback_type = case_when(feedback_type == 'Other' ~ 'Failed',
                                     TRUE ~ feedback_type),
           feedback_type = factor(feedback_type),
           feedback_type = fct_relevel(feedback_type, c('Synthesis Success', 'Timeout', 'Failed'))) %>%
    filter(feedback_type == 'Synthesis Success') %>%
    ggplot(aes(x = factor(run, levels = names(runs)), y = real)) +
    geom_boxplot(varwidth = TRUE) +
    scale_x_discrete(guide = guide_axis(angle = angle)) +
    scale_fill_ibm() +
    labs(y = 'Time (s)', x = 'Run') +
    # coord_polar("y", start = 0) +
    my_theme()
}

vbs <- function(..., timelimit = 600) {
  runs <- list(...)
  data <- bind_rows(runs, .id = 'run')
  t <- data %>%
    group_by(instance) %>%
    summarise(real = min(ifelse(feedback_type == 'Synthesis Success', real, timelimit)),
              cpu = min(ifelse(feedback_type == 'Synthesis Success', cpu, timelimit)),
              feedback_type = case_when(any(feedback_type == "Synthesis Success") ~ "Synthesis Success",
                                        .default = 'Fail')
    )
  t %>% mutate(run = 'VBS')
}

pick <- function(condition) {
  function(d) d %>% filter_(condition)
}

times_inverse_cactus <- function(..., all = F, filter_f = NULL, use_vbs = F, every_other = 50, get_data = F, get_data_cutoff = NA) {
  runs <- list2(...)
  data <- bind_rows(runs, .id = 'run')
  if (use_vbs) {
    data <- data %>% bind_rows(vbs(...))
  }
  if (!all) {
    data <- data %>%
      filter(feedback_type != 'Solution OK' &
               feedback_type != 'Parsing Error' &
               feedback_type != 'Grounding Error')
  }
  if (!is.null(filter_f)) {
    data <- data %>% filter(filter_f(instance))
  }
  tmp <- data %>%
    mutate(feedback_type = case_when(feedback_type == 'Other' ~ 'Failed',
                                     TRUE ~ feedback_type),
           feedback_type = factor(feedback_type),
           feedback_type = fct_relevel(feedback_type, c('Synthesis Success', 'Timeout', 'Failed'))) %>%
    arrange(real) %>%
    group_by(run) %>%
    # mutate(val = cumsum(feedback_type == 'Synthesis Success')) %>%
    mutate(val = cumsum(feedback_type == 'Synthesis Success') / n_distinct(instance)) %>%
    mutate(id = row_number()) %>%
    ungroup()
  if (get_data) {
    tmp <- tmp %>%
      group_by(run) %>%
      mutate(instance_count = n_distinct(instance)) %>%
      ungroup()
    if (!is.na(get_data_cutoff)) {
      tmp <- tmp %>% filter(real <= get_data_cutoff)
    }
    tmp <- tmp %>%
      group_by(run, solved) %>%
      summarise(n = n() / first(instance_count))
    return(tmp)
  }
  tmp <- tmp %>% filter(feedback_type == 'Synthesis Success')
  tmp %>% ggplot(aes(y = val, x = real, color = factor(run, levels = c(names(runs), "VBS")), shape = factor(run, levels = c(names(runs), "VBS")))) +
    geom_point(data = function(d) { d %>% filter(id %% every_other == 0 & run == "Synthetic Instances" | run == "Real Instances") }) +
    geom_line() +
    scale_color_ibm() +
    annotation_logticks(sides = 'b') +
    scale_x_continuous(trans = log_trans(10), breaks = c(5, 10, 30, 60, 180, 600)) +
    # scale_y_continuous(breaks = extended_breaks(n = 6)) +
    scale_y_continuous(breaks = extended_breaks(n = 6), labels = label_percent(accuracy = 1, suffix = '\\%')) +
    labs(y = '\\% Instances Repaired', x = 'Time (s)') +
    # coord_polar("y", start = 0) +
    my_theme()
}

fault_identified_plot_new <- function(..., all = F, full = F, drop_zero_incorrect_lines = F, angle = 0, reduced_labels = T, get_data = F, pattern = F, filter_f = NULL, facet_vars = NULL, wrap_width = 15) {
  runs <- list2(...)
  data <- bind_rows(runs, .id = 'run')
  if (!all) {
    data <- data %>%
      filter(feedback_type != 'Solution OK' &
               feedback_type != 'Parsing Error' &
               feedback_type != 'Grounding Error')
  }
  if (!is.null(filter_f)) {
    data <- data %>% filter(filter_f(instance))
  }
  if (reduced_labels) {
    data <- data %>%
      mutate(fault.identified = case_when(fault.identified == 'Yes (no incorrect lines)' ~ 'All Faults Identified',
                                          fault.identified == 'Yes (first MCS)' ~ 'All Faults Identified',
                                          fault.identified == 'Yes (not first MCS)' ~ 'Wrong Identification',
                                          fault.identified == 'Wrong (wrong lines identified)' ~ 'Wrong Identification',
                                          fault.identified == 'Wrong (no incorrect lines)' ~ 'Wrong Identification',
                                          fault.identified == 'No (no lines identified)' ~ 'Fault Not Identified',
                                          is.na(fault.identified) ~ 'Fault Not Identified',
                                          fault.identified == 'Subset (first MCS)' ~ 'Some Faults Identified',
                                          fault.identified == 'Subset (not first MCS)' ~ 'Wrong Identification',
                                          fault.identified == 'Superset (first MCS)' ~ 'Superset Faults Identified',
                                          fault.identified == 'Superset (not first MCS)' ~ 'Wrong Identification',
                                          fault.identified == 'Not Disjoint (first MCS)' ~ 'Some Faults Identified',
                                          fault.identified == 'Not Disjoint (not first MCS)' ~ 'Wrong Identification',
                                          TRUE ~ fault.identified),
             fault.identified = factor(fault.identified, levels = c('All Faults Identified', 'Superset Faults Identified', 'Some Faults Identified', 'Fault Not Identified', 'Wrong Identification')))
  } else {
    data <- data %>% mutate(fault.identified = factor(fault.identified, levels = c('Yes (no incorrect lines)', 'Yes (first MCS)', 'Yes (not first MCS)',
                                                                                   'Superset (first MCS)', 'Subset (first MCS)', 'Not Disjoint (first MCS)', 'Superset (not first MCS)', 'Subset (not first MCS)',
                                                                                   'Not Disjoint (not first MCS)', 'No (no lines identified)', 'Wrong (no incorrect lines)', 'Wrong (wrong lines identified)')))
  }

  if (drop_zero_incorrect_lines) {
    data <- data %>% filter(selfeval.lines != 'set()')
  }
  if (get_data) {
    total_instances <- (data %>%
      mutate(run = factor(run, levels = names(runs))) %>%
      group_by(run) %>%
      summarise(n = n()) %>%
      ungroup() %>%
      summarise(n = max(n)))[[1]]
    print(total_instances)
    return(data %>%
             mutate(run = factor(run, levels = names(runs))) %>%
             group_by(run, fault.identified) %>%
             count() %>%
             spread(fault.identified, n) %>%
             ungroup() %>%
             mutate_at(vars(-run), function(x) { x / total_instances }) %>%
             mutate_at(vars(-run), function(x) { ifelse(!is.na(x), ifelse(x == max(x, na.rm = T), paste0('\\textbf{', scales::percent(x, suffix = "\\%", accuracy = 0.1), '}'), scales::percent(x, suffix = "\\%", accuracy = 0.1)), 'NA') }))
  }
  if (full) {
    tmp <- data %>%
      ggplot(aes(x = factor(run, levels = names(runs)), fill = fault.identified.full))
  } else {
    tmp <- data %>%
      ggplot(aes(x = factor(run, levels = names(runs)), fill = fault.identified))
  }
  if (pattern) {
    tmp <- tmp + geom_bar_pattern(aes(pattern = fault.identified, pattern_angle = fault.identified, pattern_fill = fault.identified), pattern_colour = NA, position = position_stack(reverse = TRUE))
  } else {
    tmp <- tmp + geom_bar(position = position_stack(reverse = TRUE))
  }
  if (!is.null(facet_vars)) {
    tmp <- tmp + facet_wrap({{facet_vars}}, scales="free_y", labeller = label_both)
  }
  tmp +
    scale_x_discrete(guide = guide_axis(angle = angle), labels = label_wrap(wrap_width)) +
    scale_fill_ibm() +
    labs(y = '\\# Instances', x = NULL, fill = 'Fault Identified?', pattern = 'Fault Identified?', pattern_angle = 'Fault Identified?', pattern_fill = 'Fault Identified?') +
    # coord_polar("y", start = 0) +
    my_theme() +
    theme(legend.title = element_text(size = rel(1), colour = "#000000", face = "plain"), legend.position = 'bottom') +
    guides(fill = guide_legend(), pattern = guide_legend())
}

fault_identified_grid <- function(data, all = F) {
  if (!all) {
    data <- data %>%
      filter(feedback_type != 'Solution OK' &
               feedback_type != 'Parsing Error' &
               feedback_type != 'Grounding Error')
  }
  data <- data %>%
    mutate(fault.identified = case_when(fault.identified == 'Yes (no incorrect lines)' ~ 'Yes',
                                        fault.identified == 'Yes (first MCS)' ~ 'Yes',
                                        startsWith(fault.identified, 'No') ~ 'No',
                                        TRUE ~ fault.identified),
           fault.partial = case_when(is.na(fault.partial) ~ 'No',
                                     TRUE ~ fault.partial),
           fault.identified = factor(fault.identified),
           fault.identified = fct_relevel(fault.identified, c('Yes', 'Yes (not first MCS)', 'Subset', 'Superset', 'No')))
  data %>%
    ggplot(aes(x = fault.identified, y = fault.partial)) +
    geom_bin2d() +
    stat_bin2d(geom = "text", aes(label = ..count..)) +
    scale_x_discrete(guide = guide_axis(angle = 30)) +
    scale_fill_gradient(low = "white", high = ibm_colors[[1]], limits = c(0, NA)) +
    my_theme() +
    labs(y = 'Could program be fixed without\n removing wrong lines?', x = 'Fault identitifed?') +
    theme(legend.position = 'none')
}

feedback_fault_grid_new <- function(data, all = F, full = F, reduced_labels = T) {
  if (!all) {
    data <- data %>%
      filter(feedback_type != 'Solution OK' &
               feedback_type != 'Parsing Error' &
               feedback_type != 'Grounding Error')
  }
  total_instances <- (data %>% tally())[[1]]
  if (reduced_labels) {
    data <- data %>%
      mutate(fault.identified = case_when(fault.identified == 'Yes (no incorrect lines)' ~ 'All Faults Identified',
                                          fault.identified == 'Yes (first MCS)' ~ 'All Faults Identified',
                                          fault.identified == 'Yes (not first MCS)' ~ 'Wrong Identification',
                                          fault.identified == 'Wrong (wrong lines identified)' ~ 'Wrong Identification',
                                          fault.identified == 'Wrong (no incorrect lines)' ~ 'Wrong Identification',
                                          fault.identified == 'No (no lines identified)' ~ 'Fault Not Identified',
                                          is.na(fault.identified) ~ 'Fault Not Identified',
                                          fault.identified == 'Subset (first MCS)' ~ 'Some Faults Identified',
                                          fault.identified == 'Subset (not first MCS)' ~ 'Wrong Identification',
                                          fault.identified == 'Superset (first MCS)' ~ 'Superset Faults Identified',
                                          fault.identified == 'Superset (not first MCS)' ~ 'Wrong Identification',
                                          fault.identified == 'Not Disjoint (first MCS)' ~ 'Some Faults Identified',
                                          fault.identified == 'Not Disjoint (not first MCS)' ~ 'Wrong Identification',
                                          TRUE ~ fault.identified),
             fault.identified = factor(fault.identified, levels = c('All Faults Identified', 'Superset Faults Identified', 'Some Faults Identified', 'Fault Not Identified', 'Wrong Identification')))
  } else {
    data <- data %>% mutate(fault.identified = factor(fault.identified, levels = c('Yes (no incorrect lines)', 'Yes (first MCS)', 'Yes (not first MCS)',
                                                                                   'Superset (first MCS)', 'Subset (first MCS)', 'Not Disjoint (first MCS)', 'Superset (not first MCS)', 'Subset (not first MCS)',
                                                                                   'Not Disjoint (not first MCS)', 'No (no lines identified)', 'Wrong (no incorrect lines)', 'Wrong (wrong lines identified)')))
  }
  if (full) {
    tmp <- data %>%
      ggplot(aes(x = fault.identified.full, y = feedback_type))
  } else {
    tmp <- data %>%
      ggplot(aes(x = fault.identified, y = feedback_type))
  }
  tmp +
    geom_bin2d() +
    stat_bin2d(geom = "text", aes(label = scales::percent(..count.. / total_instances))) +
    scale_x_discrete(guide = guide_axis(angle = 30)) +
    scale_fill_gradient(low = "white", high = ibm_colors[[1]], limits = c(0, NA)) +
    my_theme() +
    labs(y = 'Outcome', x = 'Fault identitifed?') +
    theme(legend.position = 'none')
}

detailed_times_boxplot <- function(..., percentage = F) {
  runs <- list(...)
  data <- bind_rows(runs, .id = 'run')
  data <- data %>%
    pivot_longer(cols = contains(".time"), names_to = "time.name", values_to = "time")
  if (!percentage) {
    tmp <- data %>% ggplot(aes(y = time, x = time.name))
  } else {
    tmp <- data %>% ggplot(aes(y = time / real, x = time.name))
  }
  tmp +
    geom_boxplot() +
    facet_wrap(~run) +
    scale_x_discrete(guide = guide_axis(angle = 45))
}
