options(tikzDefaultEngine = 'pdftex',
        tikzLatexPackages = c(
          getOption("tikzLatexPackages"),
          "\\RequirePackage[T1]{fontenc}\n",
          "\\RequirePackage[tt=false, type1=true]{libertine}\n",
          "\\RequirePackage[varqu]{zi4}\n",
          "\\RequirePackage[libertine]{newtxmath}\n"
        ),
        tikzMetricsDictionary = './metrics_cache_acm',
        standAlone = T)
textwidth <- 7.00697
linewidth <- 3.3374


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

ibm_colors <- c("#fe6100", "#dc267f", "#785ef0", "#648fff", "#ffb000")

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

times_inverse_cactus <- function(..., all = F, angle = 0) {
  runs <- list(...)
  data <- bind_rows(runs, .id = 'run')
  if (!all) {
    data <- data %>%
      filter(feedback_type != 'Solution OK' &
               feedback_type != 'Parsing Error' &
               feedback_type != 'Grounding Error')
  }
  num_instances <- n_distinct(data$instance)
  data %>%
    mutate(feedback_type = case_when(feedback_type == 'Other' ~ 'Failed',
                                     TRUE ~ feedback_type),
           feedback_type = factor(feedback_type),
           feedback_type = fct_relevel(feedback_type, c('Synthesis Success', 'Timeout', 'Failed'))) %>%
    arrange(real) %>%
    group_by(run) %>%
    mutate(val = cumsum(feedback_type == 'Synthesis Success') / num_instances) %>%
    mutate(id = row_number()) %>%
    ungroup() %>%
    filter(feedback_type == 'Synthesis Success') %>%
    ggplot(aes(y = val, x = real, color = factor(run, levels = names(runs)), shape = factor(run, levels = names(runs)))) +
    geom_point() +
    geom_line() +
    scale_color_ibm() +
    annotation_logticks(sides = 'b') +
    scale_x_continuous(trans = log_trans(10), breaks = c(5, 10, 30, 60, 180, 600)) +
    scale_y_continuous(breaks = extended_breaks(n = 6), labels = label_percent(accuracy = 1, suffix = '\\%')) +
    labs(y = '\\% Instances Repaired', x = 'Time (s)') +
    # coord_polar("y", start = 0) +
    my_theme()
}

fault_identified_plot <- function(..., all = F, full = F, drop_zero_incorrect_lines=F, angle = 0) {
  runs <- list(...)
  data <- bind_rows(runs, .id = 'run')
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
                                        is.na(fault.identified) ~ 'No',
                                        TRUE ~ fault.identified),
           fault.identified = factor(fault.identified, levels=  c('Yes', 'Yes (not first MCS)', 'Subset', 'Superset', 'No')))
  if (drop_zero_incorrect_lines) {
    data <- data %>% filter(selfeval.lines != 'set()')
  }
  if (full) {
    tmp <- data %>%
      ggplot(aes(x = factor(run, levels = names(runs)), fill = fault.identified.full))
  } else {
    tmp <- data %>%
      ggplot(aes(x = factor(run, levels = names(runs)), fill = fault.identified))
  }
  tmp +
    geom_bar(position = position_stack(reverse = TRUE)) +
    scale_x_discrete(guide = guide_axis(angle = angle)) +
    scale_fill_ibm() +
    labs(y = '\\# Instances', x = NULL, fill='Fault Identified?') +
    # coord_polar("y", start = 0) +
    my_theme() +
    theme(legend.title = element_text(size = rel(1), colour = "#000000", face = "plain")) +
  guides(fill = guide_legend(title.position="top", title.hjust = .50, nrow = 2, byrow = TRUE))
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


feedback_fault_grid <- function(data, all = F, full = F) {
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
           fault.identified = factor(fault.identified),
           fault.identified = fct_relevel(fault.identified, c('Yes', 'Yes (not first MCS)', 'Subset', 'Superset', 'No')),
           feedback_type = case_when(feedback_type == 'Synthesis Success' ~ 'Repair Success',
                                     TRUE ~ feedback_type))
  if (full) {
    tmp <- data %>%
      ggplot(aes(x = fault.identified.full, y = feedback_type))
  } else {
    tmp <- data %>%
      ggplot(aes(x = fault.identified, y = feedback_type))
  }
  tmp +
    geom_bin2d() +
    stat_bin2d(geom = "text", aes(label = ..count..)) +
    scale_x_discrete(guide = guide_axis(angle = 30)) +
    scale_fill_gradient(low = "white", high = ibm_colors[[1]], limits = c(0, NA)) +
    my_theme() +
    labs(y = 'Outcome', x = 'Fault identitifed?') +
    theme(legend.position = 'none')
}
