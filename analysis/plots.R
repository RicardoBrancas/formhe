my_theme <- theme_bw() +
  theme(legend.position = "bottom", legend.title = element_blank(), text = element_text(size = 9), strip.text.x = element_text(size = 8))

instance_info_dist <- function(expr, y_expr, two_d = F, continuous = T, log_x = F, facet = F, group_status = F, density_adjust = 1, x_lab = NA, y_lab = NA, data = instance_info) {
  data <- data %>%
    mutate(benchmark = recode_factor(benchmark, "scythe/recent-posts" = "\\texttt{recent-posts}",
                                     "scythe/top-rated-posts" = "\\texttt{top-rated-posts}",
                                     textbook = "\\texttt{textbook}",
                                     spider = "\\texttt{spider}",
                                     kaggle = "\\texttt{kaggle}"))

  if (!facet & !group_status) {
    tmp <- data %>% ggplot(aes(x = { { expr } }, color = benchmark, fill = benchmark))
  } else if (!facet & group_status) {
    tmp <- data %>% ggplot(aes(x = { { expr } }, color = status, fill = status))
  } else {
    tmp <- data %>% ggplot(aes(x = { { expr } }))
  }

  tmp +
  { if (continuous & !two_d)  geom_density(alpha = .1, adjust = density_adjust) } +
  { if (continuous & two_d)  geom_density_2d_filled(aes(y = { { y_expr } })) } +
  { if (!continuous)  geom_bar() } +
  { if (log_x)  scale_x_log10() } +
  { if (!facet)  scale_color_brewer(palette = "Dark2") } +
  { if (!facet)  scale_fill_brewer(palette = "Dark2") } +
    # { if (log_x)  annotation_logticks(sides="b") } +
  { if (facet)  facet_wrap(~benchmark, scales = "free_y") } +
  { if (!is.na(x_lab)) labs(x = x_lab) } +
  { if (!is.na(y_lab)) labs(y = y_lab) } +
    my_theme +
  { if (continuous & !two_d)  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) }
}