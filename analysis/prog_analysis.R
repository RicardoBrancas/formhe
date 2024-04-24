library(tidygraph)
library(ggraph)
library(igraph)
library(ggforce)

bigram_counts <- read_csv("analysis/bigrams.csv") %>%
  mutate(word1 = as.character(word1), word2 = as.character(word2)) %>%
  filter(word1 != "empty" & word2 != "empty")

bigram_graph <- bigram_counts %>% as_tbl_graph()

bigram_graph

ggraph(bigram_graph, layout = "treemap") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

ggraph(bigram_graph, layout = "tree") +
  geom_edge_diagonal() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

ggraph(bigram_graph, layout = "dendogram") +
  geom_edge_elbow() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

ggraph(bigram_graph, layout = "kk") +
  geom_edge_link(aes(edge_alpha = count), show.legend = FALSE,
                 arrow = grid::arrow(type = "closed", length = unit(.15, "inches")), end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()

bigram_counts %>%
  complete(word1, word2, fill = list(count = 0)) %>%
  ggplot(aes(x = word1, y = word2, fill = log1p(count))) + geom_tile()

############################################################################################################

relax_data <- read_csv("analysis/prog_analysis.csv")
relax_data2 <- read_csv("analysis/prog_analysis2.csv")

relax_data %>% filter(instance == "analysis/data/1009/E_0/39.log") %>% ggplot(aes(x = i, y = relax)) +
  geom_point(aes( color=type, group=1))

relax_data %>% ggplot(aes(x = i, y = relax)) +
  geom_point(aes( color=type, group=1)) +
  facet_wrap_paginate(~instance, 6, 6, scales="free")

relax_data2 %>% ggplot(aes(x = i, y = relax)) +
  geom_point(aes( color=type, group=1)) +
  facet_wrap_paginate(~instance, 6, 6, scales="free")