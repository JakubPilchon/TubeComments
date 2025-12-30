# Betweenness Centrality Analysis and Visualization
# This script creates a visualization of the top 10 nodes by betweenness centrality

# Fixed code for betweenness centrality visualization
# Assuming:
# - G is a graph object (from igraph)
# - betwen should be a vector of betweenness values (likely from betweenness(G))
# - theme_dracula is defined elsewhere in the environment

# Create data frame with proper syntax
betweenness_df <- data.frame(
  Nazwa = V(G)$name,
  Pośrednictwo = betwen
)

# Create visualization with proper axis mapping and ordering
ggplot(
  betweenness_df %>% top_n(10, Pośrednictwo),
  aes(x = reorder(Nazwa, Pośrednictwo), y = Pośrednictwo)
) +
  geom_col() +
  coord_flip() +
  labs(
    x = "Nazwa",
    y = "Pośrednictwo",
    title = "Top 10 Nodes by Betweenness Centrality"
  ) +
  theme_dracula
