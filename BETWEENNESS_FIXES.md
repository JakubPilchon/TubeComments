# Betweenness Centrality Code Fixes

## Original Issues

The original code had several syntax and logical errors:

```r
betweenness_df <- data.frame(
  Nazwa <- V(G)$name,
  Pośrednictwo <-  betwen
)

ggplot(
  betweenness_df %>% top_n(10, Pośrednictwo),
  aes(x=Pośrednictwo, y=Nazwa)
) +
  coord_flip() +
  geom_col() +
  theme_dracula
```

## Issues Fixed

### 1. Data Frame Column Assignment Syntax
**Problem**: Used `<-` instead of `=` for column assignment inside `data.frame()`

**Fix**: Changed to proper syntax:
```r
betweenness_df <- data.frame(
  Nazwa = V(G)$name,
  Pośrednictwo = betwen
)
```

### 2. Incomplete Variable Name
**Problem**: Variable `betwen` appears to be incomplete/typo

**Note**: Assumed this should reference betweenness values, likely from `betweenness(G)`. The variable name is kept as-is since the actual source variable is not provided in the problem statement.

### 3. Coordinate Flip Logic
**Problem**: Using `coord_flip()` with axis mapping that would result in horizontal bars with labels on the left

**Fix**: Swapped x and y aesthetics to work correctly with `coord_flip()`:
```r
aes(x = reorder(Nazwa, Pośrednictwo), y = Pośrednictwo)
```

### 4. Missing Ordering
**Problem**: Bars were not ordered by value, making the chart less readable

**Fix**: Added `reorder(Nazwa, Pośrednictwo)` to sort bars by betweenness value

### 5. Missing Proper Labels
**Enhancement**: Added explicit labels for better visualization:
```r
labs(
  x = "Nazwa",
  y = "Pośrednictwo",
  title = "Top 10 Nodes by Betweenness Centrality"
)
```

## Final Working Code

The corrected code is available in `betweenness_analysis.R`.

## Prerequisites

To run this code, you need:
- R with the following packages:
  - `igraph` (for graph object G and V() function)
  - `ggplot2` (for visualization)
  - `dplyr` (for top_n() function)
  - Custom `theme_dracula` defined in your environment
- Variables:
  - `G`: An igraph graph object with named vertices
  - `betwen`: A vector of betweenness centrality values

## Example Setup

```r
library(igraph)
library(ggplot2)
library(dplyr)

# Example: Create a sample graph and calculate betweenness
# G <- graph_from_data_frame(your_edge_data)
# betwen <- betweenness(G)
# 
# # Define theme_dracula or load it from your theme package
# source("theme_dracula.R")  # or however you define it
```
