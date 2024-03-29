---
layout: mathjax
title:  Networkd Data Analysis with Twitter Data
date:   2023-06-02
---
## A Sociological question
- How do users influence and are influenced by other users on Twitter network.

## Network analysis methodology
## Background of the centrality measurement
- As the definition of the betweenness centrality which quantifies the number of times a node acts as a bridge along the shortest path between two other nodes, we can identify the most frequent player play the role in the network with highest value of betweenness.

- The definition of the eigenvector centrality which is a measure of the influence of a node in a network, what is based on the concept that connections to high-scoring node contribute more to the score of the node rather than the low-scoring node contribution. The high eigenvector centrality node presents the connection of that node to other high centrality score nodes. 

- The degree centrality measures the number of direct connections of a node. This would have limitation that the player have the connection to other nodes.

# Data Sources


# Analysis Methodology
- The Network Analysis Methodology:
	- Perform Exploratory Data Analysis to have the basic understanding of the data
	- Compute and present some statistic information of the data
	- From the understanding of the data, choose the appropriate centrality measures to define the influence as the centralities score.
	- Compute the applicable centrality measurement
	- Interpret and conclude the result that does it answer the initial interest question of *How do users influence and are influenced by other users on Twitter network?*, the assumptions and the limitation of the method.

## Data preparation
- To load the Twitter network data from SNAP, use the following code in the appendix 

```python
# import the libraries
!pip install snap-stanford

import numpy as np
import pandas as pd
import networkx as nx
import snap

# Load data from twitter_combined.txt
G_snap = snap.LoadEdgeList(snap.PNGraph, "twitter_combined.txt", 0, 1) 
edges = [(e.GetSrcNId(), e.GetDstNId()) for e in G_snap.Edges()] 
G_nx = nx.convert.from_edgelist(edges)
G = nx.convert.from_edgelist(edges)

!apt-get install graphviz graphviz-dev
!pip install pygraphviz
nx.draw(G, pos=nx.drawing.nx_agraph.graphviz_layout(G), with_labels=True)
```

- The statistic of the Twitter data network
	- The number of node is 25944
	- The number of edges 384998
	- The histogram of the degree distribution
	- ![Twitter](/images/Twitter_network.png)
		- As the degree distribution is very frequent for the user with less than 200 friends, and less frequent for the user with more than 200 friends is seem to follow the power law distribution
	- ![Log_log](/images/Twitter_log_log.png)
		- In log - log scale
		- we also can perform the hypothesis testing for the exponential distribution versus the power law distribution test for the Twitter network data of degree distribution, the p-value = 1.8e-05< 0.05, it means that the data is unlikely to be generated by the null model . Therefore, the power law model is a better fit for the data than the exponential model. However, this does not mean that the power law model is the best or the only possible model for the data



# Basis to choose an appropriate centrality measurement
- Assume that the connections to high-scoring users contribute more to the score than the low-scoreing users, eigenvector centrality is a measure of how influential a user is in a network. In Twitter network data, eigenvector centrality can represent the influence of a user by taking into account not only how many followers they have, but also how influential their followers are. A user with high eigenvector centrality can potentially spread information to a large and influential audience on Twitter, they can be seen as opinion leaders or information sources in the network.

- Combine the eigenvector centrality and the degree of the users to find out the important role of the user in network as:
	- Users who have high eigenvector centrality and high out degree centrality: These are the most influential and popular users in the network.
	- Users who have high eigenvector centrality and high in degree centrality: These are the most influenced users in the network

# Compute and interprete the centrality measurement
- Take the top 10 users in the eigenvector centrality value, in degree, out degree users
	- `table to compare 10 user in 3 column of `
	- the 40981798 is the most influential and popular users in the network as present in both highest
	- The 43003845 is in degree highest and eigenvector centrality highest

```python
H = G.to_directed()
# Define the in and out degree centrality
##  Eigenvector centrality data frame
ECs = nx.eigenvector_centrality(H)
df_ECs = pd.DataFrame.from_dict(ECs, orient='index', columns=['eigenvector_centrality'])

## in out degree data frame

in_DCs = nx.in_degree_centrality(H)
out_DCs = nx.out_degree_centrality(H)

df_in_DCs = pd.DataFrame.from_dict(in_DCs,  orient='index', columns=['in_degree_centrality'])
df_out_DCs = pd.DataFrame.from_dict(out_DCs,  orient='index', columns=['out_degree_centrality'])

# find n largest user
n = 10
top_in_degree = df_in_DCs['in_degree_centrality'].nlargest(n)
top_out_degree = df_out_DCs['out_degree_centrality'].nlargest(n)
top_ECs = df_ECs['eigenvector_centrality'].nlargest(n)

diff_out = top_ECs.compare(top_out_degree, align_axis=1)
influential_user =  (diff_out.isna().all(axis=0)).sum()

diff_in = top_ECs.compare(top_in_degree, align_axis=1)
influenced_user =  (diff_in.isna().all(axis=0)).sum()

print(f"Most influential user: {influential_user} \n Most influenced user: {influenced_user}")
```

# Conclusion
- By following the stated analysis methodoly, the influencer and influenced user were identified, however, there are several limitations in this method:
	- The assumption of the most influential and influenced user are the people who have high eigenvector centrality and out or in-degree may has the bias on itself, just reflects the analysis way to find what is "influential" and "influenced". In the actual situation, the most "influential" and "influenced" people may be different in our human sense.
	- In the data there are crossed user in list of 10 highest value in eigenvector centrality and in-out degree, this is not guaranteed to give the same result in other data.

```python
H = G.to_directed()
# Define the in and out degree centrality
##  Eigenvector centrality data frame
ECs = nx.eigenvector_centrality(H)
df_ECs = pd.DataFrame.from_dict(ECs, orient='index', columns=['eigenvector_centrality'])
## in out degree data frame
in_DCs = nx.in_degree_centrality(H)
out_DCs = nx.out_degree_centrality(H)
df_in_DCs = pd.DataFrame.from_dict(in_DCs,  orient='index', columns=['in_degree_centrality'])
df_out_DCs = pd.DataFrame.from_dict(out_DCs,  orient='index', columns=['out_degree_centrality'])
# find n largest user
n = 10
df_in_DCs['in_degree_centrality'].nlargest(n)
df_out_DCs['out_degree_centrality'].nlargest(n)
df_ECs['eigenvector_centrality'].nlargest(n)
```

