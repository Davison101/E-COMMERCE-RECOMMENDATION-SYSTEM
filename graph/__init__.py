"""Graph construction module for LightGCN Recommender System.

This module converts user-item interaction data into a bipartite graph
structure suitable for LightGCN message passing.
"""

from graph.build_graph import GraphBuilder, build_graph

__all__ = [
    'GraphBuilder',
    'build_graph',
]