import json

import numpy as np
from scipy import linalg


def create_adjacency_matrix(nodes):
    node_ids = set()
    for node in nodes:
        node_ids.add(nodes[node]["paper_id"])
        node_ids.update(nodes[node]["reference_ids"])

    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

    matrix_size = len(node_ids)
    adjacency_matrix = [[0] * matrix_size for _ in range(matrix_size)]

    for node in nodes:
        n = len(nodes[node]["reference_ids"])
        source_index = id_to_index[nodes[node]["paper_id"]]
        for destination_id in nodes[node]["reference_ids"]:
            destination_index = id_to_index[destination_id]
            adjacency_matrix[source_index][destination_index] = 1 / n

    return adjacency_matrix, id_to_index


from typing import Dict, List, Set, Union, Any


def pagerank(graph: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Returns the personalized PageRank scores for the nodes in the graph, given the user's preferences.

    Parameters:
    graph (Dict[str, List[str]]): The graph represented as a dictionary of node IDs and their outgoing edges.

    Returns:
    Dict[str, float]: A dictionary of node IDs and their personalized PageRank scores.
    """
    adjacency_matrix, id_to_index = create_adjacency_matrix(graph)
    adjacency_matrix = np.array(adjacency_matrix)
    P = 0.9 * adjacency_matrix + 0.1 * (1 / len(graph))
    w, vl = linalg.eig(P)
    largest_eigenvector = vl[:, np.argmax(w)]
    index_to_id = {v: k for k, v in id_to_index.items()}
    result = {index_to_id[i]: v for i, v in enumerate(largest_eigenvector)}
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}

    return result


def important_articles(Professor: str) -> List[Dict[str, Union[Union[float, str], Any]]]:
    """
    Returns the most important articles in the field of given professor, based on the personalized PageRank scores.

    Parameters:
    Professor (str): Professor's name.

    Returns:
    List[str]: A list of article IDs representing the most important articles in the field of given professor.
    """
    with open(f'../new_crawled/crawled_paper_{Professor}.json') as f:
        papers = json.load(f)
        pr = pagerank(papers)
        return [{
            'id': k,
            'url': papers[k]['url'],
            'title': papers[k]['title']} for k in
            list(pr.keys())[:10]]
