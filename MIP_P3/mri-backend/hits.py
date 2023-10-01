import json

import numpy as np
import networkx as nx
import heapq
from operator import itemgetter

crawled_papers = {}
writers = ['Rabiee', 'Rohban', 'Sharifi', 'Soleymani', 'Kasaei']

for writer in writers:
    with open(f'../new_crawled/crawled_paper_{writer}.json', "r") as f:
        crawled_papers = dict(crawled_papers, **json.load(f))


def create_authors_adjacency_matrix(nodes):
    authors = set()
    for node in nodes:
        authors.update(nodes[node]['authors'])

    name_to_index = {author: index for index, author in enumerate(authors)}

    matrix_size = len(authors)
    adjacency_matrix = [[0] * matrix_size for _ in range(matrix_size)]

    for node in nodes:
        for author in nodes[node]["authors"]:
            source_index = name_to_index[author]
            for destination_id in nodes[node]["reference_ids"]:
                if nodes.get(destination_id):
                    for ref_author in nodes[destination_id]['authors']:
                        destination_index = name_to_index[ref_author]
                        adjacency_matrix[source_index][destination_index] = 1

    return adjacency_matrix, name_to_index


def hit_algorithm(papers, n):
    """
        Implementing the HITS algorithm to score authors based on their papers and co-authors.

        Parameters
        ---------------------------------------------------------------------------------------------------
        papers: A list of paper dictionaries with the following keys:
                "id": A unique ID for the paper
                "title": The title of the paper
                "abstract": The abstract of the paper
                "date": The year in which the paper was published
                "authors": A list of the names of the authors of the paper
                "related_topics": A list of IDs for related topics (optional)
                "citation_count": The number of times the paper has been cited (optional)
                "reference_count": The number of references in the paper (optional)
                "references": A list of IDs for papers that are cited in the paper (optional)
        n: An integer representing the number of top authors to return.

        Returns
        ---------------------------------------------------------------------------------------------------
        List
        list of the top n authors based on their hub scores.
    """
    index_to_name = {v: k for k, v in name_to_index.items()}
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    hubs, authorities = nx.hits(gr, max_iter=5, normalized=True)

    topitems = heapq.nlargest(n, hubs.items(), key=itemgetter(1))  # Use .iteritems() on Py2
    top_hubs = dict(topitems)

    topitems = heapq.nlargest(n, authorities.items(), key=itemgetter(1))  # Use .iteritems() on Py2
    top_auth = dict(topitems)
    return [index_to_name[t] for t in top_auth], [index_to_name[t] for t in top_hubs]


adjacency_matrix, name_to_index = create_authors_adjacency_matrix(crawled_papers)
adjacency_matrix = np.array(adjacency_matrix)

# call the hit_algorithm function
top_authors = hit_algorithm(crawled_papers, 8)

# print the top authors
best_authors, best_hubs = top_authors
