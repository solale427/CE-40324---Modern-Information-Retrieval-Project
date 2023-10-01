# CE-40324---Modern-Information-Retrieval-Project

## Phase 1

An uncomplicated search engine for AI and Bioinformatics research papers, utilizing diverse ranking functions such as SMART and Okapi-25. The code also integrates Gamma and Variable Byte encoding to compress the index, enhancing overall performance. Furthermore, the search engine incorporates a spelling correction feature based on bigram indexing.

## Phase 2

In this phase, paper classification was accomplished using various methods, including Naive Bayes for content-based categorization, an MLP model trained with FastText word embeddings, and experiments with BERT fine-tuning, both with and without weight freezing, to enhance accuracy. Additionally, we implemented an Information Retrieval (IR) system enabling users to search for papers within specific categories. After retrieving search results, K-means clustering was applied to group similar papers, facilitating organized access to research papers in AI and Bioinformatics.

## Phase 3

Papers were gathered from Semantic Scholar by crawling their titles and abstracts, followed by the calculation of page rank scores to assess paper relevance and hub scores to evaluate author prominence. A rudimentary recommender system was then devised, employing both content-based and collaborative filtering techniques to provide users with personalized paper recommendations based on their preferences and interactions with similar content and users.
