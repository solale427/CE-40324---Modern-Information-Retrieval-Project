import math
import re, string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, PorterStemmer
import pandas as pd
from nltk.corpus import wordnet
import numpy as np

nltk.download('punkt')
nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

pos_map = {
    'JJ': wordnet.ADJ,
    'VB': wordnet.VERB,
    'RB': wordnet.ADV,
    'NN': wordnet.NOUN,
    'DT': wordnet.NOUN,
    'VBZ': wordnet.VERB,
    'IN': wordnet.NOUN
}


def pos_to_wordnet(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def pre_clean(text):
    bad_chars = ['“', '”']

    for t in bad_chars:
        text = text.replace(t, " ")

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = regex.sub(' ', text)
    text = text.lower()
    text = text.casefold()
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = " ".join(text.split())
    return word_tokenize(text)


def clean_data(text: str):
    """Preprocesses the text with tokenization, case folding, stemming and lemmatization, and punctuations

    Parameters
    ----------
    text : str
        The title or abstract of an article

    Returns
    -------
    list
        A list of tokens
    """
    tokens = pre_clean(text)
    lemmatizer = WordNetLemmatizer()
    tokens = list(map(lambda token: lemmatizer.lemmatize(token, pos=pos_to_wordnet(pos_tag([token])[0][1])), tokens))

    return tokens


def find_stop_words(all_text, num_token=30):
    """Detects stop-words

     Parameters
    ----------
    all_text : list of all tokens
        (result of clean_data(text) for all the text)

    Returns
    -------
    Return Value is optional but must print the stop words and number of their occurence
    """

    text = (' ').join(all_text)

    words = nltk.tokenize.word_tokenize(text)
    dist = nltk.FreqDist(words)
    stop_words = dist.most_common(num_token)
    return stop_words


def create_paper_id_map(df, stop_words_list):
    doc_dict = {}
    paper_id_map = {}
    max_id = 1
    for index, row in df.iterrows():
        title = row['titles']
        abstract = row['summaries']
        doc_dict[max_id] = {
            'title': [t for t in clean_data(title) if t not in stop_words_list],
            'abstract': [t for t in clean_data(abstract) if t not in stop_words_list] if abstract != -1 else None
        }
        paper_id_map[index] = max_id
        max_id += 1
    return doc_dict, paper_id_map

def add_doc_to_index(i, posting_lists, doc_dict):
    if doc_dict[i]['title']:
        for j, token in enumerate(doc_dict[i]['title']):
            if token in posting_lists.keys():
                if i in posting_lists[token]['title'].keys():
                    posting_lists[token]['title'][i].append(j + 1)
                else:
                    posting_lists[token]['title'][i] = [j + 1]
            else:
                posting_lists[token] = {
                    'title': {i: [j + 1]},
                    'abstract': {}
                }
    if doc_dict[i]['abstract']:
        for j, token in enumerate(doc_dict[i]['abstract']):
            if token in posting_lists.keys():
                if i in posting_lists[token]['abstract'].keys():
                    posting_lists[token]['abstract'][i].append(j + 1)
                else:
                    posting_lists[token]['abstract'][i] = [j + 1]

            else:
                posting_lists[token] = {
                    'title': {},
                    'abstract': {i: [j + 1]}
                }


def construct_positional_indexes(paper_id_map, doc_dict):
    posting_lists = {}
    for paper_id in paper_id_map.keys():
        add_doc_to_index(paper_id_map[paper_id], posting_lists, doc_dict)
    return posting_lists


def calculate_tf(token_list, term):
    return 1 + np.log(token_list.count(term))


def compute_tf(documents):
    tf_dict = {}
    for doc_id in documents.keys():
        tf_dict[doc_id] = {
            'title': {},
            'abstract': {}
        }
        if documents[doc_id]['title']:
            for term in documents[doc_id]['title']:
                tf_dict[doc_id]['title'][term] = calculate_tf(documents[doc_id]['title'], term)
        if documents[doc_id]['abstract']:
            for term in documents[doc_id]['abstract']:
                tf_dict[doc_id]['abstract'][term] = calculate_tf(documents[doc_id]['abstract'], term)

    return tf_dict


def min_span(k, lst):
    n = len(lst)
    freq = [0] * (k + 1)
    count = 0
    left = 0
    min_span = float('inf')

    for right in range(n):
        if freq[lst[right]] == 0:
            count += 1
        freq[lst[right]] += 1

        while count == k:
            min_span = min(min_span, right - left + 1)
            freq[lst[left]] -= 1
            if freq[lst[left]] == 0:
                count -= 1
            left += 1

    return min_span


def compute_proximity(query_tokens, doc_id, index):
    term_list = []
    for i, term in enumerate(query_tokens):
        if index[term]['title']:
            if doc_id in index[term]['title']:
                for j in index[term]['title'][doc_id]:
                    term_list.insert(j, i)
            else:
                return math.inf
    return min_span(len(query_tokens), term_list)


def compute_idf(index, paper_id_map):
    N = len(paper_id_map.keys())
    idf = {}
    for term in index.keys():
        idf[term] = {
            'title': np.log(N / len(index[term]['title'].keys())) if len(index[term]['title'].keys()) > 0 else None,
            'abstract': np.log(N / len(index[term]['abstract'].keys())) if len(
                index[term]['abstract'].keys()) > 0 else None
        }
    return idf


def calculate_cosine(q_tfidf, d_tfidf, query_tokens, method):
    dot_product = 0
    qry_mod = 0
    doc_mod = 0
    for token in query_tokens:
        qry_mod += q_tfidf[token] ** 2
        doc_mod += d_tfidf[token] ** 2
    qry_mod = np.sqrt(qry_mod)
    doc_mod = np.sqrt(doc_mod)
    if qry_mod * doc_mod == 0:
        return 0
    for token in query_tokens:
        if method == 'n':
            dot_product += q_tfidf[token] * d_tfidf[token]
        else:
            dot_product += (q_tfidf[token] / qry_mod) * (d_tfidf[token] / doc_mod)

    denominator = qry_mod * doc_mod
    cos_sim = dot_product / denominator if method == 'n' else dot_product
    return cos_sim


def compute_query_tf_idf(query_tokens):
    tf_idf = {}
    for term in query_tokens:
        tf = calculate_tf(query_tokens, term)
        tf_idf[term] = tf

    return tf_idf


def compute_similarity(title_query_tokens, abstract_query_token, weight, method, tf_doc, idf_dict):
    tf_idf = {}
    t_q_tf_idf = compute_query_tf_idf(title_query_tokens)
    a_q_tf_idf = compute_query_tf_idf(abstract_query_token)
    sim_list = []
    for doc_id in tf_doc.keys():
        tf_idf[doc_id] = {
            'title': {},
            'abstract': {}
        }
        for token in title_query_tokens:
            tfidf_title = tf_doc[doc_id]['title'][token] * idf_dict[token]['title'] if (
                    token in tf_doc[doc_id]['title'] and idf_dict[token]['title']) else 0
            tf_idf[doc_id]['title'][token] = tfidf_title
        for token in abstract_query_token:
            tfidf_abstract = tf_doc[doc_id]['abstract'][token] * idf_dict[token]['abstract'] if (
                    token in tf_doc[doc_id]['abstract'] and idf_dict[token]['abstract']) else 0
            tf_idf[doc_id]['abstract'][token] = tfidf_abstract
        cos_sim = weight * calculate_cosine(t_q_tf_idf, tf_idf[doc_id]['title'], title_query_tokens, method) + (
                1.0 - weight) * calculate_cosine(a_q_tf_idf, tf_idf[doc_id]['abstract'], abstract_query_token,
                                                 method)
        # proximity = compute_proximity(query_tokens=title_query_tokens, doc_id=doc_id, index=docs)
        # proximity_factor = 1 / (1 + 10 * proximity)
        sim_list.append((cos_sim, doc_id))
    sim_list.sort(key=lambda tup: tup[0], reverse=True)
    return [x[1] for x in sim_list]


def get_doc_df_id(paper_id_map, i):
    return list(paper_id_map.keys())[list(paper_id_map.values()).index(i)]


def print_docs(doc_ids, df, paper_id_map):
    paper_ids = []
    for i in doc_ids:
        paper_ids.append(get_doc_df_id(paper_id_map, i))
    print(df[df.index.isin(paper_ids)].titles.to_markdown())


def search(title_query: str, abstract_query: str, max_result_count: int = 10, method: str = 'ltn-lnn',
           weight: float = 0.5, stop_words_list=None, paper_id_map=None, tf_doc=None, idf_dict=None, doc_dict=None,
           okapi_idf=None, category='all', df=None):
    if stop_words_list is None:
        stop_words_list = []
    t_q_tokens = [x for x in clean_data(title_query) if x not in stop_words_list]
    a_q_tokens = [x for x in clean_data(abstract_query) if x not in stop_words_list]
    if method == 'okapi25':
        result = compute_okapi(t_q_tokens, a_q_tokens, weight, doc_dict=doc_dict, okapi_idf=okapi_idf, tf_doc=tf_doc)
    else:
        result = compute_similarity(t_q_tokens, a_q_tokens, weight, method[-1], tf_doc=tf_doc, idf_dict=idf_dict)

    if category != 'all':
        doc_ids = []
        for i in result:
            idx = get_doc_df_id(paper_id_map, i)
            cat = df.loc[idx]['terms']
            if cat == category:
                doc_ids.append(i)
                if len(doc_ids) == max_result_count:
                    break

    else:
        doc_ids = result[:max_result_count]
    print_docs(doc_ids, df, paper_id_map=paper_id_map)
    return doc_ids


def compute_idf_okapi(index, paper_id_map):
    N = len(paper_id_map.keys())
    idf = {}
    for term in index.keys():
        t_freq = len(index[term]['title'].keys())
        a_freq = len(index[term]['abstract'].keys())
        idf[term] = {
            'title': np.log(1 + (N - t_freq + 0.5) / (t_freq + 0.5)),
            'abstract': np.log(1 + (N - a_freq + 0.5) / (a_freq + 0.5)),
            't_freq': t_freq,
            'a_freq': a_freq
        }
    return idf


def compute_okapi_d_q(query_tokens, doc_id, avg_doc_len, k=1.5, b=0.75, okapi_idf=None, doc_dict=None, tf_doc=None,
                      weight=0.5):
    t_doc_len = len(doc_dict[doc_id]['title'])
    a_doc_len = len(doc_dict[doc_id].get('abstract')) if doc_dict[doc_id].get('abstract') else 0
    score = 0.0
    for term in query_tokens:
        if term in tf_doc[doc_id]['title'].keys():
            freq = okapi_idf[term]['t_freq']
            idf = okapi_idf[term]['title'] if okapi_idf[term]['title'] else 0
            numerator = idf * freq * (k + 1)
            denominator = freq + k * (1 - b + b * t_doc_len / avg_doc_len)
            score += (numerator / denominator) * weight
        if term in tf_doc[doc_id]['abstract'].keys():
            freq = okapi_idf[term]['a_freq']
            idf = okapi_idf[term]['abstract'] if okapi_idf[term]['abstract'] else 0
            numerator = idf * freq * (k + 1)
            denominator = freq + k * (1 - b + b * a_doc_len / avg_doc_len)
            score += (numerator / denominator) * (1 - weight)
    return score


def compute_okapi(title_query_tokens, abstract_quesry_token, weight, doc_dict, okapi_idf, tf_doc):
    results = []
    avg_title_len = sum([len(doc_dict[x]['title']) for x in doc_dict.keys()]) / len(doc_dict.keys())
    # avg_abstract_len = sum([len(doc_dict[x]['abstract']) for x in doc_dict.keys()]) / len(doc_dict.keys())
    for doc_id in doc_dict.keys():
        score = compute_okapi_d_q(title_query_tokens, doc_id, avg_title_len, doc_dict=doc_dict, okapi_idf=okapi_idf,
                                  tf_doc=tf_doc, weight=weight)
        results.append((score, doc_id))
    results.sort(key=lambda tup: tup[0], reverse=True)
    return [x[1] for x in results]
