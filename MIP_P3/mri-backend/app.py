from flask import Flask, request
from flask_cors import CORS, cross_origin
from page_rank import important_articles
from saerch import search, stop_words_list, all_tf_docs, all_idf_dicts, all_doc_dicts, all_okapi_idf, all_df, writers
from hits import best_authors, best_hubs

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/search', methods=['POST'])
@cross_origin()
def search_api():
    results = search(title_query=request.json.get('title_query', ''),
                     abstract_query=request.json.get('abstract_query', ''),
                     weight=float(request.json.get('weight', '1') or '1'),
                     stop_words_list=stop_words_list, tf_doc=all_tf_docs, idf_dict=all_idf_dicts, method='similarity',
                     doc_dict=all_doc_dicts, okapi_idf=all_okapi_idf, df=all_df, max_result_count=5, writers=writers)
    print(results.to_dict('records')[0])
    return {"results": results.to_dict('records')}


@app.route('/page-rank', methods=['GET'])
@cross_origin()
def page_rank():
    result = important_articles('Soleymani')
    print(result)
    return result


@app.route('/hits', methods=['GET'])
@cross_origin()
def hits():
    result = {'best_authors': best_authors, 'best_hubs': best_hubs}
    print(result)
    return result


if __name__ == '__main__':
    app.run()
