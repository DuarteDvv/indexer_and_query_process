import json
import util_processor as util
from queue import Queue
from threading import Thread
import os
import numpy as np


# -i <INDEX>: Path to inverted jsonl index.
# -q <QUERIES>: path to the txt with the queries.
# -r <RANKER>: a string informing the ranking function (either “TFIDF” or “BM25”) to be used to score documents for each query.

query_queue = Queue()
n_threads = 5  # number of worker threads


def process_query(query_terms_index, ranker, index_path):

    while not query_queue.empty():

        query_tokens, original_query = query_queue.get()

        candidate_docs = util.naive_daat_and(query_terms_index, query_tokens)

        if candidate_docs:

            top_ranking = util.rank_documents(candidate_docs, ranker, index_path, query_terms_index, query_tokens)

        else:

            print(f"No documents found for query: {original_query}")

            top_ranking = []
           

        result = {
            'Query': original_query,
            'Results': top_ranking
        }

        print(result)
        query_queue.task_done()

    return



def main():

    queries_path, index_path, ranker = util.read_args()

    # read the queries from the file
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    queries_tokens = []
    tokens_set = set()

    for query in queries:
        
        tokens = util.preprocess(query)
        tokens_set.update(tokens)
        queries_tokens.append(tokens)

    query_terms_index = {}

    n_tokens = len(tokens_set)

    print("Looking for tokens in the index...")

    inversed_index_path = os.path.join(index_path, 'complete_inversed_index.jsonl')

    # read the inverted index from the file
    with open(inversed_index_path, 'r', encoding='utf-8') as f:
        for line in f:

            if not tokens_set:
                break

            entry = json.loads(line.strip())
            term = entry['term']

            if term in tokens_set:
                query_terms_index[term] = entry['doc_list']
                tokens_set.remove(term)

    print(f"Found {len(query_terms_index)}/{n_tokens} tokens in the index.")


    for i, tokens in enumerate(queries_tokens):

        if not tokens:
            continue

        query_queue.put((tokens, queries[i]))

    # create worker threads
    threads = []
    for _ in range(n_threads):
        thread = Thread(target=process_query, args=(query_terms_index, ranker, index_path))
        thread.start()
        threads.append(thread)

    # wait for all queries to be processed
    query_queue.join()

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    return


if __name__ == "__main__":
    main()