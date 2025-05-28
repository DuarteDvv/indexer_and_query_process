from argparse import ArgumentParser
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import os
import json
import numpy as np

STOP_WORDS = set(stopwords.words('english'))

def naive_daat_and(query_terms_index, query_tokens):
   
   iterators = []
   for token in query_tokens:

      if token in query_terms_index:

         iterators.append(iter(query_terms_index[token]))
      else:
         # if any query token is missing, intersection is empty
         return []

   # initialize current_docs: fetch the first posting from each iterator
   current_docs = [next(it, None) for it in iterators]

   # list to hold the [doc_id, [tf1, tf2, ...]] of documents that contain all query terms
   candidate_docs = []

   # loop until at least one postings list is exhausted
   while True:

      # if any iterator is done, break: no further full intersections are possible
      if any(doc is None for doc in current_docs):
         break

      # extract the current doc_ids
      doc_ids = [doc[0] for doc in current_docs]

      # if all doc_ids are equal, we have a match
      if all(doc_id == doc_ids[0] for doc_id in doc_ids):

         # gather term frequencies in order
         tfs = [doc[1] for doc in current_docs]

         # Add [doc_id, tfs] to result
         candidate_docs.append([doc_ids[0], tfs])

         # advance every iterator to the next posting
         for idx, it in enumerate(iterators):
               current_docs[idx] = next(it, None)

      else:

         # not all equal: advance the iterator(s) with the smallest doc_id
         min_id = min(doc_ids)
         for idx, doc in enumerate(current_docs):
               # if this doc_id is the smallest, advance its iterator to catch up
               if doc[0] == min_id:
                  current_docs[idx] = next(iterators[idx], None)

   return candidate_docs

def look_for_document(candidate_docs, index_path):
    
   document_metadata = {}

   documento_index_name = 'document_index.jsonl'
   documento_index_path = os.path.join(index_path, documento_index_name)

   ids = set(doc[0] for doc in candidate_docs)

   n_documents_corpus = 0

   # read the index from the file

   with open(documento_index_path, 'r', encoding='utf-8') as f:
      for line in f:

         n_documents_corpus += 1

         entry = json.loads(line.strip())
         doc_id = entry['id']

         if doc_id in ids:

               document_metadata[doc_id] = {
                  'title': entry['title'],
                  'length': entry['length']                
               }

               ids.remove(doc_id)


   return document_metadata, n_documents_corpus
   

def rank_documents(candidate_docs, ranker, index_path, query_terms_index, query_tokens):
   top_ranking = []

   print("looking for documents in the document index...")
   document_metadata, n_documents_corpus = look_for_document(candidate_docs, index_path)
   print('found')

   # calculate average document length
   total_length = sum(meta['length'] for meta in document_metadata.values())
   avgDL = total_length / n_documents_corpus if n_documents_corpus > 0 else 0

   # BM25 parameters
   k1 = 1.5
   b = 0.75

   for doc in candidate_docs:
      doc_id = doc[0]
      tfs = [unary_decode(tf) for tf in doc[1]]  # decode the term frequencies from unary code

      doc_len = document_metadata[doc_id]['length']
      # number of docs containing each term
      dfs = [len(query_terms_index[term]) for term in query_tokens]

      score = 0.0
      if ranker == "TFIDF":
         query_vec = [1] * len(query_tokens)
         tf_norm = [tf / doc_len for tf in tfs]
         idfs = [np.log((n_documents_corpus + 1) / df) for df in dfs]
         doc_vec = [tf_n * idf for tf_n, idf in zip(tf_norm, idfs)]
         denom = (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
         score = float(np.dot(query_vec, doc_vec) / denom) if denom != 0 else 0.0

      elif ranker == "BM25":
         for tf, df in zip(tfs, dfs):
               # IDF with added 0.5 smoothing
               idf = np.log((n_documents_corpus - df + 0.5) / (df + 0.5) + 1)
               # term frequency component
               tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgDL))) if avgDL > 0 else 0
               score += idf * tf_component
         score = float(score)

      result = {'ID': doc_id, 'Score': score}
      top_ranking.append(result)

   # sort by descending score and take top 10
   top_ranking.sort(key=lambda x: x['Score'], reverse=True)
   top_ranking = top_ranking[:10]
   return top_ranking



def unary_decode(code: str) -> int:
   if '1' not in code:
      raise ValueError("Invalid unary code: missing terminator '1'.")
   
   count = 0
   for char in code:
      if char == '0':
         count += 1
      elif char == '1':
         break
      else:
         raise ValueError(f"Invalid character '{char}' in unary code.")
   
   return count


def preprocess(text):
   """
   Preprocess the text by tokenizing, removing stopwords, and stemming.
   """
   
   # tokenize
   tokens = word_tokenize(text.lower())
   
   # transform - and _ into space
   tokens = [re.sub(r"[-_]", " ", tok) for tok in tokens]
   
   # keep only alphanumeric tokens
   tokens = [tok for tok in tokens if re.match(r"^[a-z0-9]+$", tok)]
   
   # remove stopwords
   tokens = [word for word in tokens if word not in STOP_WORDS]
   
   # Stem
   stemmer = PorterStemmer()
   tokens = [stemmer.stem(word) for word in tokens]

   # remove single character tokens
   tokens = [word for word in tokens if len(word) > 1]
   
   return tokens

def read_args ():
   """
   Reads the command line arguments.
   """

   parser = ArgumentParser(description="Processor for the index")
   parser.add_argument("-i", "--index", type=str, required=True, help="Path to inverted index.")
   parser.add_argument("-q", "--queries", type=str, required=True, help="the path to a file with the list of queries to process.")
   parser.add_argument("-r", "--ranker", type=str, required=True, help="a string informing the ranking function (either “TFIDF” or “BM25”) to be used to score documents for each query.")

   args = parser.parse_args()

   print(f"Processing queries from {args.queries} using the {args.ranker} ranker with index at {args.index}")

   return args.queries, args.index, args.ranker