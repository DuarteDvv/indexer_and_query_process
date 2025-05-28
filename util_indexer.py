
from argparse import ArgumentParser
import os
import psutil
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import json
import heapq
from collections import defaultdict
import threading

STOP_WORDS = set(stopwords.words('english'))


def save_partial_document_index(document_index, index_path, idx_doc_index):
   """
   Save the partial document index to disk.
   """
   os.makedirs(index_path, exist_ok=True)
   filename = os.path.join(index_path, f"partial_document_index_{idx_doc_index}.jsonl")
   
   with open(filename, 'w', encoding='utf-8') as f:
      for doc_id, doc_info in sorted(document_index.items()):
         obj = {"id": doc_id, "title": doc_info['title'], "length": doc_info['length']}
         f.write(json.dumps(obj, ensure_ascii=False) + "\n")
   
   print(f"[Writer] Saved partial document index to {filename} with {len(document_index)} documents.")

def append_partial_document_indexes(index_path):
    
   all_files = os.listdir(index_path)
   
   partial_files = [
      f for f in all_files
      if f.startswith("partial_document_index_") and f.endswith(".jsonl")
   ]

   partial_files.sort()

   out_fname = os.path.join(index_path, "document_index.jsonl")
   
   with open(out_fname, 'w', encoding='utf-8') as out_f:
      for fname in partial_files:
         full_path = os.path.join(index_path, fname)
         with open(full_path, 'r', encoding='utf-8') as in_f:
               for line in in_f:
                  out_f.write(line)

   print(f"[Writer] Merged {len(partial_files)} partial document index files into {out_fname}.")

   # remove partial files
   for fname in partial_files:
      os.remove(os.path.join(index_path, fname))
                  


def save_partial_index_jsonl(index, path, idx_val, idx_lock):
   """
   Atomically save a partial inverted index to disk.
   """
   os.makedirs(path, exist_ok=True)
   with idx_lock:
      filename = os.path.join(path, f"partial_inversed_{idx_val.value}.jsonl")
      idx_val.value += 1

   with open(filename, 'w', encoding='utf-8') as f:
      for term, doc_list in sorted(index.items()): # sorting by term

         obj = {"term": term, "doc_list": doc_list}
         f.write(json.dumps(obj, ensure_ascii=False) + "\n")

   print(f"[Writer] Saved partial index to {filename} with {len(index)} terms.")
   
         

def unary_encode(f: int) -> str:
   return "0"*f + "1"


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

   parser = ArgumentParser(description="Indexer for the corpus.")
   parser.add_argument("-m", "--memory", type=int, required=True, help="Memory available to the indexer in megabytes.")
   parser.add_argument("-c", "--corpus", type=str, required=True, help="Path to the corpus file to be indexed.")
   parser.add_argument("-i", "--index", type=str, required=True, help="Path to the directory where indexes should be written.")

   args = parser.parse_args()
   
   print(f"Indexing corpus at {args.corpus} with {args.memory}MB of memory and saving to {args.index}.")

   return args.memory, args.corpus, args.index

def memory_used():
   """
   Returns the memory used by the process in megabytes.
   """

   process = psutil.Process(os.getpid())
   mem = process.memory_info().rss / (1024 * 1024)  # convert bytes to megabytes
   
   return mem


def merge_group(file_paths, output_path, final=False):
   """
   Merge a list of partial index files into a single JSONL at output_path.
   """

   if final:
      number_of_terms = 0
      sum_list_size = 0
   
   # open all partial index files for reading
   read_file_pointers = [open(path, 'r', encoding='utf-8') for path in file_paths]
   iterators = [iter(fp) for fp in read_file_pointers]

   # prepare writing
   with open(output_path, 'w', encoding='utf-8') as write_fp:
      min_heap = []  # (term, idx, doc_list)
      # initialize heap
      for i, it in enumerate(iterators):
         try:
            line = next(it)
            content = json.loads(line)
            heapq.heappush(min_heap, (content['term'], i, content['doc_list']))
         except StopIteration:
            pass

      # merge loop
      while min_heap:
         term, idx, doc_list = heapq.heappop(min_heap)
         merged_doc_list = doc_list.copy()

         # advance this iterator
         try:
            line = next(iterators[idx])
            content = json.loads(line)
            heapq.heappush(min_heap, (content['term'], idx, content['doc_list']))
         except StopIteration:
            pass

         # merge entries with the same term
         while min_heap and min_heap[0][0] == term:
            _, other_idx, other_list = heapq.heappop(min_heap)
            merged_doc_list.extend(other_list)
            try:
               line = next(iterators[other_idx])
               content = json.loads(line)
               heapq.heappush(min_heap, (content['term'], other_idx, content['doc_list']))
            except StopIteration:
               pass

         # aggregate frequencies
         agg = defaultdict(int)
         for doc_id, freq in merged_doc_list:
               agg[doc_id] += freq

         # build final or intermediate list
         if final:
               final_list = [[doc_id, unary_encode(agg[doc_id])] for doc_id in sorted(agg)]
         else:
               final_list = [[doc_id, agg[doc_id]] for doc_id in sorted(agg)]

         # write 
         write_fp.write(json.dumps({'term': term, 'doc_list': final_list}, ensure_ascii=False) + '\n')

         if final:
            number_of_terms += 1
            sum_list_size += len(final_list)

   # close all read files
   for fp in read_file_pointers:
      fp.close()

   if final:
      average_list_size = sum_list_size / number_of_terms if number_of_terms > 0 else 0

      return number_of_terms, average_list_size


def parallel_merge_partial_indexes(index_path, output_name='complete_inversed_index.jsonl'):
   """
   Split the partial index files into groups and merge in parallel using threads.
   """
   # list of partial index files
   all_files = [f for f in os.listdir(index_path) if (f.startswith('partial_inversed_') and f.endswith('.jsonl'))]
   full_paths = [os.path.join(index_path, f) for f in all_files]

   # set number of threads
   num_workers = 4
   groups = [full_paths[i::num_workers] for i in range(num_workers)]

   threads = []
   intermediate_paths = []

   # start threads for each group
   for i, group in enumerate(groups):

      if not group:
         continue

      interm_path = os.path.join(index_path, f'intermediate_{i}.jsonl')
      intermediate_paths.append(interm_path)

      t = threading.Thread(target=merge_group, args=(group, interm_path, False), name=f"Thread-{i}")
      t.start()
      threads.append(t)

   # wait for all threads to finish
   for t in threads:
      t.join()

   # final merge of intermediates
   final_path = os.path.join(index_path, output_name)
   number_of_terms, average_list_size = merge_group(intermediate_paths, final_path, final=True)

   # cleanup
   for ip in intermediate_paths:
      try:
         os.remove(ip)
      except OSError:
         pass

   print(f"Parallel merge complete. Final index at: {final_path}")

   return number_of_terms, average_list_size