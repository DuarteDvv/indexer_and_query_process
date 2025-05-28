import json
import gc
import os
import multiprocessing as mp
import util_indexer as util
from collections import Counter, defaultdict
from queue import Empty
import math
import time

# -m <MEMORY>: the memory available to the indexer in megabytes.
# -c <CORPUS>: the path to the corpus file to be indexed.
# -i <INDEX>: the path to the directory where indexes should be written.

# SPIMI like algorithm
   
def task_corpus_reader(corpus_path, doc_queue):
   """
   Reads the corpus file and puts documents into a queue.
   """
   with open(corpus_path, "rb") as f:
      for line in f:
         doc = json.loads(line)
         doc_queue.put(doc)
         
   print(f"[Reader] Finished reading corpus.")


def task_writer(memory_limit, index_path, doc_queue, idx_val, idx_lock, document_index):
   """
   Worker process: pulls docs until memory threshold or queue exhausted,
   builds a partial inverted index, flushes to disk, then exits.
   """
   inverted_index = defaultdict(list)

   while True:
      try:

         doc = doc_queue.get()

      except Empty:
         break

      doc_id = int(doc.get('id'))

      fields = {
         'title': doc.get('title', ''),
         'text': doc.get('text', ''),
         'keywords': ' '.join(doc.get('keywords', []))
      }

      content = ' '.join(fields.values()).strip()

      tokens = util.preprocess(content)

      document_index[doc_id] = {
         'title': fields['title'],
         'length': len(tokens)
      }

      freqs = Counter(tokens)

      for term, tf in freqs.items():
            
         inverted_index[term].append((doc_id, tf))


      # flush if exceeding memory threshold
      if util.memory_used() > memory_limit:

         util.save_partial_index_jsonl(inverted_index, index_path, idx_val, idx_lock)
         inverted_index.clear()
         gc.collect()

         # exit process to free memory
         return

   # final flush if entries remain
   if inverted_index:
      util.save_partial_index_jsonl(inverted_index, index_path, idx_val, idx_lock)
      

def main():

   time_start = time.time()
   
   memory_limit, corpus_path, index_path = util.read_args()
   num_workers = int(math.ceil(mp.cpu_count() / 2))
   print(f"[Main] Using up to {num_workers} concurrent writers.")

   manager = mp.Manager()
   doc_queue = manager.Queue(maxsize=1000)
   idx_val = manager.Value('i', 0)
   idx_lock = manager.Lock()
   document_index = manager.dict()
   idx_doc_index = 0

   # start reader process
   reader = mp.Process(target=task_corpus_reader, args=(corpus_path, doc_queue), name="Reader")
   reader.start()
   print("[Main] Reader process started.")

   # spawn batches of writers until all docs processed
   while reader.is_alive() or not doc_queue.empty():
      print(f"[Main] Starting new batch of writers...")

      writers = []
      for i in range(num_workers):
         p = mp.Process(
            target=task_writer,
            args=((memory_limit*0.8 / num_workers) , index_path, doc_queue, idx_val, idx_lock, document_index),
            name=f"Writer-{i}"
         )
         p.start()
         writers.append(p)

      # wait for this batch to finish
      for p in writers:
         p.join()

      util.save_partial_document_index(document_index, index_path,idx_doc_index )
      document_index.clear()
      document_index = manager.dict()  
      idx_doc_index += 1

      # ensure memory freed before next batch
      gc.collect()

      print(f"[Main] Batch of writers finished.")
      print(f"[Main] Cleaning up memory...")
 

   reader.join()
   print("[Main] All docs processed. Partial indices ready for merge.")
   
   # merge partial indexes
   n_of_terms, average_list_size = util.parallel_merge_partial_indexes(index_path)

   print("Constructing complete document index...")
   util.append_partial_document_indexes(index_path)
   print("[Main] Document index constructed.")

   # erase the partial index files
   for file in os.listdir(index_path):
      if file.startswith('partial_'):
         os.remove(os.path.join(index_path, file))

   time_end = time.time()

   index_size = os.path.getsize(os.path.join(index_path, 'complete_inversed_index.jsonl')) / (1024 * 1024)

   output = {
      'Index Size': index_size,
      'Elapsed Time': time_end - time_start, # in seconds
      'Number of Lists':n_of_terms,
      'Average List Size': average_list_size
   }

   print(output)

if __name__ == "__main__":
   main()
