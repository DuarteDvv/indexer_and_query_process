import json
import gc
import os
import multiprocessing as mp
import util_indexer as util
from collections import Counter, defaultdict
from queue import Empty

# -m <MEMORY>: the memory available to the indexer in megabytes.
# -c <CORPUS>: the path to the corpus file to be indexed.
# -i <INDEX>: the path to the directory where indexes should be written.

# SPIMI like algorithm


def merge_partial_indexes(index_path):
   """
   Merges all partial indexes into a single index file.
   """
   
   # adquire all names of the partial indexes
   partial_indexes_names = os.listdir(index_path) 
   
   
   
   
   
   
def task_corpus_reader(corpus_path, doc_queue):
   """
   Reads the corpus file and puts documents into a queue.
   """
   with open(corpus_path, "rb") as f:
      for line in f:
         doc = json.loads(line)
         doc_queue.put(doc)
         
   print(f"[Reader] Finished reading corpus. Total docs enqueued: {doc_queue.qsize()}")


def task_writer(memory_limit, index_path, doc_queue, idx_val, idx_lock):
   """
   Worker process: pulls docs until memory threshold or queue exhausted,
   builds a partial inverted index, flushes to disk, then exits.
   """
   inverted_index = defaultdict(list)
   threshold = memory_limit * 0.1

   while True:
      try:
         doc = doc_queue.get(timeout=1)
      except Empty:
         # no more docs to process
         break

      doc_id = int(doc.get('id'))
      fields = {
         'title': doc.get('title', ''),
         'text': doc.get('text', ''),
         'keywords': ' '.join(doc.get('keywords', []))
      }
      for origin, content in fields.items():
         tokens = util.preprocess(content)
         freqs = Counter(tokens)
         for term, tf in freqs.items():
               encoded_tf = util.unary_encode(tf)
               inverted_index[term].append((doc_id, encoded_tf, origin))

      # flush if exceeding memory threshold
      if util.memory_used() > threshold:
         print(f"[Writer {mp.current_process().name}] Memory threshold reached ({util.memory_used()} MB). Flushing...")
         util.save_partial_index_jsonl(inverted_index, index_path, idx_val, idx_lock)
         inverted_index.clear()
         gc.collect()
         # exit process to free memory
         return

   # final flush if entries remain
   if inverted_index:
      util.save_partial_index_jsonl(inverted_index, index_path, idx_val, idx_lock)
      
      
   
    


def main():
   
   memory_limit, corpus_path, index_path = util.read_args()
   num_workers = mp.cpu_count()
   print(f"[Main] Using up to {num_workers} concurrent writers.")

   manager = mp.Manager()
   doc_queue = manager.Queue(maxsize=1000)
   idx_val = manager.Value('i', 0)
   idx_lock = manager.Lock()

   # Start reader process
   reader = mp.Process(target=task_corpus_reader, args=(corpus_path, doc_queue), name="Reader")
   reader.start()
   print("[Main] Reader process started.")

   # Spawn batches of writers
   while reader.is_alive() or not doc_queue.empty():
      writers = []
      for i in range(num_workers):
         p = mp.Process(
               target=task_writer,
               args=(memory_limit, index_path, doc_queue, idx_val, idx_lock),
               name=f"Writer-{i}"
         )
         p.start()
         writers.append(p)
         print(f"[Main] Started {p.name}.")

      # Wait for this batch to finish
      for p in writers:
         p.join()
         print(f"[Main] {p.name} finished.")

      # Ensure memory freed before next batch
      gc.collect()
      print(f"[Main] Batch complete. Queue size: {doc_queue.qsize()}")

   reader.join()
   print("[Main] All docs processed. Partial indices ready for merge.")
   
   # Merge partial indexes
   merge_partial_indexes(index_path)
   print("[Main] Merging partial indexes complete.")
   
   

if __name__ == "__main__":
   main()
