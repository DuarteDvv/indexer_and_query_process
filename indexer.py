import json
import gc
from threading import Thread, Lock
import util_indexer as util
from collections import Counter
from queue import Queue

# -m <MEMORY>: the memory available to the indexer in megabytes.
# -c <CORPUS>: the path to the corpus file to be indexed.
# -i <INDEX>: the path to the directory where indexes should be written.

q_size = 1000 
n_workers = 1
document_queue = Queue(q_size)

partial_inverted_index = {}  # term -> (docID, frequency, origin)
inverted_index_mutex = Lock()  # Mutex for the inverted index

def task_corpus_reader(corpus_path):
   """
   Reads the corpus file and puts documents into a queue.
   """
   with open(corpus_path, "r") as f:
      for line in f:
         doc = json.loads(line)
         document_queue.put(doc)
              
   for _ in range(n_workers):  # Signal end of processing
      document_queue.put(None)
      
   print("Corpus reader finished.")
   
   return


def task_index(memory_limit, index_path):
   
   while True:
      
      doc = document_queue.get()
      
      if doc is None:
         break
      
      # Check memory limit
      if util.memory_used() > memory_limit*0.8:
         
         # Save the partial inverted index to disk
         
         return
         
      
      
   
      doc_id = doc.get('id')
      
      # extrai campos
      fields = {
         'title': doc.get('title', ''),
         'text': doc.get('text', ''),
         'keywords': ' '.join(doc.get('keywords', []))
      }

      for origin, content in fields.items():
         
         tokens = util.preprocess(content)
         freqs = Counter(tokens)

         for term, tf in freqs.items():
            # Adiciona tupla (docID, frequency, origin)
            
            with inverted_index_mutex:
               partial_inverted_index.setdefault(term, []).append((doc_id, tf, origin))

    
 
   return 

def main():
   
   # read args
   memory_limit, corpus_path, index_path = util.read_args()
   
   # Start corpus reader in a separate thread
   reader_t = Thread(target=task_corpus_reader, args=(corpus_path,))
   
   reader_t.start() # starts the thread who reads the corpus
   
   print("Started corpus reader thread.")
   
   # Create worker threads
   workers = []
   
   for _ in range(n_workers):
      worker = Thread(target=task_index, args=(memory_limit, index_path))
      workers.append(worker)
      worker.start()
      print("Started worker thread.")
      
   # Wait for the reader thread to finish
   reader_t.join()
   print("Reader thread finished.")
   
   # Wait for all worker threads to finish
   for worker in workers:
      worker.join()
      print("Worker thread finished.")
      
   
   
      
      
            
            
            
            
            
   
               
            
            
         
   
   
      
         
      
      
      
      
      


if __name__ == "__main__":
   main()
   
   
   