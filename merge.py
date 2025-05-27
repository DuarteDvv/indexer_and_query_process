import os
import json
import heapq
from collections import defaultdict
import util_indexer as util
from multiprocessing import Process


def merge_group(file_paths, output_path, final=False):
   """
   Merge a list of partial index files into a single JSONL index at output_path.
   """
   # open all partial indexes for reading
   read_file_pointers = [open(path, 'r', encoding='utf-8') for path in file_paths]
   iterators = [iter(fp) for fp in read_file_pointers]

   # prepare writer for the merged index
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

         # merge all same-term entries
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

         # build final list with unary-encoded freq or intermediary list with raw frequencies
         if final:
     
            final_list = [[doc_id, util.unary_encode(agg[doc_id])] for doc_id in sorted(agg)]
         else:
     
            final_list = [[doc_id, agg[doc_id]] for doc_id in sorted(agg)]

         # write 
         write_fp.write(json.dumps({'term': term, 'doc_list': final_list}, ensure_ascii=False) + '\n')

   # close all input file pointers
   for fp in read_file_pointers:
      fp.close()


def paralel_merge_partial_indexes(index_path, output_name='complete_index.jsonl'):
 
   # list partial index files
   all_files = [f for f in os.listdir(index_path) if f.endswith('.jsonl')]
   full_paths = [os.path.join(index_path, f) for f in all_files]

   # partition into groups
   num_workers = 8
   groups = [full_paths[i::num_workers] for i in range(num_workers)]

   processes = []
   intermediate_paths = []

   # launch parallel merges for each group
   for i, group in enumerate(groups):
      if not group:
         continue
      interm_path = os.path.join(index_path, f'intermediate_{i}.jsonl')
      intermediate_paths.append(interm_path)
      p = Process(target=merge_group, args=(group, interm_path))
      p.start()
      processes.append(p)

   # wait for all to finish
   for p in processes:
      p.join()

   # final merge of intermediate files
   final_path = os.path.join(index_path, output_name)
   merge_group(intermediate_paths, final_path, final=True)

   #cleanup intermediate files
   for ip in intermediate_paths:
      try:
         os.remove(ip)
      except OSError:
         pass

   print(f"Parallel merge complete. Final index at: {final_path}")


# Example usage:
paralel_merge_partial_indexes('./index/')
