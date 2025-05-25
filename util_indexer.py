
from argparse import ArgumentParser
import os
from psutil import Process
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import json

STOP_WORDS = set(stopwords.words('english'))

def save_partial_index_jsonl(index, path, idx_val, idx_lock):
   """
   Atomically save a partial inverted index to disk.
   """
   os.makedirs(path, exist_ok=True)
   with idx_lock:
      filename = os.path.join(path, f"partial_{idx_val.value}.jsonl")
      idx_val.value += 1

   with open(filename, 'w', encoding='utf-8') as f:
      for term, postings in sorted(index.items()):
         obj = {"term": term, "postings": postings}
         f.write(json.dumps(obj, ensure_ascii=False) + "\n")
         

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

   process = Process(os.getpid())
   mem = process.memory_info().rss / (1024 * 1024)  # Convert bytes to megabytes
   
   return mem
