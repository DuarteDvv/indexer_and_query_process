
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



def preprocess(text):
   """
   Preprocess the text by tokenizing, removing stopwords, and stemming.
   """
   
   # tokenize
   tokens = word_tokenize(text.lower())
   
   # keep only alphanumeric tokens
   tokens = [tok for tok in tokens if re.match(r"^[a-z0-9]+$", tok)]
   
   # remove stopwords
   stop_words = set(stopwords.words('english'))
   tokens = [word for word in tokens if word not in stop_words]
   
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

def memory_used() -> int:
   """
   Returns the memory used by the process in megabytes.
   """

   process = Process(os.getpid())
   mem = process.memory_info().rss / (1024 * 1024)  # Convert bytes to megabytes
   
   return mem