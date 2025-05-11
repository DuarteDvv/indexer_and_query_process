import argparse
import json

# -m <MEMORY>: the memory available to the indexer in megabytes.
# -c <CORPUS>: the path to the corpus file to be indexed.
# -i <INDEX>: the path to the directory where indexes should be written.


def main():
   
   parser = argparse.ArgumentParser(description="Indexer for the corpus.")
   parser.add_argument("-m", "--memory", type=int, required=True, help="Memory available to the indexer in megabytes.")
   parser.add_argument("-c", "--corpus", type=str, required=True, help="Path to the corpus file to be indexed.")
   parser.add_argument("-i", "--index", type=str, required=True, help="Path to the directory where indexes should be written.")

   args = parser.parse_args()

   print(f"Indexing corpus at {args.corpus} with {args.memory}MB of memory and saving to {args.index}.")
   
   # read jsonl
   
   with open(args.corpus, "r") as f:
      for line in f:
         data = json.loads(line)
         # process data
         print(data)
      
      
      
      
      


if __name__ == "__main__":
   main()
   
   
   