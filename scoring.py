import time
from minicons import scorer
#import torch
#from torch.utils.data import DataLoader
import numpy as np
#import json
#import csv
import pandas as pd

start = time.time()

model = scorer.IncrementalLMScorer('gpt2', 'cpu')

# Opens a window of text of size window_size with window_size words in it for the
# model to calculate surprisal scores from
window_size = 15
# file that we want to calculate scores from
filename = "newtext.txt"
with open(filename, "r", encoding="utf-8") as file:
  text = file.read().split()

# while loop that calcutes scores in the text until it reaches end of .txt file
scores_df = pd.DataFrame()
end = len(text)
i = 0
while i < end:
  if len(text[i:i + window_size]) < window_size:
    break
  window_str = " ".join(text[i:i + window_size])
  print("New window: " + window_str + " || end of window.")
  i = i + 1

  # Calculates the surprisal score using the gpt-2 language model
  # Alltext2 will store all the results of the calculation
  # model refers to the loaded gpt-2 model loaded
  # text is the list of strings containing the text we're analyzing 
  alltext2=model.token_score(window_str, surprisal = True, base_two = True)
  scores_df = scores_df.append(alltext2, ignore_index=True) # Append new score to data frame
  print("Calculated scores for window #" + str(i) + ".")
  #print(scores_df)
  print("\n")

print("Surprisal score calculations are done. \n")

# formats data from scores_df into two rows in alltext2
Dat=[]
for r in alltext2:
  for c in r:
    n = []
    if(c==None):
      n.append(0)
      n.append(0)
    else:
      n.append(c[0])
      n.append(c[1])
    Dat.append(n)
  alltext2 = pd.DataFrame(Dat)

# saves scores_df as a .cvs file in designated path
df = pd.DataFrame(scores_df)
csv_file_path = r"your own path" # change to your own path
df.to_csv(csv_file_path, index=False)

end = time.time()

print("Time elapsed: " + str(end - start))