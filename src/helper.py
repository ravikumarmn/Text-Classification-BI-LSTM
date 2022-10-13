import pandas as pd
import re


def read_csv(base_dir,file_name,input_column,target_columns):
  dataset = pd.read_csv(base_dir+file_name,usecols = input_column + target_columns,sep = "\t",header=None)
  return dataset

def preprocess_text(text):
    text = text.lower()
    # text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text #text.strip()


