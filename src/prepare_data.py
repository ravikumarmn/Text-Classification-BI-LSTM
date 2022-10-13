import torch
import pickle
import json
from torch.utils.data import Dataset,DataLoader
import config
import helper
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np

def build_vocabulary(data):
    words = list()
    samples = list()
    word2index = {'PAD' : 0,'SOS':1,'EOS':2,'UNK':3}
    index2word = {0:'PAD',1:'SOS',2:'EOS',3 : 'UNK'}

    for example in data:
        for word in example.split():                
            words.append(word)  
    uniq_words = set(words)
    for idx,word in enumerate(uniq_words,start = 4):
        word2index[word] = idx
        index2word[idx] = word

    for example in data:
        seq = list()
        for word in example.split():
            seq.append(word2index[word])
        samples.append(seq)
    return samples,word2index,index2word,uniq_words


class CustomDataset(Dataset):
    def __init__(self,file_name,max_seq_len,data_type = 'train'):
        super(CustomDataset,self).__init__()
        self.max_seq_len = max_seq_len
        data = pickle.load(open(config.base_dir + file_name,'rb'))
        # vocab = json.load(open(config.base_dir + config.vocab_file_name,"r"))

        self.wordvecs = KeyedVectors.load(config.base_dir + config.word2vec_file)
        self.pickle_data =pickle.load(open(config.base_dir +config.emb_vec_file,'rb'))
        self.word2index = self.pickle_data['word2index']
        # index2word  = vocab['index2word']
        self.pad_str  = 0
        self.wordvecs.wv[self.pad_str] = np.zeros(len(self.wordvecs.wv["hi"])).astype(self.wordvecs.wv["hi"].dtype)

        if data_type == 'train':
            dataset = list()
            for inp,lbl in zip(data['X_train'],data['y_train']):
                dataset.append((inp,lbl))
            self.data = dataset
        else:
            dataset = list()
            for inp,lbl in zip(data['X_test'],data['y_test']):
                dataset.append((inp,lbl))
            self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        seq,label = self.data[idx]

        seq = [self.word2index[s] for s in seq]
        seq_padded = self.padded(seq)



        return {
            "seq_padded" : torch.as_tensor(seq_padded),
            "label" : torch.tensor(label,dtype = torch.float)
        }

    def padded(self,x):
        x = x[:self.max_seq_len]
        x = x +[self.pad_str] * (self.max_seq_len - len(x))

        return x


if __name__ == "__main__":
    data = helper.read_csv(config.base_dir,config.file_name,config.input_column,config.target_columns)
    dataset = data[config.input_column[0]].apply(lambda x : helper.preprocess_text(x))
    samples,word2index,index2word,uniq_words = build_vocabulary(dataset)
    
    json.dump(
    {
      "word2index" : word2index,
      "index2word" : index2word
    },
    open(config.base_dir + config.vocab_file_name,"w")) 

    # lbl_dict  = {"spam":0,"ham":1}
    samples_lbl = data[config.target_columns[0]]#.map(lbl_dict)
    X_train, X_test, y_train, y_test = train_test_split(samples,samples_lbl,test_size=0.2,shuffle=True)

    pickle.dump({
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train" : y_train.values.tolist(),
        "y_test" : y_test.values.tolist(),
    },open(config.base_dir + config.train_test_data,'wb'))