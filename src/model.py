# input_dim = torch.size([16,500]) # batch,seq
# output_dim = torch.size([16,5]) # batch,n_labels
import torch.nn as nn
import pickle
import config
import torch

class ClassifierModel(nn.Module):
  def __init__(self,n_vocab,hidden_size,out_hidden,embedding_dim,n_labels,max_seq):
    super(ClassifierModel,self).__init__()
    pickle_data = pickle_data =pickle.load(open(config.base_dir +config.emb_vec_file,'rb'))

    self.embedding_tensor = torch.from_numpy(pickle_data['embedding_vector']).float()
    n_vocab,embedding_dim = self.embedding_tensor.shape
    self.embedding = nn.Embedding(n_vocab+1,embedding_dim,padding_idx = 0)
    self.embedding.weight = nn.Parameter(self.embedding_tensor)
    
    self.lstm = nn.LSTM(embedding_dim,hidden_size,batch_first = True,bidirectional=True)
    self.linear1  = nn.Linear(hidden_size*2,hidden_size)
    self.linear2 = nn.Linear(hidden_size,out_hidden)
    self.linear3 = nn.Linear(out_hidden,n_labels)
    # self.reduce_embed = nn.Lingear(100,embedding_dim)
    self.sigmoid = nn.Sigmoid()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.relu = nn.ReLU()

  def forward(self,x):
    embedded = self.embedding(x)
    output,_ = self.lstm(embedded)#h_in->1,b,e, output:b,s,h:16,500,32
    output_mean = output.mean(1) #16,1,32, mean removed dim 
    output = self.linear1(output_mean)
    output = self.relu(self.dropout1(output))
    output = self.linear2(output) #before applying non-linearity , you can call them logits
    output = self.relu(self.dropout2(output))
    logits = self.linear3(output)
    output = self.sigmoid(logits)
    return output.flatten()
