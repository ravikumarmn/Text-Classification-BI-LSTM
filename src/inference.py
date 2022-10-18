import json 
import config
import torch
from model import ClassifierModel
import os
import random 
from prepare_data import CustomDataset

custom_testdata = CustomDataset(config.train_test_data,config.max_seq_len,"test")

def last_train_model(model_folder_str):
    model_file_name = list()
    for x in os.listdir(model_folder_str):
        if (x.endswith(".pt")):
            model_file_name.append(x)
        else:
            continue
    return model_file_name[-1]

def eval(model,input_token):
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        out = model(input_token)
        pred_class = (out >= 0.5).float().item()
        return pred_class

def testing_data(params,test_dataloader,vocab,no_samples = 5):
    test_pairs = len(custom_testdata)
    data_indices = list(range(test_pairs))
    random.shuffle(data_indices)
    inference_data = list()
    for idx in data_indices[:no_samples]:
        data = test_dataloader[idx]
        ins_token = data['seq_padded'].to('cpu')
        labels = data['label'].to('cpu')
        lbls = {0:"spam",1:"ham"}
        true_sen = " ".join([vocab['index2word'][str(index.item())] for index in ins_token if index != 0])
        true_lbl = lbls[int(labels.item())]
        inference_data.append({
            "input_tokens" : ins_token,
            "true_sentence" : true_sen,
            "true_label" : true_lbl
        })
    return inference_data

def predict(params,samples):
    vocab = json.load(open(params.base_dir + params.vocab_file_name,"r"))
    n_vocab = len(vocab['word2index'])
    model_file_name = last_train_model(params.save_checkpoint_dir)
    model_file_dir = params.save_checkpoint_dir + model_file_name
    checkpoint = torch.load(open(model_file_dir,"rb"))
    model = ClassifierModel(n_vocab,params.HIDDEN_SIZE,params.OUT_DIM,params.EMBED_SIZE,params.n_labels)

    model.load_state_dict(checkpoint["model_state_dict"])
    inference_data = testing_data(params,test_dataloader=custom_testdata,vocab=vocab,no_samples=samples)
    for idx,data in enumerate(inference_data):
        pred_class = eval(model,data['input_tokens'])
        if pred_class == 1:
            pred_class = 'ham'
        elif pred_class == 0:
            pred_class = 'spam'
        print(f"sample:{idx} \tInput_sentence  :  {data['true_sentence']}\n\t\tTrue label\t: {data['true_label']}\n\t\tInference label : {pred_class}")
        print("="*150)


if __name__ == "__main__":
    predict(config,5)


