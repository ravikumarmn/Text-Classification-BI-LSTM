import torch.nn as nn
from torch.optim import Adam
from model import ClassifierModel
import config
from tqdm import tqdm
import helper
import prepare_data
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import pickle
import config 
import json
import inference
import wandb 
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def score(y_true,y_pred):
    tp = (y_true * y_pred).sum().to(torch.float32).item()
    tn = ((1-y_true) * (1 - y_pred)).sum().to(torch.float32).item()
    fp = ((1-y_true) * y_pred).sum().to(torch.float32).item()
    fn = (y_true * (1- y_pred)).sum().to(torch.float32).item()
    return tp,tn,fp,fn
    

    # confusion_mat = [[tp,fp],[fn,tn]]

    # precision = tp/(tp + fp)
    # recall = tp /(tp + fn)

    # f1_score = (2 * recall * precision)/(recall + precision)


    # df_cm = pd.DataFrame(confussion_mat, range(2), range(2))
    # sn.set(font_scale=1.4) # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    # plt.savefig('confusion_matrix.jpg')

def train(model,device,train_dataloader,optimizer,criterion):
    model.train()
    train_loss = list()
    tqdm_obj_loader = tqdm(enumerate(train_dataloader),total = len(iter(train_dataloader)))
    tqdm_obj_loader.set_description_str('Train')
    train_preds = list()
    target_true = 0
    predicted_true = 0
    correct_true = 0

    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0

    for batch_index,data in tqdm_obj_loader:
        optimizer.zero_grad()
        data = {k:v.to(device) for k, v in data.items()}
        pred = model(data['seq_padded'])#b,n
        target = data['label']#b,n
        
        loss = criterion(pred,target)
        # running_loss += (loss_batch - running_loss)/(batch_index + 1)
        
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        preds_bool = target == (pred.data>0.5).float()
        # target_true += torch.sum(target == 0).float()
        # predicted_true += torch.sum(preds_bool).float()
        # correct_true += torch.sum(preds_bool == target *preds_bool)

        # recall = correct_true / target_true
        # precision = correct_true / predicted_true
        # F1_score = 2 * (precision * recall)/(precision + recall)
        y_pred = preds_bool.float().to("cpu")
        y_true = target.to('cpu')

        tp,tn,fp,fn = score(y_true=y_true,y_pred=y_pred)
        true_p += tp
        true_n += tn
        false_n += fp
        false_n += fn

        train_preds.extend(preds_bool.int().tolist())

    epsilon = 1e-7
    precision = true_p/(true_p + false_p + epsilon)
    recall = true_p /(true_p + false_n+epsilon)
    f1_score = (2 * recall * precision)/(recall + precision +epsilon)

    return sum(train_loss)/len(train_loss),sum(train_preds)/len(train_preds),correct_true,target_true,predicted_true,precision,recall,f1_score

def evaluate(model,device,test_dataloader,criterion):
    model.eval()
    test_loss = list()
    test_preds = list()
    tqdm_obj_val_loader = tqdm(enumerate(test_dataloader),total = len(iter(test_dataloader)))
    tqdm_obj_val_loader.set_description_str('Val')
    target_true = 0 
    predicted_true = 0
    correct_true = 0

    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0
    all_pred = list()
    all_true = list()
    with torch.no_grad():
        for batch_index,data in tqdm_obj_val_loader:
            data = {k:v.to(device) for k, v in data.items()}
            y_pred = model(data['seq_padded'])
            loss = criterion(y_pred,data['label'])

            target = data['label']#b,n
            test_loss.append(loss.item())

            preds_bool = target == (y_pred.data>0.5).float()
            y_pred = preds_bool.float().to("cpu")
            y_true = target.to('cpu')
    
            tp,tn,fp,fn = score(y_true=y_true,y_pred=y_pred)
            true_p += tp
            true_n += tn
            false_n += fp
            false_n += fn
            all_pred.extend(y_pred.tolist())
            all_true.extend(y_true.tolist())

            test_preds.extend(preds_bool.int().tolist())
        epsilon = 1e-7

        precision = true_p/(true_p + false_p +epsilon)
        recall = true_p /(true_p + false_n+epsilon)
        f1_score = (2 * recall * precision)/(recall + precision +epsilon)
        return sum(test_loss)/len(test_loss),sum(test_preds)/len(test_preds),precision,recall,f1_score,np.array(all_pred),np.array(all_true)

def train_fn(model,train_dataloader,test_dataloader,criterion,optimizer,params):
    tqdm_obj_epoch = tqdm(range(params["EPOCHS"]),total = params["EPOCHS"],leave = False)
    tqdm_obj_epoch.set_description_str("Epoch")
    val_loss = np.inf

    for epoch in tqdm_obj_epoch:
        training_loss,training_accuracy,correct_true,target_true,predicted_true,precision,recall,f1_score = train(model,params["device"],train_dataloader,optimizer,criterion)
        validation_loss,validation_accuracy,precision,recall,f1_score,all_pred,all_true = evaluate(model,params["device"],test_dataloader,criterion)
        tp = (all_true * all_pred).sum()
        tn = ((1-all_true) * (1 - all_pred)).sum()
        fp = ((1-all_true) * all_pred).sum()
        fn = (all_true * (1- all_pred)).sum()
        confusion_mat = [[tp,fp],[fn,tn]]

        x,y = np.array(confusion_mat).shape
        if validation_loss < val_loss:
            val_loss = validation_loss

            early_stopping = 0  
            torch.save(
                {  
                    "model_state_dict":model.state_dict(),
                    "params":params
                },str(params["save_checkpoint_dir"])+\
                    f'seq2seq_hidden_{params["HIDDEN_SIZE"]}_embed_{params["EMBED_SIZE"]}_imdb_prep_word2vec.pt')
        else:
            early_stopping += 1
        if early_stopping == params["patience"]:
            df_cm = pd.DataFrame(confusion_mat, range(x), range(y))
            # df_norm_col=(df_cm-df_cm.mean())/df_cm.std()
            ax = plt.axes()
            sn.set(font_scale=1.4) # for label size
            
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},) # font size
            ax.set_title('Confusion matrix of the classifier')
            plt.savefig('/content/drive/MyDrive/DL_projects/text_classification/confusion_matrix1.jpg')
            print("Early stopping")
            break
            

        print(f'Epoch: {epoch+1}/{params["EPOCHS"]}\t,Train loss: {training_loss}\tTrain acc: {training_accuracy}\tVal loss:{validation_loss}\tVal acc:{validation_accuracy}\
            Recall: {recall}\tPrecision : {precision}\tf1_score : {f1_score}')
        
        #,Train loss: {training_loss}\tTrain acc: {training_accuracy}\tVal loss:{validation_loss}\tVal acc:{validation_accuracy}
        wandb.log({
            "epoch/validation_loss" : validation_loss,
            "epoch/validation_error" : 1 - validation_accuracy,
            "epoch/validation_accuracy" : validation_accuracy,

            "epoch/training_loss" : training_loss,
            "epoch/training_error" : 1 - training_accuracy,
            "epoch/training_accuracy" : training_accuracy,

            "epoch/recall" : recall,
            "epoch/precision" : precision,
            "epoch/f1_score" : f1_score
        })
    return model

def main(config):
    vocab = json.load(open(config["base_dir"] + config["vocab_file_name"],'r'))
    word2index = vocab['word2index']

    trains = prepare_data.CustomDataset(config["train_test_data"],config["max_seq_len"],"train")
    tests = prepare_data.CustomDataset(config["train_test_data"],config["max_seq_len"],"test")

    train_dataloader = DataLoader(trains,batch_size = config["BATCH_SIZE"],shuffle = True)
    test_dataloader = DataLoader(tests,batch_size = config["BATCH_SIZE"])
    my_model = ClassifierModel(len(word2index),config["HIDDEN_SIZE"],config["OUT_DIM"],config["EMBED_SIZE"],n_labels = config["n_labels"],max_seq=config['max_seq_len'])
    criterion = nn.BCELoss(reduction='sum')
    optimizer = Adam(my_model.parameters(), lr=config["LEARNING_RATE"])#weight_decay=1e-5
    my_model.to(config["device"])
    model = train_fn(my_model,train_dataloader,test_dataloader,criterion,optimizer,config)
    return model,test_dataloader

if __name__ == '__main__':
    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    print("Params :",params,sep="\n")
    wandb.init(project='text_classifier',
            name = params["runtime_name"] + f'_seq2seq_hidden_{params["HIDDEN_SIZE"]}_embed_{params["EMBED_SIZE"]}',
            notes = "taking mean of all hidden state, bidirectional lstm, loss reduction is sum,added recall,precision,f1_score",
            tags = ['baseline',"lstm","loss_sum"],
            config=params,
            mode = 'disabled')

    model,test_dataloader = main(params)

    # pred_class = inference.predict(config,5)