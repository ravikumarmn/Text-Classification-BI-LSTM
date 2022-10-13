import torch.nn as nn
from torch.optim import Adam,SGD
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


def train(model,device,train_dataloader,optimizer,criterion):
    model.train()
    train_loss = list()
    tqdm_obj_loader = tqdm(enumerate(train_dataloader),total = len(iter(train_dataloader)))
    tqdm_obj_loader.set_description_str('Train')
    train_preds = list()
    for batch_index,data in tqdm_obj_loader:
        optimizer.zero_grad()
        data = {k:v.to(device) for k, v in data.items()}
        pred = model(data['seq_padded'])#b,n
        target = data['label']#b,n
        
        loss = criterion(pred,target)
        # l1 = mean_square_error(pred,target)
        # running_loss += (loss_batch - running_loss)/(batch_index + 1)
        # loss = loss + l1
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        preds_bool = target == (pred.data>0.5).float()
        train_preds.extend(preds_bool.int().tolist())
    return sum(train_loss)/len(train_loss),sum(train_preds)/len(train_preds)

def evaluate(model,device,test_dataloader,criterion):
    model.eval()
    test_loss = list()
    test_preds = list()
    tqdm_obj_val_loader = tqdm(enumerate(test_dataloader),total = len(iter(test_dataloader)))
    tqdm_obj_val_loader.set_description_str('Val')
    with torch.no_grad():
        for batch_index,data in tqdm_obj_val_loader:
            data = {k:v.to(device) for k, v in data.items()}
            pred = model(data['seq_padded'])
            target = data['label']#b,n
            loss = criterion(pred,target)
            # l1 = mean_square_error(pred,target)
            # loss = loss + l1
            test_loss.append(loss.item())
            preds_bool = target == (pred.data>0.5).float()
            test_preds.extend(preds_bool.int().tolist())
        return sum(test_loss)/len(test_loss),sum(test_preds)/len(test_preds)

def train_fn(model,train_dataloader,test_dataloader,criterion,optimizer,params):
    tqdm_obj_epoch = tqdm(range(params["EPOCHS"]),total = params["EPOCHS"],leave = False)
    tqdm_obj_epoch.set_description_str("Epoch")
    val_loss = np.inf

    for epoch in tqdm_obj_epoch:
        training_loss,training_accuracy = train(model,params["device"],train_dataloader,optimizer,criterion)
        validation_loss,validation_accuracy = evaluate(model,params["device"],test_dataloader,criterion)
    
        if validation_loss < val_loss:
            val_loss = validation_loss

            early_stopping = 0  
            torch.save(
                {  
                    "model_state_dict":model.state_dict(),
                    "params":params
                },str(params["save_checkpoint_dir"])+\
                    f'seq2seq_hidden_{params["HIDDEN_SIZE"]}_embed_{params["EMBED_SIZE"]}.pt')
        else:
            early_stopping += 1
        if early_stopping == params["patience"]:
            print("Early stopping")
            break
            

        print(f'Epoch: {epoch+1}/{params["EPOCHS"]}\tTrain loss: {training_loss}\tTrain acc: {training_accuracy}\tVal loss:{validation_loss}\tVal acc:{validation_accuracy}')
        

        wandb.log({
            "epoch/validation_loss" : validation_loss,
            "epoch/validation_error" : 1 - validation_accuracy,
            "epoch/validation_accuracy" : validation_accuracy,

            "epoch/training_loss" : training_loss,
            "epoch/training_error" : 1 - training_accuracy,
            "epoch/training_accuracy" : training_accuracy,
            
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
    criterion = nn.BCELoss(reduction='mean')
    optimizer = Adam(my_model.parameters(), lr=config["LEARNING_RATE"])
    #SGD(my_model.parameters(), lr=config["LEARNING_RATE"],weight_decay=config['weight_decay'])#weight_decay=1e-5 ,momentum=0.9
    my_model.to(config["device"])
    model = train_fn(my_model,train_dataloader,test_dataloader,criterion,optimizer,config)
    return model,test_dataloader


def mean_absolute_error(pred,target):
    return torch.abs(target-pred).mean()

def mean_square_error(pred,target):
    out = (target-pred)**2
    return out.mean()


if __name__ == '__main__':
    params =  {k:v for k,v in config.__dict__.items() if "__" not in k}
    print("Params :",params,sep="\n")
    wandb.init(project='text_classifier',
            name = params["runtime_name"] + f'_seq2seq_hidden_{params["HIDDEN_SIZE"]}_embed_{params["EMBED_SIZE"]}',
            notes = "taking mean of all hidden state, bidirectional lstm, loss reduction is sum",
            tags = ['baseline',"lstm","loss_sum"],
            config=params,
            mode = 'online')

    model,test_dataloader = main(params)

    # pred_class = inference.predict(config,5)