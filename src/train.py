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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class Metrics:
    epsilon = 1e-7
    def __init__(self,y_true,y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def compute_tp_tn_fp_fn(self) -> float:
        """
        True positive  - actual = 1, predicted = 1
        False positive - actual = 1, predicted = 0
        False negative - actual = 0, predicted = 1
        True negative  - actual = 0, predicted = 0
        """
        tp = sum((self.y_true == 1) & (self.y_pred == 1))
        tn = sum((self.y_true == 0) & (self.y_pred == 0))
        fp = sum((self.y_true == 1) & (self.y_pred == 0))
        fn = sum((self.y_true == 0) & (self.y_pred == 1))

        return (tp,tn,fp,fn)

    def compute_accuracy(self) -> float:
        """
        Accuracy  = TP + TN / FP + FN + TP + TN

        """
        # assert len(self.y_true) == len(self.y_pred)
        tp,tn,fp,fn = self.compute_tp_tn_fp_fn()
        accuracy_score = (tp + tn)/(tp + tn + fp +fn + self.epsilon)
        return accuracy_score

    def compute_precision(self) -> float:
        """
        Precision = TP / TP + FP
        
        """
        tp,tn,fp,_ = self.compute_tp_tn_fp_fn()
        precision_score = tp/(tp + fp + self.epsilon)

        return precision_score

    def compute_recall(self) -> float:
        """
        Recall = TP / TP + FN

        """
        tp,tn,_,fn = self.compute_tp_tn_fp_fn()
        recall_score = tp /(tp + fn + self.epsilon)

        return recall_score

    def compute_f1_score(self) -> float:
        """
        F1-Score = (2*precision * recall)/(precision + recall)

        """
        tp,tn,fp,fn = self.compute_tp_tn_fp_fn()
        precision = self.compute_precision()
        recall = self.compute_recall()
        f1_score = (2 * precision * recall)/(precision + recall)
        
        return f1_score

    def compute_confustion_matrix(self) -> list:
        tp,tn,fp,fn = self.compute_tp_tn_fp_fn()
        confusion_mat = [[tp,fp],[fn,tn]]
        return np.array(confusion_mat)

    def metrics_report(self) -> dict:
      results = {}
      function_name = [x for x in dir(Metrics) if "__" not in x and (x!= 'metrics_report')]
      for x in function_name:
          results.update({x:getattr(Metrics,x)(self)})
      return results

# def score(y_true,y_pred):
#     tp = (y_true * y_pred).sum().to(torch.float32).item()
#     tn = ((1-y_true) * (1 - y_pred)).sum().to(torch.float32).item()
#     fp = ((1-y_true) * y_pred).sum().to(torch.float32).item()
#     fn = (y_true * (1- y_pred)).sum().to(torch.float32).item()
#     return tp,tn,fp,fn
    

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
    all_pred = list()
    all_true = list()

    for batch_index,data in tqdm_obj_loader:
        optimizer.zero_grad()
        data = {k:v.to(device) for k, v in data.items()}
        pred = model(data['seq_padded'])#b,n
        target = data['label']#b,n
        
        loss = criterion(pred,target)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        preds_bool = target == (pred.data>0.5).float()
        y_pred = preds_bool.float().to("cpu")
        y_true = target.to('cpu')

        all_pred.extend(y_pred)
        all_true.extend(y_true)

    metrics = Metrics(all_true,all_pred)
    precision = metrics.compute_precision()
    recall = metrics.compute_recall()
    f1_score = metrics.compute_f1_score()
    


    return train_loss,all_pred,metrics

def evaluate(model,device,test_dataloader,criterion):
    model.eval()
    test_loss = list()
    # test_preds = list()
    tqdm_obj_val_loader = tqdm(enumerate(test_dataloader),total = len(iter(test_dataloader)))
    tqdm_obj_val_loader.set_description_str('Val')

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
    
            

            all_pred.extend(y_pred.tolist())
            all_true.extend(y_true.tolist())

            # test_preds.extend(preds_bool.int().tolist())

        metrics = Metrics(all_true,all_pred)
        precision = metrics.compute_precision()
        recall = metrics.compute_recall()
        f1_score = metrics.compute_f1_score()
        

        return test_loss,all_pred,all_true,metrics

def train_fn(model,train_dataloader,test_dataloader,criterion,optimizer,params):
    tqdm_obj_epoch = tqdm(range(params["EPOCHS"]),total = params["EPOCHS"],leave = False)
    tqdm_obj_epoch.set_description_str("Epoch")
    val_loss = np.inf

    for epoch in tqdm_obj_epoch:
        train_loss,train_all_pred,train_metrics = train(model,params["device"],train_dataloader,optimizer,criterion)
        training_loss = sum(train_loss)/len(train_loss)
        training_accuracy = sum(train_all_pred)/len(train_all_pred)

        test_loss,test_all_pred,test_all_true,test_metrics = evaluate(model,params["device"],test_dataloader,criterion)
        validation_loss = sum(test_loss)/len(test_loss)
        validation_accuracy = sum(test_all_pred)/len(test_all_pred)

        confu_matrix = test_metrics.compute_confustion_matrix()
        
        x,y = confu_matrix.shape
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
            df_cm = pd.DataFrame(confu_matrix, range(x), range(y))
            # df_norm_col=(df_cm-df_cm.mean())/df_cm.std()
            ax = plt.axes()
            sn.set(font_scale=1.4) # for label size
            
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},) # font size
            ax.set_title('Confusion matrix of the classifier')
            plt.savefig('/content/drive/MyDrive/DL_projects/text_classification/confusion_matrix1.jpg')
            print("Early stopping")
            break
            

        print(f'Epoch: {epoch+1}/{params["EPOCHS"]}\t,Train loss: {training_loss}\tTrain acc: {training_accuracy}\tVal loss:{validation_loss}\tVal acc:{validation_accuracy}\
            Recall: {test_metrics.compute_recall()}\tPrecision : {test_metrics.compute_precision()}\tf1_score : {test_metrics.compute_f1_score()}')
        
        #,Train loss: {training_loss}\tTrain acc: {training_accuracy}\tVal loss:{validation_loss}\tVal acc:{validation_accuracy}
        wandb.log({
            "epoch/validation_loss" : validation_loss,
            "epoch/validation_error" : 1 - validation_accuracy,
            "epoch/validation_accuracy" : validation_accuracy,

            "epoch/training_loss" : training_loss,
            "epoch/training_error" : 1 - training_accuracy,
            "epoch/training_accuracy" : training_accuracy,

            "epoch/recall" : test_metrics.compute_recall(),
            "epoch/precision" : test_metrics.compute_precision(),
            "epoch/f1_score" : test_metrics.compute_f1_score(),
            "epoch/confu_matrix" : wandb.plot.confusion_matrix(
                probs=None,
                preds = all_pred,
                y_true=all_true,
                class_names= list(params['mapping'].keys())
            )
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
            notes = "taking mean of all hidden state, bidirectional lstm, loss reduction is sum,added recall,precision,f1_score,confu_matrix",
            tags = ['baseline',"lstm","loss_sum"],
            config=params,
            mode = 'disabled')

    model,test_dataloader = main(params)

    # pred_class = inference.predict(config,5)