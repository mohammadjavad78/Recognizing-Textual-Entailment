import os
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from dataloaders.DataLoader import *
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from tqdm import tqdm
from utlis.Averagemeter import *
from utlis.Read_yaml import *
from nets.model import *

yml=Getyaml()


model_name=yml['model_name']


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res



# model_name
savetocsv(name=model_name)




def train(
    train_loader,
    val_loader,
    test_loader,
    model,
    model_name,
    epochs,
    learning_rate,
    gamma,
    step_size,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
):

    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimzier
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer
        )

    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_top1_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_top1_acc_till_current_batch",
            "avg_test_loss_till_current_batch",
            "avg_test_top1_acc_till_current_batch"])

    for epoch in tqdm(range(1, epochs + 1)):
        top1_acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        top1_acc_val = AverageMeter()
        loss_avg_val = AverageMeter()
        top1_acc_test = AverageMeter()
        loss_avg_test = AverageMeter()

        model.train()
        mode = "train"
        
        
        loop_train = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc="train",
            position=0,
            leave=True)
        for batch_idx, (input_ids,token_type_ids,attention_mask, labels) in loop_train:
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            labels_pred = model(input_ids,token_type_ids,attention_mask)
            loss = criterion(labels_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1 = accuracy(labels_pred, labels)
            top1_acc_train.update(acc1[0], input_ids.size(0))
            loss_avg_train.update(loss.item(), input_ids.size(0))

            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "mode": mode,
                 "image_type":"original",
                 "epoch": epoch,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size": input_ids.size(0),
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_train_loss_till_current_batch":loss_avg_train.avg,
                 "avg_train_top1_acc_till_current_batch":top1_acc_train.avg,
                 "avg_val_loss_till_current_batch":None,
                 "avg_val_top1_acc_till_current_batch":None,
                 "avg_test_loss_till_current_batch":None,
                 "avg_test_top1_acc_till_current_batch":None},index=[0])

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                top1_accuracy_train="{:.4f}".format(top1_acc_train.avg),
                max_len=2,
                refresh=True,
            )
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}.ckpt",
                model=model,
                optimizer=optimizer,
            )

        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_idx, (input_ids,token_type_ids,attention_mask, labels) in loop_val:
                optimizer.zero_grad()
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                # inputs = inputs.to(device).float()
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                labels_pred = model(input_ids,token_type_ids,attention_mask)
                loss = criterion(labels_pred, labels)
                acc1 = accuracy(labels_pred, labels)
                top1_acc_val.update(acc1[0], input_ids.size(0))
                loss_avg_val.update(loss.item(), input_ids.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "image_type":"original",
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": input_ids.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_train_top1_acc_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg,
                     "avg_val_top1_acc_till_current_batch":top1_acc_val.avg,
                     "avg_test_loss_till_current_batch":None,
                     "avg_test_top1_acc_till_current_batch":None,},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
                    refresh=True,
                )

                
        model.eval()
        mode = "test"
        with torch.no_grad():
            loop_test = tqdm(
                enumerate(test_loader, 1),
                total=len(test_loader),
                desc="test",
                position=0,
                leave=True,
            )
            for batch_idx, (input_ids,token_type_ids,attention_mask, labels) in loop_test:
                optimizer.zero_grad()
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                labels_pred = model(input_ids,token_type_ids,attention_mask)
                loss = criterion(labels_pred, labels)
                acc1 = accuracy(labels_pred, labels)
                top1_acc_test.update(acc1[0], input_ids.size(0))
                loss_avg_test.update(loss.item(), input_ids.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "image_type":"original",
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": input_ids.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_train_top1_acc_till_current_batch":None,
                     "avg_val_loss_till_current_batch":None,
                     "avg_val_top1_acc_till_current_batch":None,
                     "avg_test_loss_till_current_batch":loss_avg_test.avg,
                     "avg_test_top1_acc_till_current_batch":top1_acc_test.avg},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_test.set_description(f"test - iteration : {epoch}")
                loop_test.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_test_loss_till_current_batch="{:.4f}".format(loss_avg_test.avg),
                    top1_accuracy_test="{:.4f}".format(top1_acc_test.avg),
                    refresh=True,
                )
        lr_scheduler.step()
        report.to_csv(f"{report_path}/{model_name}_report.csv")
    return model, optimizer, report





batch_size = yml['batch_size']
epochs = yml['num_epochs']
learning_rate = float(yml['learning_rate'])
gamma=yml['gamma']
step_size=yml['step_size']
ckpt_save_freq = yml['ckpt_save_freq']
outputlayer = yml['outputlayer']
pooler = yml['pooler']
output_attentions = yml['output_attentions']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
custom_model = BERTClassification(model_name,outputlayer,pooler,output_attentions)
print(custom_model)

from dataloaders.DataLoader import *

train_dataset = MyDataset("Train")
val_dataset = MyDataset("Val")
test_dataset = MyDataset("Test")



train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

trainer = train(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model = custom_model,
    model_name="model_name",
    epochs=epochs,
    learning_rate=learning_rate,
    gamma = gamma,
    step_size = step_size,
    device=device,
    load_saved_model=False,
    ckpt_save_freq=ckpt_save_freq,
    ckpt_save_path=yml['ckpt_save_path'],
    ckpt_path=yml['ckpt_path'],
    report_path=yml['report_path'],
)


