import os
import torch
from tqdm import tqdm 

class Trainer:
    def __init__(self, model, opt, criterion, train_dataloader, test_dataloader, out_dir, epochs, scheduler,device):
        self.model = model
        self.opt = opt
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.out_dir = out_dir
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
        

    def fit(self):
        with open(f'{self.out_dir}/logs.csv', 'w') as file:
            file.write('epoch,loss,accuracy,best_accuracy,val_loss,val_accuracy,best_val_accuracy\n')
        
        best_accuracy = 0.0 
        best_eval_accuracy = 0.0
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train() 
            
            running_loss = 0.0  
            running_acc = 0.0
            
            running_eval_loss = 0.0 
            running_eval_acc = 0.0
        
            for x,y in iter(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.opt.zero_grad() 
                
                pred = self.model(x) 
                loss = self.criterion(pred,y) 
                loss.backward() 
                self.opt.step() 
                
                running_loss += loss.detach() 
                
                accuracy = torch.div(torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)),
                                     y.size(dim=0)) 
                
                running_acc += accuracy 
                    
            running_loss = running_loss/len(self.train_dataloader) 
            running_acc = running_acc/len(self.train_dataloader)
        
            if running_acc > best_accuracy: 
                    best_accuracy = running_acc
            
            self.model.eval()
            for x,y in iter(self.test_dataloader): 
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(x) 
                loss = self.criterion(pred,y) 
                
                running_eval_loss += loss.detach()
                
                accuracy = torch.div(torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)),
                                     y.size(dim=0)) 
                running_eval_acc += accuracy
            
            running_eval_loss = running_eval_loss/len(self.test_dataloader)
            running_eval_acc = running_eval_acc/len(self.test_dataloader)
        
            if running_eval_acc>= best_eval_accuracy:
                    best_eval_accuracy = running_eval_acc
                    torch.save(self.model, f'{self.out_dir}/best.pt') 
        
            
            print(f'EPOCH {epoch+1}: Loss: {running_loss}, Accuracy: {running_acc}, Best_accuracy: {best_accuracy}')
            print(f'Val_loss: {running_eval_loss}, Val_accuracy: {running_eval_acc}, Best_val_accuracy: {best_eval_accuracy}')
            print()
                
            with open(f'{self.out_dir}/logs.csv', 'a') as file:
                file.write(f'{epoch+1},{running_loss},{running_acc},{best_accuracy},{running_eval_loss},{running_eval_acc},{best_eval_accuracy}\n')
                  
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(running_eval_loss)
            else:
                self.scheduler.step()
                
        torch.save(self.model, f'{self.out_dir}/last.pt') 
                


####### RoBERTa Trainer #########
class Trainer:
    def __init__(self, model, opt, criterion, train_dataloader,
                 test_dataloader, out_dir, epochs, scheduler,
                 device, tokenizer, pretrain=False):
        self.pretrain = pretrain
        self.model = model
        self.opt = opt
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.out_dir = out_dir
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def fit(self):
        if self.pretrain:
            for param in model.roberta.parameters():
               param.requires_grad = False
        else:
            for param in model.roberta.parameters():
               param.requires_grad = True
            
        
        with open(f'{self.out_dir}/logs.csv', 'w') as file:
            file.write('epoch,loss,accuracy,best_accuracy,val_loss,val_accuracy,best_val_accuracy\n')
            
        self.tokenizer.save_pretrained('./best_trained_model/')
        
        best_accuracy = 0.0 
        best_eval_accuracy = 0.0
        
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train() 
            
            running_loss = 0.0  
            running_acc = 0.0
            
            running_eval_loss = 0.0 
            running_eval_acc = 0.0
        
            for batch in self.train_dataloader:
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.opt.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                self.opt.step()
                
                running_loss += loss.detach() 
                
                accuracy = torch.div(torch.sum(torch.argmax(outputs, dim=-1) == torch.argmax(labels, dim=-1)),
                                     labels.size(dim=0))             
                running_acc += accuracy 
                    
            running_loss = running_loss/len(self.train_dataloader) 
            running_acc = running_acc/len(self.train_dataloader)
        
            if running_acc > best_accuracy: 
                    best_accuracy = running_acc
            
            self.model.eval()
            for batch in self.test_dataloader: 
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs, labels)

                    running_eval_loss += loss.detach()
                    accuracy = torch.div(torch.sum(torch.argmax(outputs, dim=-1) == torch.argmax(labels, dim=-1)),
                                     labels.size(dim=0)) 
                    running_eval_acc += accuracy
            
            running_eval_loss = running_eval_loss/len(self.test_dataloader)
            running_eval_acc = running_eval_acc/len(self.test_dataloader)
        
            if running_eval_acc>= best_eval_accuracy:
                best_eval_accuracy = running_eval_acc
                torch.save(self.model, f'{self.out_dir}/best.pt')

            
            print(f'EPOCH {epoch+1}: Loss: {running_loss}, Accuracy: {running_acc}, Best_accuracy: {best_accuracy}')
            print(f'Val_loss: {running_eval_loss}, Val_accuracy: {running_eval_acc}, Best_val_accuracy: {best_eval_accuracy}')
            print()
                
            with open(f'{self.out_dir}/logs.csv', 'a') as file:
                file.write(f'{epoch+1},{running_loss},{running_acc},{best_accuracy},{running_eval_loss},{running_eval_acc},{best_eval_accuracy}\n')
                  
            if self.scheduler:
                if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(running_eval_loss)
                else:
                    self.scheduler.step()
                
        torch.save(self.model, f'{self.out_dir}/last.pt') 










###### Trainer with Tensorboard ######
# start Tensorboard session with:
# tensorboard --logdir runs 

import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1,
                                           patience=10, threshold=0.0001,
                                           cooldown=0, min_lr=0)


def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        
        writer.add_scalar("Loss/train", loss, epoch)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        writer.add_scalar("Learning_rate", scheduler._last_lr[0], epoch)
        

train_model(100)
writer.flush()
writer.close()

###### Trainer with weights and biases ######
import wandb
wandb.login()

epochs=100
batch_size=16
lr=0.001


wandb.init(
    project="example_project",
    name="run_name",
    id='1',
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        })

config = wandb.config

for epoch in range(epochs):
    y1 = model(x)
    loss = criterion(y1, y)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    metrics = {"train/train_loss": loss,
               "Learning_rate" : scheduler._last_lr[0],
                   "train/epoch": epoch}

    val_metrics = {"train/train_loss": loss,
               "Learning_rate" : scheduler._last_lr[0],
                   "train/epoch": epoch}

    wandb.log({**metrics, **val_metrics})
        


torch.save(model, "my_model.pt")
wandb.log_model("./my_model.pt", "my_mnist_model", aliases=[f"epoch-{epoch+1}_loss{loss}"])

wandb.summary['test_accuracy'] = 0.8
wandb.finish()
