import torch
import time
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm
from sklearn.metrics import classification_report

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, folder='', folds=None, cv_files=None, dataset_sizes=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    checkpoint = torch.load(folder+"/model_initial.pt", map_location=torch.device('cpu'))
    
    for fold in folds:
        best_auroc = 0.0
        best_average_loss = 100
        best_val_loss = 100
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        folder_fold = folder+"/"+str(fold)
        try:
            os.mkdir(folder_fold)
        except:
            pass
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            train_epoch_loss = 100
            val_epoch_loss = 100
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_scores = []
                running_golds = []
                # Iterate over data.
                for inputs, labels in tqdm(cv_files[fold][phase]):
                    inputs = inputs.to(device)
                    labels = labels.reshape(-1,1).to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, att1, att2 = model(inputs)
                        running_scores += torch.flatten(outputs[:,1]).tolist()
#                         running_scores += torch.flatten(outputs).tolist()
                        running_golds += torch.flatten(labels).tolist()
                        _, preds = torch.max(outputs, 1)
                        preds = torch.reshape(preds, (-1,1)).float()
                        gold = torch.tensor([[1-label,label] for label in labels]).to(device).float()
#                         gold = torch.tensor([[1,0] if label==0 else [0,1] for label in labels]).to(device).float()
                        loss = criterion(outputs.float(), gold.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[fold][phase]
                epoch_auroc = roc_auc_score(running_golds, running_scores)
                print(classification_report(running_golds, pd.Series(running_scores).apply(lambda x: 1 if x>=0.5 else 0).values, labels=[0,1]))
    
                if phase == 'train':
                    train_epoch_loss = epoch_loss
                elif phase == 'val':
                    val_epoch_loss = epoch_loss
                    average_loss = (train_epoch_loss+val_epoch_loss)/2
                    print(f'{phase} Average Loss: {average_loss}')
                    
                print(f'fold: {fold}, {phase} Loss: {epoch_loss:.4f} AUROC: {epoch_auroc}')
                
                # Save the model with highest val AUROC
                if phase == 'val' and best_auroc < epoch_auroc:
                    best_auroc = epoch_auroc
                    best_average_loss = average_loss
                    best_val_loss = val_epoch_loss
#                     best_auroc = epoch_auroc
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, folder_fold+"/epoch_"+str(f"{epoch:02d}")+"loss"+str(val_epoch_loss)[:8]+"_AUROC_"+str(best_auroc)[:8]+".pt")

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        # print(f'Best val Loss: {best_val_loss:4f}')