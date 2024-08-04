import torch
import numpy as np
import pandas as pd

df = pd.read_csv(f'{out_dir}/logs.csv')


###### Loss and accuracy plotting ######
import matplotlib.pyplot as plt

plt.plot(df.epoch, df.loss, label='Training Loss')
plt.plot(df.epoch, df.val_loss, label='Validation Loss')
 
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.plot(df.epoch, df.accuracy, label='Training accuracy')
plt.plot(df.epoch, df.val_accuracy, label='Validation accuracy')
 
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


###### sklearn classification report ######

out_dir = './models/clinical'

preds = []
labels = []

model = torch.load(f'{out_dir}/best.pt')
model.eval() 
for x,y in iter(test_dataloader): 
    pred = model(x) 
    preds.append(pred.detach().cpu())
    labels.append(y)


preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)

preds = np.argmax(preds, axis=1)
labels = np.argmax(labels, axis=1)


from sklearn.metrics import classification_report
report = classification_report(y_true=labels, y_pred=preds, output_dict=True)

df = pd.DataFrame(report).transpose()
df.to_csv('file_name.csv')
print(report)


###### sklearn confusion matrix ######
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_true=labels,
                      y_pred=preds)

plt.Figure()
disp = ConfusionMatrixDisplay(cm,)

disp.plot(cmap=plt.cm.Blues, values_format='')
plt.savefig('confusion_matrix.jpg')