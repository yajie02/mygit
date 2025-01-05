# imports
import snntorch as snn
from snntorch import surrogate

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter

# plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# metric (AUC, ROC, sensitivity & specificity)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Step 1: Prepare the dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        with open(csv_file,"r") as f:
            self.data = pd.read_csv(f) # read the files
        self.labels = self.data.iloc[:,0].values - 1 # set the first line of the input data as the label (Originally 1 or 2, but we -1 here so they become 0 or 1)
        self.features = self.data.iloc[:, 1:].values # set the rest of the input data as the feature (FLUX over time)
        self.transform = transform # transformation (which is None) that will be applied to samples.

        # If you want to have a look at how does this dataset look like with pandas,
        # you can enable the line below.
        # print(data.head(5))

    def __len__(self): # function that gives back the size of the dataset (how many samples)
        return len(self.labels)

    def __getitem__(self, idx): # retrieves a data sample from the dataset
        label = self.labels[idx] # fetch label of sample
        feature = self.features[idx] # fetch features of sample

        if self.transform: # if there is a specified transformation, transform the data
            feature = self.transform(feature)

        sample = {'feature': feature, 'label': label}
        return sample

train_dataset = CustomDataset('./exo_data/exoTrain.csv') # grab the training data
test_dataset = CustomDataset('./exo_data/exoTest.csv') # grab the test data
# print(train_dataset.__getitem__(37));

# print("Class distribution in the original training dataset:", pd.Series(train_dataset.labels).value_counts())
# print("Class distribution in the original testing dataset:", pd.Series(test_dataset.labels).value_counts())
#
# label_counts = np.bincount(train_dataset.labels)
# label_names = ['Not Exoplanet','Exoplanet']

# plt.figure(figsize=(8, 6))
# plt.pie(label_counts, labels=label_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
# plt.title('Distribution of Positive and Negative Samples in the Training Dataset')
# plt.show()

# Step 2: Apply SMOTE to deal with the unbalanced data
smote = SMOTE(sampling_strategy='all') # initialize a smote, while sampling_strategy='all' means setting all the classes to the same size
train_dataset.features, train_dataset.labels = smote.fit_resample(train_dataset.features, train_dataset.labels) # update the labels and features to the resampled data

# print("Class distribution in the training dataset after SMOTE:", pd.Series(train_dataset.labels).value_counts())
# print("Class distribution in the testing dataset after SMOTE:", pd.Series(test_dataset.labels).value_counts())
#
# label_counts = np.bincount(train_dataset.labels)
# label_names = ['Not Exoplanet','Exoplanet']

# plt.figure(figsize=(8, 6))
# plt.pie(label_counts, labels=label_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
# plt.title('Distribution of Positive and Negative Samples')
# plt.show()

# Step 3: Create dataloader
batch_size = 64 # determines the number of samples in each batch during training
spike_grad = surrogate.fast_sigmoid(slope=25) #
beta = 0.5 # initialize a beta value of 0.5
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # create a dataloader for the trainset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True) # create a dataloader for the testset

# print(train_dataset.data.head(1))

# fig = make_subplots(rows=2, cols=2,subplot_titles=("Star #0 (Exoplanet)", "Star #1 (Exoplanet)",
#                                                    "Star #3000 (No-Exoplanet)", "Star #3001 (No-Exoplanet)"))
# fig.add_trace(
#     go.Scatter(y=train_dataset.__getitem__(0)['feature']),
#     row=1, col=1
# )
# fig.add_trace(
#     go.Scatter(y=train_dataset.__getitem__(1)['feature']),
#     row=1, col=2
# )
# fig.add_trace(
#     go.Scatter(y=train_dataset.__getitem__(3000)['feature']),
#     row=2, col=1
# )
# fig.add_trace(
#     go.Scatter(y=train_dataset.__getitem__(3001)['feature']),
#     row=2, col=2
# )
# for i in range(1, 5):
#     fig.update_xaxes(title_text="Time", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
#     fig.update_yaxes(title_text="Flux", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
#
# fig.update_layout(height=600, width=800, title_text="Exoplanets Flux vs No-Exoplanet Flux",showlegend=False)
# fig.show()

# Step 4: Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers (3 linear layers and 3 leaky layers)
        self.fc1 = nn.Linear(3197, 128) # takes an input of 3197 and outputs 128
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(64, 64) # takes an input of 64 and outputs 68
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(32, 2) # takes in 32 inputs and outputs our two outputs (planet with/without an exoplanet)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = F.max_pool1d(self.fc1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool1d(self.fc2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc3(spk2.view(batch_size, -1))

        # return cur3
        return cur3

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = Net() # initialize the model to the new class.

# Step 5: Define the Loss function and the Optimizer
criterion = nn.CrossEntropyLoss()  # look up binarycross entropy if we have time
optimizer = optim.SGD(model.parameters(), lr=0.001) # stochastic gradient descent with a learning rate of 0.001

# create a pandas dataframe to hold the current epoch, the accuracy， sensitivity, specificity, auc-roc and loss
results = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC-ROC', 'Test Loss'])

num_epochs = 100 # initialize a certain number of epoch iterations

for epoch in range(num_epochs): # iterate through num_epochs
    model.train() # forward pass
    for data in train_dataloader: # iterate through every data sample
        inputs, labels = data['feature'].float(), data['label']  # Float
        optimizer.zero_grad() # clear previously stored gradients
        outputs = model(inputs) #
        loss = criterion(outputs, labels) # calculates the difference (loss) between actual values and predictions
        loss.backward() # backward pass on the loss
        optimizer.step() # updates parameters

    # Test Set, evaluate the model every epoch
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        all_probs = []
        for data in test_dataloader:
            inputs, labels = data['feature'].float(), data['label']
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())


            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(outputs)[:, 1]  # Assuming 1 represents the positive class
            all_probs.extend(probabilities.cpu().numpy())
        # output the accuracy (even though it is not very useful in this case)
        accuracy = 100 * correct / total
        # calculate teat loss
        # test_loss =
        # initialize a confusing matrix
        cm = confusion_matrix(all_labels, all_predicted)
        # grab the amount of true negatives and positives, and false negatives and positives.
        tn, fp, fn, tp = cm.ravel()
        # calculate sensitivity
        sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # calculate specificity
        specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0.0
        # calculate AUC-ROC
        auc_roc = 100 * roc_auc_score(all_labels, all_probs)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] Test Loss: {test_loss / len(test_dataloader):.2f} '
            f'Test Accuracy: {accuracy:.2f}% Sensitivity: {sensitivity:.2f}% Specificity: {specificity:.2f}% AUC-ROC: {auc_roc:.4f}%'
        )

        results = results._append({
            'Epoch': epoch + 1,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Test Loss': test_loss / len(test_dataloader),
            'AUC-ROC': auc_roc
        }, ignore_index=True)

