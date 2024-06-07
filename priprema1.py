import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score # Import koji ćemo koristiti za izračunavanje krajnje metrike
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# za reproducibilnost
seed=42
torch.manual_seed(seed)

df = pd.read_csv("data/train.csv")

df = df.dropna()

df['zvanje'] = df['zvanje'].map({'AsstProf': 0, 'Prof': 1, 'AssocProf': 2})

df = pd.get_dummies(df, columns=['pol', 'oblast'], drop_first=True)

print(df.head())

train, test = train_test_split(df, test_size=0.3, random_state=42)

x_train = train.drop('zvanje', axis=1)
x_test = test.drop('zvanje', axis=1)

y_train = train['zvanje']
y_test = test['zvanje']

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLPClassifier, self).__init__()
        self.relu = nn.ReLU()
        sizes = [input_size] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.relu(out)
        return out
    
sc = StandardScaler()

x_train[x_train.columns] = sc.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = sc.transform(x_test[x_test.columns])

input_size = x_train.shape[1]
hidden_sizes = [128, 64, 32]
num_classes = len(set(y_train))
model = MLPClassifier(input_size, hidden_sizes, num_classes)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'[{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

with torch.no_grad():
    model.eval()
    outputs = model(x_test)
    _, predicted = torch.max(outputs, 1)
    f1 = f1_score(y_pred=predicted, y_true=y_test, average='micro')
    print(f'F1: {f1}')

# prikaži f1 meru
# f1 = f1_score(y_pred=predicted, y_true=y_test, average='micro')
# print(f'F1: {f1}')
