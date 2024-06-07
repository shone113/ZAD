import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from sklearn.feature_extraction.text import CountVectorizer

# za reproducibilnost
seed=42
torch.manual_seed(seed)

# Učitavanje podataka i enkodovanje labela (Cartman - 1, Stan - 2 ...)
df = pd.read_csv('./data/south_park_train.csv')

df = df.dropna()
df['Character'] = LabelEncoder().fit_transform(df['Character'])

# print(df.head())

x_train, x_test, y_train, y_test = train_test_split(df['Line'], df['Character'], test_size=0.3, random_state=42)

cv = CountVectorizer(stop_words="english")

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

torch.manual_seed(42)

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
    
input_size = x_train.shape[1]
hidden_sizes = [64, 32]
num_classes = len(set(y_train))
model = MLPClassifier(input_size, hidden_sizes, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x_train = torch.tensor(x_train.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
x_test = torch.tensor(x_test.toarray(), dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0 :
        print(f'Epoch[{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

#Evaluacija
with torch.no_grad():
    model.eval()
    outputs = model(x_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy}')
# prikaži meru tačnosti
# from sklearn.metrics import accuracy_score
# print(f'Accuracy: {accuracy_score(y_pred, y_test)}')
