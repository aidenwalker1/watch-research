import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import responses as resp
import interactions as inter
import read_data_bio

from sklearn import ensemble

class TransformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, d_model=128, num_heads=8, d_ff=512, dropout=0.1):
        super(TransformerNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Average pooling over time dimension
        x = torch.mean(x, dim=1)
        
        # Final classification layer
        x = self.fc(x)
        return x

class TransformerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, time_length, num_layers, num_heads, dropout):
        super(TransformerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*time_length, 5)  # Output size is 1 for regression
        self.fc2 = nn.Linear(64, 5)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Reshape input tensor to (batch_size, sequence_length, input_size)
        batch_size, sequence_length, input_size = x.size()
        x = x.view(batch_size, sequence_length, input_size)

        # Apply LSTM
        x, _ = self.lstm(x.float())

        # Apply Transformer encoding
        x = self.transformer(x)

        # Flatten and apply fully-connected layer
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x

class Lin(nn.Module):
    def __init__(self, input_size,):
        super(Lin, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Output size is 1 for regression
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input tensor to (batch_size, sequence_length, input_size)
        x = torch.flatten(x.float(), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)

        return x

dyad_no = 1
#all_data = []
total_supreme_data = []

saved = True

minute_data = []
hour_data = []
day_data = []
all_y = []

if not saved:
    for i in range(1,6) :
        for no in [1, 2] :
            dyad_no = i
            dyad = "dyadH0" + str(dyad_no) + "A" + str(no) + "w"
            folder = "dyads/H0" + str(dyad_no) + "/"

            d, l, daily, hour = read_data_bio.read_computed("./clean/" + dyad + "_computed.csv")
            
        
            # data_path = dyad + "_clean_sensor_data.json"
            path = folder + dyad + ".prompt_groups.json"
            # log_path = folder + dyad + ".system_logs.log"

            responses, audio_responses = resp.read_prompts(path)
                
            newdata = read_data_bio.build_data(d, l, hour, daily, responses)
            
            total_supreme_data += newdata

            #all_data += d
            #all_y += converted


            # interactions = inter.get_interactions(log_path, responses)
            # freqs, home_cords = read_data_bio.find_home(data_path, 24000)

            # data = read_data_bio.read_data(data_path, responses, interactions, 24000, home_cords)
            # all_data += data
    minute_data = [total_supreme_data[i][0] for i in range(len(total_supreme_data))]
    hour_data = [total_supreme_data[i][1] for i in range(len(total_supreme_data))]
    day_data = [total_supreme_data[i][2] for i in range(len(total_supreme_data))]
    all_y = [total_supreme_data[i][3] for i in range(len(total_supreme_data))]
    np.save('./computed_data/minute_data', np.array(minute_data))
    np.save('./computed_data/hour_data', np.array(hour_data))
    np.save('./computed_data/day_data', np.array(day_data))
    np.save('./computed_data/all_y', np.array(all_y))

minute_data = np.load('./computed_data/minute_data.npy')
hour_data = np.load('./computed_data/hour_data.npy')
day_data = np.load('./computed_data/day_data.npy')
all_y = np.load('./computed_data/all_y.npy').tolist()

selected_ema = 0

selected_data = day_data.tolist()

y = [all_y[i][selected_ema] for i in range(len(all_y))]
i = 2

while i < len(y) :
    if y[i] == -1 :
        del y[i]
        del selected_data[i]
    else :
        i += 1

selected_data = np.array(selected_data)
print('done reading')

# prepare your data, assuming you have X of shape (100, 30, 5) and y of shape (100, 5)
X = np.array(selected_data)
#X = [all_data[i][0] for i in range(len(all_data))]
X = torch.as_tensor(X)
X = torch.DoubleTensor(X)
y = torch.as_tensor(y)

# split the data into training and validation sets
train_size = int(0.5 * len(y))
train_X, val_X = X[:train_size], X[train_size:]
train_y, val_y = y[:train_size], y[train_size:]

# create DataLoader objects for training and validation sets
train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=3)

# Create the model
#model = BiometricTransformer(input_dim=121, hidden_dim=64,num_classes=5,num_layers=10,dropout=0.1)
#model = TransformerModel(121, 64, 4, 8, 5, 0.1)

# Example usage
input_size = 270  # Number of features
hidden_size = 128  # LSTM hidden size
num_layers = 8  # Number of LSTM layers
num_heads = 16  # Number of attention heads in the Transformer
dropout = 0.25  # Dropout rate
time_length = 4

#model = TransformerLSTM(input_size, hidden_size, time_length, num_layers, num_heads, dropout)
model = Lin(270)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 50
for epoch in range(num_epochs):
    # train for one epoch
    train_loss = 0
    train_acc = 0
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.shape[0]
        train_acc += (y_pred.argmax(dim=1) == y_batch).sum().item()
    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)

    # evaluate on validation set
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.shape[0]
            val_acc += (y_pred.argmax(dim=1) == y_batch).sum().item()
    val_loss /= len(val_dataset)
    val_acc /= len(val_dataset)

    # print training progress
    print(f"Epoch {epoch + 1}/{num_epochs}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")