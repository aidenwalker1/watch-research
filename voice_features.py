import librosa
import matplotlib.pyplot as plt
import numpy as np
import stat_features
import torch.nn as nn
import torch
import torch.functional
import torch.optim as optim

class CNN(nn.Module) :
    def __init__(self) :
        super(CNN, self).__init__()
         # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 5)  # Assuming 10 output classes
        self.sm = nn.Softmax()
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Linear(nn.Module) :
    def __init__(self, num_features) :
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 5)  # Assuming 10 output classes
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x) :
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
    



y, sr = librosa.load('./test.mp3')
# Get RMS value from each frame's magnitude value
S, phase = librosa.magphase(librosa.stft(y))
# Compute local onset autocorrelation
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
rms = librosa.feature.rms(S=S)
melspec = librosa.feature.melspectrogram(S=S)
chroma = librosa.feature.chroma_stft(S=S)
spectral = librosa.feature.spectral_centroid(S=S)
tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                           hop_length=512)[0]

f1 = stat_features.generate_statistical_features(rms[0])

f3s = []

for i in range(chroma.shape[0]) :
    f3s.append(stat_features.generate_statistical_features(chroma[i]))
f3s = np.mean(f3s, axis=0)
f4 = stat_features.generate_statistical_features(spectral[0])

f5 = tempo

features = f1 + f3s.tolist() + f4 + [f5]

def avg_arr(arr, to) :
    out = []
    elems = len(arr) // to

    for i in range(len(arr)) :
        x = arr[i]
        out.append([0] * to)
        for j in range(0,to) :
            sum = 0
            for k in range(elems) :
                sum += x[(j*elems) + k]
            out[i][j] = sum
        remain = len(arr) % to
        if remain != 0 :
            sum = 0
            for j in range(remain) :
                sum += x[-j - 1]
            out[i][-1] = sum
    
    return out
            
            
x = avg_arr(melspec.tolist(), 128)

model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = np.array(x)
# num batches x batch size x channels x shape
x = x.reshape(1,1,1,x.shape[0],x.shape[1])
dataloader = x

# Assuming you have your dataset and dataloader defined
# Iterate over the dataset for multiple epochs
num_epochs = 10
labels = np.array([1, 1, 1, 1, 1])
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        #inputs, labels = data
        inputs = torch.as_tensor(data).float()
        labels = torch.as_tensor(labels).float()
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}')
            running_loss = 0.0

print('Training finished.')

#dimensions
#rms = n
#mel = 128 x n
#chroma = 12 x n
#spectral = n
#tempo = 1