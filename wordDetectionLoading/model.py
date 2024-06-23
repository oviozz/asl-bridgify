
import torch
import torch.nn as nn
import json


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.fc1 = nn.Linear(hidden_size3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.relu(self.fc1(x[:, -1, :]))  # Take the output of the last time step
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


input_size = 84
hidden_size1 = 64
hidden_size2 = 128
hidden_size3 = 64
output_size = 420

model = LSTMModel(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
model.load_state_dict(torch.load('wordDetectionLoading/lstm_model_weights.pth'))
model.eval()


def callModel(handPoses):
    single_example_tensor = torch.tensor(handPoses, dtype=torch.float32)
    single_example_tensor = single_example_tensor.unsqueeze(0)  # Shape: (1, 200, 84)
    model.eval()
    with torch.no_grad():
        output = model(single_example_tensor)
        _, predicted = torch.max(output, 1)
        with open('wordDetectionLoading/id_to_gloss_mapping.json', 'r') as file:
            data = json.load(file)
        value = predicted.tolist()[0]
        predictedWord = data[str(value)]

        return predictedWord


randTensor = torch.randn(200, 84)
print(callModel(randTensor))





