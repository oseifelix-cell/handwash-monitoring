import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class HandwashLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.5):  # 🔥 Increased dropout
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = AttentionLayer(hidden_size * 2)
        
        # 🔥 NEW: Add Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.6),  # 🔥 Increased dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 🔥 Increased dropout
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        context, _ = self.attention(out)
        
        # Apply batch norm
        context = self.bn1(context)
        
        # First layer
        x = self.fc[0](context)  # Linear
        x = self.bn2(x)
        x = self.fc[1](x)  # ReLU
        x = self.fc[2](x)  # Dropout
        
        # Second layer
        x = self.fc[3](x)  # Linear
        x = self.bn3(x)
        x = self.fc[4](x)  # ReLU
        x = self.fc[5](x)  # Dropout
        
        # Output
        x = self.fc[6](x)  # Linear
        
        return x