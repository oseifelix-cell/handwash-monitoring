import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    """Attention mechanism to focus on important temporal features"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context, attention_weights


class BaselineLSTM(nn.Module):
    """
    Baseline LSTM model with attention mechanism.
    Hyperparameters matched to ensemble configuration.
    """
    def __init__(self, input_size, hidden_size=128, num_classes=9, num_layers=2, dropout=0.5):
        super().__init__()
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer (input is hidden_size * 2 due to bidirectional)
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Fully connected layers with dropout matching ensemble
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.6),   # Increased to match ensemble
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),   # Increased to match ensemble
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        context, attention_weights = self.attention(lstm_out)  # (batch, hidden_size*2)
        
        # Batch normalization
        context = self.bn1(context)
        
        # First FC layer
        x = self.fc[0](context)  # Linear
        x = self.bn2(x)
        x = self.fc[1](x)  # ReLU
        x = self.fc[2](x)  # Dropout
        
        # Second FC layer
        x = self.fc[3](x)  # Linear
        x = self.bn3(x)
        x = self.fc[4](x)  # ReLU
        x = self.fc[5](x)  # Dropout
        
        # Output layer
        x = self.fc[6](x)  # Linear
        
        return x
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            _, attention_weights = self.attention(lstm_out)
        return attention_weights


if __name__ == "__main__":
    # Test the updated baseline
    input_size = 63
    hidden_size = 128
    num_classes = 9
    batch_size = 32
    seq_len = 30

    model = BaselineLSTM(input_size, hidden_size, num_classes)
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
