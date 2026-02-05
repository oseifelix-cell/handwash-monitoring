"""
Baseline LSTM Model for WHO Handwashing Step Classification
Independent single model for comparison with ensemble approach.
"""

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
    Uses optimal hyperparameters similar to best ensemble configuration.
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.4):
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
        
        # Fully connected layers with progressive dimension reduction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Apply attention
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
        x = self.fc[6](x)  # Linear (no activation, will use CrossEntropyLoss)
        
        return x
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            _, attention_weights = self.attention(lstm_out)
        return attention_weights


if __name__ == "__main__":
    # Test the model
    print("Testing Baseline LSTM Model...")
    
    # Model parameters
    input_size = 63  # 21 landmarks × 3 coordinates
    hidden_size = 128  # Optimal configuration
    num_classes = 9  # WHO steps 0-8
    batch_size = 32
    seq_len = 30
    
    # Create model
    model = BaselineLSTM(input_size, hidden_size, num_classes)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_classes})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nBaseline LSTM model test passed!")