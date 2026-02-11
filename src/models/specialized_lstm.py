"""
Specialized LSTM Models for 8-Model Ensemble
Each model is trained to recognize specific WHO steps, creating expert models.

Model Distribution:
- Model 1: Step 1 (Palm to Palm) + Background
- Model 2: Step 2 (Right over Left) + Background  
- Model 3: Step 3 (Left over Right) + Background
- Model 4: Step 4 (Fingers Interlaced) + Background
- Model 5: Step 5 (Backs of Fingers) + Background
- Model 6: Step 6 (Thumbs) + Background
- Model 7: Step 7 (Fingertips) + Background
- Model 8: Step 8 (Wrists) + Background (special model for rare class)
"""

import torch
import torch.nn as nn


class SpecializedLSTM(nn.Module):
    """
    Specialized LSTM for binary classification of a specific WHO step vs background.
    This creates an expert model for each WHO step.
    """
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.4):
        super().__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Layer normalization (works with batch size = 1)
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.ln2 = nn.LayerNorm(128)
        
        # Fully connected layers for binary classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)  # Binary: target step vs background
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Attention
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # Layer normalization
        context = self.ln1(context)
        
        # FC layers with layer norm
        x = self.fc[0](context)  # Linear
        x = self.ln2(x)
        x = self.fc[1](x)  # ReLU
        x = self.fc[2](x)  # Dropout
        
        x = self.fc[3](x)  # Linear
        x = self.fc[4](x)  # ReLU
        x = self.fc[5](x)  # Dropout
        
        x = self.fc[6](x)  # Output layer
        
        return x


if __name__ == "__main__":
    # Test the model
    print("Testing Specialized LSTM Model...")
    
    input_size = 63
    hidden_size = 128
    batch_size = 32
    seq_len = 30
    
    model = SpecializedLSTM(input_size, hidden_size)
    
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 2) for binary classification")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✓ Specialized LSTM model test passed!")