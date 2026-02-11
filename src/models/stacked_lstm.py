"""
Stacked LSTM Architecture
Multiple LSTM layers stacked vertically where each layer's output feeds into the next.
This creates hierarchical feature learning.
"""

import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """
    Deep LSTM with stacked layers for hierarchical feature learning.
    
    Architecture:
    Input → LSTM Layer 1 → LSTM Layer 2 → ... → LSTM Layer N → Attention → FC → Output
    
    Each layer learns different levels of abstraction:
    - Layer 1: Low-level features (hand position, movement direction)
    - Layer 2: Mid-level features (gesture patterns)
    - Layer 3+: High-level features (WHO step semantics)
    """
    
    def __init__(self, input_size, hidden_size, num_classes, num_lstm_layers=2, dropout=0.4):
        """
        Args:
            input_size: Number of input features (63 for hand landmarks)
            hidden_size: Hidden units in each LSTM layer
            num_classes: Number of output classes (9 for WHO steps)
            num_lstm_layers: Number of stacked LSTM layers (2, 3, 4, or 5)
            dropout: Dropout rate between LSTM layers
        """
        super().__init__()
        
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        
        # Stacked Bidirectional LSTM Layers
        # Each layer's output becomes the next layer's input
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism (operates on final LSTM layer output)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Layer normalization (works with any batch size)
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        
        # Fully connected layers
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
        """
        Forward pass through stacked LSTM layers.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Pass through all stacked LSTM layers
        # Layer 1 output → Layer 2 input → ... → Layer N output
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        
        # Apply attention to final layer's output
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # Layer normalization
        context = self.ln1(context)
        
        # Fully connected layers with layer norm
        x = self.fc[0](context)  # Linear
        x = self.ln2(x)
        x = self.fc[1](x)  # ReLU
        x = self.fc[2](x)  # Dropout
        
        x = self.fc[3](x)  # Linear
        x = self.ln3(x)
        x = self.fc[4](x)  # ReLU
        x = self.fc[5](x)  # Dropout
        
        x = self.fc[6](x)  # Output layer
        
        return x
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
        return attention_weights


class StackedLSTMConfig:
    """Configuration presets for different stacking depths"""
    
    @staticmethod
    def get_config(num_layers):
        """
        Get optimal configuration for a given number of layers.
        
        Deeper networks need:
        - Lower learning rate (to prevent exploding gradients)
        - Higher dropout (to prevent overfitting)
        - More training epochs (to converge)
        """
        configs = {
            2: {
                'num_lstm_layers': 2,
                'hidden_size': 128,
                'learning_rate': 0.001,
                'dropout': 0.4,
                'batch_size': 32,
                'num_epochs': 50
            },
            3: {
                'num_lstm_layers': 3,
                'hidden_size': 128,
                'learning_rate': 0.0008,
                'dropout': 0.45,
                'batch_size': 32,
                'num_epochs': 60
            },
            4: {
                'num_lstm_layers': 4,
                'hidden_size': 112,
                'learning_rate': 0.0006,
                'dropout': 0.5,
                'batch_size': 32,
                'num_epochs': 70
            },
            5: {
                'num_lstm_layers': 5,
                'hidden_size': 96,
                'learning_rate': 0.0005,
                'dropout': 0.55,
                'batch_size': 32,
                'num_epochs': 80
            }
        }
        return configs.get(num_layers, configs[2])


if __name__ == "__main__":
    print("="*70)
    print("TESTING STACKED LSTM ARCHITECTURES")
    print("="*70)
    
    input_size = 63
    num_classes = 9
    batch_size = 32
    seq_len = 30
    
    # Test different depths
    for num_layers in [2, 3, 4, 5]:
        print(f"\n{'='*70}")
        print(f"Testing {num_layers}-Layer Stacked LSTM")
        print(f"{'='*70}")
        
        config = StackedLSTMConfig.get_config(num_layers)
        
        model = StackedLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_classes=num_classes,
            num_lstm_layers=num_layers,
            dropout=config['dropout']
        )
        
        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nConfiguration:")
        print(f"  LSTM Layers: {num_layers}")
        print(f"  Hidden Size: {config['hidden_size']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Dropout: {config['dropout']}")
        
        print(f"\nModel Statistics:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Estimate model size
        model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
        print(f"  Estimated size: {model_size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)