"""
LSTM model architecture for audio classification.
"""

import torch
import torch.nn as nn


class AudioLSTM(nn.Module):
    """
    LSTM network for audio classification.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Number of hidden units
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
        bidirectional (bool): Use bidirectional LSTM
    """
    
    def __init__(self, input_size=40, hidden_size=256, num_layers=2, 
                 num_classes=50, dropout=0.3, bidirectional=True):
        super(AudioLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        output = self.classifier(last_output)
        return output
