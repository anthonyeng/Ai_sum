import torch
import torch.nn as nn


class BiLSTMSummarizer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # bidirectional → hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths=None):
        """
        x: (batch, max_seq_len, input_dim)
        lengths: (batch,) actual sequence lengths for packing
        returns: (batch, max_seq_len) importance scores
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x)

        lstm_out = self.dropout(lstm_out)
        scores = self.fc(lstm_out).squeeze(-1)

        return scores
