import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        # features: (batch, seq_len, input_dim)
        outputs, hidden = self.gru(features)
        # outputs: (batch, seq_len, hidden_dim * 2)
        outputs = self.dropout(outputs)
        # Combine last layer forward + backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_dim * 2)
        hidden = torch.tanh(self.fc(hidden))                 # (batch, hidden_dim)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch, seq_len, encoder_dim)
        # decoder_hidden:  (batch, decoder_dim)
        enc_att = self.encoder_att(encoder_outputs)           # (batch, seq_len, attn_dim)
        dec_att = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch, 1, attn_dim)
        scores = self.full_att(torch.tanh(enc_att + dec_att)).squeeze(2)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)                    # (batch, seq_len)
        context = (encoder_outputs * weights.unsqueeze(2)).sum(dim=1)     # (batch, encoder_dim)
        return context, weights


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, hidden_dim, attention_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)
        self.gru = nn.GRU(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, encoder_outputs):
        # token:            (batch,)
        # hidden:           (1, batch, hidden_dim)
        # encoder_outputs:  (batch, seq_len, encoder_dim)
        embedded = self.dropout(self.embedding(token))           # (batch, embed_dim)
        context, weights = self.attention(encoder_outputs, hidden.squeeze(0))
        gru_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # (batch, 1, embed+enc)
        output, hidden = self.gru(gru_input, hidden)
        prediction = self.fc_out(self.dropout(output.squeeze(1)))  # (batch, vocab_size)
        return prediction, hidden, weights


class VideoCaptionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        encoder_hidden=512,
        decoder_hidden=512,
        attention_dim=256,
        input_dim=512,
        encoder_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        self.encoder = VideoEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
        )
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_hidden * 2,   # bidirectional
            hidden_dim=decoder_hidden,
            attention_dim=attention_dim,
            dropout=dropout,
        )

    def forward(self, features, captions, teacher_forcing_ratio=0.5):
        # features: (batch, seq_len, input_dim)
        # captions: (batch, max_len)  — token ids incl. <sos> and <eos>
        batch_size = features.size(0)
        max_len = captions.size(1)
        vocab_size = self.decoder.fc_out.out_features

        encoder_outputs, hidden = self.encoder(features)
        hidden = hidden.unsqueeze(0)   # (1, batch, hidden_dim)

        outputs = torch.zeros(batch_size, max_len, vocab_size, device=features.device)
        input_token = captions[:, 0]   # <sos>

        for t in range(1, max_len):
            pred, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = pred
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            input_token = captions[:, t] if use_teacher else pred.argmax(1)

        return outputs
