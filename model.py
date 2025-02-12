import torch
import torch.nn as nn


class GukAkLSTM(nn.Module):
    def __init__(self,
                 pitch_vocab_size, octave_vocab_size, duration_vocab_size,
                 pitch_embed_dim=16, octave_embed_dim=9, duration_embed_dim=19,
                 hidden_size=16, num_layers=2):
        """
        After embedding each channel separately, concatenate them to input to the LSTM,
        and make predictions for the pitch, octave, and duration channels from the output of the LSTM.
        """
        super(GukAkLSTM, self).__init__()
        # Embedding layer for each channel
        self.pitch_embedding = nn.Embedding(pitch_vocab_size, pitch_embed_dim)
        self.octave_embedding = nn.Embedding(octave_vocab_size, octave_embed_dim)
        self.duration_embedding = nn.Embedding(duration_vocab_size, duration_embed_dim)

        # Concatenate the embedding vectors of each channel for input to the LSTM
        input_dim = pitch_embed_dim + octave_embed_dim + duration_embed_dim
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)

        # Fully Connected (FC) heads for making predictions for each channel from the LSTM's hidden state
        self.fc_pitch = nn.Linear(hidden_size, pitch_vocab_size)
        self.fc_octave = nn.Linear(hidden_size, octave_vocab_size)
        self.fc_duration = nn.Linear(hidden_size, duration_vocab_size)

    def forward(self, pitch_seq, octave_seq, duration_seq, hidden=None):
        """
        Inputs:
          - pitch_seq: Tensor (batch, seq_length) of pitch indices
          - octave_seq: Tensor (batch, seq_length) of octave indices
          - duration_seq: Tensor (batch, seq_length) of duration indices
        Outputs:
          - output_pitch: (batch, seq_length, pitch_vocab_size)
          - output_octave: (batch, seq_length, octave_vocab_size)
          - output_duration: (batch, seq_length, duration_vocab_size)
          - hidden: Last hidden state of the LSTM (used for generation)
        """
        # Embed each channel
        pitch_emb = self.pitch_embedding(pitch_seq)         # (batch, seq_length, pitch_embed_dim)
        octave_emb = self.octave_embedding(octave_seq)      # (batch, seq_length, octave_embed_dim)
        duration_emb = self.duration_embedding(duration_seq)  # (batch, seq_length, duration_embed_dim)

        # Concatenate all channels: (batch, seq_length, input_dim)
        x = torch.cat([pitch_emb, octave_emb, duration_emb], dim=-1)

        # Input to LSTM: output shape of x -> (batch, seq_length, hidden_size)
        x, hidden = self.lstm(x, hidden)

        # Pass through Fully Connected heads for each channel to make predictions
        output_pitch = self.fc_pitch(x)       # (batch, seq_length, pitch_vocab_size)
        output_octave = self.fc_octave(x)     # (batch, seq_length, octave_vocab_size)
        output_duration = self.fc_duration(x) # (batch, seq_length, duration_vocab_size)

        return output_pitch, output_octave, output_duration, hidden
