import torch
import torch.nn as nn


class GukAkLSTM(nn.Module):
    def __init__(self,
                 pitch_vocab_size, octave_vocab_size, duration_vocab_size,
                 pitch_embed_dim=16, octave_embed_dim=9, duration_embed_dim=19,
                 hidden_size=16, num_layers=2):
        """
        각 채널별 임베딩 후, 이들을 concat하여 LSTM에 입력하고,
        LSTM의 출력으로부터 피치, 옥타브, duration 각 채널에 대한 예측을 수행합니다.
        """
        super(GukAkLSTM, self).__init__()
        # 채널별 임베딩 레이어
        self.pitch_embedding = nn.Embedding(pitch_vocab_size, pitch_embed_dim)
        self.octave_embedding = nn.Embedding(octave_vocab_size, octave_embed_dim)
        self.duration_embedding = nn.Embedding(duration_vocab_size, duration_embed_dim)

        # 각 채널의 임베딩 벡터를 concat하여 LSTM의 입력으로 사용
        input_dim = pitch_embed_dim + octave_embed_dim + duration_embed_dim
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)

        # LSTM의 은닉 상태로부터 각 채널별로 예측을 수행하기 위한 FC 헤드들
        self.fc_pitch = nn.Linear(hidden_size, pitch_vocab_size)
        self.fc_octave = nn.Linear(hidden_size, octave_vocab_size)
        self.fc_duration = nn.Linear(hidden_size, duration_vocab_size)

    def forward(self, pitch_seq, octave_seq, duration_seq, hidden=None):
        """
        입력:
          - pitch_seq: (batch, seq_length) 텐서 (피치 인덱스)
          - octave_seq: (batch, seq_length) 텐서 (옥타브 인덱스)
          - duration_seq: (batch, seq_length) 텐서 (duration 인덱스)
        출력:
          - output_pitch: (batch, seq_length, pitch_vocab_size)
          - output_octave: (batch, seq_length, octave_vocab_size)
          - output_duration: (batch, seq_length, duration_vocab_size)
          - hidden: LSTM의 마지막 hidden state (생성 시 사용)
        """
        # 각 채널별 임베딩
        pitch_emb = self.pitch_embedding(pitch_seq)         # (batch, seq_length, pitch_embed_dim)
        octave_emb = self.octave_embedding(octave_seq)         # (batch, seq_length, octave_embed_dim)
        duration_emb = self.duration_embedding(duration_seq)   # (batch, seq_length, duration_embed_dim)

        # 세 채널을 concat: (batch, seq_length, input_dim)
        x = torch.cat([pitch_emb, octave_emb, duration_emb], dim=-1)

        # LSTM에 입력: x의 출력 shape -> (batch, seq_length, hidden_size)
        x, hidden = self.lstm(x, hidden)

        # 각 채널별 Fully Connected 헤드를 통과하여 예측 수행
        output_pitch = self.fc_pitch(x)       # (batch, seq_length, pitch_vocab_size)
        output_octave = self.fc_octave(x)       # (batch, seq_length, octave_vocab_size)
        output_duration = self.fc_duration(x)   # (batch, seq_length, duration_vocab_size)

        return output_pitch, output_octave, output_duration, hidden
