import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

class GukAkDataset(Dataset):
    def __init__(self, Song_informations,
                 pitch_token_to_idx,
                 octave_token_to_idx,
                 duration_token_to_idx,
                 pad=True,
                 max_length=300):

        self.samples = []
        self.pad = pad
        self.max_length = max_length
        self.pitch_token_to_idx = pitch_token_to_idx
        self.octave_token_to_idx = octave_token_to_idx
        self.duration_token_to_idx = duration_token_to_idx
        self.Song_informations = Song_informations

        if self.pad and self.max_length is None:
            raise ValueError("max_length must be provided if pad is True.")


        self.pitch_pad_idx = pitch_token_to_idx["<PAD>"]
        self.octave_pad_idx = octave_token_to_idx["<PAD>"]
        self.duration_pad_idx = duration_token_to_idx["<PAD>"]

        for song_info in self.Song_informations:
            # predefined 토큰화 함수 호출: 각 채널의 토큰 시퀀스를 반환합니다.
            pitch_tokens, octave_tokens, duration_tokens = song_info['Song_pitch'], song_info['Song_octave'], song_info['Song_duration']

            # 토큰을 정수 인덱스로 변환 (없는 토큰은 패딩 토큰으로 대체)
            pitch_indices = [pitch_token_to_idx.get(tok, self.pitch_pad_idx) for tok in pitch_tokens]
            octave_indices = [octave_token_to_idx.get(tok, self.octave_pad_idx) for tok in octave_tokens]
            duration_indices = [duration_token_to_idx.get(tok, self.duration_pad_idx) for tok in duration_tokens]

            # pad 옵션이 True이면 고정 길이로 패딩 혹은 잘라내기
            if self.pad:
                pitch_indices = self._pad_sequence(pitch_indices, self.max_length, self.pitch_pad_idx)
                octave_indices = self._pad_sequence(octave_indices, self.max_length, self.octave_pad_idx)
                duration_indices = self._pad_sequence(duration_indices, self.max_length, self.duration_pad_idx)

            self.samples.append((
                torch.tensor(pitch_indices, dtype=torch.long),
                torch.tensor(octave_indices, dtype=torch.long),
                torch.tensor(duration_indices, dtype=torch.long)
            ))

    def _pad_sequence(self, seq, target_length, pad_token):
        """시퀀스의 길이를 target_length로 맞춥니다.
           target_length보다 짧으면 pad_token을 뒤쪽에 추가, 길면 잘라냅니다."""
        if len(seq) < target_length:
            return seq + [pad_token] * (target_length - len(seq))
        else:
            return seq[:target_length]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]