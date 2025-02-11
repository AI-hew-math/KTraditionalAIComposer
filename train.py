import torch
import torch.nn as nn
import torch.optim as optim


def train(model,
          n_epochs, learning_rate, weight_decay,
          device, train_loader):

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    idx_of_PAD_token = 0
    criterion_pitch = nn.CrossEntropyLoss(ignore_index=idx_of_PAD_token)
    criterion_octave = nn.CrossEntropyLoss(ignore_index=idx_of_PAD_token)
    criterion_duration = nn.CrossEntropyLoss(ignore_index=idx_of_PAD_token)
    model.train()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            pitch_batch, octave_batch, duration_batch = batch

            # 텐서를 device로 이동 (각 텐서 shape: [batch_size, max_length])
            pitch_batch = pitch_batch.to(device)
            octave_batch = octave_batch.to(device)
            duration_batch = duration_batch.to(device)

            # 입력과 타겟 시퀀스 구성:
            # 입력: 시퀀스의 처음 ~ (T-1)번째 토큰 [<SOS>, first_note, ...., last_note]
            # 타겟: 시퀀스의 1번째 ~ T번째 토큰 (즉, 한 스텝 뒤의 정답) [first_note, ..., last_note, <EOS>]
            input_pitch = pitch_batch[:, :-1]      # (batch_size, max_length-1)
            target_pitch = pitch_batch[:, 1:]      # (batch_size, max_length-1)

            input_octave = octave_batch[:, :-1]
            target_octave = octave_batch[:, 1:]

            input_duration = duration_batch[:, :-1]
            target_duration = duration_batch[:, 1:]

            optimizer.zero_grad()

            # 모델에 입력을 전달하여 각 채널별 예측 로짓을 얻습니다.
            # 각 출력의 shape:
            #   output_pitch: (batch_size, seq_len, pitch_vocab_size)
            #   output_octave: (batch_size, seq_len, octave_vocab_size)
            #   output_duration: (batch_size, seq_len, duration_vocab_size)
            output_pitch, output_octave, output_duration, _ = model(input_pitch, input_octave, input_duration)

            loss_pitch = criterion_pitch(output_pitch.view(-1, pitch_vocab_size), target_pitch.view(-1))
            loss_octave = criterion_octave(output_octave.view(-1, octave_vocab_size), target_octave.view(-1))
            loss_duration = criterion_duration(output_duration.view(-1, duration_vocab_size), target_duration.view(-1))

            loss = loss_pitch + loss_octave + loss_duration
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    return model



