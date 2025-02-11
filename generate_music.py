"""
- generate_music (function) : 학습된 model과 vocabulary 정보를 입력 받아 곡을 생성합니다.
                              후술할 JungGanBoInputUI를 이용해 user_phrase_token를 생성하여
                              입력하면 사용자가 직접 작곡한 음표를 시작으로 하여, 이후에 모델이 곡을 생성할 수 있습니다.

- JungGanBoInputUI (class) : 사용자가 직접 정간보의 음표를 선택해 최대 10개까지의 음표를 입력하면,
                             모델에 입력 가능한 토큰으로 변환하여 출력하는 IPython 위젯 입니다.
                             사용자가 코딩이 아니라, 마우스 클릭으로 쉽게 입력할 수 있는 유저인터페이스입니다.
"""

import torch
import ipywidgets as widgets
from IPython.display import display, clear_output
import torch
import numpy as np

def generate_musics(model,
                   pitch_token_to_idx, octave_token_to_idx, duration_token_to_idx,
                   pitch_idx_to_token, octave_idx_to_token, duration_idx_to_token,
                   device, max_gen_length=300, user_phrase_token=None, user_phrase=None):
    """
    학습된 모델을 이용해 음악을 생성합니다.

    반환:
      - 생성된 피치, 옥타브, duration 토큰 시퀀스 (각각 리스트 형태).
    """
    model.to(device)
    model.eval()


    hidden = None  # LSTM의 초기 hidden state (None이면 0으로 초기화됨)


    # 결과 시퀀스 저장 (초기 <SOS>는 나중에 제외할 수 있음)
    generated_pitch = [pitch_token_to_idx["<SOS>"]]
    generated_octave = [octave_token_to_idx["<SOS>"]]
    generated_duration = [duration_token_to_idx["<SOS>"]]


    if user_phrase_token is not None:
        pitch_token, octave_token, duration_token = user_phrase_token
        token_length = len(pitch_token)
        # 마지막 직전 음표까지는 hidden state만 update 시키고, 생성된 음표는 무시함.
        for i in range(token_length):  # Time-step 별로 순차 입력
            current_pitch = torch.tensor([[pitch_token_to_idx.get(pitch_token[i], pitch_token_to_idx["<PAD>"])]], dtype=torch.long, device=device)
            current_octave = torch.tensor([[octave_token_to_idx.get(octave_token[i], octave_token_to_idx["<PAD>"])]], dtype=torch.long, device=device)
            current_duration = torch.tensor([[duration_token_to_idx.get(duration_token[i], duration_token_to_idx["<PAD>"])]], dtype=torch.long, device=device)

            _, _, _, hidden = model(current_pitch, current_octave, current_duration, hidden)

            # 입력된 user_phrase_token을 그대로 결과에 저장
            generated_pitch.append(pitch_token_to_idx.get(pitch_token[i], pitch_token_to_idx["<PAD>"]))
            generated_octave.append(octave_token_to_idx.get(octave_token[i], octave_token_to_idx["<PAD>"]))
            generated_duration.append(duration_token_to_idx.get(duration_token[i], duration_token_to_idx["<PAD>"]))
            #print(i, current_pitch)

    else:
        token_length = 0
        current_pitch = torch.tensor([[pitch_token_to_idx["<SOS>"]]], dtype=torch.long, device=device)
        current_octave = torch.tensor([[octave_token_to_idx["<SOS>"]]], dtype=torch.long, device=device)
        current_duration = torch.tensor([[duration_token_to_idx["<SOS>"]]], dtype=torch.long, device=device)

    #print(generated_pitch)


    for i in range(max_gen_length):
        # 현재 입력(각 채널)으로 모델 실행. 입력 shape: (1, 1)
        output_pitch, output_octave, output_duration, hidden = model(current_pitch, current_octave, current_duration, hidden)
        # 각 출력의 shape: (1, 1, vocab_size)
        # 마지막 타임스텝의 출력 벡터를 추출하고 squeeze하여 (vocab_size,)로 만듭니다.
        logits_pitch = output_pitch[:, -1, :].squeeze(0)   # shape: (pitch_vocab_size)
        logits_octave = output_octave[:, -1, :].squeeze(0)
        logits_duration = output_duration[:, -1, :].squeeze(0)

        probs_pitch = torch.softmax(logits_pitch, dim=-1)
        probs_octave = torch.softmax(logits_octave, dim=-1)
        probs_duration = torch.softmax(logits_duration, dim=-1)

        # 확률 분포에 따라 다음 토큰 샘플링 (multinomial sampling)
        next_pitch = torch.multinomial(probs_pitch, num_samples=1).item()
        next_octave = torch.multinomial(probs_octave, num_samples=1).item()
        next_duration = torch.multinomial(probs_duration, num_samples=1).item()

        # 생성된 토큰을 결과 시퀀스에 추가
        generated_pitch.append(next_pitch)
        generated_octave.append(next_octave)
        generated_duration.append(next_duration)
        #print(generated_pitch)
        # 피치 채널에서 <EOS> 토큰이 생성되면 종료 (혹은 모든 채널이 <EOS>가 되면 종료)
        if next_pitch == pitch_token_to_idx["<EOS>"]:
            break

        # 다음 반복을 위한 입력 구성 (새로 생성된 토큰을 현재 입력으로 사용)
        current_pitch = torch.tensor([[next_pitch]], dtype=torch.long, device=device)
        current_octave = torch.tensor([[next_octave]], dtype=torch.long, device=device)
        current_duration = torch.tensor([[next_duration]], dtype=torch.long, device=device)

    # 초기 <SOS> 토큰은 제외하고, 생성된 토큰 인덱스를 실제 토큰 문자열로 변환합니다.
    gen_pitch_tokens = [pitch_idx_to_token[idx] for idx in generated_pitch[1:]]
    gen_octave_tokens = [octave_idx_to_token[idx] for idx in generated_octave[1:]]
    gen_duration_tokens = [duration_idx_to_token[idx] for idx in generated_duration[1:]]

    return gen_pitch_tokens, gen_octave_tokens, gen_duration_tokens

class JungGanBoInputUI:
    def __init__(self):
        # 음계 매핑
        self.pitch_mapping = {
            "황종": "E-", "대려": "E", "태주": "F", "협종": "G-",
            "고선": "G", "중려": "A-", "유빈": "A", "임종": "B-",
            "이칙": "B", "남려": "C", "무역": "D-", "응종": "D",
            "쉼표": "rest"  # 쉼표 추가
        }

        # 옵션 리스트
        self.pitch_options = list(self.pitch_mapping.keys())
        self.octave_options = [2, 3, 4, 5, 6]
        self.duration_options = [f"{n}/12" for n in range(1, 37)]

        # 입력 데이터 저장 리스트
        self.current_input = []
        self.generated_data = None  # 생성된 데이터를 저장할 변수

        # UI 위젯 생성
        self.pitch_selector = widgets.Dropdown(options=self.pitch_options, description="음정:")
        self.octave_selector = widgets.Dropdown(options=self.octave_options, description="옥타브:")
        self.duration_selector = widgets.Dropdown(options=self.duration_options, description="정간:")

        self.add_button = widgets.Button(description="➕ 추가")
        self.remove_button = widgets.Button(description="🗑 삭제")
        self.clear_button = widgets.Button(description="🔄 전체 삭제")
        self.generate_button = widgets.Button(description="💾 데이터 생성")

        self.output = widgets.Output()

        # 버튼 동작 설정
        self.add_button.on_click(self.add_entry)
        self.remove_button.on_click(self.remove_entry)
        self.clear_button.on_click(self.clear_entries)
        self.generate_button.on_click(self.generate_data)

        # UI 표시
        display(
            self.pitch_selector, self.octave_selector, self.duration_selector,
            self.add_button, self.remove_button, self.clear_button, self.generate_button,
            self.output
        )
        self.update_display()

    def update_display(self):
        """현재 입력된 값을 출력"""
        with self.output:
            clear_output()
            if self.current_input:
                print("📜 현재 입력된 목록:")
                for i, (p, o, d) in enumerate(self.current_input, 1):
                    print(f"{i}. 음: {p}, 옥타브: {o}, Duration: {d}")
            else:
                print("❗ 아직 입력된 값이 없습니다.")

    def add_entry(self, _):
        """선택한 값을 리스트에 추가 (최대 10개)"""
        if len(self.current_input) < 10:
            pitch = self.pitch_selector.value
            octave = 0 if pitch == "쉼표" else self.octave_selector.value  # 쉼표일 경우 옥타브 0 고정
            self.current_input.append((pitch, octave, self.duration_selector.value))
        else:
            with self.output:
                print("⚠ 최대 10개까지 입력 가능합니다.")
        self.update_display()

    def remove_entry(self, _):
        """마지막 입력값을 삭제"""
        if self.current_input:
            self.current_input.pop()
        self.update_display()

    def clear_entries(self, _):
        """모든 입력값을 초기화"""
        self.current_input.clear()
        self.generated_data = None  # 데이터도 초기화
        self.update_display()

    def generate_music_data(self):
        """음악 데이터를 변환하여 저장"""
        if not self.current_input:
            print("⚠ 입력된 값이 없습니다.")
            return None

        pitch_tokens = [self.pitch_mapping[p] for p, _, _ in self.current_input]
        octave_tokens = [o for _, o, _ in self.current_input]
        duration_tokens = [d for _, _, d in self.current_input]

        self.generated_data = (pitch_tokens, octave_tokens, duration_tokens)  # 내부 변수에 저장
        return self.generated_data

    def generate_data(self, _):
        """버튼 클릭 시 데이터를 생성하고 UI에 출력"""
        data = self.generate_music_data()
        if data:
            pitch_tokens, octave_tokens, duration_tokens = data
            with self.output:
                clear_output()
                print("✅ 데이터가 저장되었습니다!")
                print("🎵 변환된 토큰:")
                print(f"Pitch: {pitch_tokens}")
                print(f"Octave: {octave_tokens}")
                print(f"Duration: {duration_tokens}")

    def get_token(self):
        """객체 내부에 저장된 데이터를 반환"""
        return self.generated_data
        
def remove_special_tokens(generated_music) :

    gen_pitch, gen_octave, gen_duration = generated_music

    gen_pitch = np.array(gen_pitch)
    gen_octave = np.array(gen_octave)
    gen_duration = np.array(gen_duration)


    # 검사할 토큰들, 문자열과 숫자 포함
    tokens = {'<EOS>', '<PAD>', '<SOS>', 0, '0'}

    # 결과 인덱스 집합 초기화
    indices = set()

    # 각 배열에 대해 토큰이 있는지 검사하고 인덱스를 추가
    for array in [gen_pitch, gen_octave, gen_duration]:
        # array가 숫자로 구성된 경우
        if array.dtype.kind in {'i', 'u', 'f'}:  # 데이터 타입이 정수형, 부호 없는 정수형, 실수형인 경우
            numeric_tokens = {int(tok) for tok in tokens if isinstance(tok, int) or (isinstance(tok, str) and tok.isdigit())}
            indices.update(np.where(np.isin(array, list(numeric_tokens)))[0])
        # array가 문자열로 구성된 경우
        elif array.dtype.kind in {'U', 'S'}:  # 데이터 타입이 유니코드 문자열, 바이트 문자열인 경우
            string_tokens = {str(tok) for tok in tokens if isinstance(tok, str)}
            indices.update(np.where(np.isin(array, list(string_tokens)))[0])

    remove_ind = indices

    # 3개의 리스트를 2D 배열로 결합
    data = np.array([gen_pitch, gen_octave, gen_duration])

    # 유효한 컬럼 인덱스 생성 (데이터 범위 내에 있는 인덱스만 고려)
    valid_indices = [i for i in range(data.shape[1]) if i not in remove_ind]

    # 유효한 컬럼만 선택
    filtered_gen_pitch, filtered_gen_octave, filtered_gen_duration = map(list,data[:, valid_indices])

    return filtered_gen_pitch, filtered_gen_octave, filtered_gen_duration