from utils import *
import os
from music21 import converter, spanner
from fractions import Fraction
from music21 import stream
from music21 import meter
from music21 import note

## dur_to_Junggan 함수 만들기 위한 곡 별 분류 기준 (그룹마다, 길이 해석이 다름)
criterion_1 = ['01 G-Sangnyeongsan_Haegeum_part(0807)','01 G-Sangnyeongsan_Piri_part(0807)','01 J-Sangnyeongsan_Gayageum_part(0719)','01 J-Sangnyeongsan_Geomungo_part(0719)','01 P-Sangnyeongsan_Daegeum_part(0807)',
'02 G-Jungnyeongsan_Haegeum_part(0807)','02 G-Jungnyeongsan_Piri_part(0807)','02 J-Jungnyeongsan_Gayageum_part(0722)','02 J-Jungnyeongsan_Geomungo_part(0722)','02 P-Jungnyeongsan_Daegeum_part(0807)',
'03 G-Seryeongsan_Haegeum_part(0807)','03 G-Seryeongsan_Piri_part(0807)','03 J-Seryeongsan_Gayageum_part(0722)','03 J-Seryeongsan_Geomungo_part(0722)','03 P-Seryeongsan_Daegeum_part(0807)',
'04 G-Garakdeori_Haegeum_part(0807)','04 G-Garakdeori_Piri_part(0807)','04 J-Garakdeori_Gayageum_part(0722)','04 J-Garakdeori_Geomungo_part(0722)','04 P-Garakdeori_Daegeum_part(0807)','05 G-Samhyeondodeuri_Haegeum_part(0807)',
'05 G-Samhyeondodeuri_Piri_part(0807)','05 P-Sanghyeondodeuri_Daegeum_part(0807)','06 G-Yeombuldodeuri_Haegeum_part(0807)','06 G-Yeombuldodeuri_Piri_part(0807)','06 P-Yeombuldodeuri_Daegeum_part(0807)',]

criterion_2 = ['03 Ch-Giltaryeong_Haegeum_part(0722)','03 Ch-Giltaryeong_Piri_part(0722)','03 Cheon-Ujogarakdodeuri_Daegeum_part(0807)','04 Ch-Byeorujotaryeong_Haegeum_part(0722)',
'04 Ch-Byeorujotaryeong_Piri_part(0722)','07 G-Taryeong_Haegeum_part(0807))','07 G-Taryeong_Piri_part(0807)','07 P-Taryeong_Daegeum_part(0807)','08 P-Gunak_Daegeum_parts(0807)']

criterion_3 = ['Jajin 02 Yeomyangchun_Haegeum_part(0807)','Jajin 02 Yeomyangchun_Piri_part(0807) Piri','Mitdodeuri_Daegeum_part(0807)',
'Utdodeuri_Daegeum_part(0807)','Yeomillak 1_Daegeum_part(0807)','Yeomillak 1_Gayageum_part(0807)','Yeomillak 1_Geomungo_part(0807)','Yeomillak 2_Daegeum_part(0807)',
'Yeomillak 2_Gayageum_part(0807)','Yeomillak 2_Geomungo_part(0807)','Yeomillak 3_Daegeum_part(0807)','Yeomillak 3_Gayageum_part(0807)','Yeomillak 3_Geomungo_part(0807)']

Adjust_criterion = {}
for cri in criterion_1:
    Adjust_criterion[cri] = ['C1', 1.5] # [기준 1, Adjust_duration ratio]
for cri in criterion_2:
    Adjust_criterion[cri] = ['C2', 1]   # [기준 2, Adjust_duration ratio]
for cri in criterion_3:
    Adjust_criterion[cri] = ['C3', 1] # [기준 3, Adjust_duration ratio]

# data_processing의 핵심 함수입니다.
def extract_notes_from_musicxml(score, fraction=True):
    """
    이 함수는 다음과 같은 데이터 변환 절차를 functional programming style로 구현하였습니다.
    (엄밀한 FP는 아님)

    musicxml -> music21 object -> (note, tie, slur) triple list -> merged_notes

    구현 : 허은우 박사 (hew0920@postech.ac.kr)
    리펙토링 : 이성헌 (shlee0125@postech.ac.kr)
    """
    #score = load_musicxml(file_path)
    nts_list = score_to_nts_list(score)
    merged_nts_list = merge_notes(nts_list)
    return merged_nts_list


def music_to_tokens(file_directory):
    """
    음악 데이터를 받아서 3개의 채널로 구성된 시퀀스로 변환합니다.

    각 시점은 다음 정보를 갖습니다:
      - pitch: 음의 이름 (예: "C", "D", "E", ..., "rest")
      - octave: 옥타브 정보 (note인 경우 정수, rest인 경우 특별한 값, 예를 들어 0 또는 -1)
      - duration: 음의 길이 (quarterLength 값)

    또한, 시퀀스의 시작(<SOS>)과 종료(<EOS>)를 표시하기 위한 특별 토큰을 각 채널에 추가합니다.

    반환:
      pitch_tokens: pitch 채널의 시퀀스 (리스트)
      octave_tokens: octave 채널의 시퀀스 (리스트)
      duration_tokens: duration 채널의 시퀀스 (리스트)
    """

    music_data = load_musicxml(file_directory)

    pitch_tokens = []
    octave_tokens = []
    duration_tokens = []

    # 시퀀스 시작 토큰 추가
    pitch_tokens.append("<SOS>")
    octave_tokens.append("<SOS>")
    duration_tokens.append("<SOS>")

    ## 서로 다른 곡 끼리 Duration 비율 기준 맞추는 코드
    # 파일의 기본 이름만 추출 (경로 제거)
    base_name = os.path.basename(file_directory)
    # 파일 확장자 제거 (.musicxml.xml 제거)
    clean_name = os.path.splitext(os.path.splitext(base_name)[0])[0]
    _, adjust_duration_ratio = Adjust_criterion[clean_name]

    for pit, dur in extract_notes_from_musicxml(music_data, fraction = True):
        if pit == 'rest':
            pitch_tokens.append(pit)
            octave_tokens.append(0)
        else:
            pitch_tokens.append(pit[:-1])
            octave_tokens.append(int(pit[-1]))

        junggan = Find_fraction_form_along_base_denominator(dur/adjust_duration_ratio, base = 36)
        # base = 12 로 바꾸기
        junggan = Find_fraction_form_along_base_denominator(junggan, base = 12, strict_expression = True)
        duration_tokens.append(junggan)

    # 시퀀스 종료 토큰 추가
    pitch_tokens.append("<EOS>")
    octave_tokens.append("<EOS>")
    duration_tokens.append("<EOS>")

    return pitch_tokens, octave_tokens, duration_tokens


def build_vocab_mapping(vocab_list):
    """주어진 vocab_list로 token->index, index->token dict를 생성."""
    token_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token



# pre-defined vocabulary
def build_predefined_vocabs():
    # Pitch: 12음 + "rest"와 특수 토큰
    pitch_vocab = ["<PAD>", "<SOS>", "<EOS>",
                   "C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B", "rest"]

    # Octave: 1~5, rest는 "0", 그리고 특수 토큰
    octave_vocab = ["<PAD>", "<SOS>", "<EOS>", 0, 2, 3, 4, 5, 6]


    # duration_tokens = [d for d in [0.25 * i for i in range(1, int(4.0/0.25)+1)]]
    # Duration: n/12, 문자열로 표현
    duration_tokens = [ f"{i}/12" for i in range(1, 37)]
    duration_vocab = ["<PAD>", "<SOS>", "<EOS>"] + duration_tokens

    # Junggan
    junggan_tokens = [f"{n}/12" for n in range(1, 37)]
    junggan_vocab = ["<PAD>", "<SOS>", "<EOS>"] + junggan_tokens

    # Hangul
    return pitch_vocab, octave_vocab, junggan_vocab

def Insert_accummlated_Note(accummulation, music_stream):
    for acc_p_token, acc_o_token, acc_d_token in accummulation:
        # 피치가 "rest"이면 Rest 객체 생성
        if acc_p_token.lower() == "rest":
            r = note.Rest()
            r.duration.quarterLength = float(acc_d_token)
            music_stream.append(r)
        else:
            # 피치와 옥타브를 결합하여 Note 객체 생성
            # 예를 들어, 피치가 "C#"이고 옥타브가 "4"이면 "C#4"가 됨
            n = note.Note(f"{acc_p_token}{acc_o_token}")
            n.duration.quarterLength = float(acc_d_token)
            music_stream.append(n)

    return music_stream


def tokens_to_music21(generated_music, author='Anonymous', key_signature='-5',title='AI Generated GukAk', time_signature='18/8', cutting_number = 5):

    pitch_tokens, octave_tokens, duration_tokens = generated_music

    gen_pitch_copy = pitch_tokens.copy()
    gen_octave_copy = octave_tokens.copy()
    gen_duration_copy = duration_tokens.copy()

    Frac_to_str={Fraction(n,12): f'{n}/12' for n in range(1,73)} 
    duration_tokens_to_fraction = { f"{i}/12": Fraction(i,12) for i in range(1, 73)}


    adjust_ratio = Fraction(3,2)
    cutting_number *= adjust_ratio 

    # music21의 stream 객체 생성
    music_stream = stream.Stream()
    time_sig = meter.TimeSignature(time_signature)
    key_sig = key.KeySignature(key_signature)
    music_stream.append(time_sig)
    music_stream.append(key_sig)
    music_stream.metadata.title = title
    music_stream.metadata.composer = author

    length_acc=[]
    accummulation = []

    for ind, (p_token, o_token, d_token) in enumerate(zip(gen_pitch_copy, gen_octave_copy, gen_duration_copy)):
        
        # 제일 마지막 노드 처리
        if ind == len(gen_pitch_copy)-1:
            music_stream = Insert_accummlated_Note(accummulation, music_stream)
            break

        real_duration = duration_tokens_to_fraction[d_token] * adjust_ratio

        length_acc.append(real_duration)
        length_acc_sum = sum(length_acc)

        if length_acc_sum < cutting_number: ### 아직 초과하지 않았을 때는, 일단 Note 쌓기, 초기화는 하지 않기.
            accummulation.append((p_token, o_token, real_duration))

        elif length_acc_sum == cutting_number: ### 딱 맞아 떨어 질때는, 정상적으로 Note 쌓은후, music_stream에 넣고 초기화 하기.
            
            accummulation.append((p_token, o_token, real_duration))
            ### for acc에 있는거 music stream에 넣고, acc 초기화
            ######## accummulation에 넣어져 있는 것 stream에 넣기
            music_stream = Insert_accummlated_Note(accummulation, music_stream)

            # 현재노드를 오른쪽 노드 accumulation에 넣으면서 초기화 하기
            length_acc = []
            accummulation = []

        elif length_acc_sum > cutting_number: ### 조정하고, 초기화 하기

            if real_duration - length_acc_sum + cutting_number >= length_acc_sum - cutting_number: # 왼쪽 >= 오른쪽
                left_duration = real_duration - length_acc_sum + cutting_number
                accummulation.append((p_token, o_token, left_duration))

                ######## accummulation에 넣어져 있는 것 stream에 넣기
                music_stream = Insert_accummlated_Note(accummulation, music_stream)

                new_right_duration = Fraction(gen_duration_copy[ind+1])+Find_fraction_form_along_base_denominator((length_acc_sum - cutting_number)/adjust_ratio, base = 12, strict_expression = False)

                gen_duration_copy[ind+1] = Frac_to_str[new_right_duration] # 오른쪽에 대입.
                length_acc = []
                accummulation = []
                
            else: # 왼쪽 < 오른쪽
                # 왼쪽 이전 노드의 gen_duration 조정
                right_duration = length_acc_sum - cutting_number ## 18/12
                p_tok ,o_tok , previous_left_duration = accummulation[-1]
                previous_left_duration += (real_duration - length_acc_sum + cutting_number) # 왼쪽 파트 만큼 늘리기, 6/12
                accummulation[-1] = (p_tok ,o_tok , previous_left_duration)

                ######## accummulation에 넣어져 있는 것 stream에 넣기
                music_stream = Insert_accummlated_Note(accummulation, music_stream)
                # 현재노드를 오른쪽 노드 accumulation에 넣으면서 초기화 하기
                accummulation = [(p_token, o_token, right_duration)]
                length_acc = [right_duration]
    return music_stream

