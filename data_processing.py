from utils import *
import os
from music21 import converter, spanner
from fractions import Fraction
from music21 import stream
from music21 import meter
from music21 import note
from music21 import key, metadata

## Criteria for Classifying Songs by Group for Creating the dur_to_Jungganbo (Each Group Interprets Duration Differently)
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

# This is a core function of data_processing.
def extract_notes_from_musicxml(score, fraction=True):
    """
    This function implements the following data transformation procedure in a functional programming style (though not strictly FP):

    musicxml -> music21 object -> (note, tie, slur) triple list -> merged_notes

    Implemented by: Dr. Heo Eunwoo (hew0920@postech.ac.kr)
    Refactored by: Lee Seongheon (shlee0125@postech.ac.kr)
    """
    #score = load_musicxml(file_path)
    nts_list = score_to_nts_list(score)
    merged_nts_list = merge_notes(nts_list)
    return merged_nts_list


def music_to_tokens(file_directory):
    """
    Transforms music data into a sequence composed of three channels.

    Each point in the sequence contains the following information:
      - pitch: Name of the note (e.g., "C", "D", "E", ..., "rest")
      - octave: Octave information (integer for notes, special value for rest, e.g., 0 or -1)
      - duration: Length of the note (quarterLength value)

    Special tokens to indicate the start (<SOS>) and end (<EOS>) of the sequence are also added to each channel.

    Returns:
      pitch_tokens: Sequence of the pitch channel (list)
      octave_tokens: Sequence of the octave channel (list)
      duration_tokens: Sequence of the duration channel (list)
    """

    music_data = load_musicxml(file_directory)

    pitch_tokens = []
    octave_tokens = []
    duration_tokens = []

    # Add sequence start token
    pitch_tokens.append("<SOS>")
    octave_tokens.append("<SOS>")
    duration_tokens.append("<SOS>")

    ## Code to align Duration ratio standards between different songs
    # Extract the base name of the file (remove path)
    base_name = os.path.basename(file_directory)
    # Remove the file extension (.musicxml.xml removal)
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
        # Change base to 12
        junggan = Find_fraction_form_along_base_denominator(junggan, base = 12, strict_expression = True)
        duration_tokens.append(junggan)

    # Add sequence end token
    pitch_tokens.append("<EOS>")
    octave_tokens.append("<EOS>")
    duration_tokens.append("<EOS>")

    return pitch_tokens, octave_tokens, duration_tokens

def build_vocab_mapping(vocab_list):
    """Creates token-to-index and index-to-token dictionaries from the given vocab_list."""
    token_to_idx = {token: idx for idx, token in enumerate(vocab_list)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token

# pre-defined vocabulary
def build_predefined_vocabs():
    # Pitch: 12 notes + "rest" and special tokens
    pitch_vocab = ["<PAD>", "<SOS>", "<EOS>",
                   "C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B", "rest"]

    # Octave: 2 ~ 6, "0" for rest, and special tokens
    octave_vocab = ["<PAD>", "<SOS>", "<EOS>", 0, 2, 3, 4, 5, 6]


    # Duration: n/12, represented as strings
    duration_tokens = [ f"{i}/12" for i in range(1, 37)]
    duration_vocab = ["<PAD>", "<SOS>", "<EOS>"] + duration_tokens

    # Junggan
    junggan_tokens = [f"{n}/12" for n in range(1, 37)]
    junggan_vocab = ["<PAD>", "<SOS>", "<EOS>"] + junggan_tokens

    return pitch_vocab, octave_vocab, junggan_vocab

def Insert_accummlated_Note(accummulation, music_stream):
    for acc_p_token, acc_o_token, acc_d_token in accummulation:
        # Create a Rest object if the pitch is "rest"
        if acc_p_token.lower() == "rest":
            r = note.Rest()
            r.duration.quarterLength = float(acc_d_token)
            music_stream.append(r)
        else:
            # Combine pitch and octave to create a Note object
            # For example, if the pitch is "C#" and the octave is "4", it becomes "C#4"
            n = note.Note(f"{acc_p_token}{acc_o_token}")
            n.duration.quarterLength = float(acc_d_token)
            music_stream.append(n)

    return music_stream


def tokens_to_music21(generated_music, author='Anonymous', key_signature=-5,title='AI Generated GukAk', time_signature='18/8', cutting_number = 5):

    pitch_tokens, octave_tokens, duration_tokens = generated_music

    gen_pitch_copy = pitch_tokens.copy()
    gen_octave_copy = octave_tokens.copy()
    gen_duration_copy = duration_tokens.copy()

    Frac_to_str={Fraction(n,12): f'{n}/12' for n in range(1,73)} 
    duration_tokens_to_fraction = { f"{i}/12": Fraction(i,12) for i in range(1, 73)}


    adjust_ratio = Fraction(3,2)
    cutting_number *= adjust_ratio 

    music_stream = stream.Stream()
    music_stream.metadata = metadata.Metadata()

    time_sig = meter.TimeSignature(time_signature)
    key_sig = key.KeySignature(key_signature)
    music_stream.append(time_sig)
    music_stream.append(key_sig)
    music_stream.metadata.title = title
    music_stream.metadata.composer = author

    length_acc=[]
    accummulation = []

    for ind, (p_token, o_token, d_token) in enumerate(zip(gen_pitch_copy, gen_octave_copy, gen_duration_copy)):

        if ind == len(gen_pitch_copy)-1:
            music_stream = Insert_accummlated_Note(accummulation, music_stream)
            break

        real_duration = duration_tokens_to_fraction[d_token] * adjust_ratio

        length_acc.append(real_duration)
        length_acc_sum = sum(length_acc)

        if length_acc_sum < cutting_number: 
            accummulation.append((p_token, o_token, real_duration))

        elif length_acc_sum == cutting_number: 
            
            accummulation.append((p_token, o_token, real_duration))
            music_stream = Insert_accummlated_Note(accummulation, music_stream)

            length_acc = []
            accummulation = []

        elif length_acc_sum > cutting_number: 

            if real_duration - length_acc_sum + cutting_number >= length_acc_sum - cutting_number: 
                left_duration = real_duration - length_acc_sum + cutting_number
                accummulation.append((p_token, o_token, left_duration))

                music_stream = Insert_accummlated_Note(accummulation, music_stream)

                new_right_duration = Fraction(gen_duration_copy[ind+1])+Find_fraction_form_along_base_denominator((length_acc_sum - cutting_number)/adjust_ratio, base = 12, strict_expression = False)

                gen_duration_copy[ind+1] = Frac_to_str[new_right_duration] 
                length_acc = []
                accummulation = []
                
            else: 
                right_duration = length_acc_sum - cutting_number 
                p_tok ,o_tok , previous_left_duration = accummulation[-1]
                previous_left_duration += (real_duration - length_acc_sum + cutting_number) 
                accummulation[-1] = (p_tok ,o_tok , previous_left_duration)

                music_stream = Insert_accummlated_Note(accummulation, music_stream)
                accummulation = [(p_token, o_token, right_duration)]
                length_acc = [right_duration]
    return music_stream

