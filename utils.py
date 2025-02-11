from music21 import converter
from music21 import note
from music21 import spanner
from fractions import Fraction

# helper functions
def load_musicxml(file_path):
    """
    MusicXML 파일(악보 데이터)을 읽어옵니다.

        입력 : 파일 경로
        출력 : 음악 데이터
    """
    score = converter.parse(file_path)
    return score

# helper functions
def get_tie_type(element, previous_tie_type):
    """
    음표의 Tie 정보를 분석하여 반환
    구현 : 허은우 박사 (hew0920@postech.ac.kr)
    """
    if element.tie is not None:
        if element.tie.type == 'start' and previous_tie_type in ['start', 'continue']:
            raise Exception('Do not use tie notation in musicxml file for notes with different pitches')
        return element.tie.type
    return None


def get_slur_type(element):
    """
    음표의 Slur 정보를 분석하여 반환
    구현 : 허은우 박사 (hew0920@postech.ac.kr)
    """
    slurs = [s for s in element.getSpannerSites() if isinstance(s, spanner.Slur)]
    if slurs:
        for slur in slurs:
            if slur.isFirst(element):
                return 'start'
            elif slur.isLast(element):
                return 'stop'
        return 'continue'
    return None


def all_notes_have_same_pitch(notes_list: list) -> bool:
    """list에 들어있는 Note들이 같은 음높이인지 확인"""
    if not notes_list:
        return False  # 빈 리스트일 경우 False 반환

    first_note = notes_list[0]

    # 첫 번째 요소가 음표인지 확인 (쉼표가 포함되었을 경우 대비)
    if not isinstance(first_note, note.Note):
        return False

    for n in notes_list:
        if not isinstance(n, note.Note):  # 쉼표가 있으면 무시
            continue
        if n.nameWithOctave != first_note.nameWithOctave:
            return False  # 하나라도 다르면 False 반환

    return True


# def get_duration(note, fraction=True: bool):
def get_duration(note, fraction: bool=True):
    """
    note(음표)의 duration 값을 Fraction 형태 혹은 float 형태로 출력합니다
    """
    if fraction:
        return Fraction(note.duration.quarterLength)
    else:
        return note.duration.quarterLength


def score_to_nts_list(score) -> list:
    """
    score(악보) 데이터에서 (note, tie, slur) triple을 추출해 list로 변환
    """
    nts_list = []
    previous_tie_type = None

    for element in score.flatten().notesAndRests:
        tie_type = get_tie_type(element, previous_tie_type)
        slur_type = get_slur_type(element)

        nts_list.append((element, tie_type, slur_type))
        previous_tie_type = tie_type

    return nts_list


def merge_notes(nts_list:list, fraction=True):
    """
    slur나 tie로 묶인 Note들을 하나의 Note로 합쳐주는 함수입니다.
    구현 : 허은우 박사 (hew0920@postech.ac.kr)

    입력
        - nts_list(list) : (note, tie_type, slur_type) 의 triple 정보를 원소로 담고 있는 list 입니다.

    """
    results = []
    tmp_accumul = []  # 병합할 음표 저장용

    for element, tie_is, slur_is in nts_list:
        # 쉬는 음표(Rest) 처리
        if isinstance(element, note.Rest):
            rest_duration = Fraction(element.duration.quarterLength) if fraction else element.duration.quarterLength
            results.append(('rest', rest_duration))
            continue

        # 타이/슬러 'stop' → 'merge'
        if tie_is == 'stop' or slur_is == 'stop':
            tmp_accumul.append(element)
            # 'tmp_accumul' list에 들어있는 객체들이 모두 같은 음인지 반별

            if all_notes_have_same_pitch(tmp_accumul):
                # tmp_accumul에 있는 note들의 accumulated duration을 계산.
                total_duration = sum([get_duration(n, fraction) for n in tmp_accumul])
                results.append((tmp_accumul[0].pitch.nameWithOctave, total_duration))
            else:
                for note_ in tmp_accumul:
                    duration = get_duration(note_, fraction)
                    results.append((note_.pitch.nameWithOctave, duration))
            ### 추가 파트
            if tie_is == 'start' or slur_is == 'start':
                tmp_accumul = [element]

        # 타이/슬러 'start' or 'continue'
        elif tie_is =='start' or slur_is == 'start':
            tmp_accumul = [element]
        elif tie_is =='continue' or slur_is == 'continue':
            tmp_accumul.append(element)

        # 독립된 음 처리
        else:
            duration = get_duration(element, fraction)
            results.append((element.pitch.nameWithOctave, duration))

    return results

def Find_fraction_form_along_base_denominator(original_fraction, base = 36, strict_expression = False):
    """
    고정된 분모의 형태의 분수 배수중, 가장 가까운 분모 표현 계산해내는 코드입니다.
    strict_expression을 True로 하면, 약분 없이 분모표현을 base로 고정합니다.
    """

    reduced_fraction = original_fraction - int(original_fraction) # 소수 부분만 뽑기.

    if reduced_fraction == 0:
        if strict_expression == True :
            return f'{int(original_fraction * base)}/{base}'
        return original_fraction

    # 소수 부분만 근사. (코드의 효율성을 위해)
    # n을 1부터 base 까지 증가시키면서 n/base 가 reduced_fraction과 가장 가까운 값 찾기
    closest_n = None
    min_difference = float('inf')

    for n in range(1, base+1):
        current_fraction = Fraction(n, base)
        current_difference = abs(current_fraction - reduced_fraction)
        if current_difference < min_difference:
            min_difference = current_difference
            closest_n = n

    result = int(original_fraction) + Fraction(closest_n,base)

    if strict_expression == True :
        return f'{int(result.numerator*(base/result.denominator))}/{ int(result.denominator*(base/result.denominator))}'
    return result


def pitch_normalize(note):
    """
    #(샵)음을 b(플랫)으로 정규화 합니다.
    참고 : music21에서는 플랫을 -로 표기하고 있음.
    """
    pitch = note.pitch.name

    sharp_to_flat = {
        "C#": "D-",
        "D#": "E-",
        "F#": "G-",
        "G#": "A-",
        "A#": "B-"
    }

    if "#" in pitch:
        return sharp_to_flat.get(pitch)
    else:
        return pitch