from music21 import converter
from music21 import note
from music21 import spanner
from fractions import Fraction

# helper functions
def load_musicxml(file_path):
    """
    Reads a MusicXML file (sheet music data).

        Input : file path
        Output: music data
    """
    score = converter.parse(file_path)
    return score

# helper functions
def get_tie_type(element, previous_tie_type):
    """
    Analyzes and returns the Tie information of a note.
    """
    if element.tie is not None:
        if element.tie.type == 'start' and previous_tie_type in ['start', 'continue']:
            raise Exception('Do not use tie notation in musicxml file for notes with different pitches')
        return element.tie.type
    return None

def get_slur_type(element):
    """
    Analyzes and returns the Slur information of a note.
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
    """Checks if all Notes in a list have the same pitch."""
    if not notes_list:
        return False  # Return False for an empty list

    first_note = notes_list[0]

    # Check if the first element is a note (to handle rests included)
    if not isinstance(first_note, note.Note):
        return False

    for n in notes_list:
        if not isinstance(n, note.Note):  # Ignore rests
            continue
        if n.nameWithOctave != first_note.nameWithOctave:
            return False  # Return False if any are different

    return True


# def get_duration(note, fraction=True: bool):
def get_duration(note, fraction: bool=True):
    """
    Outputs the duration of a note either in Fraction form or as a float.
    """
    if fraction:
        return Fraction(note.duration.quarterLength)
    else:
        return note.duration.quarterLength


def score_to_nts_list(score) -> list:
    """
    Converts score (sheet music) data into a list of (note, tie, slur) triples.
    """
    nts_list = []
    previous_tie_type = None

    for element in score.flatten().notesAndRests:
        tie_type = get_tie_type(element, previous_tie_type)
        slur_type = get_slur_type(element)

        nts_list.append((element, tie_type, slur_type))
        previous_tie_type = tie_type

    return nts_list

def merge_notes(nts_list: list, fraction=True):
    """
    Merges notes that are connected by slurs or ties into a single note.
    
    Input:
        - nts_list (list): List containing triples of (note, tie_type, slur_type).
    """
    results = []
    tmp_accumul = []  # Storage for notes to merge

    for element, tie_is, slur_is in nts_list:
        # Handling rests
        if isinstance(element, note.Rest):
            rest_duration = Fraction(element.duration.quarterLength) if fraction else element.duration.quarterLength
            results.append(('rest', rest_duration))
            continue

        # Tie/Slur 'stop' → 'merge'
        if tie_is == 'stop' or slur_is == 'stop':
            tmp_accumul.append(element)
            # Check if all notes in 'tmp_accumul' have the same pitch

            if all_notes_have_same_pitch(tmp_accumul):
                # Calculate the accumulated duration of the notes in 'tmp_accumul'.
                total_duration = sum([get_duration(n, fraction) for n in tmp_accumul])
                results.append((tmp_accumul[0].pitch.nameWithOctave, total_duration))
            else:
                for note_ in tmp_accumul:
                    duration = get_duration(note_, fraction)
                    results.append((note_.pitch.nameWithOctave, duration))
            ### Additional part
            if tie_is == 'start' or slur_is == 'start':
                tmp_accumul = [element]

        # Tie/Slur 'start' or 'continue'
        elif tie_is == 'start' or slur_is == 'start':
            tmp_accumul = [element]
        elif tie_is == 'continue' or slur_is == 'continue':
            tmp_accumul.append(element)

        # Handle independent notes
        else:
            duration = get_duration(element, fraction)
            results.append((element.pitch.nameWithOctave, duration))

    return results

def Find_fraction_form_along_base_denominator(original_fraction, base=36, strict_expression=False):
    """
    Computes the closest fraction representation with a fixed denominator among multiples of a fixed fraction.
    If strict_expression is True, the denominator is strictly set to 'base' without reduction.
    """

    reduced_fraction = original_fraction - int(original_fraction)  # Extract the fractional part.

    if reduced_fraction == 0:
        if strict_expression:
            return f'{int(original_fraction * base)}/{base}'
        return original_fraction

    # Approximate only the fractional part for efficiency.
    closest_n = None
    min_difference = float('inf')

    for n in range(1, base + 1):
        current_fraction = Fraction(n, base)
        current_difference = abs(current_fraction - reduced_fraction)
        if current_difference < min_difference:
            min_difference = current_difference
            closest_n = n

    result = int(original_fraction) + Fraction(closest_n, base)

    if strict_expression:
        return f'{int(result.numerator * (base / result.denominator))}/{int(result.denominator * (base / result.denominator))}'
    return result


def pitch_normalize(note):
    """
    Normalizes sharp notes to flat notes.
    Note: In music21, flats are denoted with a '-'.
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