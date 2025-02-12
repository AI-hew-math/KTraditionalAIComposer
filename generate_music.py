"""
- generate_music (function): Receives a trained model and vocabulary information to generate music.
                             By using the JungGanBoInputUI described below to create user_phrase_tokens,
                             the user can input their own composed notes as a starting point, and the model can then generate the rest of the music.

- JungGanBoInputUI (class): This is an IPython widget that allows users to select notes from JungGanBo and input up to 10 notes.
                            It converts these notes into tokens that can be input into the model and outputs them.
                            This user interface enables input through mouse clicks rather than coding, making it accessible and easy to use.
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
    Generates music using a trained model.

    Returns:
      - Generated sequences of pitch, octave, and duration tokens (each as a list).
    """
    model.to(device)
    model.eval()


    hidden = None  # Initial hidden state for the LSTM (if None, it is initialized to zero)

    # Store the result sequences (initial <SOS> can be excluded later)
    generated_pitch = [pitch_token_to_idx["<SOS>"]]
    generated_octave = [octave_token_to_idx["<SOS>"]]
    generated_duration = [duration_token_to_idx["<SOS>"]]


    if user_phrase_token is not None:
        pitch_token, octave_token, duration_token = user_phrase_token
        token_length = len(pitch_token)
        # Update the hidden state up to the last note, ignoring the generated notes.
        for i in range(token_length):  # Sequential input per time-step
            current_pitch = torch.tensor([[pitch_token_to_idx.get(pitch_token[i], pitch_token_to_idx["<PAD>"])]], dtype=torch.long, device=device)
            current_octave = torch.tensor([[octave_token_to_idx.get(octave_token[i], octave_token_to_idx["<PAD>"])]], dtype=torch.long, device=device)
            current_duration = torch.tensor([[duration_token_to_idx.get(duration_token[i], duration_token_to_idx["<PAD>"])]], dtype=torch.long, device=device)

            _, _, _, hidden = model(current_pitch, current_octave, current_duration, hidden)

            # Save the inputted user_phrase_token directly into the results
            generated_pitch.append(pitch_token_to_idx.get(pitch_token[i], pitch_token_to_idx["<PAD>"]))
            generated_octave.append(octave_token_to_idx.get(octave_token[i], octave_token_to_idx["<PAD>"]))
            generated_duration.append(duration_token_to_idx.get(duration_token[i], duration_token_to_idx["<PAD>"]))
            #print(i, current_pitch)

    else:
        token_length = 0
        current_pitch = torch.tensor([[pitch_token_to_idx["<SOS>"]]], dtype=torch.long, device=device)
        current_octave = torch.tensor([[octave_token_to_idx["<SOS>"]]], dtype=torch.long, device=device)
        current_duration = torch.tensor([[duration_token_to_idx["<SOS>"]]], dtype=torch.long, device=device)

    for i in range(max_gen_length):
        # Execute the model with the current input for each channel. Input shape: (1, 1)
        output_pitch, output_octave, output_duration, hidden = model(current_pitch, current_octave, current_duration, hidden)
        # Each output's shape: (1, 1, vocab_size)
        # Extract and squeeze the last timestep's output vector to make it (vocab_size,)
        logits_pitch = output_pitch[:, -1, :].squeeze(0)   # shape: (pitch_vocab_size)
        logits_octave = output_octave[:, -1, :].squeeze(0)
        logits_duration = output_duration[:, -1, :].squeeze(0)

        probs_pitch = torch.softmax(logits_pitch, dim=-1)
        probs_octave = torch.softmax(logits_octave, dim=-1)
        probs_duration = torch.softmax(logits_duration, dim=-1)

        # Sample the next token according to the probability distribution (multinomial sampling)
        next_pitch = torch.multinomial(probs_pitch, num_samples=1).item()
        next_octave = torch.multinomial(probs_octave, num_samples=1).item()
        next_duration = torch.multinomial(probs_duration, num_samples=1).item()

        # Add the generated tokens to the result sequence
        generated_pitch.append(next_pitch)
        generated_octave.append(next_octave)
        generated_duration.append(next_duration)
        #print(generated_pitch)
        # End generation if the <EOS> token is produced in the pitch channel (or in all channels)
        if next_pitch == pitch_token_to_idx["<EOS>"]:
            break

        # Configure the next input for the following iteration (use newly generated tokens as current input)
        current_pitch = torch.tensor([[next_pitch]], dtype=torch.long, device=device)
        current_octave = torch.tensor([[next_octave]], dtype=torch.long, device=device)
        current_duration = torch.tensor([[next_duration]], dtype=torch.long, device=device)

    # Exclude the initial <SOS> token and convert the generated token indices to actual token strings.
    gen_pitch_tokens = [pitch_idx_to_token[idx] for idx in generated_pitch[1:]]
    gen_octave_tokens = [octave_idx_to_token[idx] for idx in generated_octave[1:]]
    gen_duration_tokens = [duration_idx_to_token[idx] for idx in generated_duration[1:]]

    return gen_pitch_tokens, gen_octave_tokens, gen_duration_tokens

class JungGanBoInputUI:
    def __init__(self):
        # Pitch mapping
        self.pitch_mapping = {
            "황종": "E-", "대려": "E", "태주": "F", "협종": "G-",
            "고선": "G", "중려": "A-", "유빈": "A", "임종": "B-",
            "이칙": "B", "남려": "C", "무역": "D-", "응종": "D",
            "쉼표": "rest"  # 쉼표 추가
        }

        # Option lists
        self.pitch_options = list(self.pitch_mapping.keys())
        self.octave_options = [2, 3, 4, 5, 6]
        self.duration_options = [f"{n}/12" for n in range(1, 37)]

        # List for storing input data
        self.current_input = []
        self.generated_data = None  # 생성된 데이터를 저장할 변수

        # Create UI widgets
        self.pitch_selector = widgets.Dropdown(options=self.pitch_options, description="음정:")
        self.octave_selector = widgets.Dropdown(options=self.octave_options, description="옥타브:")
        self.duration_selector = widgets.Dropdown(options=self.duration_options, description="정간:")

        self.add_button = widgets.Button(description="➕ 추가")
        self.remove_button = widgets.Button(description="🗑 삭제")
        self.clear_button = widgets.Button(description="🔄 전체 삭제")
        self.generate_button = widgets.Button(description="💾 데이터 생성")

        self.output = widgets.Output()

        # Button actions
        self.add_button.on_click(self.add_entry)
        self.remove_button.on_click(self.remove_entry)
        self.clear_button.on_click(self.clear_entries)
        self.generate_button.on_click(self.generate_data)

        # Display UI
        display(
            self.pitch_selector, self.octave_selector, self.duration_selector,
            self.add_button, self.remove_button, self.clear_button, self.generate_button,
            self.output
        )
        self.update_display()

    def update_display(self):
        """Display the currently entered values"""
        with self.output:
            clear_output()
            if self.current_input:
                print("📜 Currently entered list:")
                for i, (p, o, d) in enumerate(self.current_input, 1):
                    print(f"{i}. Pitch: {p}, Octave: {o}, Duration: {d}")
            else:
                print("❗ No values have been entered yet.")

    def add_entry(self, _):
        """Add the selected values to the list (up to 10 entries)"""
        if len(self.current_input) < 10:
            pitch = self.pitch_selector.value
            octave = 0 if pitch == "rest" else self.octave_selector.value  # Fix octave to 0 for rest
            self.current_input.append((pitch, octave, self.duration_selector.value))
        else:
            with self.output:
                print("⚠ You can enter up to 10 items.")
        self.update_display()

    def remove_entry(self, _):
        """Remove the last entered value"""
        if self.current_input:
            self.current_input.pop()
        self.update_display()

    def clear_entries(self, _):
        """Clear all entered values"""
        self.current_input.clear()
        self.generated_data = None  # 데이터도 초기화
        self.update_display()

    def generate_music_data(self):
        """Transform and save music data"""
        if not self.current_input:
            print("⚠ No values have been entered.")
            return None

        pitch_tokens = [self.pitch_mapping[p] for p, _, _ in self.current_input]
        octave_tokens = [o for _, o, _ in self.current_input]
        duration_tokens = [d for _, _, d in self.current_input]

        self.generated_data = (pitch_tokens, octave_tokens, duration_tokens)  # Save to internal variable
        return self.generated_data

    def generate_data(self, _):
        """Generate data on button click and display in UI"""
        data = self.generate_music_data()
        if data:
            pitch_tokens, octave_tokens, duration_tokens = data
            with self.output:
                clear_output()
                print("✅ Data has been saved!")
                print("🎵 Converted tokens:")
                print(f"Pitch: {pitch_tokens}")
                print(f"Octave: {octave_tokens}")
                print(f"Duration: {duration_tokens}")

    def get_token(self):
        """Return the data stored inside the object"""
        return self.generated_data
        
def remove_special_tokens(generated_music):
    gen_pitch, gen_octave, gen_duration = generated_music

    gen_pitch = np.array(gen_pitch)
    gen_octave = np.array(gen_octave)
    gen_duration = np.array(gen_duration)

    # Tokens to check, including strings and numbers
    tokens = {'<EOS>', '<PAD>', '<SOS>', 0, '0'}

    indices = set()

    # Check each array for tokens and add indices
    for array in [gen_pitch, gen_octave, gen_duration]:
        # If the array consists of numbers
        if array.dtype.kind in {'i', 'u', 'f'}:  # Data type is integer, unsigned integer, or floating point
            numeric_tokens = {int(tok) for tok in tokens if isinstance(tok, int) or (isinstance(tok, str) and tok.isdigit())}
            indices.update(np.where(np.isin(array, list(numeric_tokens)))[0])
        # If the array consists of strings
        elif array.dtype.kind in {'U', 'S'}:  # Data type is unicode string or byte string
            string_tokens = {str(tok) for tok in tokens if isinstance(tok, str)}
            indices.update(np.where(np.isin(array, list(string_tokens)))[0])

    remove_ind = indices

    # Combine the three lists into a 2D array
    data = np.array([gen_pitch, gen_octave, gen_duration])

    # Create valid column indices (only consider indices within the data range)
    valid_indices = [i for i in range(data.shape[1]) if i not in remove_ind]

    # Select only valid columns
    filtered_gen_pitch, filtered_gen_octave, filtered_gen_duration = map(list, data[:, valid_indices])

    return filtered_gen_pitch, filtered_gen_octave, filtered_gen_duration
