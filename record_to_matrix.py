import torch
import torchaudio
import os
import numpy as np
import pandas as pd
import mido
import math

def separate_hv_list(hv_list):
    """
    Convert a hit-velocity list to a 3-band representation (low, mid, high)
    
    Parameters:
    hv_list (list): List of note/velocity pairs organized by step
    
    Returns:
    numpy.ndarray: 3x16 matrix with rows representing low, mid, high frequency bands
    """
    # Create a 3×16 matrix (assuming hv_list is 16 steps)
    matrix = np.zeros((3, len(hv_list)))
    
    # List of MIDI instruments by category
    lows = [35, 36, 41, 45, 47, 64, 66]
    mids = [37, 38, 39, 40, 43, 48, 50, 61, 62, 65, 68, 77]
    highs = [22, 26, 42, 44, 46, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81]
    
    # Process each step in the bar
    for i, step in enumerate(hv_list):
        low_energy = 0
        mid_energy = 0
        high_energy = 0
        
        # Process each onset (drum hit) in the step
        for onset in step:
            note = onset[0]  # MIDI note number
            vel = onset[1]   # Velocity (intensity)
            
            # Add energy to appropriate frequency band
            if note in lows:
                low_energy += vel
            elif note in mids:
                mid_energy += vel
            else:  # Assume it's high
                high_energy += vel
        
        # Store values in matrix
        matrix[0, i] = low_energy
        matrix[1, i] = mid_energy
        matrix[2, i] = high_energy
    
    # Normalize each row to have max value of 1
    for j in range(3):
        if np.max(matrix[j]) > 0:  # Avoid division by zero
            matrix[j] = matrix[j] / np.max(matrix[j])
    
    return matrix

# Define General MIDI drum key map
GM_dict = {
    # key is midi note number
    # values are:
    # [0] name (as string)
    # [1] name category low mid or high (as string)
    # [2] substiture midi number for simplified MIDI (all instruments)
    # [3] name of instrument for 8 note conversion (as string)
    # [4] number of instrument for 8 note conversion
    # [5] substiture midi number for conversion to 8 note
    # [6] substiture midi number for conversion to 16 note
    # [7] substiture midi number for conversion to 3 note
    22: ["Closed Hi-Hat edge", "high", 42, "CH", 3, 42, 42, 42],
    26: ["Open Hi-Hat edge", "high", 46, "OH", 4, 46, 46, 42],
    35: ["Acoustic Bass Drum", "low", 36, "K", 1, 36, 36, 36],
    36: ["Bass Drum 1", "low", 36, "K", 1, 36, 36, 36],
    37: ["Side Stick", "mid", 37, "RS", 6, 37, 37, 38],
    38: ["Acoustic Snare", "mid", 38, "SN", 2, 38, 38, 38],
    39: ["Hand Clap", "mid", 39, "CP", 5, 39, 39, 38],
    40: ["Electric Snare", "mid", 38, "SN", 2, 38, 38, 38],
    41: ["Low Floor Tom", "low", 45, "LT", 7, 45, 45, 36],
    42: ["Closed Hi Hat", "high", 42, "CH", 3, 42, 42, 42],
    43: ["High Floor Tom", "mid", 45, "HT", 8, 45, 45, 38],
    44: ["Pedal Hi-Hat", "high", 46, "OH", 4, 46, 46, 42],
    45: ["Low Tom", "low", 45, "LT", 7, 45, 45, 36],
    46: ["Open Hi-Hat", "high", 46, "OH", 4, 46, 46, 42],
    47: ["Low-Mid Tom", "low", 47, "MT", 7, 45, 47, 36],
    48: ["Hi-Mid Tom", "mid", 47, "MT", 7, 50, 50, 38],
    49: ["Crash Cymbal 1", "high", 49, "CC", 4, 46, 42, 42],
    50: ["High Tom", "mid", 50, "HT", 8, 50, 50, 38],
    51: ["Ride Cymbal 1", "high", 51, "RC", -1, 42, 51, 42],
    52: ["Chinese Cymbal", "high", 52, "", -1, 46, 51, 42],
    53: ["Ride Bell", "high", 53, "", -1, 42, 51, 42],
    54: ["Tambourine", "high", 54, "", -1, 42, 69, 42],
    55: ["Splash Cymbal", "high", 55, "OH", 4, 46, 42, 42],
    56: ["Cowbell", "high", 56, "CB", -1, 37, 56, 42],
    57: ["Crash Cymbal 2", "high", 57, "CC", 4, 46, 42, 42],
    58: ["Vibraslap", "mid", 58, "VS", 6, 37, 37, 42],
    59: ["Ride Cymbal 2", "high", 59, "RC", 3, 42, 51, 42],
    60: ["Hi Bongo", "high", 60, "LB", 8, 45, 63, 42],
    61: ["Low Bongo", "mid", 61, "HB", 7, 45, 64, 38],
    62: ["Mute Hi Conga", "mid", 62, "MC", 8, 50, 62, 38],
    63: ["Open Hi Conga", "high", 63, "HC", 8, 50, 63, 42],
    64: ["Low Conga", "low", 64, "LC", 7, 45, 64, 36],
    65: ["High Timbale", "mid", 65, "", 8, 45, 63, 38],
    66: ["Low Timbale", "low", 66, "", 7, 45, 64, 36],
    67: ["High Agogo", "high", 67, "", -1, 37, 56, 42],
    68: ["Low Agogo", "mid", 68, "", -1, 37, 56, 38],
    69: ["Cabasa", "high", 69, "MA", -1, 42, 69, 42],
    70: ["Maracas", "high", 69, "MA", -1, 42, 69, 42],
    71: ["Short Whistle", "high", 71, "", -1, 37, 56, 42],
    72: ["Long Whistle", "high", 72, "", -1, 37, 56, 42],
    73: ["Short Guiro", "high", 73, "", -1, 42, 42, 42],
    74: ["Long Guiro", "high", 74, "", -1, 46, 46, 42],
    75: ["Claves", "high", 75, "", -1, 37, 75, 42],
    76: ["Hi Wood Block", "high", 76, "", 8, 50, 63, 42],
    77: ["Low Wood Block", "mid", 77, "", 7, 45, 64, 38],
    78: ["Mute Cuica", "high", 78, "", -1, 50, 62, 42],
    79: ["Open Cuica", "high", 79, "", -1, 45, 63, 42],
    80: ["Mute Triangle", "high", 80, "", -1, 37, 75, 42],
    81: ["Open Triangle", "high", 81, "", -1, 37, 75, 42],
}

def midifile2hv_list(file_name, mapping="all"):
    '''
    pattern name must include .mid
    get a MIDI file and convert it to an hv_list (a list of note numbers and velocity)
    use the "mapping" variable to define the type of instrument mapping
    that will be used in the hv_list "all", "16", "8", "3"
    '''
    pattern=[]
    mid=mido.MidiFile(file_name) #create a mido file instance
    sixteenth= mid.ticks_per_beat/4 #find the length of a sixteenth note

    # time: inside a track, it is delta time in ticks (integrer).
    # A delta time is how long to wait before the next message.
    acc=0 #use this to keep track of time

    # depending on the instruments variable select a notemapping
    if mapping=="all":
        column=2
    elif mapping=="16":
        column=6
    elif mapping=="8":
        column=5
    elif mapping=="3":
        column=7
    else: column = 2 # if no mapping is selected use "allinstrument" mapping

    for i, track in enumerate(mid.tracks):
        for msg in track: #process all messages
            acc += msg.time # accumulate time of any message type
            if msg.type == "note_on" and msg.note in GM_dict:
                midinote = GM_dict[msg.note][column] #remap msg.note by demand
                rounded_step = int((acc/sixteenth)+0.45)
                midivelocity = float(msg.velocity)/127 # normalize upfront
                pattern.append((int(acc/sixteenth), midinote, midivelocity)) # step, note, velocity
        if len(pattern)>0: #just proceed if analyzed pattern has at least one onset
            #round the pattern to the next multiple of 16
            pattern_len_in_steps = 16 * ((rounded_step//16)+((rounded_step%16)+16)//16)
            #create an empty list of lists the size of the pattern
            output_pattern=[[]]*pattern_len_in_steps
            # group the instruments and their velocity that played at a specific step
            i = 0
            for step in range(pattern_len_in_steps):
                step_content = [(x[1],x[2]) for x in pattern if x[0]==step]
                # remove repeated notes at the same step
                result = {}
                for tup in step_content:
                    note, vel = tup
                    if note not in result or vel > result[note][1]:
                        result[note] = tup

                step_content = list(result.values()) # dictionary to tuple list
                step_content.sort() #sort by note ascending
                output_pattern[step] = step_content

    ##################################
    # split the pattern every 16 steps
    ##################################
    hv_lists_split=[]
    for x in range(len(output_pattern)//16):
        patt_fragment = output_pattern[x*16:(x*16)+16]
        patt_density = sum([1 for x in patt_fragment if x!=[]])
        #############################################################
        # filter out patterns that have less than 4 events with notes
        #############################################################
        if patt_density > 4:
            hv_lists_split.append(patt_fragment)
    
    return hv_lists_split

def audio_to_tensor_matrix(audio_file_path, bars_length=None):
    """
    Convert an audio recording to a tensor matrix representation.
    
    Parameters:
    audio_file_path (str): Path to the audio file
    bars_length (int, optional): Number of bars to return. If None, returns all available bars.
    
    Returns:
    list: List of 3-band tensor matrices representing the drum pattern, one matrix per bar.
           Each matrix is a 3x16 tensor where the rows represent low, mid, and high frequency bands.
    """
    # Step 1: Determine the directory containing the audio file
    audio_dir = os.path.dirname(audio_file_path)
    audio_filename = os.path.basename(audio_file_path)
    
    # Step 2: Look for corresponding MIDI file with the same base name but .mid extension
    base_name = os.path.splitext(audio_filename)[0]
    midi_path = None
    
    # Try a few common patterns for midi file naming
    midi_candidates = [
        os.path.join(audio_dir, f"{base_name}.mid"),
        os.path.join(audio_dir, f"{base_name}.midi"),
        os.path.join(audio_dir, f"{base_name}_midi.mid")
    ]
    
    for candidate in midi_candidates:
        if os.path.exists(candidate):
            midi_path = candidate
            break
    
    if not midi_path:
        raise FileNotFoundError(f"Could not find corresponding MIDI file for {audio_filename}")
    
    # Step 3: Load audio file to get BPM if needed
    try:
        audio, fs = torchaudio.load(audio_file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}")
    
    # Step 4: Process MIDI file to get hit-velocity lists
    try:
        hv_lists = midifile2hv_list(midi_path, "all")
    except Exception as e:
        raise RuntimeError(f"Error processing MIDI file: {e}")
    
    if not hv_lists:
        raise ValueError("No valid drum patterns found in the MIDI file")
    
    # Step 5: Convert hv_lists to 3-band representations
    midi_reps = []
    for hv_list in hv_lists:
        # Use the separate_hv_list function to get 3-band representation
        band_matrix = separate_hv_list(hv_list)
        # Convert to PyTorch tensor
        tensor_matrix = torch.tensor(band_matrix, dtype=torch.float32)
        midi_reps.append(tensor_matrix)
    
    # Step 6: Limit number of bars if specified
    if bars_length is not None and bars_length > 0:
        midi_reps = midi_reps[:bars_length]
    
    return midi_reps

# Usage example:
# tensor_matrices = audio_to_tensor_matrix("path/to/audio.wav", bars_length=4)



if __name__ == "__main__":
    import sys
    import os
    audio_file_path="/Users/danielmartinezvillegas/Downloads/groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.wav"
    matrices = audio_to_tensor_matrix(audio_file_path=audio_file_path, bars_length=8)
    print(matrices)
