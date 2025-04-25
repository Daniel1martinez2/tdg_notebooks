import numpy as np
import torch
import librosa
import os
import matplotlib.pyplot as plt

def extract_onsets_from_audio(audio_path, sr=22050, hop_length=512):
    """
    Extract onset information from audio file
    
    Parameters:
    audio_path (str): Path to audio file
    sr (int): Sample rate
    hop_length (int): Hop length for onset detection
    
    Returns:
    onset_times (np.ndarray): Onset times in seconds
    onset_strengths (np.ndarray): Strength of each onset
    onset_backtrack (np.ndarray): Precise onset timing by backtracking
    y (np.ndarray): Audio signal
    sr (int): Sample rate
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Detect onsets
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, 
                                              sr=sr, 
                                              hop_length=hop_length,
                                              backtrack=False)
    
    # Get onset times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    # Get precise onset times through backtracking
    onset_backtrack = librosa.onset.onset_detect(onset_envelope=onset_env,
                                               sr=sr,
                                               hop_length=hop_length,
                                               backtrack=True)
    
    onset_backtrack_times = librosa.frames_to_time(onset_backtrack, sr=sr, hop_length=hop_length)
    
    # Get onset strengths
    onset_strengths = onset_env[onset_frames]
    
    return onset_times, onset_strengths, onset_backtrack_times, y, sr

def classify_onsets_by_frequency(y, sr, onset_frames, hop_length=512, 
                               low_thresh=250, mid_thresh=2500):
    """
    Classify onsets into low, mid, and high frequency bands
    
    Parameters:
    y (np.ndarray): Audio signal
    sr (int): Sample rate
    onset_frames (np.ndarray): Frame indices of onsets
    hop_length (int): Hop length used for onset detection
    low_thresh (int): Upper threshold for low frequency band (Hz)
    mid_thresh (int): Upper threshold for mid frequency band (Hz)
    
    Returns:
    classifications (np.ndarray): Array of classifications (0: low, 1: mid, 2: high)
    energies (np.ndarray): Energy of each onset in all bands
    """
    classifications = []
    energies = []
    
    # Compute spectrogram
    D = np.abs(librosa.stft(y, hop_length=hop_length))
    
    # Map frequency bins to Hz
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Get frequency band indices
    low_band = freqs <= low_thresh
    mid_band = (freqs > low_thresh) & (freqs <= mid_thresh)
    high_band = freqs > mid_thresh
    
    for frame in onset_frames:
        # Get spectrum at this frame
        spec = D[:, frame]
        
        # Calculate energy in each band
        low_energy = np.sum(spec[low_band])
        mid_energy = np.sum(spec[mid_band])
        high_energy = np.sum(spec[high_band])
        
        # Store energy in all three bands
        energies_array = np.array([low_energy, mid_energy, high_energy])
        
        # Determine dominant band
        dominant_band = np.argmax(energies_array)
        
        classifications.append(dominant_band)
        energies.append(energies_array)  # Store all three energy values
    
    return np.array(classifications), np.array(energies)

def detect_tempo_and_beats(y, sr, start_bpm=120.0):
    """
    Detect tempo and beat positions from audio
    
    Parameters:
    y (np.ndarray): Audio signal
    sr (int): Sample rate
    start_bpm (float): Initial tempo estimate
    
    Returns:
    tempo (float): Estimated tempo in BPM
    beat_frames (np.ndarray): Frame indices of beats
    beat_times (np.ndarray): Times of beats in seconds
    """
    # Detect tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=start_bpm)
    
    # Convert beat frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    return tempo, beat_frames, beat_times

def quantize_onsets_to_16th_notes(onset_times, tempo, bars=1):
    """
    Quantize onsets to 16th note grid
    
    Parameters:
    onset_times (np.ndarray): Times of onsets in seconds
    tempo (float): Tempo in BPM
    bars (int): Number of bars to quantize
    
    Returns:
    quantized_steps (np.ndarray): Indices of 16th notes (0-15) for each onset
    """
    # Make sure tempo is a scalar
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo)
    
    # Duration of a 16th note in seconds
    sixteenth_duration = 60.0 / (tempo * 4)
    
    # Duration of a bar in seconds
    bar_duration = 16 * sixteenth_duration
    
    # Initialize array for quantized steps
    quantized_steps = []
    
    for onset_time in onset_times:
        # Determine which bar this onset belongs to
        bar_number = int(onset_time / bar_duration)
        
        # If we've exceeded our desired number of bars, stop
        if bar_number >= bars:
            break
            
        # Time since the beginning of this bar
        time_in_bar = onset_time - (bar_number * bar_duration)
        
        # Calculate nearest 16th note (0-15) within the bar
        step_in_bar = int(np.round(time_in_bar / sixteenth_duration))
        
        # Handle rounding to exactly 16 (which should be 0 of next bar)
        if step_in_bar == 16:
            step_in_bar = 0
        
        # Only add steps that fit within our bars
        if step_in_bar < 16:
            quantized_steps.append(step_in_bar + (bar_number * 16))
    
    return np.array(quantized_steps) % 16

def audio_to_tensor_matrix(audio_path, bars_length=1, start_bpm=None, quantize=True, all_bars=False):
    """
    Create a 3-band matrix representation from audio
    
    Parameters:
    audio_path (str): Path to audio file
    bars_length (int): Number of bars to analyze
    start_bpm (float): Initial tempo estimate. If None, will be auto-detected
    quantize (bool): Whether to quantize onsets to 16th notes grid
    all_bars (bool): If True, analyze all bars in the audio file
    
    Returns:
    list: List of 3-band tensor matrices representing the drum pattern, one matrix per bar.
          Each matrix is a 3x16 tensor where the rows represent low, mid, and high frequency bands.
    """
    try:
        # Extract onset information
        onset_times, onset_strengths, precise_onset_times, y, sr = extract_onsets_from_audio(audio_path)
        
        if len(onset_times) == 0:
            raise ValueError("No onsets detected in audio")
        
        # Convert onset times to frames for frequency analysis
        onset_frames = librosa.time_to_frames(onset_times, sr=sr)
        
        # Classify onsets by frequency band
        band_classifications, band_energies = classify_onsets_by_frequency(y, sr, onset_frames)
        
        # Detect tempo and beats
        if start_bpm is None:
            tempo, beat_frames, beat_times = detect_tempo_and_beats(y, sr)
        else:
            tempo, beat_frames, beat_times = detect_tempo_and_beats(y, sr, start_bpm=start_bpm)
        
        print(f"Detected tempo: {tempo} BPM")
        
        # Make sure tempo is a scalar
        if isinstance(tempo, np.ndarray):
            tempo_value = float(tempo)
        else:
            tempo_value = tempo
            
        # Calculate total bars in the audio
        audio_duration = librosa.get_duration(y=y, sr=sr)
        bar_duration = 60.0 / tempo_value * 4  # Duration of a bar in seconds
        total_bars = int(audio_duration / bar_duration) + 1  # +1 to include partial bar
        
        # Determine how many bars to process
        if all_bars:
            bars_to_process = total_bars
            print(f"Processing all {total_bars} bars in audio")
        else:
            bars_to_process = bars_length
            print(f"Processing {bars_to_process} bars")
        
        # Create a mapping from onsets to steps
        if quantize:
            steps = quantize_onsets_to_16th_notes(onset_times, tempo_value, bars_to_process)
        else:
            # If not quantizing, estimate the step based on onset time and tempo
            sixteenth_duration = 60.0 / (tempo_value * 4)
            steps = (onset_times / sixteenth_duration).astype(int) % 16
        
        # Initialize matrices (one per bar)
        matrices = []
        for bar in range(bars_to_process):
            # Create empty 3x16 matrix for this bar
            matrix = np.zeros((3, 16))
            
            # Fill matrix based on onsets in this bar
            for i, (onset_time, classification, energy_array) in enumerate(zip(onset_times, band_classifications, band_energies)):
                # Check if this onset belongs to the current bar
                if isinstance(tempo, np.ndarray):
                    tempo_value = float(tempo)
                else:
                    tempo_value = tempo
                bar_duration = 60.0 / tempo_value * 4  # Duration of a bar in seconds
                
                if bar * bar_duration <= onset_time < (bar + 1) * bar_duration:
                    # Get the step (0-15) for this onset within the bar
                    if quantize:
                        if i < len(steps):  # Check if index is valid
                            step = steps[i]
                        else:
                            continue
                    else:
                        sixteenth_duration = 60.0 / (tempo_value * 4)
                        step = int((onset_time - bar * bar_duration) / sixteenth_duration) % 16
                    
                    # Only proceed if step is valid
                    if 0 <= step < 16:
                        # Set a threshold for detecting significant energy in each band
                        # This helps populate the mid band even if it's not dominant
                        energy_threshold = 0.3 * np.max(energy_array)
                        
                        # Add energy to all bands that have significant energy
                        for band in range(3):
                            if energy_array[band] > energy_threshold:
                                matrix[band, step] += energy_array[band]
            
            # Normalize each row to have max value of 1
            for j in range(3):
                max_val = np.max(matrix[j])
                if max_val > 0:  # Avoid division by zero
                    matrix[j] = matrix[j] / max_val
            
            # Convert numpy matrix to PyTorch tensor
            tensor_matrix = torch.tensor(matrix, dtype=torch.float32)
            matrices.append(tensor_matrix)
        
        return matrices
    
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise

def plot_3band_matrix(matrix, title="3-Band Drum Pattern"):
    """
    Visualize a 3-band matrix as a heatmap
    
    Parameters:
    matrix: numpy array or torch tensor of shape (3, 16)
    title (str): Plot title
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
        
    plt.figure(figsize=(12, 4))
    plt.imshow(matrix, aspect='auto', cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    
    # Add grid to show 4 beats
    for beat in range(5):  # 0, 4, 8, 12, 16
        plt.axvline(x=beat*4-0.5, color='white', linestyle='-', linewidth=1, alpha=0.3)
    
    # Set labels
    plt.yticks([0, 1, 2], ['Low', 'Mid', 'High'])
    plt.xticks(range(16), [(i%4+1) if i%4==0 else "" for i in range(16)])
    plt.xlabel('Beat')
    plt.ylabel('Frequency Band')
    plt.title(title)
    plt.colorbar(label='Normalized Energy')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    audio_file_path = "/Users/macbookpro/Downloads/tdg_notebooks/1_funk_80_beat_4-4_aligned.wav"
    
    # Test with specific number of bars
    matrices = audio_to_tensor_matrix(audio_file_path, bars_length=2, start_bpm=138)
    
    # Print the first matrix
    print(f"Generated {len(matrices)} bar matrices")
    print("First matrix shape:", matrices[0].shape)
    print("First matrix:")
    print(matrices[0])
    
    # Test with all bars
    print("\n\nTesting with all_bars=True:")
    all_matrices = audio_to_tensor_matrix(audio_file_path, start_bpm=138, all_bars=True)
    
    # Print information about all matrices
    print(f"Generated {len(all_matrices)} bar matrices")
    print("Example matrix:")
    print(all_matrices[5])
    
    # Visualize the first matrix
    plot_3band_matrix(all_matrices[5], title="Drum Pattern - Bar 1")
