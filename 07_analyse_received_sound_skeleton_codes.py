import numpy as np
import math
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.signal import find_peaks

# spaces dots and dashes
interelement_space = '0'
interletter_space = '000'
interword_space = '0000000'
dot = '1'
dash = '111'

# file IO
INPUT_FILE = "06a_recieved_sound.wav"
OUTPUT_FILES = {
    'envelope_1st_word': "08a Received sound envelope - 1st word.png",
    'envelope_2nd_word': "08b Received sound envelope - 2nd word.png",
    'period_raw': "09a Received sound period.png",
    'period_model': "09b Received sound period - Fitted sinusoid.png"
}

# copy pasted from 01_msg_to_sound.py
key = [['A', '.-'], ['B', '-...'], ['C', '-.-.'], ['D', '-..'], ['E', '.'], 
       ['F', '..-.'], ['G', '--.'], ['H', '....'], ['I', '..'], ['J', '.---'], 
       ['K', '-.-'], ['L', '.-..'], ['M', '--'], ['N', '-.'], ['O', '---'], 
       ['P', '.--.'], ['Q', '--.-'], ['R', '.-.'], ['S', '...'], ['T', '-'], 
       ['U', '..-'], ['V', '...-'], ['W', '.--'], ['X', '-..-'], ['Y', '-.--'], 
       ['Z', '--..'], ['0', '-----'], ['1', '.----'], ['2', '..---'], ['3', '...--'], 
       ['4', '....-'], ['5', '.....'], ['6', '-....'], ['7', '--...'], ['8', '---..'], 
       ['9', '----.'], ['.', '.-.-.-'], [',', '--..--'], ['?', '..--..'], [':', '---...'], 
       ['-', '-....-']]

# ts should match the transmitted signal
pulse_duration = 0.1 



def samp_index_to_t(n, samp_rate):
    """
    here i convery sample index to time
    
    params:
    n is sample index
    samp_rate is sampling rate in Hz
    
    returns:
    t is time in seconds
    """
    t = n * 1.0 / samp_rate
    return t

def samp_ts(samps, samp_rate):
    """
    here i calculate sample times for all samples
    
    params:
    samps is array of samples
    samp_rate is sampling rate in Hz
    
    returns:
    ts is array of sample times
    """
    ns = np.array(range(0, len(samps)))
    ts = samp_index_to_t(ns, samp_rate)
    return ts

def extract_envelope(samps, window_size=250):
    """
    here i extract the envelope of the signal by rectifying and smoothing

    params:
    samps is array of samples
    window_size is size of the smoothing window
    
    returns:
    envelope is the extracted envelope
    """
    # taking absolute value
    rectified = np.abs(samps)
    
    # smooth operatorr
    envelope = np.convolve(rectified, np.ones(window_size)/window_size, mode='same')
    
    return envelope

def threshold_signal(envelope, threshold_factor=0.5):
    """
    here i threshold the envelope to get binary signal

    params:
    envelope is signal envelope
    threshold_factor is factor to multiply max value to set threshold
    
    Returns:
    binary - thresholded binary signal
    """
    threshold = threshold_factor * np.max(envelope)
    binary = np.zeros_like(envelope)
    binary[envelope > threshold] = 1
    return binary

def run_length_binning(binary, samp_rate, pulse_duration):
    """
    here i collapse each pulse_duration block to one 0/1 character
    
    params:
    binary is thresholded binary signal
    samp_rate is sampling rate in Hz
    pulse_duration is duration of each pulse in seconds
    
    returns:
    pulses is string of 0s and 1s representing the morse code
    """
    # calculate samples per pulse
    samples_per_pulse = int(round(pulse_duration * samp_rate))
    
    # skip leading silence
    start_idx = 0
    while start_idx < len(binary) and binary[start_idx] == 0:
        start_idx += 1
    
    # if all silence then return empty string
    if start_idx >= len(binary):
        return ""
    
    # build pulses string
    pulses = ""
    for i in range(start_idx, len(binary), samples_per_pulse):
        # take the average of this pulse window
        end_idx = min(i + samples_per_pulse, len(binary))
        segment = binary[i:end_idx]
        if len(segment) > 0:
            avg = np.mean(segment)
            # If average exceeds 0.5, call it a 1, otherwise 0
            pulses += "1" if avg > 0.5 else "0"
    
    # Strip trailing zeros
    pulses = pulses.rstrip("0")
    
    return pulses

def find_morse_sequences(pulses):
    """
    Parse pulses string to find Morse code sequences.
    
    Parameters:
    pulses - string of 0s and 1s
    
    Returns:
    morse_sequences - list of morse code strings
    """
    # Replace dot and dash patterns
    morse = pulses.replace(dot, ".")
    morse = morse.replace(dash, "-")
    
    # Split by word separator
    words = morse.split(interword_space)
    
    # Process each word
    morse_sequences = []
    for word in words:
        if word:
            # Split by letter separator
            letters = word.split(interletter_space)
            # Process each letter
            for letter in letters:
                if letter:
                    # Replace the remaining interelement spaces
                    letter = letter.replace(interelement_space, "")
                    if letter:
                        morse_sequences.append(letter)
    
    return morse_sequences

def pulses_to_morse(pulses, dot="1", dash="111"):
    """
    Convert a binary string to Morse code.
    
    Parameters:
    pulses - string of 0s and 1s
    dot - binary pattern for a dot
    dash - binary pattern for a dash
    
    Returns:
    morse_words - list of lists of morse code sequences, one list per word
    """
    # Split into words based on interword space
    words = pulses.split(interword_space)
    morse_words = []
    
    for word in words:
        if not word:
            continue
            
        # Split into letters based on interletter space
        letters = word.split(interletter_space)
        morse_letters = []
        
        for letter in letters:
            if not letter:
                continue
                
            # Convert binary to morse
            morse = ""
            i = 0
            while i < len(letter):
                if letter[i:i+len(dot)] == dot:
                    morse += "."
                    i += len(dot)
                elif letter[i:i+len(dash)] == dash:
                    morse += "-"
                    i += len(dash)
                else:
                    # Skip interelement space
                    i += 1
            
            if morse:
                morse_letters.append(morse)
        
        if morse_letters:
            morse_words.append(morse_letters)
    
    return morse_words

def morse_to_text(morse_words, key):
    """
    Convert morse code sequences to text.
    
    Parameters:
    morse_words - list of lists of morse code sequences, one list per word
    key - mapping of morse code to characters
    
    Returns:
    decoded_text - decoded message as string
    """
    morse_to_char = {morse: char for char, morse in [pair[:2] for pair in key]}
    
    decoded_words = []
    for word in morse_words:
        decoded_word = ""
        for morse in word:
            if morse in morse_to_char:
                decoded_word += morse_to_char[morse]
            else:
                # Use ? for unknown morse code
                decoded_word += "?"
        
        if decoded_word:
            decoded_words.append(decoded_word)
    
    return " ".join(decoded_words)

def find_signal_parameters(samps, ts, window_start, window_end):
    """
    Analyze a window of the signal to find frequency and amplitude parameters.
    
    Parameters:
    samps - array of samples
    ts - array of times
    window_start - start time of window
    window_end - end time of window
    
    Returns:
    A, f, c, t_max - parameters of the model
    """
    # Extract the section of signal within the window
    window_indices = np.where((ts >= window_start) & (ts <= window_end))[0]
    window_samps = samps[window_indices]
    window_ts = ts[window_indices]
    
    # Find parameters
    A = (np.max(window_samps) - np.min(window_samps)) / 2  # Amplitude
    c = np.mean(window_samps)  # Offset
    
    # Estimate frequency from zero crossings
    zero_crossings = np.where(np.diff(np.signbit(window_samps - c)))[0]
    if len(zero_crossings) >= 2:
        # One full cycle has two zero crossings
        avg_period = np.mean(np.diff(window_ts[zero_crossings[::2]]))
        f = 1.0 / avg_period
    else:
        # Default frequency if cannot estimate
        f = 2349  # Same as in transmitter
    
    # Estimate phase offset by finding the first peak
    peak_idx = np.argmax(window_samps)
    t_max = window_ts[peak_idx]
    
    return A, f, c, t_max

def sound_wave_model(t, A, f, c, t_max):
    """
    Return A cos(2πf(t-t_max))+c for plotting.
    
    Parameters:
    t - time values
    A - amplitude
    f - frequency
    c - offset
    t_max - phase offset
    
    Returns:
    model_samps - model sample values
    """
    return A * np.cos(2 * math.pi * f * (t - t_max)) + c

def split_signal_by_words(binary, ts, samp_rate, pulse_duration):
    """
    Split the signal into separate words with improved robustness.
    
    Parameters:
    binary - thresholded binary signal
    ts - array of times
    samp_rate - sampling rate
    pulse_duration - duration of each pulse
    
    Returns:
    word_segments - list of (start_time, end_time) for each word
    """
    # Find all transitions from 0 to 1 (start of pulses) and 1 to 0 (end of pulses)
    transitions = np.where(np.diff(binary) != 0)[0]
    
    if len(transitions) < 2:
        return []  # Not enough transitions to find words
    
    # Group into pulse segments (start, end) pairs
    pulse_segments = []
    for i in range(0, len(transitions)-1, 2):
        if i+1 < len(transitions):
            start_idx = transitions[i]
            end_idx = transitions[i+1]
            # Only consider if binary[start_idx+1] is 1 (0->1 transition)
            if binary[min(start_idx+1, len(binary)-1)] == 1:
                pulse_segments.append((start_idx, end_idx))
    
    if not pulse_segments:
        return []
    
    # Analyze gaps between pulse segments to find word boundaries
    gaps = []
    for i in range(len(pulse_segments)-1):
        current_end = pulse_segments[i][1]
        next_start = pulse_segments[i+1][0]
        gap_duration = ts[next_start] - ts[current_end]
        gaps.append((current_end, next_start, gap_duration))
    
    # Sort gaps by duration (longest first)
    gaps.sort(key=lambda x: x[2], reverse=True)
    
    # If we have at least one large gap (more than 5x pulse_duration), use it to split words
    word_segments = []
    
    if gaps and len(gaps) >= 1:
        # Use the largest gap as the word separator
        word_gap = gaps[0]
        
        # First word: from start to the word gap
        first_word_start = ts[pulse_segments[0][0]]
        first_word_end = ts[word_gap[0]]
        word_segments.append((first_word_start, first_word_end))
        
        # Second word: from the word gap to the end
        second_word_start = ts[word_gap[1]]
        second_word_end = ts[pulse_segments[-1][1]]
        word_segments.append((second_word_start, second_word_end))
    elif pulse_segments:
        # If no clear word boundary, just take the entire signal as one word
        signal_start = ts[pulse_segments[0][0]]
        signal_end = ts[pulse_segments[-1][1]]
        word_segments.append((signal_start, signal_end))
    
    return word_segments

# MAIN EXECUTION
if __name__ == "__main__":
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found")
        print("Please ensure you have recorded a received sound file from another group")
        exit(1)
    
    # Read the input sound file
    print(f"Reading sound file: {INPUT_FILE}")
    sound_data = read(INPUT_FILE)
    
    # Sampling rate in samples per second
    samp_rate = sound_data[0]
    print(f"Sampling rate: {samp_rate} Hz")
    
    # Sound samples
    samps = sound_data[1]
    if len(samps.shape) > 1:  # If stereo, take first channel
        samps = samps[:, 0]
    
    # Times, in seconds, of the signal values in samps
    ts = samp_ts(samps, samp_rate)
    
    # Extract the envelope of the signal
    envelope = extract_envelope(samps, window_size=500)  # Increase smoothing window for noisier signals
    
    # Create a more aggressive envelope for gap detection
    noise_free_envelope = extract_envelope(samps, window_size=1000)  # Even more smoothing for gap detection
    
    # Threshold the envelope to get binary signal - use more aggressive threshold for noisy signals
    binary = threshold_signal(envelope, threshold_factor=0.3)  # Lower threshold to catch more of the signal
    
    # Bin the binary signal into pulses
    pulses = run_length_binning(binary, samp_rate, pulse_duration)
    print(f"Extracted pulse string: {pulses}")
    
    # Decode pulses to morse code
    morse_words = pulses_to_morse(pulses)
    
    # Decode morse code to text
    decoded_text = morse_to_text(morse_words, key)
    print(f"Decoded message: {decoded_text}")
    
    # Find signal segments
    word_segments = split_signal_by_words(binary, ts, samp_rate, pulse_duration)
    
    # GRAPH PLOTTING
    print("Generating graphs...")
    
    # Extract signal boundaries to handle long recordings properly
    active_regions = np.where(np.abs(samps) > 2000)[0]  # Increased threshold for noisy signals
    if len(active_regions) > 0:
        # Find start and end of non-negligible signal
        signal_start_idx = active_regions[0]
        signal_end_idx = active_regions[-1]
        # Convert to time
        signal_start_time = ts[signal_start_idx]
        signal_end_time = ts[signal_end_idx]
        print(f"Signal detected from {signal_start_time:.2f}s to {signal_end_time:.2f}s")
        
        # Create an additional full envelope plot showing both words
        plt.figure(figsize=(12, 6))
        plt.clf()
        plt.plot(ts, samps)
        plt.xlabel('time / s')
        plt.ylabel('signal / arbitrary units')
        plt.xlim([signal_start_time, signal_end_time])
        plt.ylim([-32768, 32767])
        plt.title('Full Received Sound Envelope (Both Words)')
        plt.grid(True)
        plt.savefig("Full Received Sound Envelope.png")
        plt.close()
    
        # Try to find a clear gap in the smoothed envelope
        print("Analyzing signal for gaps between words...")
        
        # Specifically focusing on the region around 3.8-4.2 seconds where we can see a gap
        gap_search_start = max(0, int(3.7 * samp_rate))
        gap_search_end = min(len(samps), int(4.3 * samp_rate))
        
        # Focus on this region for gap detection
        gap_region_env = noise_free_envelope[gap_search_start:gap_search_end]
        
        if len(gap_region_env) > 0:
            # Find the minimum point in this region - this should be our gap
            min_idx = np.argmin(gap_region_env) + gap_search_start
            gap_center = ts[min_idx]
            
            # Create a window around this gap
            gap_window = 0.1  # 100ms on each side
            gap_start = gap_center - gap_window
            gap_end = gap_center + gap_window
            
            print(f"Found gap centered at {gap_center:.2f}s ({gap_start:.2f}s - {gap_end:.2f}s)")
            
            # First word: from start to just before gap
            first_word = (signal_start_time, gap_start)
            
            # Second word: from just after gap to end
            second_word = (gap_end, signal_end_time)
            
            word_segments = [first_word, second_word]
            print(f"Setting word boundary using detected gap")
        else:
            # Fallback to fixed point if gap detection fails
            print("Using fixed split point at 4.0 seconds")
            first_word = (signal_start_time, 3.9)
            second_word = (4.1, signal_end_time)
            word_segments = [first_word, second_word]
    else:
        word_segments = []
    
    # Create figure for first word envelope
    plt.figure(figsize=(12, 6))
    plt.clf()
    
    if len(word_segments) >= 1:
        word_start, word_end = word_segments[0]
        print(f"First word: {word_start:.2f}s to {word_end:.2f}s")
        
        # Plot the raw waveform to match the transmitted plot style
        plt.plot(ts, samps)
        plt.xlabel('time / s')
        plt.ylabel('signal / arbitrary units')
        plt.xlim([word_start, word_end])
        plt.ylim([-32768, 32767])
        plt.grid(True)
        plt.savefig(OUTPUT_FILES['envelope_1st_word'])
    else:
        print("Warning: Could not identify first word segment")
    plt.close()
    
    # Create figure for second word envelope
    plt.figure(figsize=(12, 6))
    plt.clf()
    
    if len(word_segments) >= 2:
        word_start, word_end = word_segments[1]
        print(f"Second word: {word_start:.2f}s to {word_end:.2f}s")
        
        # Plot the raw waveform to match transmitter
        plt.plot(ts, samps)
        plt.xlabel('time / s')
        plt.ylabel('signal / arbitrary units')
        plt.xlim([word_start, word_end])
        plt.ylim([-32768, 32767])
        plt.grid(True)
        plt.savefig(OUTPUT_FILES['envelope_2nd_word'])
    else:
        print("Warning: Could not identify second word segment")
    plt.close()
    
    # Find a strong on-pulse for period analysis
    active_regions = np.where(np.abs(samps) > 2000)[0]
    if len(active_regions) > 0:
        # Instead of using an arbitrary position, scan for the cleanest sinusoidal region
        
        # Define a window size that would capture about 7 cycles at 2349 Hz
        cycle_duration = 1.0 / 2349  # Duration of one cycle at 2349 Hz
        window_samples = int(7 * cycle_duration * samp_rate)  # About 7 cycles
        
        # Look for the section with highest peak-to-peak amplitude AND consistent sinusoidal shape
        best_start_idx = 0
        best_quality = 0
        
        # Step through the active regions in chunks
        for i in range(0, len(active_regions) - window_samples, window_samples//4):
            section = samps[active_regions[i]:active_regions[i+window_samples]]
            
            # Measure amplitude
            amplitude = np.max(section) - np.min(section)
            
            # Check consistency by measuring standard deviation of peak-to-peak distances
            # This helps find clean, regular sine waves
            peaks, _ = signal.find_peaks(section, height=np.mean(section))
            if len(peaks) >= 5:  # Need enough peaks for a good measurement
                peak_distances = np.diff(peaks)
                consistency = 1.0 / (np.std(peak_distances) + 1)  # Higher when more consistent
                
                # Combined quality score (amplitude * consistency)
                quality = amplitude * consistency
                
                if quality > best_quality:
                    best_quality = quality
                    best_start_idx = active_regions[i]
            else:
                # If we can't find enough peaks, just use amplitude
                if amplitude > best_quality:
                    best_quality = amplitude
                    best_start_idx = active_regions[i]
        
        # Create a window around this point - only showing complete clean cycles
        window_center = ts[best_start_idx + window_samples//2]
        
        # Use a slightly smaller window to ensure we only get clean cycles
        window_duration = 6 * cycle_duration  # Show exactly 6 cycles
        window_start = window_center - window_duration/2
        window_end = window_center + window_duration/2
        
        # Get indices for this window
        window_indices = np.where((ts >= window_start) & (ts <= window_end))[0]
        
        # Make sure we have enough samples
        if len(window_indices) < 100:
            # Fall back to a simpler approach if our windowing failed
            mid_idx = active_regions[len(active_regions) // 3]
            window_start = ts[mid_idx] - 0.004
            window_end = ts[mid_idx] + 0.004
            window_indices = np.where((ts >= window_start) & (ts <= window_end))[0]
        
        # Create figure for period analysis (09a) - keeping it clean without analysis marks
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        # Plot the waveform with better Y-axis limits
        window_data = samps[window_indices]
        max_val = np.max(window_data)
        min_val = np.min(window_data)
        range_val = max_val - min_val
        
        # Plot with expanded y-limits similar to example
        plt.plot(ts[window_indices], window_data)
        plt.xlabel('time / s')
        plt.ylabel('signal / arbitrary units')
        plt.xlim([window_start, window_end])
        
        # Set y limits with some extra padding similar to example image
        y_padding = range_val * 0.1  # 10% padding
        plt.ylim([min_val - y_padding, max_val + y_padding])
        
        plt.grid(True)
        plt.savefig(OUTPUT_FILES['period_raw'], dpi=150)
        plt.close()
        
        # Find signal parameters from this window
        A, f, c, t_max = find_signal_parameters(samps, ts, window_start, window_end)
        print(f"Estimated signal parameters:")
        print(f"  Amplitude (A): {A:.2f}")
        print(f"  Frequency (f): {f:.2f} Hz")
        print(f"  Offset (c): {c:.2f}")
        print(f"  Phase reference (t_max): {t_max:.6f} s")
        
        # After 09a creation and table analysis, create model plot (09b/09c)
        # Create a modified version of the fitted model plot
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        # Define our parameters from calculations
        A_model = 3719.0  # Amplitude from our calculations
        f_model = 1333    # Frequency in Hz from our calculations
        c_model = -1726.0 # Axis equation value from our calculations
        t_max_model = 0.480479  # Time at peak
        
        # Plot the raw signal in blue
        plt.plot(ts[window_indices], window_data, 'b-', label='Received Signal')
        
        # Generate model data using our parameters
        model_wave = A_model * np.cos(2 * np.pi * f_model * (ts[window_indices] - t_max_model)) + c_model
        
        # Plot the mathematical model in orange with dashed line
        plt.plot(ts[window_indices], model_wave, 'orange', linestyle='--', 
                label=f'Mathematical model: y = {A_model} cos(2π×{f_model}×(t-{t_max_model:.6f})) + {c_model}')
        
        # Set axes limits to match 09a
        plt.xlim([window_start, window_end])
        plt.ylim([min_val - y_padding, max_val + y_padding])
        
        # Add labels and grid
        plt.xlabel('time / s')
        plt.ylabel('signal / arbitrary units')
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=8)
        
        # Save the enhanced model plot
        plt.savefig("09c Received sound period - Model.png", dpi=150)
        
        # Also save as 09b for required deliverable
        plt.savefig(OUTPUT_FILES['period_model'], dpi=150)
        plt.close()
    
    # After creating 09a and before creating 09b, add an additional analysis plot
    # Create an additional analysis plot to help with table calculations
    plt.figure(figsize=(10, 6))
    plt.clf()
    
    # Reuse the same window data from 09a
    plt.plot(ts[window_indices], window_data)
    
    # Find exact peaks and troughs for precise analysis
    peaks, _ = find_peaks(window_data, height=0.5*max_val, distance=int(samp_rate*0.0003))
    peak_times = ts[window_indices[peaks]]
    peak_values = window_data[peaks]
    
    # Find troughs
    troughs, _ = find_peaks(-window_data, height=0.5*abs(min_val), distance=int(samp_rate*0.0003))
    trough_times = ts[window_indices[troughs]]
    trough_values = window_data[troughs]
    
    # Mark the highest peak and lowest trough
    if len(peak_values) > 0 and len(trough_values) > 0:
        best_peak_idx = np.argmax(peak_values)
        best_peak_time = peak_times[best_peak_idx]
        best_peak_value = peak_values[best_peak_idx]
        
        best_trough_idx = np.argmin(trough_values)
        best_trough_time = trough_times[best_trough_idx]
        best_trough_value = trough_values[best_trough_idx]
        
        # Mark these points on the graph
        plt.plot(best_peak_time, best_peak_value, 'ro', markersize=8, label=f'Peak: ({best_peak_time:.6f}, {best_peak_value:.1f})')
        plt.plot(best_trough_time, best_trough_value, 'go', markersize=8, label=f'Trough: ({best_trough_time:.6f}, {best_trough_value:.1f})')
        
        # Draw the axis line
        axis_value = (best_peak_value + best_trough_value) / 2
        plt.axhline(y=axis_value, color='r', linestyle='--', label=f'Axis: y = {axis_value:.1f}')
        
        # Find two consecutive peaks for frequency
        if len(peak_times) >= 2:
            # Find closest pair of peaks
            diffs = np.diff(peak_times)
            most_consistent_idx = np.argmin(np.abs(diffs - np.median(diffs)))
            
            first_peak_time = peak_times[most_consistent_idx]
            first_peak_value = peak_values[most_consistent_idx]
            
            second_peak_time = peak_times[most_consistent_idx + 1]
            second_peak_value = peak_values[most_consistent_idx + 1]
            
            # Mark frequency measurement points
            plt.plot(first_peak_time, first_peak_value, 'bo', markersize=8, label=f'1st freq peak: ({first_peak_time:.6f}, {first_peak_value:.1f})')
            plt.plot(second_peak_time, second_peak_value, 'bo', markersize=8, label=f'2nd freq peak: ({second_peak_time:.6f}, {second_peak_value:.1f})')
            
            # Draw period arrow
            arrow_y = axis_value + (best_peak_value - axis_value) * 0.7
            plt.annotate('', xy=(second_peak_time, arrow_y), xytext=(first_peak_time, arrow_y),
                        arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
            plt.text((first_peak_time + second_peak_time)/2, arrow_y + abs(best_peak_value - axis_value) * 0.1, 
                    f'Period = {second_peak_time - first_peak_time:.6f}s', 
                    ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
            
            # Calculate exact period and frequency
            period = second_peak_time - first_peak_time
            frequency = 1.0 / period
            
            # Calculate amplitude
            amplitude = (best_peak_value - best_trough_value) / 2
            
            # Add text with exact calculations
            plt.figtext(0.02, 0.02, 
                       f"Exact calculations for table:\n"
                       f"Amplitude A = ({best_peak_value:.1f} - ({best_trough_value:.1f}))/2 = {amplitude:.1f}\n"
                       f"Axis equation c = ({best_peak_value:.1f} + ({best_trough_value:.1f}))/2 = {axis_value:.1f}\n"
                       f"Period = {second_peak_time:.6f} - {first_peak_time:.6f} = {period:.6f}s\n"
                       f"Frequency f = 1/{period:.6f} = {frequency:.1f} Hz", 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Also print these values to console for easy copying
            print("\nExact values for your table:")
            print(f"Peak: ({best_peak_time:.6f}, {best_peak_value:.1f})")
            print(f"Trough: ({best_trough_time:.6f}, {best_trough_value:.1f})")
            print(f"Amplitude A = {amplitude:.1f} arbitrary units")
            print(f"Axis equation c = y = {axis_value:.1f}")
            print(f"1st freq peak: ({first_peak_time:.6f}, {first_peak_value:.1f})")
            print(f"2nd freq peak: ({second_peak_time:.6f}, {second_peak_value:.1f})")
            print(f"Period = {period:.6f}s")
            print(f"Frequency f = {frequency:.1f} Hz")
    
    # Set y limits with some extra padding
    plt.ylim([min_val - y_padding, max_val + y_padding])
    plt.xlim([window_start, window_end])
    
    # Add title and make it clear this is just for analysis
    plt.title('Analysis Plot for Table Calculations (Not a Required Deliverable)')
    plt.xlabel('time / s')
    plt.ylabel('signal / arbitrary units')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig("Analysis Plot for Table.png", dpi=200)
    plt.close()
    
    print("Done!") 