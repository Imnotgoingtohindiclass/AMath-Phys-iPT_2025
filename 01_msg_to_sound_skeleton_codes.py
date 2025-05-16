# getting things ready

# import os

# import pip
# pip.main(["install","numpy"])
import numpy as np
import math
import os
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

msg = "  SOUND WAVE  "

clean_msg = ''.join(c for c in msg.strip() if c.isalnum())

OUTPUT_FILES = {
    'morse_sound': "02_sound.wav",
    'envelope': "03 Transmitted sound envelope.png",
    'period': "04 Transmitted sound period.png"
}

interelement_space = '0';
interletter_space = '000';
interword_space = '0000000';
dot = '1';
dash = '111';

key = \
[['A', '.-'], ['B', '-...'], ['C', '-.-.'], ['D', '-..'], ['E', '.'], \
 ['F', '..-.'], ['G', '--.'], ['H', '....'], ['I', '..'], ['J', '.---'], \
 ['K', '-.-'], ['L', '.-..'], ['M', '--'], ['N', '-.'], ['O', '---'], \
 ['P', '.--.'], ['Q', '--.-'], ['R', '.-.'], ['S', '...'], ['T', '-'], \
 ['U', '..-'], ['V', '...-'], ['W', '.--'], ['X', '-..-'], ['Y', '-.--'], \
 ['Z', '--..'], \
 ['0', '-----'], ['1', '.----'], ['2', '..---'], ['3', '...--'], ['4', '....-'], \
 ['5', '.....'], ['6', '-....'], ['7', '--...'], ['8', '---..'], ['9', '----.'], \
 ['.', '.-.-.-'], [',', '--..--'], ['?', '..--..'], [':', '---...'], ['-', '-....-']];

# how long each pulse should be
pulse_duration = 0.1;  # seconds
pulse_rate = 1.0/pulse_duration;  # pulses per second

# sound settings
sound_freq = 2349;  # the beep frequency in Hz (from the assignment)
sound_amplitude = 20000;  # how loud the beep is (max 32767)

# audio quality settings
samp_rate = 44100;  # samples per second (CD quality)

# all the functions we need

def morse_to_pulses(morse, interelement_space, dot, dash):
    """
    Convert a dot/dash string to the corresponding 0/1 pulse string.
    
    Parameters:
    morse - string containing only dots and dashes
    interelement_space - character used for space between elements
    dot - character used for dot
    dash - character used for dash
    
    Returns:
    pulses - string of 0s and 1s representing the Morse code
    """
    # turns dots and dashes into 1s and 0s
    # NO SPACES!!!!
    pulses = '';
    j = 0;
    j_max = len(morse) - 1;
    while j <= j_max:

        if j != 0:
            pulses = pulses + interelement_space;

        if morse[j] == '.':
            pulses = pulses + dot;
        elif morse [j] == '-':
            pulses = pulses + dash;
        else:
            print('whoops! only dots and dashes allowed here');
            [0][1];  # crash the program
        j = j + 1;

    return pulses;

def add_pulses_to_key(key, interelement_space, dot, dash):
    """
    Append the pulse string to every [letter, morse] pair and return updated key.
    
    Parameters:
    key - list of lists, each inner list containing a character and its Morse code
    interelement_space - character used for space between elements
    dot - character used for dot
    dash - character used for dash
    
    Returns:
    key - updated key with pulse strings appended
    """
    # adds the binary pattern to each letter in our lookup table
    j = 0;
    j_max = len(key) - 1;
    while j <= j_max:
        morse = key[j][1];
        key[j].append(morse_to_pulses(morse, interelement_space, dot, dash));
        j = j + 1;

    return key;

def letter_to_pulses(letter, key):
    """
    Look up one letter in key and return its pulse string.
    
    Parameters:
    letter - single character to look up
    key - list of lists containing characters, Morse codes, and pulse strings
    
    Returns:
    pulses - string of 0s and 1s for the letter
    """
    # converts a single letter to its beep pattern
    ok_letters = [row[0] for row in key];
    pulses = [row[2] for row in key];
    j = ok_letters.index(letter);
    return pulses[j];

def msg_to_pulses(msg, key, interletter_space, interword_space):
    """
    Convert the whole padded message to one long pulse string.
    
    Parameters:
    msg - string message to convert
    key - list of lists containing characters, Morse codes, and pulse strings
    interletter_space - space between letters
    interword_space - space between words
    
    Returns:
    pulses - string of 0s and 1s for the entire message
    """
    # turns your whole message into a beep pattern
    pulses = '';
    j = 0;
    j_max = len(msg) - 1;
    while j <= j_max:

        if msg[j] == ' ':
            pulses = pulses + interword_space;
        else:
            if j != 0:
                if msg[j-1] != ' ':
                    pulses = pulses + interletter_space;
            
            pulses = pulses + letter_to_pulses(msg[j], key);

        j = j + 1;
    
    return pulses;

def pulse_index_to_start_t(j, pulse_duration):
    """
    Return the start-time (s) of the j-th pulse in the overall string.
    
    Parameters:
    j - index of the pulse
    pulse_duration - duration of each pulse in seconds
    
    Returns:
    t - start time in seconds
    """
    # figures out when a pulse should start
    t = j * pulse_duration;
    return t;

def t_to_samp_index(t, samp_rate):
    """
    Convert time to sample index.
    
    Parameters:
    t - time in seconds
    samp_rate - sampling rate in Hz
    
    Returns:
    n - sample index
    """
    # converts time to sample number
    n = int(round(t * samp_rate));
    return n;

def samp_ts(start_t, end_t, samp_rate):
    """
    Calculate sample times between start and end time.
    
    Parameters:
    start_t - start time in seconds
    end_t - end time in seconds
    samp_rate - sampling rate in Hz
    
    Returns:
    ts - array of sample times
    """
    # makes a list of times for all our samples
    start_n = t_to_samp_index(start_t, samp_rate);
    end_n = t_to_samp_index(end_t, samp_rate);
    
    # make a list from start to end
    ns = np.arange(start_n, end_n + 1);
    
    # convert to actual times
    ts = ns / samp_rate;
    
    return ts;

def tone(sound_freq, sound_amplitude, start_t, end_t, samp_rate):
    """
    Generate a NumPy array of the sinusoid that spans exactly start_t to end_t.
    
    Parameters:
    sound_freq - frequency of the tone in Hz
    sound_amplitude - amplitude of the tone
    start_t - start time in seconds
    end_t - end time in seconds
    samp_rate - sampling rate in Hz
    
    Returns:
    samps - array of sample values
    """
    # makes a beep sound between start and end time
    
    # get all the sample times
    ts = samp_ts(start_t, end_t, samp_rate);
    
    # math stuff for making waves
    omega = 2 * math.pi * sound_freq;
    
    # make the actual wave
    samps = sound_amplitude * np.sin(omega * ts);
    
    return samps;

def pulses_to_samps(pulses, pulse_duration, sound_freq, sound_amplitude, samp_rate):
    """
    Walk the whole 0/1 string, call tone for every pulse, concatenate, return samps.
    
    Parameters:
    pulses - string of 0s and 1s
    pulse_duration - duration of each pulse in seconds
    sound_freq - frequency of the tone in Hz
    sound_amplitude - amplitude of the tone
    samp_rate - sampling rate in Hz
    
    Returns:
    samps - array of sample values for the entire signal
    """
    # turns our beep pattern into actual sound data
    
    # start with empty sound
    samps = np.array([]);
    
    # go through each pulse
    for j in range(len(pulses)):
        # when should this bit start and end?
        start_t = pulse_index_to_start_t(j, pulse_duration);
        end_t = pulse_index_to_start_t(j+1, pulse_duration);
        
        # make the sound for this bit
        if pulses[j] == '1':
            # make a beep for 1s
            pulse_samps = tone(sound_freq, sound_amplitude, start_t, end_t, samp_rate);
        else:
            # make silence for 0s
            pulse_samps = tone(sound_freq, 0, start_t, end_t, samp_rate);
        
        # add it to our sound
        samps = np.append(samps, pulse_samps);
    
    # round everything to whole numbers
    samps = np.round(samps);
    
    return samps;

# Main execution
if __name__ == '__main__':
    print("Input message:", msg)

    # make the morse code
    add_pulses_to_key(key, interelement_space, dot, dash)
    pulses = msg_to_pulses(msg, key, interletter_space, interword_space)
    print("Morse code pattern:", pulses)

    # make the sound
    samps = pulses_to_samps(pulses, pulse_duration, sound_freq, sound_amplitude, samp_rate)

    # figure out the timing
    ts = np.linspace(0, len(samps)/samp_rate, len(samps))

    # save the sound file
    write(OUTPUT_FILES['morse_sound'], samp_rate, samps.astype(np.int16))

    # make some cool graphs

    # first show the whole signal
    plt.figure(1, figsize=(12, 6))  # Wider figure for better visibility
    plt.clf()
    x_axis = ts
    y_axis = samps
    plt.plot(x_axis, y_axis)
    plt.ylim([-32768, 32767])
    plt.xlabel('time / s')
    plt.ylabel('signal / arbitrary units')
    plt.grid(True)
    plt.savefig(OUTPUT_FILES['envelope'])
    plt.close(1)

    # then zoom in to see the waves
    plt.figure(2)
    plt.clf()
    # find where the signal starts making noise
    non_zero_indices = np.where(np.abs(samps) > 1000)[0]
    if len(non_zero_indices) > 0:
        start_idx = non_zero_indices[0]
        start_time = ts[start_idx]
        plt.plot(ts[ts >= start_time][0:441], samps[start_idx:start_idx+441])
        plt.xlabel('time / s')
        plt.ylabel('signal / arbitrary units')
        plt.grid(True)
        plt.savefig(OUTPUT_FILES['period'])
    plt.close(2)

    print("Done!")
