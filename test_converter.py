MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
    'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
    '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----'
}

def msg_to_pulses(msg):
    pulses = []
    words = msg.upper().split(' ')

    for word in words:
        word_pulses = []
        for char in word:
            if char in MORSE_CODE_DICT:
                morse = MORSE_CODE_DICT[char]

                char_pulses_with_zeros = []
                for i, sym in enumerate(morse):
                    if i > 0:
                        char_pulses_with_zeros.append('0')
                    if sym == '.':
                        char_pulses_with_zeros.append('1')
                    else:
                        char_pulses_with_zeros.append('111')

                word_pulses.append(''.join(char_pulses_with_zeros))
        pulses.append('0'.join(word_pulses))
    return '0'.join(pulses)

print(msg_to_pulses("SOS"))