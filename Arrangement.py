import socket
import threading
import os
import sys
import math

from random import *

from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note, Rest
from midi2audio import FluidSynth

originalKeys = [
    [1, 'A0'], [2, 'A#0'], [3, 'B0'],
    [4, 'C1'], [5, 'C#1'], [6, 'D1'], [7, 'D#1'], [8, 'E1'], [9, 'F1'],
    [10, 'F#1'], [11, 'G1'], [12, 'G#1'], [13, 'A1'], [14, 'A#1'], [15, 'B1'],
    [16, 'C2'], [17, 'C#2'], [18, 'D2'], [19, 'D#2'], [20, 'E2'], [21, 'F2'],
    [22, 'F#2'], [23, 'G2'], [24, 'G#2'], [25, 'A2'], [26, 'A#2'], [27, 'B2'],
    [28, 'C3'], [29, 'C#3'], [30, 'D3'], [31, 'D#3'], [32, 'E3'], [33, 'F3'],
    [34, 'F#3'], [35, 'G3'], [36, 'G#3'], [37, 'A3'], [38, 'A#3'], [39, 'B3'],
    [40, 'C4'], [41, 'C#4'], [42, 'D4'], [43, 'D#4'], [44, 'E4'], [45, 'F4'],
    [46, 'F#4'], [47, 'G4'], [48, 'G#4'], [49, 'A4'], [50, 'A#4'], [51, 'B4'],
    [52, 'C5'], [53, 'C#5'], [54, 'D5'], [55, 'D#5'], [56, 'E5'], [57, 'F5'],
    [58, 'F#5'], [59, 'G5'], [60, 'G#5'], [61, 'A5'], [62, 'A#5'], [63, 'B5'],
    [64, 'C6'], [65, 'C#6'], [66, 'D6'], [67, 'D#6'], [68, 'E6'], [69, 'F6'],
    [70, 'F#6'], [71, 'G6'], [72, 'G#6'], [73, 'A6'], [74, 'A#6'], [75, 'B6'],
    [76, 'C7'], [77, 'C#7'], [78, 'D7'], [79, 'D#7'], [80, 'E7'], [81, 'F7'],
    [82, 'F#7'], [83, 'G7'], [84, 'G#7'], [85, 'A7'], [86, 'A#7'], [87, 'B7'],
    [88, 'C8']
]
tempKeys = [
    ['r', -1], ['f3', 0], ['g3', 1], ['a3', 2], ['b3', 3], ['c4', 4], ['d4', 5], ['e4', 6],
    ['f4', 7], ['g4', 8], ['a4', 9], ['b4', 10], ['c5', 11], ['d5', 12], ['e5', 13],
    ['f5', 14], ['g5', 15], ['a5', 16], ['b5', 17], ['c6', 18], ['d6', 19], ['e6', 20]
]
tempKeys2 = [
    ['r', -1], ['f3', 0], ['g3', 1], ['a3', 2], ['a#3', 3], ['c4', 4], ['d4', 5], ['e4', 6],
    ['f4', 7], ['g4', 8], ['a4', 9], ['a#4', 10], ['c5', 11], ['d5', 12], ['e5', 13],
    ['f5', 14], ['g5', 15], ['a5', 16], ['a#5', 17], ['c6', 18], ['d6', 19], ['e6', 20]
]
tempKeys3 = [
    ['r', -1], ['f#3', 0], ['g3', 1], ['a3', 2], ['b3', 3], ['c4', 4], ['d4', 5], ['e4', 6],
    ['f#4', 7], ['g4', 8], ['a4', 9], ['b4', 10], ['c5', 11], ['d5', 12], ['e5', 13],
    ['f#5', 14], ['g5', 15], ['a5', 16], ['b5', 17], ['c6', 18], ['d6', 19], ['e6', 20]
]
tempKeys4 = [
    ['r', -1], ['f3', 0], ['g3', 1], ['g#3', 2], ['a#3', 3], ['c4', 4], ['d4', 5], ['d#4', 6],
    ['f4', 7], ['g4', 8], ['g#4', 9], ['a#4', 10], ['c5', 11], ['d5', 12], ['d#5', 13],
    ['f5', 14], ['g5', 15], ['g#5', 16], ['a#5', 17], ['c6', 18], ['d6', 19], ['d#6', 20]
]
tempKeys5 = [
    ['r', -1], ['f3', 0], ['g3', 1], ['a3', 2], ['a#3', 3], ['c4', 4], ['d4', 5], ['d#4', 6],
    ['f4', 7], ['g4', 8], ['a4', 9], ['a#4', 10], ['c5', 11], ['d5', 12], ['d#5', 13],
    ['f5', 14], ['g5', 15], ['a5', 16], ['a#5', 17], ['c6', 18], ['d6', 19], ['d#6', 20]
]

song = [['c5', 8], ['r', 8], ['c5', 8], ['r', 8], ['c5', -8], ['a#4', 16], ['a4', 8], ['a#4', 8], ['c5', -8],
        ['c5', 16], ['c5', 8], ['d5', 8], ['c5', 4], ['r', 4], ['a#4', -8], ['a#4', 16], ['a#4', 8], ['a#4', 8],
        ['a4', -8], ['a4', 16], ['a4', 4], ['g4', -8], ['g4', 16], ['g4', 8], ['a4', 8], ['g4', 4], ['r', 4],
        ['c5', 8], ['r', 8], ['c5', 8], ['r', 8], ['c5', -8], ['a#4', 16], ['a4', 8], ['a#4', 8], ['c5', -8],
        ['c5', 16], ['c5', 8], ['d5', 8], ['c5', 4], ['r', 4], ['d5', -8], ['d5', 16], ['d5', 8], ['d5', 8],
        ['c5', -8], ['c5', 16], ['c5', 4], ['c5', -8], ['a#4', 16], ['a4', 8], ['g4', 8], ['f4', 4], ['r', 4],
        ['c4', -8], ['c4', 16], ['f4', 8], ['a4', 8], ['c5', -8], ['c5', 16], ['c5', 4], ['d5', -8], ['d5', 16],
        ['c5', 8], ['a#4', 8], ['a4', 4], ['r', 4], ['c4', -8], ['c4', 16], ['e4', 8], ['g4', 8], ['a#4', -8],
        ['a#4', 16], ['a#4', 4], ['a4', -8], ['a4', 16], ['g4', 8], ['g4', 8], ['f4', 4], ['r', 4]]

def make_midi(midi_path, notes, bpm, instrument, beat):
    note_names = 'c c# d d# e f f# g g# a  a# b'.split()

    result = NoteSeq()
    melody_dur = 0
    for n in notes:
        if (n[1] < 0):
            duration = (1.0 / -n[1]) + (1.0 / -n[1] / 2)
        else:
            duration = 1.0 / n[1]
        melody_dur += duration

        if n[0].lower() == 'r':
            result.append(Rest(dur=duration))
        else:
            pitch = n[0][:-1]
            octave = int(n[0][-1]) + 1
            pitch_number = note_names.index(pitch.lower())

            result.append(Note(pitch_number, octave=octave, dur=duration, volume=100))

    duration = 1.0 / beat
    harmony_len = math.ceil(melody_dur / duration)
    harmony_len += (4 - harmony_len % 4)

    pitch = [[0, 4, 7], [9, 0, 4], [2, 5, 9], [7, 11, 2, 5]]
    octave = [[5, 5, 5], [5, 6, 6], [5, 5, 5], [5, 5, 6, 6]]

    guitar0 = NoteSeq()
    for n in range(harmony_len):
        index = int(math.floor(n % 16 / 4))
        if (n >= math.ceil(melody_dur / duration)):
            volume = 50
        else:
            volume = 70
        guitar0.append(Note(pitch[index][0], octave=octave[index][0], dur=duration, volume=volume))

    guitar1 = NoteSeq()
    for n in range(harmony_len):
        index = int(math.floor(n % 16 / 4))
        if (n >= math.ceil(melody_dur / duration)):
            volume = 50
        else:
            volume = 70
        guitar1.append(Note(pitch[index][1], octave=octave[index][1], dur=duration, volume=volume))

    guitar2 = NoteSeq()
    for n in range(harmony_len):
        index = int(math.floor(n % 16 / 4))
        if (n >= math.ceil(melody_dur / duration)):
            volume = 50
        else:
            volume = 70
        guitar2.append(Note(pitch[index][2], octave=octave[index][2], dur=duration, volume=volume))

    guitar3 = NoteSeq()
    for n in range(harmony_len):
        index = int(math.floor(n % 16 / 4))
        if (n >= math.ceil(melody_dur / duration)):
            volume = 50
        else:
            volume = 70
        if (index == 3):
            guitar3.append(Note(pitch[index][3], octave=octave[index][3], dur=duration, volume=volume))
        else:
            guitar3.append(Rest(dur=duration))

    cymbal = NoteSeq()
    for n in range(harmony_len):
        index = int(math.floor(n % 16 / 4))
        if (n >= math.ceil(melody_dur / duration)):
            volume = 70
        else:
            volume = 100
        pitch = 10
        octave = 3
        cymbal.append(Note(pitch, octave=octave, dur=duration, volume=volume))

    kick = NoteSeq()
    for n in range(harmony_len):
        index = int(math.floor(n % 16 / 4))
        if (n >= math.ceil(melody_dur / duration)):
            volume = 70
        else:
            volume = 100
        pitch = 0
        octave = 3

        if (n % 4 == 0):
            kick.append(Note(pitch, octave=octave, dur=duration, volume=volume))
        else:
            kick.append(Rest(dur=duration))

    snare = NoteSeq()
    for n in range(harmony_len):
        index = int(math.floor(n % 16 / 4))
        if (n >= math.ceil(melody_dur / duration)):
            volume = 70
        else:
            volume = 100
        pitch = 4
        octave = 3

        if (n % 4 == 2):
            snare.append(Note(pitch, octave=octave, dur=duration, volume=volume))
        else:
            snare.append(Rest(dur=duration))

    midi = Midi(number_tracks=8, tempo=bpm, instrument=[instrument, 25, 25, 25, 25, 0, 0, 0])
    midi.seq_notes(result, track=0)
    midi.seq_notes(guitar0, track=1)
    midi.seq_notes(guitar1, track=2)
    midi.seq_notes(guitar2, track=3)
    midi.seq_notes(guitar3, track=4)
    midi.seq_notes(cymbal, track=5, channel=9)
    midi.seq_notes(kick, track=6, channel=9)
    midi.seq_notes(snare, track=7, channel=9)
    midi.write(midi_path)

# setting
bpm = 120
instrument = 1
beat = 4

# Make Midi File
randomNum = randint(10000000, 99999999)
songName = "Temp/0" + str(randomNum)
make_midi(midi_path=songName + ".mid", notes=song, bpm=bpm, instrument=instrument, beat=beat)

# Make Audio File
fs = FluidSynth('FluidR3_GM.sf2', 44100)
fs.midi2audio(songName + ".mid", songName + ".wav")
print(songName)
