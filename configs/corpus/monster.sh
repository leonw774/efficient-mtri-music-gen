#!/bin/bash
# midi preprocessing
TPQ=12
MAX_TRACK_NUMBER=40
MAX_DURATION=48
VELOCITY_STEP=16
CONTINUING_NOTE=true
TEMPO_MIN=16
TEMPO_MAX=240
TEMPO_STEP=16
MIDI_WORKER_NUMBER=32
MIDI_DIR_PATH="../Monster-MIDI-Dataset/MIDIs/"
DATA_NAME="monster"
TEST_PATHS_FILE='configs/split/empty.txt'
VALID_PATHS_FILE='configs/split/monster_valid.txt'

