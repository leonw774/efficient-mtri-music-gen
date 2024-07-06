import json

from tqdm import tqdm

from . import tokens
from .tokens import (
    itob36str,
    get_largest_possible_position,
    get_supported_time_signatures
)
from .corpus import CorpusReader, to_vocabs_file_path

class Vocabs:

    class AttrVocab:
        size: int
        id2text: dict
        text2id: dict
        def __init__(self, init_obj) -> None:
            if isinstance(init_obj, list):
                self.size = len(init_obj)
                self.id2text = {i: t for i, t in enumerate(init_obj)}
                self.text2id = {t: i for i, t in enumerate(init_obj)}
            elif isinstance(init_obj, dict):
                self.size = init_obj['size']
                self.id2text = {
                    int(k): v for k, v in init_obj['id2text'].items()
                }
                self.text2id = init_obj['text2id']
            else:
                raise TypeError(
                    f'AttrVocab\'s init_obj can only be list or dict.'
                    f'Get {type(init_obj)}'
                )
        def __len__(self) -> int:
            return self.size

    paras: dict
    bpe_contours_list: list
    padding_token: str
    events: AttrVocab
    pitchs: AttrVocab
    durations: AttrVocab
    velocities: AttrVocab
    track_numbers: AttrVocab
    instruments: AttrVocab
    max_mps_number: int
    tempos: AttrVocab
    time_signatures: AttrVocab

    def __init__(
            self, paras: dict, bpe_contours_list: list, padding_token: str,
            events, pitchs, durations, velocities, track_numbers, instruments,
            max_mps_number) -> None:
        self.paras = paras
        self.bpe_contours_list = bpe_contours_list
        self.padding_token = padding_token
        self.events = Vocabs.AttrVocab(events)
        self.pitchs = Vocabs.AttrVocab(pitchs)
        self.durations = Vocabs.AttrVocab(durations)
        self.velocities = Vocabs.AttrVocab(velocities)
        self.track_numbers = Vocabs.AttrVocab(track_numbers)
        self.instruments = Vocabs.AttrVocab(instruments)
        self.max_mps_number = max_mps_number

    @classmethod
    def from_dict(cls, d):
        return cls(
            d['paras'], d['bpe_contours_list'], d['padding_token'],
            d['events'], d['pitchs'], d['durations'], d['velocities'],
            d['track_numbers'], d['instruments'], d['max_mps_number']
        )

    def to_dict(self) -> dict:
        return {
            'paras': self.paras,
            'bpe_contours_list': self.bpe_contours_list,
            'padding_token': self.padding_token,
            'events': vars(self.events),
            'pitchs': vars(self.pitchs),
            'durations': vars(self.durations),
            'velocities': vars(self.velocities),
            'track_numbers': vars(self.track_numbers),
            'instruments': vars(self.instruments),
            'max_mps_number': self.max_mps_number,
            'tempos': vars(self.tempos),
            'time_signatures': vars(self.time_signatures),
        }


def get_corpus_vocabs(corpus_dir_path: str) -> Vocabs:
    vocab_path = to_vocabs_file_path(corpus_dir_path)
    with open(vocab_path, 'r', encoding='utf8') as vocabs_file:
        corpus_vocabs_dict = json.load(vocabs_file)
    corpus_vocabs = Vocabs.from_dict(corpus_vocabs_dict)
    return corpus_vocabs


def build_vocabs(
        corpus_reader: CorpusReader,
        paras: dict,
        bpe_contours_list: list):
    """
    The `bpe_contours_list` is list of multinote contour are learned by
    bpe algorithm. If bpe is not performed, `bpe_contours_list` should
    be empty.
    """

    # Events include:
    # - begin-of-score, end-of-score and seperator
    # - contour of multi-note/note
    # - track
    # - position
    # - measure and time_signature
    # - tempo change

    supported_time_signatures = get_supported_time_signatures()
    largest_possible_position = get_largest_possible_position(
        paras['tpq'], supported_time_signatures
    )

    event_multi_note_contours = [
        tokens.NOTE_EVENTS_CHAR, tokens.NOTE_EVENTS_CHAR+'~'
    ]
    event_multi_note_contours += [
        tokens.MULTI_NOTE_EVENTS_CHAR + contour
        for contour in bpe_contours_list
    ]
    event_track_instrument = [
        tokens.TRACK_EVENTS_CHAR + itob36str(i)
        for i in range(129)
    ]
    event_position = [
        tokens.POSITION_EVENTS_CHAR+itob36str(i)
        for i in range(largest_possible_position)
    ]
    tempo_min, tempo_max, tempo_step = paras['tempo_quantization']
    event_tempo = [
        tokens.TEMPO_EVENTS_CHAR + itob36str(t)
        for t in range(tempo_min, tempo_max+tempo_step, tempo_step)
    ]
    event_measure_time_sig = [
        f'{tokens.MEASURE_EVENTS_CHAR}{itob36str(n)}/{itob36str(d)}'
        for n, d in supported_time_signatures
    ]

    measure_substr  = ' ' + tokens.MEASURE_EVENTS_CHAR
    position_substr = ' ' + tokens.POSITION_EVENTS_CHAR
    tempo_substr    = ' ' + tokens.TEMPO_EVENTS_CHAR
    max_mps_number = 0
    corpus_measure_time_sigs = set()
    token_count_per_piece = []
    for piece in tqdm(corpus_reader, desc='Find max mps number', ncols=80):
        measure_count = piece.count(measure_substr)
        position_count = piece.count(position_substr)
        tempo_count = piece.count(tempo_substr)
        # +4 because head (3) and eos (1)
        piece_max_mps_number = (
            2 * (measure_count + position_count) + tempo_count + 4
        )
        max_mps_number = max(max_mps_number, piece_max_mps_number)

        text_list = piece.split(' ')
        token_count_per_piece.append(len(text_list))
        corpus_measure_time_sigs.update([
            text
            for text in text_list
            if text[0] == tokens.MEASURE_EVENTS_CHAR
        ])
    # remove time signatures that dont appears in target corpus
    event_measure_time_sig = [
        t
        for t in event_measure_time_sig
        if t in corpus_measure_time_sigs
    ]

    # padding token HAVE TO be at first
    event_vocab = (
        tokens.SPECIAL_TOKENS_STR
        + event_multi_note_contours
        + event_track_instrument
        + event_position
        + event_measure_time_sig
        + event_tempo
    )

    pad_token = [tokens.PADDING_TOKEN_STR]
    pitch_vocab         = pad_token + list(
        map(itob36str, range(128))
    )
    duration_vocab      = pad_token + list(
         # add 1 because range end is exclusive
        map(itob36str, range(1, paras['max_duration']+1))
    )
    velocity_vocab      = pad_token + list(
        map(itob36str, range(
            paras['velocity_step'], 128, paras['velocity_step']
        ))
    )
    track_number_vocab  = pad_token + list(
        map(itob36str, range(paras['max_track_number']))
    )
    instrument_vocab    = pad_token + list(
        map(itob36str, range(129))
    )
    # we don't need mps number vocab because they are just monotonically
    # increasing integers generated dynamically in corpus.text_to_array

    avg_token_number = sum(token_count_per_piece) / len(token_count_per_piece)
    summary_string = (
        f'Average tokens per piece: {avg_token_number}\n'
        f'Event vocab size: {len(event_vocab)}\n'
        f'- Measure-time signature: {len(event_measure_time_sig)}\n'
        f'  - Supported: {len(supported_time_signatures)}\n'
        f'  - Existed in corpus: {len(corpus_measure_time_sigs)}\n'
        f'- Position: {len(event_position)}\n'
        f'- Tempo: {len(event_tempo)}\n'
        f'- Contour: {len(event_multi_note_contours)}\n'
        f'Duration vocab size: {len(duration_vocab)}\n'
        f'Velocity vocab size: {len(velocity_vocab)}\n'
        f'Track number vocab size: {paras["max_track_number"]}\n'
        f'Max MPS number: {max_mps_number}'
    )

    # build a dictionary for every token list
    vocabs = Vocabs(
        paras=paras,
        bpe_contours_list=bpe_contours_list,
        padding_token=tokens.PADDING_TOKEN_STR,
        events=event_vocab,
        pitchs=pitch_vocab,
        durations=duration_vocab,
        velocities=velocity_vocab,
        track_numbers=track_number_vocab,
        instruments=instrument_vocab,
        max_mps_number=max_mps_number
    )

    return vocabs, summary_string
