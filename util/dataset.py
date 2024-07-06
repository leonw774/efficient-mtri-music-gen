import bisect
import io
from itertools import tee
import os
import sys
from typing import List, Callable, Iterable
import zipfile

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from . import tokens
from .corpus import to_pathlist_file_path
from .vocabs import get_corpus_vocabs
from .arrays import ATTR_NAME_INDEX, get_full_array_string


def deeper_getsizeof(obj, depth = 1):
    """sys.getsizeof but deeper"""
    assert isinstance(depth, int) and depth >= 1

    cur_stack = [obj]
    next_stack = []
    checked_id = set()
    obj_size = 0

    for _ in range(depth):
        while len(cur_stack) > 0:
            x = cur_stack.pop()
            checked_id.add(id(x))

            if isinstance(x, (tuple, list, set, frozenset)):
                obj_size += sys.getsizeof(x)
                next_stack.extend(elem for elem in x)

            elif isinstance(x, dict):
                obj_size += sys.getsizeof(x)
                next_stack.extend(key for key in x.keys())
                next_stack.extend(value for value in x.values())

            elif isinstance(obj, np.ndarray):
                obj_size += obj.nbytes

        cur_stack = next_stack
        next_stack = []

    return obj_size

def pairwise(iterable: Iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_seq_length: int,
            excluded_path_list: List[str] = None,
            virtual_piece_step_ratio: float = 0,
            flatten_virtual_pieces: bool = False,
            permute_mps: bool = False,
            permute_track_number: bool = False,
            pitch_augmentation_range: int = 0,
            verbose: bool = False) -> None:
        """
        Parameters:
        - `data_dir_path`: Expect path to corpus directory

        - `virtual_piece_step_ratio`: Should be not less than 0.
           Default is 0.
            - If > 0, split over-length pieces into multiple virtual
              pieces. The start of a virtual pieces will be from the
              measure token that has the smallest index greater than
              `max_seq_length * virtual_piece_step_ratio * N`, where N
              is positive integer. Any virtual piece starting from the
              same index are merged as one.
            - If `virtual_piece_step_ratio` is `1 / max_seq_length`,
              then its the same as using all possible measures as the
              starting index.
            - If == 0, each piece has only one virtual piece, starting
              from the index 0.
        
        - `flatten_virtual_pieces`: Default is False.
            - If True, all virtual pieces has distinct index number.
            - If False, virtual pieces within same real piece shares
              the same index number, It will be randomly one of the
              virtual pieces when accessed by `__getitem__`.

        - `excluded_path_list`: The list of relative or absolute paths
          of the files to exclude.

        - `permute_mps`: Whether or not the dataset should permute all
          the maximal permutable subarrays before returning in
          `__getitem__`, as data augmentation. Default is False.

        - `permute_track_number`: Permute all the track numbers
          relative to the instruments, as data augmentation.
          Default is False.

        - `pitch_augmentation_range`: If set to P, will add a random
          value from -P to +P (all inclusive) on all pitch values as
          data augmentation. Default is 0.
        
        - `verbose`: show tqdm or not.
        """
        self.vocabs = get_corpus_vocabs(data_dir_path)
        assert max_seq_length >= 0
        self.max_seq_length = max_seq_length
        assert virtual_piece_step_ratio >= 0
        self.virtual_piece_step_ratio = virtual_piece_step_ratio
        assert isinstance(flatten_virtual_pieces, bool)
        self.flatten_virtual_pieces = flatten_virtual_pieces
        assert isinstance(permute_mps, bool)
        self.permute_mps = permute_mps
        assert isinstance(permute_track_number, bool)
        self.permute_track_number = permute_track_number
        assert pitch_augmentation_range >= 0
        self.pitch_augmentation_range = pitch_augmentation_range

        npz_path = os.path.join(data_dir_path, 'arrays.npz')
        pathlist_file_path = to_pathlist_file_path(data_dir_path)
        with open(pathlist_file_path, 'r', encoding='utf8') as pathlist_file:
            all_paths_list = [
                p.strip()
                for p in pathlist_file.readlines()
            ]
        if excluded_path_list is not None and len(excluded_path_list) != 0:
            # assert os.path.exists(test_pathlist_file_path)
            excluded_path_tuple = tuple(p.strip() for p in excluded_path_list)
        else:
            excluded_path_tuple = tuple()
        if verbose:
            print('Reading', npz_path)
        self.included_path_list = []
        self.included_piece_id = set()
        tqdm_enum_all_path_list = tqdm(
            enumerate(all_paths_list),
            desc='Exclude paths',
            disable=not verbose,
            ncols=0
        )
        for piece_id, midi_path in tqdm_enum_all_path_list:
            # use endswith because pathlist contain relative paths
            # from project's root
            # while excluded_path_list may contain relative paths
            # from dataset's root
            if not midi_path.endswith(excluded_path_tuple):
                self.included_path_list.append(midi_path)
                self.included_piece_id.add(piece_id)

        available_memory_size = psutil.virtual_memory().available
        npz_zipfile = zipfile.ZipFile(npz_path)
        npz_zipinfo_list = npz_zipfile.infolist()
        array_memory_size = sum(zinfo.file_size for zinfo in npz_zipinfo_list)
        if array_memory_size >= available_memory_size - 1e9:
            if verbose:
                print(
                    f'Memory not enough. Need {array_memory_size//1000} KB. '
                    f'Only {available_memory_size//1000} KB available.'
                )
            raise OSError('Memory not enough.')
        # load numpy arrays into memory
        tqdm_array_names = tqdm(
            npz_zipfile.namelist(),
            desc='Loading arrays',
            disable=not verbose,
            ncols=0
        )
        self.pieces = [
            np.load(io.BytesIO(npz_zipfile.read(array_name)))
            for array_name in tqdm_array_names
            # array_name[:-4] to remove the '.npy' in filename
            if int(array_name[:-4]) in self.included_piece_id
        ]
        npz_zipfile.close()

        # The seperators of maximal permutable subarray are:
        self._sep_id = self.vocabs.events.text2id[tokens.SEP_TOKEN_STR]
        # numpy.isin with sorted 1d array helps search speed
        self._measure_ids = np.sort(np.array([
            index
            for text, index in self.vocabs.events.text2id.items()
            if text[0] == tokens.MEASURE_EVENTS_CHAR
        ]))

        # preprocessing
        pieces_num = len(self.pieces)
        self._piece_range_end = [0] * pieces_num
        self._body_start_indices = [0] * pieces_num
        self._virtual_piece_getitem_cursur = [0] * pieces_num

        if virtual_piece_step_ratio > 0:
            vp_step_size = int(
                max_seq_length * virtual_piece_step_ratio
            )
            vp_step_size_min = self.vocabs.track_numbers.size + 2
            assert vp_step_size > vp_step_size_min, \
                f'vp_step_size ({{vp_step_size}}) is ' \
                f'smaller than longest head ({vp_step_size_min}).'

        # zeros will be replaced by body start index
        self._virtual_piece_indices = [[0] for _ in range(pieces_num)]
        self._mps_sep_indices = [[] for _ in range(pieces_num)]
        self._augmentable_pitches = [
            np.empty((0,), dtype=np.bool8)
            for _ in range(pieces_num)
        ]
        # length of (-pa, ..., -1, 0, 1, ..., pa)
        self._pitch_aug_factor = self.pitch_augmentation_range * 2 + 1


        max_index = -1 # because the first element we add should be index 0
        tqdm_enum_pieces = tqdm(
            enumerate(self.pieces),
            desc='Processing',
            disable=not verbose,
            ncols=0
        )
        for pidx, piece_array in tqdm_enum_pieces:
            piece_len = int(piece_array.shape[0])

            j = 2 # because first token is BOS and second token is track token
            while piece_array[j][ATTR_NAME_INDEX['evt']] != self._sep_id:
                j += 1
            first_midx = j + 1
            self._body_start_indices[pidx] = first_midx
            self._virtual_piece_indices[pidx][0] = first_midx
            measure_indices = np.flatnonzero(
                np.isin(
                    piece_array[:, ATTR_NAME_INDEX['evt']],
                    self._measure_ids
                )
            )

            # create more virtual pieces if pieces length > max_seq_length
            if (virtual_piece_step_ratio > 0
                and piece_array.shape[0] > self.max_seq_length):
                vp_steps = np.arange(vp_step_size, piece_len, vp_step_size)
                # foreach vp_step find largest measure indices that <= vp_step
                step_positions = np.searchsorted(
                    measure_indices,
                    vp_steps,
                    side='right'
                )
                vp_indices = np.where(
                    step_positions > 0,
                    measure_indices[step_positions - 1],
                    None
                )
                # have repeat means measure length > vp_step_size
                # replace the repetition with the next meaure
                measures_number = len(measure_indices)
                for n, (l, r) in enumerate(pairwise(vp_indices)):
                    if l == r and step_positions[n] < measures_number:
                        vp_indices[n+1] = measure_indices[step_positions[n]]
                # if there's still repeat then remove them
                vp_indices = np.unique(vp_indices)
                # add measure for zeroth step
                vp_indices = np.concatenate(([first_midx], vp_indices))

            if permute_mps:
                # consider sequence: M  P  N  N  N  P  N  N  P  N  M  P  N  N
                #             index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13
                #   mps numbers are: 1  2  3  3  3  4  5  5  6  7  8  9  10 10
                # we want the beginning indices of all mps:
                #                   [0, 1, 2,       5, 6,    8, 9, 10,11,12]
                mps_numbers = piece_array[:, ATTR_NAME_INDEX['mps']]
                # just find where the number increases
                body_mps_sep_indices = np.where(
                    mps_numbers[:-1] < mps_numbers[1:]
                )[0]
                self._mps_sep_indices[pidx] = np.concatenate([
                    body_mps_sep_indices,
                    np.array([piece_array.shape[0]]) # the EOS
                ])

            if self.pitch_augmentation_range != 0:
                has_pitch = (piece_array[:, ATTR_NAME_INDEX['pit']]) != 0
                # the pitch value of drums/percussion is not real note pitch
                # but different percussion sound
                # assume instrument vocabulary is:
                #   0:PAD, 1:0, 2:1, ... 128:127, 129:128(drums)
                not_drum = (piece_array[:, ATTR_NAME_INDEX['ins']]) != 129
                augmentable_pitches = np.logical_and(has_pitch, not_drum)
                self._augmentable_pitches[pidx] = augmentable_pitches

            if self.flatten_virtual_pieces:
                virtual_piece_count = len(
                    self._virtual_piece_indices[pidx]
                )
                max_index += virtual_piece_count * self._pitch_aug_factor
            else:
                max_index += self._pitch_aug_factor
            self._piece_range_end[pidx] = max_index
        self._length = max_index + 1

        if verbose:
            if permute_mps:
                mps_lengths = []
                for mps_sep_indices in self._mps_sep_indices:
                    mps_lengths.extend([
                        j2 - j1
                        for j1, j2 in zip(
                            mps_sep_indices[:-1], mps_sep_indices[1:]
                        )
                        if j2 > j1 + 1
                    ])
                print('Number of non-singleton mps:', len(mps_lengths))
                print('Average mps length:', sum(mps_lengths)/len(mps_lengths))
            dataset_memsize = sum(
                deeper_getsizeof(self.__getattribute__(attr_name), 2)
                for attr_name in dir(self)
                if not attr_name.startswith('__')
            )
            print(f'Dataset object size: {dataset_memsize} bytes')

    def __getitem__(self, index):
        assert index < self._length
        pidx = bisect.bisect_left(self._piece_range_end, index)
        body_start_index = self._body_start_indices[pidx]

        index_offset = (
            index
            if pidx == 0 else
            index - self._piece_range_end[pidx-1] - 1
        )
        virtual_piece_indices = self._virtual_piece_indices[pidx]
        if self.flatten_virtual_pieces:
            vpidx = index_offset // self._pitch_aug_factor
            start_index = virtual_piece_indices[vpidx]
            index_offset -= vpidx * self._pitch_aug_factor
        else:
            cur_cursor = self._virtual_piece_getitem_cursur[pidx]
            start_index = virtual_piece_indices[cur_cursor]
            self._virtual_piece_getitem_cursur[pidx] = (
                (cur_cursor + 1) % len(virtual_piece_indices)
            )

        # preserve space for head
        end_index = start_index + self.max_seq_length - body_start_index
        if end_index < start_index:
            end_index = start_index
            body_start_index = self.max_seq_length

        # dont need to check end_index because if end_index > piece_length,
        # then numpy will just end the slice at piece_length
        head_array = np.array(self.pieces[pidx][:body_start_index])
        body_array = np.array(self.pieces[pidx][start_index:end_index])
        sampled_array = np.concatenate([head_array, body_array], axis=0)
        # was int16 to save space, but torch want int32
        sampled_array = sampled_array.astype(np.int32)

        # tidy up mps numbers
        # make sure all mps number smaller are than max_mps_number
        # and all of them are non-decreasing
        if body_start_index != self.max_seq_length:
            mps_index = ATTR_NAME_INDEX['mps']
            min_body_mps = np.min(
                sampled_array[body_start_index:, mps_index]
            )
            # magic number 4 because the first measure must have mps number 4
            if min_body_mps < 4:
                raise ValueError('MPS number in body is less than 4.')
            sampled_array[body_start_index:, mps_index] -= (min_body_mps-4)

        # pitch augmentation
        # pitch vocabulary is 0:PAD, 1:0, 2:1, ... 128:127
        pitch_augment = index_offset - self.pitch_augmentation_range
        if pitch_augment != 0 and body_start_index != self.max_seq_length:
            augmentables = (
                self._augmentable_pitches[pidx][
                    body_start_index:end_index]
            )
            # sometimes the sampled array does not contain any
            # pitch-augmentable token
            body_pitch_col = sampled_array[
                body_start_index:, ATTR_NAME_INDEX['pit']
            ]
            if np.any(augmentables):
                if pitch_augment > 0:
                    max_pitch = np.max((body_pitch_col)[augmentables])
                    if max_pitch + pitch_augment > 128:
                        pitch_augment -= max_pitch - 128
                else: # pitch_augment < 0
                    min_pitch = np.min((body_pitch_col)[augmentables])
                    if min_pitch + pitch_augment < 1:
                        pitch_augment += 1 - min_pitch
                body_pitch_col[augmentables] += pitch_augment

        if self.permute_track_number:
            max_track_number = self.vocabs.track_numbers.size - 1
            # add one because there is a padding token at index 0
            perm = np.concatenate((
                [0],
                np.random.permutation(max_track_number) + 1
            ))
            # view track number col
            trn_column = sampled_array[:, ATTR_NAME_INDEX['trn']]
            for i, trn in enumerate(trn_column):
                trn_column[i] = perm[trn]

        if self.permute_mps and body_start_index != self.max_seq_length:
            mps_sep_indices = []
            mps_tokens_ranges = []
            # BOS, first track token and body_start_index-1 is SEP
            mps_sep_indices_head = [0, 1, body_start_index - 1]
            mps_sep_indices_body = [
                i - start_index + body_start_index
                for i in self._mps_sep_indices[pidx]
                if start_index <= i < end_index
            ]
            mps_sep_indices = mps_sep_indices_head + mps_sep_indices_body
            mps_sep_indices_end = mps_sep_indices + [sampled_array.shape[0]]
            for i in range(len(mps_sep_indices_end)-1):
                start = mps_sep_indices_end[i]
                end = mps_sep_indices_end[i+1]
                if end - start >= 2:
                    mps_tokens_ranges.append((start, end))

            # print(mps_tokens_ranges)
            for start, end in mps_tokens_ranges:
                p = np.random.permutation(end-start) + start
                sampled_array[start:end] = sampled_array[p]

        # checking for bad numbers
        for attr_name, attr_idx in ATTR_NAME_INDEX.items():
            if len(attr_name) == 3:
                continue
            assert np.min(sampled_array[:, attr_idx]) >= 0, \
                (f'Number in {attr_name} is below zero\n'
                    f'{get_full_array_string(sampled_array, self.vocabs)}')
            max_attr_value = np.max(sampled_array[:, attr_idx])
            if attr_name == 'mps_numbers':
                vocabs_size = min(
                    self.max_seq_length,
                    self.vocabs.max_mps_number
                ) + 1
                assert max_attr_value < vocabs_size, \
                    (f'Number in {attr_name} larger than vocab size: '
                        f'{max_attr_value} >= {vocabs_size}\n'
                        f'{get_full_array_string(sampled_array, self.vocabs)}'
                    )
            else:
                vocabs_size = getattr(self.vocabs, attr_name).size
                assert max_attr_value < vocabs_size, \
                    (f'Number in {attr_name} not smaller than vocab size: '
                        f'{max_attr_value} >= {vocabs_size}\n'
                        f'{get_full_array_string(sampled_array, self.vocabs)}'
                    )

        return torch.from_numpy(sampled_array)

    def __len__(self):
        return self._length


def collate_left(piece_list: List[torch.Tensor]) -> torch.Tensor:
    """Stack pieces batch-first to a 3-d array and left-pad them"""
    # print('\n'.join(map(str, piece_list)))
    batched_pieces = pad_sequence(piece_list, batch_first=True, padding_value=0)
    return batched_pieces

def collate_right(piece_list: List[torch.Tensor]) -> torch.Tensor:
    """Stack pieces batch-first to a 3-d array and right-pad them"""
    # print('\n'.join(map(str, seq_list)))
    # Find the maximum length among all tensors
    max_length = max(tensor.size(0) for tensor in piece_list)

    # Pad tensors to the right with zeros and stack them
    F.pad: Callable # because pylint
    padded_piece_list = [
        F.pad(
            input=piece,
            pad=(max_length - piece.size(0), 0),
            mode='constant',
            value=0
        )
        for piece in piece_list
    ]
    batched_pieces = torch.stack(padded_piece_list)
    return batched_pieces

collate_mididataset = collate_left
