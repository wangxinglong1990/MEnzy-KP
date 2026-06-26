"""
SMILES Enumerator — Enumeration, Iterator, and SmilesEnumerator.

Migrated from kcat/src/utils/enumerator.py.
No internal imports — self-contained.
"""

from rdkit import Chem
import numpy as np
import threading


class Iterator(object):
    """Abstract base class for data iterators."""

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        if n < batch_size:
            raise ValueError('Input data length is shorter than batch_size\nAdjust batch_size')

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class SmilesIterator(Iterator):
    """Iterator yielding data from a SMILES array."""

    def __init__(self, x, y, smiles_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32):
        if y is not None and len(x) != len(y):
            raise ValueError(
                'X and y should have the same length. '
                'Found: X.shape = %s, y.shape = %s' %
                (np.asarray(x).shape, np.asarray(y).shape)
            )
        self.x = np.asarray(x)
        self.y = np.asarray(y) if y is not None else None
        self.smiles_data_generator = smiles_data_generator
        self.dtype = dtype
        super(SmilesIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(
            tuple([current_batch_size] + [self.smiles_data_generator.pad,
                                           self.smiles_data_generator._charlen]),
            dtype=self.dtype,
        )
        for i, j in enumerate(index_array):
            smiles = self.x[j:j + 1]
            x = self.smiles_data_generator.transform(smiles)
            batch_x[i] = x
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer."""

    def __init__(self, charset='@C)(=cOn1S2/H[N]\\', pad=120,
                 leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset

    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c, i) for i, c in enumerate(charset))
        self._int_to_char = dict((i, c) for i, c in enumerate(charset))

    def fit(self, smiles, extra_chars=None, extra_pad=5):
        extra_chars = extra_chars or []
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad

    def randomize_smiles(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        one_hot = np.zeros((smiles.shape[0], self.pad, self._charlen), dtype=np.int8)
        if self.leftpad:
            for i, ss in enumerate(smiles):
                if self.enumerate:
                    ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j, c in enumerate(ss):
                    one_hot[i, j + diff, self._char_to_int[c]] = 1
            return one_hot
        else:
            for i, ss in enumerate(smiles):
                if self.enumerate:
                    ss = self.randomize_smiles(ss)
                for j, c in enumerate(ss):
                    one_hot[i, j, self._char_to_int[c]] = 1
            return one_hot

    def reverse_transform(self, vect):
        smiles = []
        for v in vect:
            v = v[v.sum(axis=1) == 1]
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)
