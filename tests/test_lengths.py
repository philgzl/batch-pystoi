"""
Here we test the "lengths" argument which contains the original length of the
signals in the batch.

Namely the batched output should be the same as the output from the cropped
mixtures.

"""
import itertools

import numpy as np
import pytest

from batch_pystoi.stoi import N_FRAME, N, HOP, stoi


BATCH_SIZE = 4


def _test_lengths(x, y, lengths, fs, extended, batched):
    if x.ndim == 1:
        x = x[None, :]
        y = y[None, :]
    if np.ndim(lengths) == 0:
        lengths = [lengths]
    if batched:
        ours_batched = stoi(x, y, fs, extended, lengths=lengths)
    for i in range(x.shape[0]):
        if batched:
            ours = ours_batched[i]
        else:
            ours = stoi(x[i], y[i], fs, extended, lengths=lengths[i])
        theirs = stoi(x[i, :lengths[i]], y[i, :lengths[i]], fs, extended)
        assert np.isclose(theirs, ours)


@pytest.fixture(scope="module")
def _inputs_and_lengths(request):
    seed, length, batched = request.param

    generator = np.random.RandomState(seed)
    if batched:
        x = generator.randn(BATCH_SIZE, length)
        y = generator.randn(BATCH_SIZE, length)
    else:
        x = generator.randn(length)
        y = generator.randn(length)

    min_length = (N+2)*N_FRAME
    if batched:
        lengths = generator.randint(min_length, length, size=BATCH_SIZE)
    else:
        lengths = generator.randint(min_length, length)

    return (x, y), lengths, request.param


@pytest.mark.parametrize(
    '_inputs_and_lengths',
    itertools.product(
        range(3),
        [
            50*N_FRAME,
            50*N_FRAME-1,
            50*N_FRAME+1,
            50*N_FRAME+HOP,
            50*N_FRAME+HOP-1,
            50*N_FRAME+HOP+1,
        ],
        [False, True],
    ),
    indirect=True,
    scope="module",
)
@pytest.mark.parametrize('extended', [False, True])
@pytest.mark.parametrize('fs', [10000, 16000])
@pytest.mark.parametrize('silence', ['none', 'start', 'mid', 'end'])
@pytest.mark.parametrize('silence_length', [
    5*N_FRAME,
    5*N_FRAME-1,
    5*N_FRAME+1,
    5*N_FRAME+HOP,
    5*N_FRAME+HOP-1,
    5*N_FRAME+HOP+1,
])
def test_lengths(_inputs_and_lengths, extended, fs, silence, silence_length):
    (x, y), lengths, params = _inputs_and_lengths
    seed, length, batched = params

    if silence == 'start':
        silence_start = 0
    elif silence == 'mid':
        generator = np.random.RandomState(seed)
        silence_start = generator.randint(0, length - silence_length)
    elif silence == 'end':
        silence_start = length - silence_length
    elif silence != 'none':
        raise ValueError('silence must be start, mid, end or none, '
                         f'got {silence}')
    if silence != 'none':
        silence_end = silence_start + silence_length
        x[..., silence_start:silence_end] = 0
        y[..., silence_start:silence_end] = 0

    _test_lengths(x, y, lengths, fs, extended, batched)


@pytest.mark.parametrize(
    '_inputs_and_lengths',
    itertools.product(
        range(10),
        [
            50*N_FRAME,
            50*N_FRAME-1,
            50*N_FRAME+1,
            50*N_FRAME+HOP,
            50*N_FRAME+HOP-1,
            50*N_FRAME+HOP+1,
        ],
        [True],
    ),
    indirect=True,
    scope="module",
)
@pytest.mark.parametrize('extended', [False, True])
@pytest.mark.parametrize('fs', [10000, 16000])
def test_lengths_batched_random_silence(_inputs_and_lengths, extended, fs):
    (x, y), lengths, params = _inputs_and_lengths
    seed, length, batched = params
    assert batched

    generator = np.random.RandomState(seed)
    for i in range(x.shape[0]):
        silence_length = generator.randint(0, length - ((N+2)*N_FRAME))
        silence_start = generator.randint(0, length - silence_length)
        silence_end = silence_start + silence_length
        x[i, silence_start:silence_end] = 0
        y[i, silence_start:silence_end] = 0

    _test_lengths(x, y, lengths, fs, extended, batched)


@pytest.mark.parametrize(
    '_inputs_and_lengths',
    itertools.product(
        range(3),
        [
            50*N_FRAME,
            50*N_FRAME-1,
            50*N_FRAME+1,
            50*N_FRAME+HOP,
            50*N_FRAME+HOP-1,
            50*N_FRAME+HOP+1,
        ],
        [False, True],
    ),
    indirect=True,
    scope="module",
)
@pytest.mark.parametrize('which', ['x', 'y', 'both'])
@pytest.mark.parametrize('extended', [False, True])
@pytest.mark.parametrize('fs', [10000, 16000])
def test_lengths_full_silence(_inputs_and_lengths, which, extended, fs):
    (x, y), lengths, params = _inputs_and_lengths
    seed, length, batched = params

    if which == 'x':
        x[:] = 0
    elif which == 'y':
        y[:] = 0
    elif which == 'both':
        x[:] = 0
        y[:] = 0
    else:
        raise ValueError(f'which must be x, y or both, got {which}')

    _test_lengths(x, y, lengths, fs, extended, batched)
