"""
Here we test the batched implementation (ours) against the original unbatched
implementation (pystoi).

"""
import itertools

import numpy as np
import pytest
from pystoi import stoi

from batch_pystoi import stoi as batch_stoi
from batch_pystoi.stoi import HOP, N_FRAME, N

BATCH_SIZE = 4


def _test_stoi(x, y, fs, extended, batched):
    if x.ndim == 1:
        x = x[None, :]
        y = y[None, :]
    if batched:
        ours_batched = batch_stoi(x, y, fs, extended)
    for i in range(x.shape[0]):
        if batched:
            ours = ours_batched[i]
        else:
            ours = batch_stoi(x[i], y[i], fs, extended)
        assert np.isclose(stoi(x[i], y[i], fs, extended), ours)


@pytest.fixture(scope="module")
def _inputs(request):
    seed, length, batched = request.param

    generator = np.random.RandomState(seed)
    if batched:
        x = generator.randn(BATCH_SIZE, length)
        y = generator.randn(BATCH_SIZE, length)
    else:
        x = generator.randn(length)
        y = generator.randn(length)

    return (x, y), request.param


@pytest.mark.parametrize(
    '_inputs',
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
def test_stoi(_inputs, extended, fs, silence, silence_length):
    (x, y), params = _inputs
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
    _test_stoi(x, y, fs, extended, batched)


@pytest.mark.parametrize(
    '_inputs',
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
def test_stoi_batched_random_silence(_inputs, extended, fs):
    (x, y), params = _inputs
    seed, length, batched = params
    assert batched

    generator = np.random.RandomState(seed)
    for i in range(x.shape[0]):
        silence_length = generator.randint(0, length - ((N+2)*N_FRAME))
        silence_start = generator.randint(0, length - silence_length)
        silence_end = silence_start + silence_length
        x[i, silence_start:silence_end] = 0
        y[i, silence_start:silence_end] = 0

    _test_stoi(x, y, fs, extended, batched)


@pytest.mark.skip(reason=(
    'When extended=True, the original pystoi adds a small random noise to the '
    'signal when normalizing, which results in a non-zero output when the '
    'signal is all zeros. Note this is only the case when the signal is all '
    'zeros, since the VAD will remove any zeros if there is any non-zero '
    'value in the signal.'
))
@pytest.mark.parametrize(
    '_inputs',
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
def test_stoi_full_silence(_inputs, which, extended, fs):
    (x, y), params = _inputs
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

    _test_stoi(x, y, fs, extended, batched)
