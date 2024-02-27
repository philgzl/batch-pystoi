import warnings

import numpy as np

from . import utils

# Constant definition
FS = 10000                          # Sampling frequency
N_FRAME = 256                       # Window support
NFFT = 512                          # FFT Size
NUMBAND = 15                        # Number of 13 octave band
MINFREQ = 150                       # Center frequency of 1st octave band (Hz)
OBM, CF = utils.thirdoct(FS, NFFT, NUMBAND, MINFREQ)  # 1/3 octave band matrix
N = 30                              # Minimum number of frames
BETA = -15.                         # Lower SDR bound
DYN_RANGE = 40                      # Speech dynamic range
HOP = int(N_FRAME / 2)              # Hop length


def stoi(x, y, fs_sig, extended=False, lengths=None):
    """Short Term Objective Intelligibility (STOI).

    Computes the STOI (See [1][2]) of a denoised signal compared to a clean
    signal. The output is expected to have a monotonic relation with the
    subjective speech-intelligibility, where a higher score denotes better
    speech intelligibility. Accepts either a single waveform or a batch of
    waveforms stored in a numpy array.

    # Arguments
        x (np.ndarray): Clean original speech.
        y (np.ndarray): Denoised speech.
        fs_sig (int): Sampling rate of x and y.
        extended (bool): Whether to use the extended STOI described in [3].
        lengths (int or list of int): Length of the original files. This is
            useful if padding was performed before batching.

    # Returns
        float or np.ndarray: Short time objective intelligibility measure
        between clean and denoised speech. Returns float if called with a
        single waveform, np.ndarray if called with a batch of waveforms.

    # Raises
        AssertionError:
            - if x and y have different shape.
            - if x or y are not 1 or 2 dimensional
            - if lengths does not have the same length as the batch size

    # Reference
        [1] C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen "A short-time
            objective intelligibility measure for time-frequency weighted noisy
            speech", Proc. ICASSP, 2010.
        [2] C. H. Taal, R. C. Hendriks, R. Heusdens and J.Jensen "An Algorithm
            for Intelligibility Prediction of Time–Frequency Weighted Noisy
            Speech", IEEE Trans. Audio, Speech, and Language Process., 2011.
        [3] J. Jensen and C. H. Taal, "An Algorithm for Predicting the
            Intelligibility of Speech Masked by Modulated Noise Maskers",
            IEEE/ACM Trans Audio, Speech, and Language Process., 2016.
    """
    if x.shape != y.shape:
        raise AssertionError('x and y should have the same shape,'
                             f'got {x.shape} and {y.shape}')

    output_shape = x.shape[:-1]
    if x.ndim == 0 or x.ndim > 2:
        raise AssertionError('inputs should be 1 or 2 dimensional, '
                             f'got {x.ndim}')
    elif x.ndim == 1:
        x = x[None, :]
        y = y[None, :]

    if lengths is not None:
        if np.ndim(lengths) == 0:
            lengths = [lengths]
        if len(lengths) != x.shape[0]:
            raise AssertionError('lengths must have the same length as the '
                                 f'batch size, got {len(lengths)} and '
                                 f'{x.shape[0]}')

    # Resample if fs_sig is different than fs
    if fs_sig != FS:
        x = utils.resample_oct(x, FS, fs_sig)
        y = utils.resample_oct(y, FS, fs_sig)
        if lengths is not None:
            lengths = [int(np.ceil(l_ * FS / fs_sig)) for l_ in lengths]

    # Set the samples after the end of the original signal to NaN
    if lengths is not None:
        x, y = x.copy(), y.copy()
        for i, length in enumerate(lengths):
            # start at length - 1 minus one due to
            # https://github.com/mpariente/pystoi/issues/32
            x[i, length - 1:] = np.nan
            y[i, length - 1:] = np.nan

    # Remove silent frames
    x, y, mask = utils.remove_silent_frames(x, y, DYN_RANGE, N_FRAME, HOP)

    # Take STFT, shape (batch, num_frames, n_fft//2 + 1)
    x_spec = utils.stft(x, N_FRAME, NFFT, overlap=2)
    y_spec = utils.stft(y, N_FRAME, NFFT, overlap=2)

    # set to zero the last non-zero frame due to a bug in pystoi
    # the bug is caused by a matlab-like indexing which removes a frame
    # see https://github.com/mpariente/pystoi/issues/32
    mask = np.sort(mask, axis=-1)[:, ::-1]
    mask = mask[:, 1:]
    x_spec *= mask[..., None]
    y_spec *= mask[..., None]

    # Ensure at least 30 frames for intermediate intelligibility
    not_enough = np.sum(mask, axis=-1) < N
    if np.any(not_enough):
        warnings.warn('Not enough STFT frames to compute intermediate '
                      'intelligibility measure after removing silent '
                      'frames. Returning 1e-5.',
                      RuntimeWarning)
        if np.all(not_enough):
            return np.full(x.shape[0], 1e-5)
        x_spec = x_spec[~not_enough]
        y_spec = y_spec[~not_enough]
        mask = mask[~not_enough]

    # Apply OB matrix to the spectrograms as in Eq. (1)
    x_tob = np.sqrt(np.matmul(np.square(np.abs(x_spec)), OBM.T))
    y_tob = np.sqrt(np.matmul(np.square(np.abs(y_spec)), OBM.T))

    # Take segments of x_tob, y_tob
    x_segments = utils.segment_frames(x_tob, mask, N)
    y_segments = utils.segment_frames(y_tob, mask, N)
    mask = mask[:, N-1:]

    # From now on the shape is always (batch, num_segments, seg_size, bands)
    if extended:
        x_n = utils.row_col_normalize(x_segments)
        y_n = utils.row_col_normalize(y_segments)
        d_n = np.mean(np.sum(x_n * y_n, axis=3), axis=2)
        output = np.sum(d_n, axis=1) / np.sum(mask, axis=1)

    else:
        # Find normalization constants and normalize
        normalization_consts = (
            np.linalg.norm(x_segments, axis=2, keepdims=True) /
            (np.linalg.norm(y_segments, axis=2, keepdims=True) + utils.EPS)
        )
        y_segments_normalized = y_segments * normalization_consts

        # Clip as described in [1]
        clip_value = 10 ** (-BETA / 20)
        y_primes = np.minimum(
            y_segments_normalized, x_segments * (1 + clip_value)
        )

        # Subtract mean vectors
        y_primes = y_primes - np.mean(y_primes, axis=2, keepdims=True)
        x_segments = x_segments - np.mean(x_segments, axis=2, keepdims=True)

        # Divide by their norms
        y_primes /= (
            np.linalg.norm(y_primes, axis=2, keepdims=True) + utils.EPS
        )
        x_segments /= (
            np.linalg.norm(x_segments, axis=2, keepdims=True) + utils.EPS
        )

        # Find a matrix with entries summing to sum of correlations of vectors
        correlations_components = np.sum(y_primes * x_segments, axis=-2)

        # Find the mean of all correlations
        d = np.mean(correlations_components, axis=-1)
        output = np.sum(d, axis=1) / np.sum(mask, axis=1)

    if np.any(not_enough):
        output_ = np.empty(x.shape[0])
        output_[not_enough] = 1e-5
        output_[~not_enough] = output
        output = output_

    return output.reshape(output_shape)
