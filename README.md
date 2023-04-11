# batch-pystoi

This is a fork of `pystoi`, the original python implementation of classical and extended Short Term Objective Intelligibility (STOI) metrics [1-3]. This fork adds support for batched inputs and substantially improves processing time. The original author expressed that batched processing will not be implemented in the original repository, despite the interest and an open pull request. So it was implemented here.

## Notable changes
- As pointed out in this pull request in `pystoi`, the original implementation for the extended STOI uses expensive `np.random.standard_normal` calls to add noise to the signal before normalizing. However since the noise is then scaled very small, this does not change the results unless the inputs are all-zeros, in which case the STOI output is non-zero (which does not seem to make much sense anyway). The `np.random.standard_normal` calls were removed here and for all-zero inputs the STOI output is zero. Note that any non-zero sample makes the VAD discard zeros, so the user should not worry about the different behavior in this very special case with all-zero inputs.
- `stoi` now accepts an extra `lengths` optional arguments containing the original length of the waveforms before batching. This is useful if padding is performed to match the length of the waveforms in the batch, since the padding should not affect the STOI results. Note that simply padding with zeros hoping that the VAD will discard them without providing `lengths` is not enough, since the zeros can still leak onto the final frames of the original signal due to overlap-and-add.

## Tests
- New tests under `tests/` extensively compare the outputs from the original `pystoi` implementation with the implementation here. Results show the two implementations are equivalent (except for all-zeros inputs as described abovez, which can be neglected).
- The original tests were moved to `tests.old/`. These tests use `matlab.engine` and require Python 2.7. I was unable to run the tests despite my efforts. So the reference now is the original `pystoi` implementation.

## Benchmark

<img src="assets/times.png" alt="times" width=400> <img src="assets/speed_improvements.png" alt="speed_improvements" width=400>

## Install

`pip install batch-pystoi`

## Usage

### Same usage as in `pystoi`

```python
import soundfile as sf
from batch_pystoi import stoi

clean, fs = sf.read('path/to/clean/audio')
denoised, fs = sf.read('path/to/denoised/audio')

# clean and denoised must have same shape
# they can be 1D (single-channel) or 2D (multi-channel) since stoi now supports batched inputs
# swap axis from (samples, channels) to (channels, samples) if 2D
clean, denoised = clean.T, denoised.T

# compute stoi
d = stoi(clean, denoised, fs, extended=False)
```

### Using the `lengths` argument

```python
import soundfile as sf
import numpy as np
from batch_pystoi import stoi

clean_1, fs = sf.read('path/to/clean/audio/1')
clean_2, fs = sf.read('path/to/clean/audio/2')
denoised_1, fs = sf.read('path/to/denoised/audio/1')
denoised_2, fs = sf.read('path/to/denoised/audio/2')

# clean_1 and denoised_1 are 1D and have same length
# clean_2 and denoised_2 are 1D and have same length
# however clean_1 and clean_2 might have different lengths
# we pad to match the lengths and allow batched processing
max_length = max(clean_1.shape[-1], clean_2.shape[-1])
clean_1 = np.pad(clean_1, (0, max_length - clean_1.shape[-1]))
clean_2 = np.pad(clean_2, (0, max_length - clean_2.shape[-1]))
denoised_1 = np.pad(denoised_1, (0, max_length - denoised_1.shape[-1]))
denoised_2 = np.pad(denoised_2, (0, max_length - denoised_2.shape[-1]))

# make the batch
clean = np.stack([clean_1, clean_2])
denoised = np.stack([denoised_1, denoised_2])

# store original lengths
lengths = [clean_1.shape[-1], clean_2.shape[-1]]

# compute stoi
d = stoi(clean, denoised, fs, extended=False, lengths=lengths)
```

## References
* [1] C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen "A short-time objective intelligibility measure for time-frequency weighted noisy speech", Proc. ICASSP, 2010.
* [2] C. H. Taal, R. C. Hendriks, R. Heusdens and J.Jensen "An Algorithm for Intelligibility Prediction of Time–Frequency Weighted Noisy Speech", IEEE Trans. Audio, Speech, and Language Process., 2011.
* [3] J. Jensen and C. H. Taal, "An Algorithm for Predicting the Intelligibility of Speech Masked by Modulated Noise Maskers", IEEE/ACM Trans Audio, Speech, and Language Process., 2016.