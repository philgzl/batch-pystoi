import argparse
import timeit
import itertools

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16])
    parser.add_argument('--wav-length', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16],)
    parser.add_argument('--fs', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()

    pystoi_times = np.empty((len(args.batch_size), len(args.wav_length)))
    batch_pystoi_times = np.empty((len(args.batch_size), len(args.wav_length)))

    for i, j in itertools.product(
        range(len(args.batch_size)),
        range(len(args.wav_length)),
    ):
        batch_size = args.batch_size[i]
        wav_length = args.wav_length[j]
        print(f'batch_size={batch_size}, wav_length={wav_length}')

        np.random.seed(42)
        x = np.random.randn(batch_size, wav_length*args.fs)
        y = np.random.randn(batch_size, wav_length*args.fs)

        pystoi_times[i, j] = timeit.timeit(
            "for k in range(batch_size): stoi(x[k], y[k], args.fs, True)",
            number=args.repeats,
            setup=(
                "from __main__ import args, x, y, batch_size;"
                "from pystoi import stoi"
            )
        )

        batch_pystoi_times[i, j] = timeit.timeit(
            "stoi(x, y, args.fs, True)",
            number=args.repeats,
            setup=(
                "from __main__ import args, x, y, batch_size;"
                "from batch_pystoi import stoi"
            )
        )

    # raw times
    for j, wav_length in enumerate(args.wav_length):
        plt.figure()
        for times, label in [
            (pystoi_times[:, j], 'pystoi'),
            (batch_pystoi_times[:, j], 'batch_pystoi'),
        ]:
            plt.plot(times, 'o-', label=label)
        plt.title(
            f'repeats={args.repeats}, '
            f'wav_length={wav_length}s, '
            f'fs={args.fs}' + (' (no resampling)' if args.fs == 10000 else '')
        )
        plt.xlabel('Batch size')
        plt.ylabel('Time (s)')
        plt.xticks(range(len(args.batch_size)), args.batch_size)
        plt.legend()

    # speed fold increases
    plt.figure()
    for j, wav_length in enumerate(args.wav_length):
        plt.plot(
            pystoi_times[:, j]/batch_pystoi_times[:, j],
            'o-', label=f'wav_length={wav_length}s'
        )
    plt.title(
        f'repeats={args.repeats}, '
        f'fs={args.fs}' + (' (no resampling)' if args.fs == 10000 else '')
    )
    plt.xlabel('Batch size')
    plt.ylabel('Speed fold increase (x)')
    plt.xticks(range(len(args.batch_size)), args.batch_size)
    plt.legend()

    plt.show()
