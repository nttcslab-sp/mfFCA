# mfFCA: multi-frame Full-rank spatial Covariance Analysis
# sample code for speech mixtures

# import our own code
import fca_permu
import spatial_white
import util


# function for blind source separation
def perform_bss(stft_mixture, num_sources, nloop_per_block, adding_delays, num_extra_blocks):
    whitening = spatial_white.SpatialWhitening(stft_mixture)
    whitened_stft_mixture = whitening.apply(stft_mixture)
    bss = fca_permu.mfFCAp(whitened_stft_mixture, num_sources, use_cupy=True)
    bss.optimizationEM(nloop_per_block)
    bss.align_permutation(whitening, global_clusters=2, local_optimization=True)
    freq_width = 4
    bss.set_freqGroup(freq_width)
    for dl in adding_delays:
        print(f'adding {dl}')
        bss.append_delay_list(dl)
        bss.optimizationEM(nloop_per_block)
    for i in range(num_extra_blocks):
        print('.', end='')
        bss.optimizationEM(nloop_per_block)
    print('mfFCA finished')
    stft_early, stft_late = bss.early_late_separated_signals()
    stft_early = whitening.recover(stft_early).transpose(1, 3, 2, 0)
    stft_late = whitening.recover(stft_late).transpose(1, 3, 2, 0)
    return stft_early, stft_late


def main():
    # some settings
    num_sources = 3
    nloop_per_block = 50
    adding_delays = ((2,), (4,), ())
    num_extra_blocks = 6
    # read mixture file
    file_name = 'mixture_2microphones_3speeches.wav'
    mixture = util.wave_read(file_name)
    fftSize, frameShift = 1024, 256
    stft_mixture = util.stft(mixture, fftSize, frameShift)
    stft_mixture = stft_mixture.transpose(2, 1, 0)
    # perform BSS
    stft_early, stft_late = perform_bss(stft_mixture, num_sources, nloop_per_block, adding_delays, num_extra_blocks)
    # separated signals
    y_total = util.istft(stft_early + stft_late, fftSize, frameShift)
    y_early = util.istft(stft_early, fftSize, frameShift)
    y_late = util.istft(stft_late, fftSize, frameShift)
    util.wave_out(y_total, 'y_total')
    util.wave_out(y_early, 'y_early')
    util.wave_out(y_late, 'y_late')


if __name__ == "__main__":
    main()
