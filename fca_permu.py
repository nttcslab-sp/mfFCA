import permu
import fca


class mfFCAp(fca.mfFCA):
    def __init__(self, X, nSig, param_floor=1e-4, use_cupy=False):
        super().__init__(X, nSig, param_floor, use_cupy)
        self.permutation = permu.Permu(nSig, self.nFreq)

    def align_permutation(self, whitening, global_clusters, local_optimization):
        ''' Permutation alignment based on the characteristics of separated signals '''
        YAearly, _ = self.early_late_separated_signals()
        YAearly = whitening.recover(YAearly).transpose(1, 3, 2, 0)
        Ychar = permu.powRatio(YAearly)
        allpermu = self.permutation.permuYchar(Ychar, global_clusters, local_optimization)
        for f in range(self.nFreq):
            self.A[:, f] = self.A[allpermu[f], f]
            self.s[:, :, f] = self.s[:, allpermu[f], f]
