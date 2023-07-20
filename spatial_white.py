import numpy as np


class SpatialWhitening():
    def __init__(self, signals_to_be_whitened):
        self.X = signals_to_be_whitened
        (nTime, nFreq, nMic) = self.X.shape
        Cov = np.einsum('tfm,tfk->fmk', self.X, self.X.conj()) / nTime
        D, E = np.linalg.eigh(Cov)
        self.mat = np.einsum('fm,fmk->fmk', 1 / np.sqrt(D), E.transpose(0, 2, 1).conj())
        self.matinv = np.linalg.inv(self.mat)

    def apply(self, signal):
        return np.einsum('fmk,...fk->...fm', self.mat, signal)

    def recover(self, signal):
        return np.einsum('fmk,...fk->...fm', self.matinv, signal)
