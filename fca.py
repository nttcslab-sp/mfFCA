class mfFCA():
    def __init__(self, X, nSig, param_floor=1e-4, use_cupy=False):
        if use_cupy is True:
            import cupy as cp
            self.use_cupy = True
        else:
            import numpy as cp
            self.use_cupy = False
        self.cp = cp
        (self.nTime, self.nFreq, self.nMic) = X.shape  # tfm
        self.X = self.cp.asarray(X)
        self.nSig = nSig
        self.span_list = ()
        self._makeXspan()
        Ascale = self.cp.ones((nSig, self.nFreq))
        self._initialize_variables(Ascale)
        self._initAfromX()
        #
        noise_power = 1e-3
        self.beta = noise_power * self.cp.ones((self.nFreq))  # f
        self.param_floor = param_floor
        self.s = self.cp.ones((self.nTime, nSig, self.nFreq))  # tnf
        self.losshist = []
        self.freq_groups = None

    def append_delay_list(self, dlist, scale_factor=0.1):
        self.span_list += dlist
        self._makeXspan()
        [nSig, nFreq, nDimOld, nDimOld] = self.A.shape
        Aold = self.A.copy()
        Ascale = self.cp.einsum('nfmm->nf', Aold).real / nDimOld
        Ascale *= scale_factor
        self._initialize_variables(Ascale)
        self.A[:, :, :nDimOld, :nDimOld] = Aold

    def _makeXspan(self):
        self.nSpan = len(self.span_list) + 1
        self.Xspan = self._concatenate_shifted_signals(self.X)

    def _concatenate_shifted_signals(self, X):
        ''' concatenate time shifted signals along the last dimension,
            which is intended to be the microphone and delay axis '''
        mlist = [X,]
        for d in self.span_list:
            mlist.append(self._shifted_matrix(X, d))
        return self.cp.concatenate(mlist, axis=-1)

    def _shifted_matrix(self, matrix, shift):
        ''' shift along the first dimension, which is intended to be the time axis '''
        if shift == 0:
            return matrix
        elif shift > 0:
            zeros_shape = (shift,) + matrix.shape[1:]
            return self.cp.concatenate((matrix[shift:], self.cp.zeros(zeros_shape)))     
        elif shift < 0:
            zeros_shape = (-shift,) + matrix.shape[1:]
            return self.cp.concatenate((self.cp.zeros(zeros_shape), matrix[:shift]))

    def _initialize_variables(self, Ascale):
        nDim = self.nMic * self.nSpan
        imat = self.cp.eye(nDim, dtype='complex128')
        self.eyeD = self.cp.tile(imat, (self.nSig, self.nFreq, 1, 1))
        self.A = self.cp.einsum('nf,mk->nfmk', Ascale, imat)

    def _initAfromX(self):
        XX = self.cp.einsum('tfm,tfk->tfmk', self.X, self.X.conj())
        for t in range(self.nTime):
            xt = self.X[t]
            xAx = self.cp.einsum('fm,nfmk,fk->nf', xt.conj(), self.A, xt)
            trace = self.cp.einsum('nfmm->nf', self.A)
            quad = xAx / trace
            idx = self.cp.argmax(quad, axis=0)
            for f in range(self.nFreq):
                self.A[idx[f], f] += XX[t, f]
        # makes A's diagonals one on average
        self.A = self.A / self.cp.einsum('nfmm->nf', self.A)[:, :, None, None] * self.nMic

    def optimizationEM(self, nLoop):
        for i in range(nLoop):
            self._calcXbar(), self.calc_loss(), self._calcCtilde(), self._updateSem()
            self._calcXbar(), self._calcCtilde(), self._updateAem()

    def _calcXbar(self):
        noise = self.cp.einsum('f,fde->fde', self.beta, self.eyeD[0])
        self.Cbar = self.cp.einsum('nfde,tnf->tnfde', self.A, self.s)
        Cbarsum = self.Cbar.sum(axis=1)  # tfde
        self.Xbar = Cbarsum + noise[None, :]
        for i, d in enumerate(self.span_list):
            self._add_diagshift_matrix(self._shifted_matrix(Cbarsum, -d), -(i+1))
            self._add_diagshift_matrix(self._shifted_matrix(Cbarsum, d), (i+1))
        self.Xbi = self._inv_Hermitian(self.Xbar)  # tfde

    def _add_diagshift_matrix(self, matrix, shift):
        ms = shift * self.nMic
        if shift < 0:
            self.Xbar[..., :ms, :ms] += matrix[..., -ms:, -ms:]  # ms < 0
        elif shift > 0:
            self.Xbar[..., ms:, ms:] += matrix[..., :-ms, :-ms]  # ms > 0

    def _calcCtilde(self):
        Cmu = self.cp.einsum('tnfkl,tflm,tfm->tnfk', self.Cbar, self.Xbi, self.Xspan)
        Csigma = self.Cbar - self._calcXAX(self.Xbi[:, None], self.Cbar)
        self.Ctilde = self.cp.einsum('tnfm,tnfk->tnfmk', Cmu, Cmu.conj()) + Csigma

    def _inv_Hermitian(self, A):
        invA = self.cp.linalg.inv(A)
        return self._force_Hermitian_mean(invA)

    def _calcXAX(self, A, X):
        XAX = self.cp.einsum('...kj,...kl,...lm->...jm', X.conj(), A, X)
        return self._force_Hermitian_mean(XAX)

    def _force_Hermitian_mean(self, A):
        AH = self.cp.swapaxes(A, -2, -1).conj()
        return (A + AH) / 2

    def _updateAem(self):
        self.A = (self.Ctilde / self.s[:, :, :, None, None]).mean(axis=0)
        self.A += self.param_floor * self.eyeD

    def _updateSem(self):
        Ainv = self._inv_Hermitian(self.A)
        trace = self.cp.einsum('nfmk,tnfkm->tnf', Ainv, self.Ctilde)
        self.s = trace.real / (self.nMic * self.nSpan)
        self.s = self.cp.maximum(self.s, self.param_floor)
        if self.freq_groups is not None:
            self._average_s_freqGroup()

    def set_freqGroup(self, freq_width):
        self.freq_groups = self.cp.array([int(self.cp.floor(i/freq_width)) for i in range(self.nFreq)])
        self._average_s_freqGroup()

    def _average_s_freqGroup(self):
        maxGroup = max(self.freq_groups).item() + 1
        for g in range(maxGroup):
            idx = (self.freq_groups == g)
            self.s[..., idx] = self.s[..., idx].mean(axis=-1)[..., None]

    def calc_loss(self):
        trace = self.cp.einsum('tfm,tfmk,tfk->tf', self.Xspan.conj(), self.Xbi, self.Xspan).real
        det = self.cp.linalg.det(self.Xbar)
        logdet = self.cp.log(det.real)
        total_loss = trace.sum() + logdet.sum()
        return self.losshist.append(total_loss.item())

    def report_scale(self):
        print('s', self.s.mean(), 'A', self.cp.linalg.eigh(self.A)[0].mean())

    def early_late_separated_signals(self, new_input=None):
        self._multichannel_wiener_filter()
        if new_input is not None:
            input = self._concatenate_shifted_signals(self.cp.asarray(new_input))
        else:
            input = self.Xspan
        Cspan = self.cp.einsum('ntfkm,t...fm->tn...fk', self.WF, input)
        # decompose Cspan into Yearly and Ylate
        idxs, idxe = 0, self.nMic
        Yearly = Cspan[..., idxs:idxe]
        Ylate = self.cp.zeros_like(Yearly)
        for d in self.span_list:
            idxs += self.nMic
            idxe += self.nMic
            Ylate += self._shifted_matrix(Cspan[..., idxs:idxe], -d)
        return self._tonumpy(Yearly), self._tonumpy(Ylate)

    def _multichannel_wiener_filter(self):
        self._calcXbar()
        self.WF = []  # nftde, multi-frame multichannel Wiener filter
        for n in range(self.Cbar.shape[1]):
            self.WF.append(self.cp.matmul(self.Cbar[:, n], self.Xbi))
        self.WF = self.cp.array(self.WF)

    def _tonumpy(self, array):
        if self.use_cupy is False:
            return array
        else:
            return array.get()
