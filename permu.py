import numpy as np


def powRatio(YA, freq_groups=None):
    ''' Calculate power ratio for separated signals
        YA: scaled separated signals with shape (nSig, nMic, nFreq, nTime)
    '''
    if freq_groups is not None:
        YA2 = []
        maxGroup = max(freq_groups) + 1
        for g in range(maxGroup):
            idx = (freq_groups == g)
            YAg = YA[:, :, idx]
            YA2.append(np.real(YAg * YAg.conj()).sum(axis=1).mean(axis=1)[:, None])
        YA2 = np.concatenate(YA2, axis=1)
    else:
        YA2 = np.real(YA * YA.conj()).sum(axis=1)  # shape (nSig, nFreq, nTime)
    YA2 /= YA2.sum(axis=0)[None, :, :] + np.finfo(YA2.dtype).eps
    powRatio = YA2.transpose(1, 0, 2)  # shape (nFreq, nSig, nTime)
    return powRatio


class Permu():
    def __init__(self, nSig, nFreq):
        self.nSig, self.nFreq = nSig, nFreq
        self.clear_centroids()
        self.from_scratch = True
        self.adj_bins = np.array([-3, -2, -1, 1, 2, 3])
        self.harm_adj_bins = np.array([-1, 0, 1])

    def clear_centroids(self):
        self.centroids, self.labels = None, None

    def set_adj_bins(self, adj_bins):
        self.adj_bins = np.array(adj_bins)

    def set_harm_adj_bins(self, harm_adj_bins):
        if harm_adj_bins is not None:
            self.harm_adj_bins = np.array(harm_adj_bins)
        else:
            self.harm_adj_bins = None

    def permuYchar(self, Ychar, global_clusters, local_optimization):
        ''' Align permutation based on the characteristics of separated signals
            Ychar: characteristics of separated signals Y
            global_clusters: specify #clusters for each signal in the global optimization
            local_optimization: True or False, perform local optimization or not
        '''
        (nFreq, nSig, nTime) = Ychar.shape
        self.allpermu = np.tile(np.arange(nSig)[None, :], (nFreq, 1))  # shape (nFreq, nSig)
        Ychar = _zero_mean_unit_norm(Ychar, axis=2)
        # global optimization
        if global_clusters >= 1:
            Ychar = self._global_permu(Ychar, global_clusters)
        # local optimization
        if local_optimization:
            self._local_permu(Ychar)
        self.from_scratch = False
        return self.allpermu

    def _global_permu(self, Ychar, max_clusters):
        (nFreq, nSig, nTime) = Ychar.shape
        identity = np.arange(self.nSig)
        obj_hist = []
        # C: cluster centroids with shape (nSig, nCl, nTime), here nCl=1
        if self.from_scratch:  # init_cluster == 'heuristic'
            finit = np.rint(nFreq / 3).astype(np.int32) - 1
            self.centroids = Ychar[finit, :, :][:, None, :]
        elif self.centroids is None:  # init_cluster == 'mean':
            self.centroids = None
            self._cluster_Ychar(Ychar)
        else:  # init_cluster == 'previous':
            print('inherited previous centroids')
        nCl_init = self.centroids.shape[1]
        # main loop
        maxLoop = 30
        for nCl in range(nCl_init, max_clusters + 1):
            for lo in range(maxLoop):
                changed = 0
                # cosine between Ychar and C
                cosd = np.einsum('fyt,sct->cfys', Ychar, self.centroids) / nTime
                permu, pow = decide_permu(cosd.reshape((nCl * nFreq, nSig, nSig)))
                permu = permu.reshape((nCl, nFreq, nSig))
                objs = _diag_offdiag(pow).reshape((nCl, nFreq))
                selected_cl = objs.argmax(axis=0)
                obj_hist.append(objs.max(axis=0).sum() / nSig / nFreq)
                # permutations for frequency bins
                for f in range(nFreq):
                    cl = selected_cl[f]
                    if not np.all(permu[cl, f] == identity):
                        changed += 1
                        self.allpermu[f] = self.allpermu[f][permu[cl, f]]
                        Ychar[f] = Ychar[f][permu[cl, f]]
                # update C
                self._cluster_Ychar(Ychar)
                if changed == 0:
                    break
            print(f'_global_permu(), nCl={nCl}, finished after {lo} iterations, obj={obj_hist[-1]}')
            # increase clusters
            if nCl < max_clusters:
                self._increase_cluster(Ychar, objs.max(axis=0))
        return Ychar

    def _cluster_Ychar(self, Ychar):
        (nFreq, nSig, nTime) = Ychar.shape
        self.labels = np.zeros((nSig, nFreq), dtype=np.uint8)
        if self.centroids is None:
            nCl = 1
        else:
            nCl = self.centroids.shape[1]
        ####
        if nCl == 1:
            self.centroids = Ychar.mean(axis=0)[:, None, :]  # shape (nSig, 1, nTime)
            self.centroids = _zero_mean_unit_norm(self.centroids, axis=2)
        else:  # nCl >= 2
            old_centroids = self.centroids.copy()
            self.centroids = np.empty((nSig, nCl, nTime))
            for s in range(nSig):
                self.centroids[s], self.labels[s] = _kmeans(Ychar[:, s, :], old_centroids[s])

    def _increase_cluster(self, Ychar, objs):
        (nSig, nCl, nTime) = self.centroids.shape
        med = np.median(objs)
        new_cluster_freqs = objs < med
        self.labels[:, new_cluster_freqs] = nCl
        self.centroids = np.empty((nSig, nCl + 1, nTime))
        for s in range(nSig):
            for c in range(nCl + 1):
                freqs = (self.labels[s] == c)
                self.centroids[s, c] = Ychar[freqs, s].mean(axis=0)
        self.centroids = _zero_mean_unit_norm(self.centroids, axis=2)

    def _local_permu(self, Ychar):
        (nFreq, nSig, nTime) = Ychar.shape
        affected, reverse = self._affected_frequencies(nFreq)
        # initial calculation of permu and gains for all frequency bins
        corr_all = np.empty((nFreq, nSig, nSig))
        for f in range(nFreq):
            aff = affected[f]
            corr_all[f] = np.einsum('yt,ast->ys', Ychar[f], Ychar[aff]) / aff.shape[0]
        permu, pow = decide_permu(corr_all)  # permu: shape (nFreq, nSig)
        gains = _diag_offdiag(pow) - _diag_offdiag(corr_all)  # gains: shape (nFreq,)
        # main loop
        maxLoop = 500
        for lo in range(maxLoop):
            maxGain = np.max(gains)
            if maxGain <= 0:
                break
            k = np.argmax(gains)
            # print(f'maxScore={gains[k]}, k={k}, permu={permu[k]}')
            self.allpermu[k] = self.allpermu[k][permu[k]]
            Ychar[k] = Ychar[k][permu[k]]
            gains[k] = 0
            # update of permu and gains for affected frequencies
            affk = reverse[k]
            corr_affk = []
            for f in affk:
                aff = affected[f]
                corr = np.einsum('yt,ast->ys', Ychar[f], Ychar[aff]) / aff.shape[0]
                corr_affk.append(corr)
            corr_affk = np.array(corr_affk)
            permu[affk], powk = decide_permu(corr_affk)
            gains[affk] = _diag_offdiag(powk) - _diag_offdiag(corr_affk)
        print(f'_local_permu() finished after {lo} iterations')

    def _affected_frequencies(self, nFreq):
        # adjacent frequency bins and harmonic frequency bins
        affected = []
        reverse = [set() for f in range(nFreq)]
        for f in range(nFreq):
            fadj = f + self.adj_bins
            if self.harm_adj_bins is not None:
                fharm0 = int(round(f / 2)) + self.harm_adj_bins
                fharm2 = f * 2 + self.harm_adj_bins
            else:
                fharm0 = fharm2 = np.array([], dtype=np.int32)
            # unification
            aff = np.unique(np.concatenate((fharm0, fadj, fharm2)))
            # delete out-of-bounds and self
            conditions = np.logical_or(np.logical_or(aff < 0, nFreq <= aff), aff == f)
            if conditions.any():
                aff = np.delete(aff, np.nonzero(conditions))
            affected.append(aff)
            for k in aff:
                reverse[k].add(f)
        for k in range(nFreq):
            arr = np.array(list(reverse[k]))
            arr.sort()
            reverse[k] = arr
        return affected, reverse


def decide_permu(A):
    ''' A: shape (nMat, nDim, nDim)
          For A[f], f = 0, ..., nMat-1, calclate a permutation which maximize the obj[f],
          obj[f] = diag(pA[f]).sum() - offdiag(pA[f]).sum() with pA[f] = A[f][permu[f],:].
        permu: permutations with shape (nMat, nDim)
        pA: permutated A with shape (nMat, nDim, nDim)
    '''
    (nMat, nDim, nDim) = A.shape
    pA = A.copy()
    permu = -1 * np.ones((nMat, nDim), dtype=np.int16)
    rrow = np.tile(np.arange(nDim)[None, :], (nMat, 1))  # remaining row, shape (nMat, nDim)
    # For the first nDim-2 rows/colums, greedy select the maximum one-by-one
    for o in range(nDim - 2):
        ind = np.argmax(pA.reshape(nMat, nDim * nDim), axis=1)
        (iii, jjj) = np.unravel_index(ind, (nDim, nDim))
        for p in range(nDim):
            fidx = np.nonzero(iii == p)
            pA[fidx, p, :] = -np.inf
            fidx = np.nonzero(jjj == p)
            pA[fidx, :, p] = -np.inf
        for f in range(nMat):
            permu[f, jjj[f]] = iii[f]
            rrow[f, iii[f]] = -1
    # Making the remaining 2x2 matrix by extracting elements that are not -np.inf
    pA_flat = pA.flatten()
    pA22 = pA_flat[pA_flat != -np.inf].reshape(nMat, 2, 2)
    rrow_flat = rrow.flatten()
    rrow2 = rrow_flat[rrow_flat >= 0].reshape(nMat, 2)
    # For the remaining 2x2 matrix, select the best
    obj = _diag_offdiag(pA22)
    for f in range(nMat):
        idx = (permu[f] == -1)
        if obj[f] >= 0:
            permu[f, idx] = rrow2[f]
        else:
            permu[f, idx] = rrow2[f][::-1]
        # The final result
        pA[f] = A[f][permu[f]]
    return permu, pA


def _kmeans(samples, old_centroids):
    ''' find K clusters based on correlation for zero-mean, unit-norm data
        samples: shape (nFreq, nTime)
        init_centroids: shape (nCl, nTime)
    '''
    (nCl, nTime) = old_centroids.shape
    centroids = np.empty((nCl, nTime))
    maxLoop = 10
    for lo in range(maxLoop):
        corr = np.einsum('ft,ct->fc', samples, old_centroids)
        labels = np.argmax(corr, axis=1)
        for cl in range(nCl):
            centroids[cl] = samples[labels == cl, :].sum(axis=0)
        centroids = _zero_mean_unit_norm(centroids, axis=1)
        if np.all(centroids == old_centroids):
            break
        old_centroids = centroids.copy()
    return centroids, labels


def _zero_mean_unit_norm(A, axis):
    A -= np.expand_dims(A.mean(axis=axis), axis=axis)
    A /= np.expand_dims(np.sqrt((A**2).mean(axis=axis)), axis=axis) + np.finfo(A.dtype).eps
    return A


def _diag_offdiag(A):
    ''' A: shape (nMat, nDim, nDim)
        obj: diag - offdiag
             diag = np.einsum('fii->f', A)
             offdiag = np.einsum('fij->f', A) - diag
    '''
    obj = 2 * np.einsum('fii->f', A) - np.einsum('fij->f', A)
    return obj
