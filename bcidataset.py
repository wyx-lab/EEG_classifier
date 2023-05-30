import torch
import os
import mne
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from mne.io import concatenate_raws
from model import bcinet
import pywt
from mne.decoding import CSP # Common Spatial Pattern Filtering


def wpd(X):
    coeffs = pywt.WaveletPacket(X, 'db4', mode='symmetric', maxlevel=5)
    return coeffs

def feature_bands(x):

    Bands = np.empty(
                (8, x.shape[0], x.shape[1], 30))  # 8 freq band coefficients are chosen from the range 4-32Hz

    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            pos = []
            C = wpd(x[i, ii, :])
            pos = np.append(pos, [node.path for node in C.get_level(5, 'natural')])
            for b in range(1, 9):
                 Bands[b - 1, i, ii, :] = C[pos[b]].data

    return Bands
class bcidataset(Dataset):
    def __init__(self,root):
        self.root  = root
        bcis = list(sorted(os.listdir(os.path.join(root))))
        is_contain = "E"
        for bci in bcis:
            if is_contain in bci:
                bcis.remove(bci)

        self.bcis = bcis
        self.datalist = os.listdir(root)
        path = os.path.join(self.root, "A03T.gdf")
        path2 = os.path.join(self.root, "A01T.gdf")
        raw = mne.io.read_raw_gdf(path)
        #raw = mne.io.read_raw_gdf(path2)
        raw = concatenate_raws([mne.io.read_raw_gdf(os.path.join(root, da)) for da in self.datalist])
        events, _ = mne.events_from_annotations(raw)
        # Pre-load the data
        raw.load_data()
        # Filter the raw signal with a band pass filter in 7-35 H
        raw.filter(4., 40., fir_design='firwin')
        # Remove the EOG channels and pick only desired EEG channels
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                               exclude='bads')
        tmin, tmax = 1., 4.
        # left_hand = 769,right_hand = 770,foot = 771,tongue = 772
        event_id = dict({'769': 7, '770': 8})#'771':9,'772':10})
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)
        labels = epochs.events[:, -1] - 6
        data = epochs.get_data()
        csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
        data = csp.fit_transform(data,labels)





        self.labels = labels
        self.data = data

    def __getitem__(self, idx):
        bcidata = self.data[idx]
        label = np.array(self.labels[idx])

        orient = [0,0]
        #Csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        orient[label-1] = 1
        orient = np.array(orient)
        #Csp.fit_transform(self.data, self.labels)
        bcidata = torch.from_numpy(bcidata)
        orient = torch.from_numpy(orient)




        return bcidata,orient
    def __len__(self):
        return len(self.labels)



