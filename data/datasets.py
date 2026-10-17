import os
import torch
import torchaudio
from torch.utils.data import Dataset
from utils.features import FeatureExtractor

LABEL_MAP = {
    # IEMOCAP / RAVDESS / EMO-DB / Urdu -> unified {angry, happy, sad, neutral}
    "ang": 0, "angry": 0, "A": 0,
    "hap": 1, "happy": 1, "H": 1,
    "sad": 2, "S": 2,
    "neu": 3, "neutral": 3, "N": 3,
}

class EmotionDataset(Dataset):
    def __init__(self, items, feat_extractor, sample_rate=16000):
        # items: list of (wav_path, label_int)
        self.items = items
        self.fx = feat_extractor
        self.sr = sample_rate

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        F = self.fx(wav)                      # (T, 103)
        return F, y

def collate_pad(batch):
    feats, labels = zip(*batch)
    T = max(f.size(0) for f in feats)
    d = feats[0].size(1)
    out = torch.zeros(len(feats), T, d)
    for i, f in enumerate(feats):
        out[i, :f.size(0)] = f
    return out, torch.tensor(labels, dtype=torch.long)

# ---- corpus parsers: EDIT to match your local file layout ----
def parse_iemocap(root):
    items = []
    for sess in sorted(os.listdir(root)):
        if not sess.startswith("Session"): continue
        lab = os.path.join(root, sess, "dialog", "EmoEvaluation")
        wav_dir = os.path.join(root, sess, "sentences", "wav")
        # TODO: parse EmoEvaluation/*.txt for utterance id + emotion,
        #       map to LABEL_MAP, locate wav in wav_dir
    return items

def parse_ravdess(root):
    items = []
    for actor in sorted(os.listdir(root)):
        adir = os.path.join(root, actor)
        if not os.path.isdir(adir): continue
        for f in os.listdir(adir):
            if not f.endswith(".wav"): continue
            emo_code = f.split("-")[2]       # 03=happy,04=sad,05=angry,01=neutral
            code_map = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry"}
            if emo_code not in code_map: continue
            items.append((os.path.join(adir, f), LABEL_MAP[code_map[emo_code]]))
    return items

def parse_emodb(root):
    items = []
    # EMO-DB filenames: 03a01Fa.wav -> emotion letter at position 5
    code_map = {"A": "angry", "F": "happy", "H": "happy",
                "S": "sad", "N": "neutral", "W": "angry"}
    for f in os.listdir(root):
        if not f.endswith(".wav"): continue
        letter = f[5]
        if letter not in code_map: continue
        items.append((os.path.join(root, f), LABEL_MAP[code_map[letter]]))
    return items

def parse_urdu(root):
    items = []
    # TODO: adapt to the released Urdu corpus directory layout and labels
    return items
