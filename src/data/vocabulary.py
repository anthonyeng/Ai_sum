from collections import Counter

SPECIAL_TOKENS = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}


class Vocabulary:
    def __init__(self):
        self.word2idx = dict(SPECIAL_TOKENS)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.freq: Counter = Counter()

    def build(self, captions, min_freq=2):
        for cap in captions:
            for word in cap.lower().split():
                self.freq[word] += 1
        for word, count in self.freq.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, caption, max_len=30):
        tokens = ["<sos>"] + caption.lower().split()[: max_len - 2] + ["<eos>"]
        ids = [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        words = []
        for i in ids:
            word = self.idx2word.get(i, "<unk>")
            if word == "<eos>":
                break
            if word not in ("<pad>", "<sos>", "<unk>"):
                words.append(word)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)
