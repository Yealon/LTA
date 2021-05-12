import torch

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

PAD = 0
UNK = 1
BOS = 2
EOS = 3


def flatten(l):
    """
    :param l: 需要拉平的list
    将list拉平：[1, 2, [3, 4, [5, 6]], ["abc", "def"]] --> [1, 2, 3, 4, 5, 6, "abc", "def"]
    """
    for el in l:
        if hasattr(el, "__iter__"):
            for sub in flatten(el):
                yield sub
        else:
            yield el


class Dict(object):

    def __init__(self, data=None, lower=False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    def loadDict(self, idxToLabel):
        for i in range(len(idxToLabel)):
            label = idxToLabel[i]
            self.add(label, i)

    # Load entries from a file.
    def loadFile(self, filename):
        for line in open(filename):
            fields = line.split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord=False, bosWord=False, eosWord=False):
        vec = []

        if bosWord is True:
            vec += [self.lookup(BOS_WORD)]

        unk = ''
        if unkWord is True:
            unk = self.lookup(UNK_WORD)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is True:
            vec += [self.lookup(EOS_WORD)]

        vec = [x for x in flatten(vec)]

        return torch.LongTensor(vec)

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertTonumpyIdx(self, labels, unkWord=False, bosWord=False, eosWord=False):
        vec = []

        if bosWord is True:
            vec += [self.lookup(BOS_WORD)]

        unk = ''
        if unkWord is True:
            unk = self.lookup(UNK_WORD)
            vec += [self.lookup(label, default=unk) for label in labels]
        else:
            vec += [self.lookup(label) for label in labels]

        if eosWord is True:
            vec += [self.lookup(EOS_WORD)]

        vec = [x for x in flatten(vec)]

        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        labels = []

        for i in idx:
            i = int(i)
            if i == stop:
                break
            labels += [self.getLabel(i)]

        return labels

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)
