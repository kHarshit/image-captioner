import pickle
from collections import Counter
from nltk.tokenize import word_tokenize
from pycocotools.coco import COCO


class Vocabulary(object):
    """Vocabulary wrapper"""

    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word_to_idx:
            self.word_to_idx[word] = self.idx
            self.idx_to_word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word_to_idx:
            return self.word_to_idx['<unk>']
        return self.word_to_idx[word]

    def __len__(self):
        """returns vocab length"""
        return len(self.word_to_idx)


coco = COCO()
# counter variable of word frequencies
word_freq = Counter()
# indices
idx = coco.anns.keys()
for i, id in enumerate(idx):
    caption = str(coco.anns[id]['caption'])
    tokens = word_tokenize(caption.lower())
    # update word frequency
    word_freq.update(tokens)
    if (i+1) % 1000 == 0:
        print(f'[{(i+1)/len(ids)}] Tokenized the captions.')

# discard word if freq < min_word_freq
words = [w for w in w, freq in word_freq.items() if freq >= min_word_freq]

# add special tokens to vocab wrapper.
vocab = Vocabulary()
vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')

# add words to vocab
for i, word in enumerate(words):
    vocab.add_word(word)
print(f'vocab size: {len(vocab)}')

# save the vocab
with open('vocab.pickle', 'wb') as f:
    pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
