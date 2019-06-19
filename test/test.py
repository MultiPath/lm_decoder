import unittest
import torch
import lm_decoder

class Test(unittest.TestCase):

    def setUp(self):
        model_path = "/private/home/jgu/data/wmt16/en-de/lm/lm.train.bpe.4gram.de.bin"
        self.vocab = ['<init>', '<eos>'] + 'Frau Präsidentin , zur Geschäftsordnung .'.split()
        self.decoder = lm_decoder.KenLMDecoder(model_path=model_path, vocab=self.vocab, beam_width=2)

    def test_a(self):
        import time
        t = time.time()
        for _ in range(1000):
            seqs  = torch.LongTensor([[[4, 2], [6, 3], [4, 5], [5, 1], [5, 6]]])
            masks = torch.ByteTensor([[1, 1, 1, 1, 1]])
            probs = torch.FloatTensor([[[0.7, 0.2], [0.3, 0.3], [0.4, 0.2], [0.6, 0.2], [0.2, 0.2]]])
            lens = masks.sum(1)

            seqs = seqs.expand(20, 5, 2)
            probs = probs.expand(20, 5, 2)
            lens = lens.expand(20)
            seqss, scores = self.decoder.test(seqs, probs, lens)
            #print(seqss[0], scores)
        print(time.time() - t)
if __name__ == '__main__':
    unittest.main()
