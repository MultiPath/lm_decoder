import torch
from ._ext import lm_decoder


class KenLMDecoder(object):
    def __init__(self, vocab=None, model_path=None):
        self.vocab = vocab
        self.lm_scorer = lm_decoder.get_kenlm_scorer(model_path, vocab)

    def language_model_scores(self, targets, masks, workers=20):
        """
        inputs:
            targets:  batch x seqlen
            masks:    batch x seqlen
        """
        lm_scores = targets.new_zeros(targets.size(0),
                                      targets.size(1) + 1).cpu().float()
        lm_decoder.get_kenlm_scores(self.lm_scorer,
                                    targets.cpu().int(),
                                    masks.sum(1).cpu().int(), lm_scores,
                                    workers)
        if targets.is_cuda:
            lm_scores = lm_scores.cuda(targets.get_device())
        return lm_scores

    def beam_search_with_language_model(self,
                                        mt_probs,
                                        gates,
                                        masks,
                                        short_list=30,
                                        beam_size=5,
                                        workers=20):
        """
        inputs:
            mt_probs:  batch x seqlen x vocab (softmax results over a vocabulary)
            gates:     batch x seqlen 
            masks:     batch x seqlen
            short_list:  we do not check all the possible tokens. Only topN from the MT system is considered.
        
        return:
            best translation
        """
        probs, seqs = mt_probs.topk(short_list, 2)
        _seqs = seqs.cpu().int()
        scores = lm_decoder.beam_search(self.lm_scorer,
                                        probs.cpu().float(), _seqs,
                                        gates.cpu().float(),
                                        masks.sum(1).cpu().int(), beam_size,
                                        workers)
        if seqs.is_cuda:
            _seqs = _seqs.cuda(seqs.get_device())
        _seqs = _seqs.type_as(seqs)
        return scores, _seqs