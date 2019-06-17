# lm_decoder
PyTorch Wrapper for simple fusion with an n-gram language model (KenLM)

### Installation

```
pip install .
```

### Basic Usage
```
from lm_decoder import KenLMDecoder

# setup the decoder
lm_dec = KenLMDecoder(vocab=vocab, model_path=lm_path)

# compute the language model scores given a sentence
lm_dec.language_model_scores(targets, masks, workers=20)

# beam search from the language model jointly with MT scores
# here we assume a mixture distribution controlled with a gate
lm_dec.beam_search_with_language_model(mt_probs, gates, masks, short_list=30, beam_size=10, workers=20)

```
