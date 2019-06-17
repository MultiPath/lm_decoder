#ifndef KENLM_SCORER_H_
#define KENLM_SCORER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"
#include "lm/enumerate_vocab.hh"
#include "lm/virtual_interface.hh"
#include "lm/word_index.hh"
#include "util/string_piece.hh"

const double OOV_SCORE = -1000.0;
const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "<unk>";
const std::string END_TOKEN = "</s>";

const float NUM_FLT_INF  = std::numeric_limits<float>::max();
const float NUM_FLT_MIN  = std::numeric_limits<float>::min();
const float NUM_FLT_LOGE = 0.4342944819;

using namespace lm::ngram;

// Implement a callback to retrive the dictionary of language model.
class RetriveStrEnumerateVocab : public lm::EnumerateVocab {
public:
  RetriveStrEnumerateVocab() {}

  void Add(lm::WordIndex index, const StringPiece &str) {
    vocabulary.push_back(std::string(str.data(), str.length()));
  }

  std::vector<std::string> vocabulary;
};

/* External scorer to query score for n-gram or sentence. */
class Scorer {

public:
  Scorer(const std::string &lm_path, const std::vector<std::string> &vocabs);
  ~Scorer();

  // return the max order
  size_t get_max_order() const { return max_order_; }

  // return the dictionary size of language model
  size_t get_dict_size() const { return dict_size_; }

  void start(State &state);
  float get_base_log_prob(State &prev_state, const std::string &word, State &out_state);
  std::vector<float> get_sent_log_prob(const std::vector<std::string> &words);
  std::string get_word(const int index);

protected:
  void setup(const std::string &lm_path, const std::vector<std::string> &vocab);

private:
  void *language_model_;
  size_t max_order_;
  size_t dict_size_;
  std::vector<std::string> vocabulary_;
};

#endif  // KENLM_SCORER_H_