#include "kenlm_scorer.h"
#include <unistd.h>
#include <iostream>

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"
#include "util/string_piece.hh"
#include "util/tokenize_piece.hh"

using namespace lm::ngram;

Scorer::Scorer(const std::string &lm_path, const std::vector<std::string> &vocabs)
{

  language_model_ = nullptr;

  max_order_ = 0;
  dict_size_ = 0;
  setup(lm_path, vocabs);
}

Scorer::~Scorer()
{
  if (language_model_ != nullptr)
  {
    delete static_cast<lm::base::Model *>(language_model_);
  }
}

void Scorer::setup(const std::string &lm_path, const std::vector<std::string> &vocab)
{
  const char *filename = lm_path.c_str();
  RetriveStrEnumerateVocab enumerate;
  lm::ngram::Config config;
  config.enumerate_vocab = &enumerate;
  language_model_ = lm::ngram::LoadVirtual(filename, config);
  max_order_ = static_cast<lm::base::Model *>(language_model_)->Order();
  vocabulary_ = vocab;
  dict_size_ = vocabulary_.size();
}

void Scorer::start(State &state)
{
  lm::base::Model* model = static_cast<lm::base::Model*>(language_model_);
  model->BeginSentenceWrite(&state);
}

float Scorer::get_base_log_prob(State &prev_state, const std::string &word, State &out_state)
{
  lm::base::Model *model = static_cast<lm::base::Model *>(language_model_);
  lm::WordIndex word_index = model->BaseVocabulary().Index(word);
  if (word_index == 0){
    return OOV_SCORE;
  }
  float cond_prob = model->BaseScore(&prev_state, word_index, &out_state);
  return cond_prob / NUM_FLT_LOGE;  // log10 --> loge
}

std::string Scorer::get_word(const int index)
{
  return vocabulary_[index];
}

std::vector<float> Scorer::get_sent_log_prob(const std::vector<std::string> &words)
{
  lm::base::Model* model = static_cast<lm::base::Model*>(language_model_);
  lm::ngram::State state, tmp_state, out_state;
  std::vector<float> scores;

  model->BeginSentenceWrite(&state);
  for (size_t i = 0; i < words.size(); i++)
  {
    scores.push_back(get_base_log_prob(state, words[i], out_state));

    // make sure pointers are not mixed
    tmp_state = state;
    state = out_state;
    out_state = tmp_state;
  }
  scores.push_back(get_base_log_prob(state, END_TOKEN, out_state));
  return scores;
}