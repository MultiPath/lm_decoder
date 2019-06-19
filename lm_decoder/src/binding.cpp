#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <set>
#include <numeric>   // std::iota
#include <algorithm> // std::sort
#include "lm/model.hh"
#include "lm/config.hh"
#include "util/tokenize_piece.hh"
#include "util/string_piece.hh"
#include "util/string_stream.hh"
#include "kenlm_scorer.h"
#include "ThreadPool.h"

using namespace lm::ngram;

template <typename T>
std::vector<size_t> argtopk(const std::vector<T> &v, int k)
{

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // from large to small
  std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                   [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

template <typename T>
std::vector<size_t> argsort(const std::vector<T> &v)
{

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // from large to small
  std::sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

float log_sum_exp(std::vector<float> &v, int begin, int end)
{
  float init = 0;
  float max_elem = *std::max_element(v.begin() + begin, v.begin() + end);
  float sum = std::accumulate(v.begin() + begin, v.begin() + end, init, 
     [max_elem](float a, float b) { return a + std::exp(b - max_elem); });
  return max_elem + std::log(sum);
}


float weighted_logsumexp(float x, float y, float g)
{
  if (x > y){
    return x + std::log(g + (1 - g) * std::exp(y - x));
  }
  else{
    return y + std::log((1 - g) + g * std::exp(x - y));
  }
}

std::vector<float>
 _beam_search_(Scorer *lm_scorer,
               at::Tensor &probs,
               at::Tensor &seqs,
               at::Tensor &gates,
               at::Tensor &lens,
               const int beam_width,
               int index,
               const int type)
{
  // type = 0: mixture of experts: log( \alpha * p_TM + (1 - \alpha) * p_LM )
  // type = 1: shallow fusion: \alpha * log p_TM + (1 - \alpha) * log p_LM (as scores)
  // type = 2: simple fusion: log( softmax(\alpha * TM_scores + (1 - \alpha) * log p_LM (as scores)))

  auto probs_acc = probs.accessor<float, 3>();
  auto seqs_acc = seqs.accessor<int, 3>();
  auto lens_acc = lens.accessor<int, 1>();
  auto gates_acc = gates.accessor<float, 2>();

  const int32_t num_candidates = probs.size(2);
  const int32_t length = (int)lens_acc[index];

  // for simplification
  assert (~(beam_width > num_candidates));  // beam size cannot be bigger than candidates.

  std::vector<std::vector<int>> prefixes(beam_width);
  std::vector<std::vector<int>> temp_prefixes(beam_width);
  std::vector<float> cum_scores(beam_width, 0);
  std::vector<float> temp_scores(beam_width * num_candidates);
  std::vector<size_t> idx(beam_width);

  // KenLM states
  std::vector<State> states(beam_width);
  std::vector<State> tmp_states(beam_width);
  std::vector<State> out_states(beam_width * num_candidates);

  // start the language model.
  lm_scorer->start(states[0]);

  for (size_t t = 0; t < length; t++)
  {
    int max_width = (t > 0) ? beam_width : 1;
    for (size_t b = 0; b < max_width; b++)
    {
      for (size_t c = 0; c < num_candidates; c++)
      {
        if (type == 0){
          temp_scores[b * num_candidates + c] = cum_scores[b] + 
            weighted_logsumexp(
              std::log(probs_acc[index][t][c]), 
              lm_scorer->get_base_log_prob(
                states[b], 
                lm_scorer->get_word(seqs_acc[index][t][c]), 
                out_states[b * num_candidates + c]),
              gates_acc[index][t]);
        }
        else if (type == 1){
          temp_scores[b * num_candidates + c] = cum_scores[b] + 
            gates_acc[index][t] * std::log(probs_acc[index][t][c]) +
            (1 - gates_acc[index][t]) * lm_scorer->get_base_log_prob(
              states[b], lm_scorer->get_word(seqs_acc[index][t][c]), out_states[b * num_candidates + c]);

        }
        else if (type == 2){
          temp_scores[b * num_candidates + c] = 
            gates_acc[index][t] * std::log(probs_acc[index][t][c]) +
            (1 - gates_acc[index][t]) * lm_scorer->get_base_log_prob(
              states[b], lm_scorer->get_word(seqs_acc[index][t][c]), out_states[b * num_candidates + c]);
        }
      }

      if (type == 2){
        float Z = log_sum_exp(temp_scores, b * num_candidates, (b + 1) * num_candidates);
        for (size_t c = 0; c < num_candidates; c++)
        {
          temp_scores[b * num_candidates + c] = cum_scores[b] + temp_scores[b * num_candidates + c] - Z;
        }
      }
    }
     
    // top-k selection
    if (t > 0){
      idx = argtopk(temp_scores, beam_width);
    } else{
      std::iota(idx.begin(), idx.end(), 0);
    }
    
    // re-ordering
    for (size_t b = 0; b < beam_width; b++)
    {
      if (t > 0)
      {
        temp_prefixes[b] = prefixes[idx[b] / num_candidates];
      }
      temp_prefixes[b].push_back(seqs_acc[index][t][idx[b] % num_candidates]);
      cum_scores[b] = temp_scores[idx[b]];

      // make sure pointers are not mixed
      tmp_states[b] = states[b];
      states[b] = out_states[idx[b]];
      out_states[idx[b]] = tmp_states[b];
    }

    for (size_t b = 0; b < beam_width; b++)
    {
      prefixes[b] = temp_prefixes[b];
    }
  }
  // final sort rank again with the final score  
  for (size_t b = 0; b < beam_width; b++){
    temp_scores[b] = cum_scores[b] + lm_scorer->get_base_log_prob(states[b], END_TOKEN, out_states[b]);
  }

  idx = argsort(std::vector<float>(temp_scores.begin(), temp_scores.begin() + beam_width));
  
  // write the searched results back to seqs (overrides)
  for (size_t b = 0; b < beam_width; b++){
    for (size_t t = 0; t < length; t++)
    {
      seqs_acc[index][t][b] = prefixes[idx[b]][t];
    }
    cum_scores[b] = temp_scores[idx[b]];
  }
  
  return cum_scores;
}

std::vector<std::vector<float>> 
beam_search(void *scorer,
            at::Tensor probs,
            at::Tensor seqs,
            at::Tensor gates,
            at::Tensor lens,
            const int beam_width,
            const int workers,
            const int type)
{
  Scorer *lm_scorer = NULL;
  if (scorer != NULL){
    lm_scorer = static_cast<Scorer *>(scorer);
  }

  const int32_t batch_size = probs.size(0);
  std::vector<std::vector<float>> all_scores;

  if (batch_size == 1)
  {
    // single sentence testing.. no need to do multi-threads
    all_scores.push_back(_beam_search_(lm_scorer, probs, seqs, gates, lens, beam_width, 0, type));
    return all_scores;
  }

  ThreadPool pool(workers);
  std::vector<std::future<std::vector<float>>> res;
  for (size_t i = 0; i < batch_size; i++)
  {
    res.emplace_back(pool.enqueue(
        _beam_search_, lm_scorer, probs, seqs, gates, lens, beam_width, i, type));
  }

  // get decoding results
  for (size_t i = 0; i < batch_size; ++i)
  {
    all_scores.emplace_back(res[i].get());
  }

  return all_scores;
}

int _get_kenlm_scores_(Scorer *lm_scorer,
                        at::Tensor &seqs,
                        at::Tensor &lens,
                        at::Tensor &outs,
                        int index)
{
  auto outs_acc = outs.accessor<float, 2>();
  auto seqs_acc = seqs.accessor<int, 2>();
  auto lens_acc = lens.accessor<int, 1>();
  const int32_t length = (int)lens_acc[index];

  std::vector<std::string> words;
  std::vector<float> lm_scores;

  for (size_t t = 0; t < length; t++)
  {
    words.push_back(lm_scorer->get_word(seqs_acc[index][t]));
  }
  lm_scores = lm_scorer->get_sent_log_prob(words);
  for (size_t t = 0; t < lm_scores.size(); t++)
  {
    outs_acc[index][t] = lm_scores[t];
  }

  return 1;
}

void get_kenlm_scores(void *scorer,
                      at::Tensor seqs,
                      at::Tensor lens,
                      at::Tensor outs,
                      const int workers)
{
  Scorer *lm_scorer = NULL;
  if (scorer != NULL){
    lm_scorer = static_cast<Scorer *>(scorer);
  }

  const int32_t batch_size = seqs.size(0);
  ThreadPool pool(workers);

  std::vector<std::future<int>> res;
  for (size_t i = 0; i < batch_size; i++)
  {
    res.emplace_back(pool.enqueue(
        _get_kenlm_scores_, lm_scorer, seqs, lens, outs, i));
  }
  for (size_t i = 0; i < batch_size; ++i)
  {
    res[i].get();
  }
}

void *get_kenlm_scorer(const char *lm_path, const std::vector<std::string> vocab)
{
  Scorer *scorer = new Scorer(lm_path, vocab);
  return static_cast<void *>(scorer);
}

size_t get_max_order(void *scorer)
{
  Scorer *ext_scorer = static_cast<Scorer *>(scorer);
  return ext_scorer->get_max_order();
}

size_t get_dict_size(void *scorer)
{
  Scorer *ext_scorer = static_cast<Scorer *>(scorer);
  return ext_scorer->get_dict_size();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("beam_search", &beam_search, "beam_search");
  m.def("get_kenlm_scorer", &get_kenlm_scorer, "get_kenlm_scorer");
  m.def("get_kenlm_scores", &get_kenlm_scores, "get_kenlm_scores");
  m.def("get_max_order", &get_max_order, "get_max_order");
  m.def("get_dict_size", &get_dict_size, "get_dict_size");
}