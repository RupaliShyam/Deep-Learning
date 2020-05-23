import collections
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import IMDBReviewDataset, imdb_collate_fn
from sentiment_analysis import SentimentClassification

PADDING_TOKEN = 0
CKPT_VOCABULARY_SIZE = 82
CKPT_EMBEDDING_DIM = 256
CKPT_HIDDEN_SIZE = 128


class VisualizeInternalGates(nn.Module):

  def __init__(self):
    super().__init__()
    vocabulary_size = CKPT_VOCABULARY_SIZE
    embedding_dim = CKPT_EMBEDDING_DIM
    hidden_size = CKPT_HIDDEN_SIZE

    self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=PADDING_TOKEN)
    self.rnn_model = VisualizeGRUCell(input_size=embedding_dim,
                                      hidden_size=hidden_size)
    self.classifier = nn.Linear(hidden_size, vocabulary_size)
    return

  def forward(self, batch_reviews):
    data = self.embedding(batch_reviews)

    state = None
    batch_size, total_steps, _ = data.shape
    internals = []
    for step in range(total_steps):
      next_h, gate_signals = self.rnn_model(data[:, step, :], state)
      internals.append(gate_signals)
      state = next_h

    logits = self.classifier(state)

    internals = list(zip(*internals))
    outputs = {
        'update_signals': internals[0],
        'reset_signals': internals[1],
        'cell_state_candidates': internals[2],
    }
    return logits, outputs

class VisualizeIMDBGates(nn.Module):

  def __init__(self, vocabulary_size, embedding_dim, hidden_size, bias):
    super().__init__()

    self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=PADDING_TOKEN)
    self.rnn_model = VisualizeGRUCell(input_size=embedding_dim,
                                      hidden_size=hidden_size)
    self.classifier = nn.Linear(hidden_size, 2)
    return

  def forward(self, batch_reviews, batch_lengths):
    data = self.embedding(batch_reviews)
    state = None
    batch_size, total_steps, _ = data.shape
    internals = []
    full_outputs = []
    for step in range(total_steps):
      next_state, gate_signals = self.rnn_model(data[:, step, :], state)
      if isinstance(next_state, tuple):
        h, c = next_state
        full_outputs.append(h)
      else:
        full_outputs.append(next_state)
      internals.append(gate_signals)
      state = next_state

    full_outputs = torch.stack(full_outputs, dim=1)
    outputs = full_outputs[torch.arange(batch_size), batch_lengths - 1, :]
    logits = self.classifier(outputs)

    internals = list(zip(*internals))
    outputs = {
        #'update_signals': internals[0],  # this is used for coupled LSTM and the index order needs to be changed
        'forget_signals': internals[0],
        'input_signals': internals[1],
        'output_signals': internals[2],
        'cell_state_candidates': internals[3],
    }
    return logits, outputs


class VisualizeGRUCell(nn.Module):

  def __init__(self, input_size, hidden_size):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    # Uncomment for GRU gates visualization
    # self.W_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.W_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))

    # Uncomment for LSTM gates visualization
    # self.W_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.W_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.W_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    # self.b_i = nn.Parameter(torch.Tensor(hidden_size))
    # self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    # self.b_c = nn.Parameter(torch.Tensor(hidden_size))

    # Uncomment for peephole LSTM gates visualization
    self.W_f = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size + input_size))
    self.W_i = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size + input_size))
    self.W_o = nn.Parameter(torch.Tensor(hidden_size, 2 * hidden_size + input_size))
    self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    self.b_i = nn.Parameter(torch.Tensor(hidden_size))
    self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    self.b_c = nn.Parameter(torch.Tensor(hidden_size))

    #Uncomment for coupled LSTM visualization
    # self.W_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.W_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    # self.b_f = nn.Parameter(torch.Tensor(hidden_size))
    # self.b_o = nn.Parameter(torch.Tensor(hidden_size))
    # self.b_c = nn.Parameter(torch.Tensor(hidden_size))

    self.reset_parameters()

  def forward(self, x, prev_state):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
      prev_c = torch.zeros((batch, self.hidden_size), device=x.device) # Uncomment for LSTM variants
    else:
      #prev_h = prev_state # this is used when there is only one previous state
      # Comment the previous line and Uncomment the following lines for LSTM variants
      prev_h = prev_state[0]
      prev_c = prev_state[1]

    # GRU
    # concat_hx = torch.cat((prev_h, x), dim=1)
    # z = torch.sigmoid(F.linear(concat_hx, self.W_z))
    # r = torch.sigmoid(F.linear(concat_hx, self.W_r))
    # h_tilde = torch.tanh(F.linear(torch.cat((r * prev_h, x), dim=1), self.W))
    # next_h = (1 - z) * prev_h + z * h_tilde
    # return next_h, (z, r, h_tilde)

    #lstm
    # concat_hx = torch.cat((prev_h, x), dim=1)
    # f = torch.sigmoid(F.linear(concat_hx, self.W_f, self.b_f))
    # i = torch.sigmoid(F.linear(concat_hx, self.W_i, self.b_i))
    # o = torch.sigmoid(F.linear(concat_hx, self.W_o, self.b_o))
    # c_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    # next_c = f * prev_c + i * c_tilde
    # next_h = o * torch.tanh(next_c)
    # return (next_h, next_c), (f, i, o, c_tilde)

    #peepholed
    concat_hx = torch.cat((prev_h, x), dim=1)
    concat_chx = torch.cat((prev_c, prev_h, x), dim=1)
    f = torch.sigmoid(F.linear(concat_chx, self.W_f, self.b_f))
    i = torch.sigmoid(F.linear(concat_chx, self.W_i, self.b_i))
    o = torch.sigmoid(F.linear(concat_chx, self.W_o, self.b_o))
    c_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    next_c = f * prev_c + i * c_tilde
    next_h = o * torch.tanh(next_c)
    return (next_h, next_c), (f, i, o, c_tilde)

    #coupled
    # concat_hx = torch.cat((prev_h, x), dim=1)
    # f = torch.sigmoid(F.linear(concat_hx, self.W_f, self.b_f))
    # o = torch.sigmoid(F.linear(concat_hx, self.W_o, self.b_o))
    # c_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
    # next_c = f * prev_c + (1 - f) * c_tilde
    # next_h = o * torch.tanh(next_c)
    # return (next_h, next_c) , (f, o, c_tilde)

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}'.format(self.input_size,
                                                  self.hidden_size)


class VisualizeWarAndPeaceDataset(Dataset):

  def __init__(self, vocabulary):
    self.vocabulary = vocabulary

    # Hardcode the parameters to match the provided checkpoint
    txt_path = 'data/war_and_peace_visualize.txt'

    with open(txt_path, 'rb') as fp:
      raw_text = fp.read().strip().decode(encoding='utf-8')

    self.data = raw_text.split('\n')

    self.char2index = {x: i for (i, x) in enumerate(self.vocabulary)}
    self.index2char = {i: x for (i, x) in enumerate(self.vocabulary)}

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return np.array([self.char2index[x] for x in self.data[index]]), -1

  def convert_to_chars(self, sequence):
    if isinstance(sequence, torch.Tensor):
      sequence = sequence.squeeze(0).detach().numpy().tolist()
    return [self.index2char[x] for x in sequence]


def visualize_internals(sequence_id,
                        sequence,
                        gate_name,
                        states,
                        saving_dir='visualize/'):
  states = torch.cat(states, dim=0).cpu().detach().numpy().T
  hidden_size, time_stamps = states.shape
  fig, ax = plt.subplots(figsize=(time_stamps / 5, hidden_size / 5))

  if gate_name in ['update_signals', 'reset_signals', 'forget_signals', 'input_signals', 'output_signals']:
    vmin = 0
  elif gate_name == 'cell_state_candidates':
    vmin = -1
  else:
    raise ValueError

  sns.heatmap(states,
              cbar=False,
              square=True,
              linewidth=0.05,
              xticklabels=sequence,
              yticklabels=False,
              vmin=vmin,
              vmax=1,
              cmap='bwr',
              ax=ax)

  plt.xlabel('Sequence')
  plt.ylabel('Hidden Cells')

  ax.xaxis.set_ticks_position('top')
  ax.xaxis.set_ticklabels(sequence, rotation=90)

  plt.tight_layout()
  os.makedirs(saving_dir, exist_ok=True)
  plt.savefig(
      os.path.join(saving_dir,
                   'peep2_S%02d_' % sequence_id + gate_name.lower() + '.png'))
  return


def war_and_peace_visualizer():
  #####################################################################
  # Implement here following the given signature                      #
  model = VisualizeInternalGates()
  state_dict = torch.load('data/war_and_peace_model_checkpoint.pt')
  vocabulary = state_dict['vocabulary']
  model.load_state_dict(state_dict['model'])
  model.eval()

  dataset = VisualizeWarAndPeaceDataset(vocabulary=vocabulary)
  data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=8)

  progress_bar = tqdm(enumerate(data_loader))
  for step, (sequences, _) in progress_bar:

    _, outputs = model(sequences)

    for gate_name, states in outputs.items():
      print('gate name: ', gate_name)
      char_sequence = dataset.convert_to_chars(sequences)
      visualize_internals(step, char_sequence, gate_name, states, saving_dir='visualize/')
  #####################################################################

  return


def IMDB_visualizer():
  #####################################################################

  state_dict = torch.load('data/best_model_peephole.pt')
  vocabulary = state_dict['vocabulary']

  dataset = IMDBReviewDataset(csv_path='data/imdn_visualize.csv', vocabulary=vocabulary)
  data_loader = DataLoader(dataset,
                           batch_size=1,
                           shuffle=True,
                           num_workers=8,
                           collate_fn=imdb_collate_fn)

  model = VisualizeIMDBGates(vocabulary_size=len(vocabulary),
                                  embedding_dim=128,
                                  hidden_size=100,
                                  bias=True)

  model.load_state_dict(state_dict['model'])
  model.eval()


  for step, (reviews, lengths, labels) in tqdm(enumerate(data_loader)):

    _, outputs = model(reviews, lengths)

    for gate_name, states in outputs.items():
      print('gate name: ', gate_name)
      idx2wrd = dataset.index2word
      if isinstance(reviews, torch.Tensor):
        reviews = reviews.squeeze(0).detach().numpy().tolist()
      word_sequence = [idx2wrd[x] for x in reviews]
      visualize_internals(step, word_sequence, gate_name, states, saving_dir='visualize/')
  #####################################################################

  return


def main(unused_argvs):
  #war_and_peace_visualizer()
  IMDB_visualizer() # For LSTM variants visualized on IMDB text


if __name__ == '__main__':
  app.run(main)
