import torch
import torch.autograd as autograd


def get_input_tensor(word, index):
    ten = torch.zeros(1, 1, len(word_to_ix))
    ten[0][0][index[word]] = 1
    return autograd.Variable(ten)


def get_output_tensor(word, index):
    ten = autograd.Variable(torch.LongTensor([index[word]]))
    return ten


def prepare_sequence(seq, word_to_ix):
    idxs = [word_to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def get_lstm_input(s, output_tensor):
    tmpIn = []
    tmpOut = []
    for i in range(1, len(s)):
        w = s[i]
        w_b = s[i - 1]
        tmpIn.append(output_tensor[w_b])
        tmpOut.append(output_tensor[w])
    return torch.cat(tmpIn), torch.cat(tmpOut)