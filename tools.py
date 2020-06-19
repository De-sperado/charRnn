import torch
import torch.autograd as autograd

#一些辅助函数，如根据单词构造tensor
#构造训练用的数据对
def get_lstm_input(s, output_tensor):
    train_in = []
    train_out = []
    for i in range(1, len(s)):
        w = s[i]
        w_b = s[i - 1]
        train_in.append(output_tensor[w_b])
        train_out.append(output_tensor[w])
    return torch.cat(train_in), torch.cat(train_out)
#构造输入tensor
def get_input_tensor(word, index):
    ten = torch.zeros(1, 1, len(word_to_ix))
    ten[0][0][index[word]] = 1
    return autograd.Variable(ten)
#构造输出tensor
def get_output_tensor(word, index):
    return autograd.Variable(torch.LongTensor([index[word]]))
#构造测试用的输入tensor
def get_input(word, index):
    train_in = []
    for i in word:
        train_in.append(autograd.Variable(torch.LongTensor([index[i]])))
    return torch.cat(train_in)

