import torch
import pickle as p
from tools import *
import string
#前边这些为初始化一些变量，便于与预训练模型配合 
index = {}
for ch in string.ascii_letters+'-':
    index[ch]=len(index)
index['\n'] = len(index)
index['#'] = len(index)

max_length = 100
def invert_dict(d):
    return dict((v, k) for k, v in d.items())

num2word = invert_dict(index)

output_tensor = {}
for w in index:
    output_tensor.setdefault(w, get_output_tensor(w, index))
    
def sample(model,startWord='#',reverse=0):
    if reverse:
        startWord=startWord[::-1]
    if startWord!='#' and len(startWord)>1:
        input= get_lstm_input(startWord, output_tensor)[0]
    else:
        input = get_output_tensor(startWord[-1], index)
    hidden = model.initHidden()
    output_name = "";
    all_letters=[]
    if (startWord != "#"):
        output_name = startWord
        all_letters=[startWord]
    for i in range(max_length):
        output, hidden = model(input, hidden)
        topv, topi = output.data.topk(5)
        letters=[]
        for i in topi.data.tolist()[0]:
            if num2word[i]=='\n':
                letters.append('EOF')
            else:
                letters.append(num2word[i])
        all_letters.append(letters)
        w = num2word[topi.data.tolist()[0][0]]
        if (w >='A' and w <='Z'):
            output_name += w
            break
        elif w=='\n':
            break
        else:
            output_name += w
        input = get_output_tensor(w, index)
    all_letters.append(output_name)
    if reverse:
        output_name=output_name[::-1]
    return output_name,all_letters

model1 = torch.load('name-forward.pt',map_location=torch.device('cpu'))
model2 = torch.load('name-back.pt',map_location=torch.device('cpu'))

for i in range(10):
    #此行是输入开头的字符
    name1,all_letters=sample(model1,'Mo',reverse=0)
    print(name1)
    #此行是输入结尾的字符
#     name2,all_letters=sample(model2,'re',reverse=1)
#     print(name2)



#以下为可视化结果函数，支持输入开头的字符的情况
from graphviz import Digraph

u = Digraph('unix', filename='unix.gv',
            node_attr={'color': 'lightblue2', 'style': 'filled'})
u.attr(size='6,6')


u.node(all_letters[0])
for i in range(1,len(all_letters)-1):
    temp=all_letters[i][0]
    all_letters[i][0]=all_letters[i][2]
    all_letters[i][2]=temp
    for j in range(5):
        u.node(str(i)+str(j),label=all_letters[i][j])
        if i==1:
            u.edge(all_letters[0],str(i)+str(j))
        else:
            u.edge(str(i-1)+'2',str(i)+str(j))

u.node(all_letters[-1])

u.edge(str(len(all_letters)-2)+'2',all_letters[-1])


u.view()
