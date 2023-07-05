from parsinorm import General_normalization,Abbreviation, Special_numbers
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
import pandas as pd

def savetocsv(name,type_of_data=['Train','Test','Val'],onwhat=['premise','hypothesis'],Mychanging=[General_normalization().semi_space_correction,Abbreviation().replace_persian_label_abbreviation,General_normalization().alphabet_correction,Special_numbers().convert_numbers_to_text]):
  maxlen=0
  for k in type_of_data:
    datas=pd.read_csv(f'./datasets/{k}-word.csv', sep='\t')
    for j in onwhat:
      for i in Mychanging:
        datas[j] =  datas.apply(lambda x: i(sentence=x[j]),axis=1)
    tokenizer = AutoTokenizer.from_pretrained(name)
    datas['maxlen']=datas.apply(lambda x: tokenizer(x['premise'],x['hypothesis'], return_tensors='pt')['input_ids'].shape[1],axis=1)
    if(datas['maxlen'].max()>maxlen):
        maxlen=datas['maxlen'].max()
    datas=datas.drop('maxlen',axis=1)  
    

  for k in type_of_data:
    datas=pd.read_csv(f'./datasets/{k}-word.csv', sep='\t')
    for j in onwhat:
      for i in Mychanging:
        datas[j] =  datas.apply(lambda x: i(sentence=x[j]),axis=1)
    tokenizer = AutoTokenizer.from_pretrained(name)
    print(maxlen)
    datas['input_ids'] =       datas.apply(lambda x: tokenizer(x['premise'],x['hypothesis'], return_tensors='pt',max_length=maxlen,padding='max_length')['input_ids'].squeeze(0),axis=1)
    datas['token_type_ids'] =  datas.apply(lambda x: tokenizer(x['premise'],x['hypothesis'], return_tensors='pt',max_length=maxlen,padding='max_length')['token_type_ids'].squeeze(0),axis=1)
    datas['attention_mask'] =  datas.apply(lambda x: tokenizer(x['premise'],x['hypothesis'], return_tensors='pt',max_length=maxlen,padding='max_length')['attention_mask'].squeeze(0),axis=1)
    datas=datas.drop('premise',axis=1)
    datas=datas.drop('hypothesis',axis=1)
    datas.label = pd.Categorical(datas.label)
    datas.label = datas.label.cat.codes
    # print(datas['input_ids'].values[0].shape)
    
    datas.to_pickle(f'./datasets/change_{k}.pkl')


# savetocsv()



class MyDataset(Dataset):
    def __init__(self,type_of_data="Train"):
        super(MyDataset, self).__init__()
        self.datas=pd.read_pickle(f'./datasets/change_{type_of_data}.pkl')
        # print(self.datas.memory_usage(deep=True))


    def __getitem__(self, index):
        iloc=self.datas.iloc[index]
        # print(iloc)
        # print(iloc['input_ids'].shape)
        return iloc['input_ids'],iloc['token_type_ids'],iloc['attention_mask'],iloc['label']


    def __len__(self):
        return len(self.datas)

# dataset=MyDataset("Train")
# train_loader = DataLoader(dataset,batch_size=32,shuffle=True)
# kk=next(iter(train_loader))
# kk=next(iter(train_loader))
# kk
# kk.dtype()