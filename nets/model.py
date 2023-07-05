
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

class BERTClassification(nn.Module):
    def __init__ (self,name,outputlayer=-1,pooler=False,output_attentions=False):
        super(BERTClassification, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.bert = BertModel.from_pretrained(name,output_hidden_states=True, add_pooling_layer=pooler,output_attentions=output_attentions)
        # for param in self.bert.parameters():
        #   param.requires_grad = False
        self.bat1 = nn.BatchNorm1d(768)
        self.rel1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.4)
        self.out1 = nn.Linear(768, 1024)
        self.bat2 = nn.BatchNorm1d(1024)
        self.rel2 = nn.ReLU()
        self.out2 = nn.Linear(1024, 3)
        self.rel3 = nn.ReLU()
        self.outputlayer=outputlayer
        self.pooler=pooler

    def forward(self,input_ids,token_type_ids,attention_mask):
        # print(input_ids.shape)
        if(not self.pooler):
            output = self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)['hidden_states']
            output=output[self.outputlayer]
        else:
            output = self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)['pooler_output']
        # print(output.shape)
        # print(len(output))
        # [:,-1]
        # print(output.shape)
        if(not self.pooler):
            output = self.bat1(output[:,0])
        else:
            output = self.bat1(output)
        # print(output.shape)

        output = self.rel1(output)
        output = self.drop1(output)
        output = self.out1(output)
        output = self.bat2(output)
        output = self.rel2(output)
        output = self.out2(output)
        output = self.rel3(output)


        return output
    
# print(BERTClassification())
    