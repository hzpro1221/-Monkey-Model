import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

class BERT(nn.Module): 
	def __init__(self, ckpt_dir='./checkpoint/Language_Model', name='BERT'):
		super(BERT, self).__init__()
		self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
		self.BERT = AutoModel.from_pretrained("bert-base-uncased")

		# Max sequence length of input is 512 token
		self.max_length = 512

		self.name = name
		self.checkpoint_dir = ckpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+".pt")
		
		# Đóng băng tham số
		for param in self.BERT.parameters():
			param.requires_grad = False

	def forward(self, sentences):
		# Add [CLS] token, [SEP] token and padding
		for sentence in sentences:
			sentence.insert(0, 101) # Add [CLS] token
			sentence.insert(len(sentence), 102) # Add [SEP] token
			sentence += [0] * (self.max_length - len(sentence)) # Add padding

		# Create 'input_ids'
		input_ids = torch.tensor(sentences)

		# Create 'token_type_ids'
		token_type_ids_sample = [0 for _ in range(self.max_length)]
		token_type_ids = torch.tensor([token_type_ids_sample for _ in range(len(sentences))])

		# Create 'attention_mask'
		attention_mask = []
		for sentence in sentences:
			attention_mask_sample = [int(i != 0) for i in sentence]
			attention_mask.append(attention_mask_sample)
		attention_mask = torch.tensor(attention_mask)

		encoded_input = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
			
		output = self.BERT(**encoded_input)
		return output

	def save_checkpoint(self):
		print('------saving checkpoint------')
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('-------loading checkpoint------')
		self.load_state_dict(torch.load(self.checkpoint_file))
