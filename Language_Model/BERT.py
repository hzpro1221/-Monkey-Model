import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

class BERT(nn.Module):
	def __init__(self, ckpt_dir='./checkpoint/Language_Model', name='BERT'):
		super(BERT, self).__init__()
		self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
		self.BERT = AutoModel.from_pretrained("bert-base-uncased")

		self.name = name
		self.checkpoint_dir = ckpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+".pt")
		
		# Đóng băng tham số
		for param in self.BERT.parameters():
			param.requires_grad = False

	def forward(text):
		encoded_input = self.tokenizer(text, return_tensors="pt")
		output = model(**encoded_input)
		return output

	def save_checkpoint(self):
		print('------saving checkpoint------')
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('-------loading checkpoint------')
		self.load_state_dict(torch.load(self.checkpoint_file))
