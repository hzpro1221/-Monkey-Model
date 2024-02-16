import torch
import torch.nn as nn
from Layers import layer_norm

# Đầu vào là vector start, end của một Span -> Đầu ra cho ra kết quả dự đoán liệu span đó có phải Relation hay không
class Relation_Classifier(nn.Module):
	def __init__(self, last_hidden_states, d_model):
		super(Argent_Classifier, self).__init__()

		self.last_hidden_states = last_hidden_state
		self.d_model = d_model

		# Layer Normalize 
		self.layer_norm = LayerNorm1(d_model)

		# Feed forward 
		self.FeedForward1_layer_1 = nn.Linear(d_model, 1024)
		self.FeedForward1_activation_1 = nn.ReLU() 
		self.FeedForward1_layer_2 = nn.Linear(1024, 1024)
		self.FeedForward1_activation_2 = nn.ReLU()
		self.FeedForward1_layer_3 = nn.Linear(1024, 1024)
		self.FeedForward1_activation_3 = nn.ReLU()
		self.FeedForward1_layer_4 = nn.Linear(1024, d_model)

		# Layer Normalize
		self.layer_norm = LayerNorm2(d_model)

		# Feed forward
		self.FeedForward2_layer_1 = nn.Linear(d_model, 1024)
		self.FeedForward2_activation_1 = nn.ReLU() 
		self.FeedForward2_layer_2 = nn.Linear(1024, 1024)
		self.FeedForward2_activation_2 = nn.ReLU()
		self.FeedForward2_layer_3 = nn.Linear(1024, 1024)
		self.FeedForward2_activation_3 = nn.ReLU()
		self.FeedForward2_layer_4 = nn.Linear(layer_size, 1) # Point: "x >= 0.5" là Yes; "x < 0.5" là No


		self.init_weights()

	def forward(self, span_mask):

		# Tổng tất cả các vector biểu diễn các token trong span theo chiều row  
		x = torch.sum((last_hidden_state * span_mask), dim=0, keepdim=True) 

		# Layer Normalize 
		x = self.layer_norm1.forward(x)

		# Feed forward
		x = self.FeedForward1_layer_1(x)
		x = self.FeedForward1_activation_1(x)
		x = self.FeedForward1_layer_2(x)
		x = self.FeedForward1_activation_2(x)
		x = self.FeedForward1_layer_3(x)
		x = self.FeedForward1_activation_3(x)
		x = self.FeedForward1_layer_4(x)

		# Tổng tất cả các vector biểu diễn các token trong span theo chiều column
		x = torch.sum(x, dim=1, keepdim=True)

		# Layer Normalize  
		x = self.layer_norm2().forward(x)

		# Feed forward
		x = self.FeedForward2_layer_1(x)
		x = self.FeedForward2_activation_1(x)
		x = self.FeedForward2_layer_2(x)
		x = self.FeedForward2_activation_2(x)
		x = self.FeedForward2_layer_3(x)
		x = self.FeedForward2_activation_3(x)
		x = self.FeedForward2_layer_4(x)

		return x 