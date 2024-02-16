import torch
import torch.nn as nn
from KGRAPH.Layers import layer_norm

# Đầu vào là vector start, end của một Span -> Đầu ra cho ra kết quả dự đoán liệu span đó có phải Agent hay không
class Agent_Classifier(nn.Module):
	def __init__(self, d_model, layer_size=256, ckpt_dir='./checkpoint/Heads', name='KGRAPH'):
		super(Argent_Classifier, self).__init__()

		self.name = name
		self.checkpoint_dir = ckpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+".pt")

		self.d_model = d_model
		self.layer_size = layer_size

		# Layer Normalize 
		self.layer_norm = LayerNorm1(d_model)

		# Feed forward 
		self.FeedForward1_layer_1 = nn.Linear(d_model, layer_size)
		self.FeedForward1_activation_1 = nn.ReLU() 
		self.FeedForward1_layer_2 = nn.Linear(layer_size, layer_size)
		self.FeedForward1_activation_2 = nn.ReLU()
		self.FeedForward1_layer_3 = nn.Linear(layer_size, layer_size)
		self.FeedForward1_activation_3 = nn.ReLU()
		self.FeedForward1_layer_4 = nn.Linear(layer_size, d_model)

		# Layer Normalize
		self.layer_norm = LayerNorm2(d_model)

		# Feed forward
		self.FeedForward2_layer_1 = nn.Linear(d_model, layer_size)
		self.FeedForward2_activation_1 = nn.ReLU() 
		self.FeedForward2_layer_2 = nn.Linear(layer_size, layer_size)
		self.FeedForward2_activation_2 = nn.ReLU()
		self.FeedForward2_layer_3 = nn.Linear(layer_size, layer_size)
		self.FeedForward2_activation_3 = nn.ReLU()
		self.FeedForward2_layer_4 = nn.Linear(layer_size, 1) # Point: "x >= 0.5" là Yes; "x < 0.5" là No 

		# Softmax
		# self.softmax = nn.Softmax(dim=1)

		self.init_weights()

	def forward(self, span_masks):

		# Tổng tất cả các vector biểu diễn các token trong span theo chiều row  
		x = torch.sum(last_hidden_states_mask, dim=1) 

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
		x = torch.sum(x, dim=1)

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

	def save_checkpoint(self):
		print('------saving checkpoint------')
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('-------loading checkpoint------')
		self.load_state_dict(torch.load(self.checkpoint_file))