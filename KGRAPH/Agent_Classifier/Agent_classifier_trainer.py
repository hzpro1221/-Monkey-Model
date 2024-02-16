# Set sys.path for python to look for package and module on Google Colab
import sys
sys.path.append('/content/MonkeyModel')

import torch
import torch.nn as nn
from torch.optim import AdamW
from KGRAPH.Agent_Classifier.Agent_classifier_model import Agent_Classifier
from dataset.argent_detection_dataset.conll04_preprocess import conll04_preprocess
from Language_Model.BERT import BERT
from time import time

if __name__ == '__main__':

	# Thời điểm bắt đầu training
	t1 = time()

	# Device 
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Định nghĩa các mô hình
	language_model = BERT().to(device) # d_model = 768
	model = Agent_Classifier(d_model=768).to(device)

	# Hyperparameter
	num_eps = 10
	batch_size = 32
	lr = 5e-5

	# Put data to dataloader
	train_dataloader = conll04_preprocess(batch_size=4)

	# Optimizer
	optim = AdamW(model.parameters(), lr=lr)

	# Loss
	loss = nn.CrossEntropyLoss()

	# trainer
	for epoch in range(num_eps):
		for i, batch in enumerate(train_dataloader):
			optim.zero_grad()
			document_ids = batch["dataset_ids"]
			document_candidates = batch["dataset_candidates"]
			document_labels = batch["dataset_labels"]
			sentence = []

			# Đưa token qua Mô hình ngôn ngữ -> Lấy ra last hidden state 
			for token in document_ids:
				sentence += token.ids
			last_hidden_states = language_model.forward(sentence)

			# Xây dựng mask cho các candidate, chồng nó lên nhau
			# Tạo mask full 0
			candidate_mask = torch.zeros(len(document_candidates),document_ids.length)

			# Thay thế các vị trí có token của span -> 1
			for i, candidate in enumerate(document_candidates):
				for pos in range(cadidate.start, candidate.end):
					candidate_mask[i][pos] = 1

			# Thêm chiều mới vào last_hidden_states + Repeat nó số lần bằng số candidate 
			last_hidden_states_mask = last_hidden_state.unsqueeze(0).repeat(len(document_candidates), 1, 1)

			# Nhân candidate mask với last_hidden_states mask
			span_masks = candidate_mask.view(len(document_candidates),document_ids.length, 1).repeat(1, 1, 768) * last_hidden_states_mask # d_model = 768

			# Đưa vào mô hình dự đoán 
			logits = model.forward(spans_masks)

			# Tính loss giữa dự đoán và label
			output = loss(logits, document_labels).to(device)

			# Backpropagation
			output.backward()

			if (i % 4 == 0):
				print(f"Epoch: {epoch} \t Iter: {i} \t Loss: {output}")

			# Adjust learning weights based on current gradient
			optim.step()


	model.save_checkpoint()

	# Thời điểm hoàn tất training
	t2 = time()

	elapsed = t2 - t1
	print('Elapsed time for training %f seconds.' % elapsed)
