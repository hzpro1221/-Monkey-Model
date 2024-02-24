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
	batch_size = 4
	lr = 5e-5

	# Put data to dataloader
	train_dataloader = conll04_preprocess(batch_size=batch_size)

	# Optimizer
	optim = AdamW(model.parameters(), lr=lr)

	# Loss
	loss = nn.CrossEntropyLoss()

	# trainer
	for epoch in range(num_eps):
		for i, batch in enumerate(train_dataloader):
			optim.zero_grad()
			# print(batch)
			# print(len(batch))
			document_ids = []
			document_candidates = []
			document_labels = []

			for sample in batch:
				document_ids.append(sample["dataset_ids"])
				document_candidates.append(sample["dataset_candidates"])
				document_labels.append(sample["dataset_labels"])

			sentences = [] 

			# Đưa token qua Mô hình ngôn ngữ -> Lấy ra last hidden state
			for document in document_ids:
				sentence = []
				for token in document.tokens:
					sentence += token.ids
				sentences.append(sentence) 

			last_hidden_states = language_model.forward(sentences)

			# Xây dựng mask cho các candidate, chồng nó lên nhau
			padding = [[0 for _ in range(512)]] # 512 is the maximum length of input
			# Tạo mask full 0
			candidate_masks = []
			for i, document_candidate in enumerate(document_candidates):  
				candidate_mask = []
				candidate_mask += len(document_candidate) * padding
				candidate_masks.append(candidate_mask)

			# Thay thế các vị trí có token của span -> 1
			for i, candidate_mask in enumerate(candidate_masks):
				for j, candidate in enumerate(document_candidates[i]):
					for pos in range(candidate.start.start, candidate.end.end):
						candidate_mask[j][pos] = 1

			# Tìm sample có số candidate lớn nhất trong batch
			max_num_candidate = 0
			for candidate_mask in candidate_masks:
				if len(candidate_mask) > max_num_candidate:
					max_num_candidate = len(candidate_mask)

			# Thêm padding để kích cỡ ma trận các candidate của các sample bằng nhau
			for candidate_mask in candidate_masks:
				candidate_mask += (max_num_candidate - len(candidate_mask)) * padding

			# print(f"Shape for candidates: {len(candidate_masks[0])} {len(candidate_masks[1])} {len(candidate_masks[2])} {len(candidate_masks[3])}")

			# Convert candidate masks into tensor
			candidate_masks = torch.tensor(candidate_masks)

			# print(f"Shape: {candidate_masks.shape}")

			# Thêm chiều mới vào last_hidden_states + Repeat nó số lần bằng số candidate 
			last_hidden_states_masks = last_hidden_states.unsqueeze(1).repeat(1, max_num_candidate, 1, 1) # batch_size * num_candidate * 512 * 768

			# Nhân candidate mask với last_hidden_states mask
			span_masks = candidate_mask.view(batch_size, max_num_candidate, 512, 1).repeat(1, 1, 1, 768) * last_hidden_states_mask # d_model = 768, max_sequence_len = 512

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
