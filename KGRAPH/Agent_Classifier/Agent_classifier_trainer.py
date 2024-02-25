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
	language_model = BERT(device=device).to(device) # d_model = 768
	model = Agent_Classifier(d_model=768).to(device)

	# Hyperparameter
	num_eps = 10
	batch_size = 2
	lr = 5e-5

	# print("Put data to dataloader")
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

			# print("Đưa token qua Mô hình ngôn ngữ -> Lấy ra last hidden state")
			for document in document_ids:
				sentence = []
				for token in document.tokens:
					sentence += token.ids
				sentences.append(sentence) 

			last_hidden_states = language_model.forward(sentences)

			# print("Xây dựng mask cho các candidate, chồng nó lên nhau")
			candidate_masks = []
			for i, document_candidate in enumerate(document_candidates):  
				candidate_mask = []
				candidate_masks.append(candidate_mask)


			# Tạo mask full 0, thay thế các vị trí có token của span -> 1
			for i, candidate_mask in enumerate(candidate_masks):
				for j, candidate in enumerate(document_candidates[i]):
					padding = [0 for _ in range(512)] # 512 is the maximum length of input
					for pos in range(candidate.start.start, candidate.end.end):
						padding[pos] = 1
					candidate_mask.append(padding)
					# print(f"sample {i}, candidate {j}: {candidate_mask}")

			# Số candidate lớn nhất trong batch
			max_num_candidate = 500 

			padding_list = [[0 for _ in range(512)]]
			# Thêm padding để kích cỡ ma trận các candidate/Xóa bớt sample để bằng max_num_candidate
			for candidate_mask in candidate_masks:
				# print(f"padding: {padding}")
				if (max_num_candidate >= len(candidate_mask)):
					candidate_mask += (max_num_candidate - len(candidate_mask)) * padding_list
				else:
					for _ in range(len(candidate_mask) - max_num_candidate):
						candidate_mask.pop()	

			# Thêm padding tăng kích cỡ/Xóa bớt sample để bằng max_num_candidate
			for label in document_labels:
				if (max_num_candidate >= len(label)):
					label += (max_num_candidate - len(label)) * [0]
				else:
					for _ in range(len(label) - max_num_candidate):
						label.pop()

			# print(f"Shape for candidates: {len(candidate_masks[0])} {len(candidate_masks[1])} {len(candidate_masks[2])} {len(candidate_masks[3])}")

			# Convert candidate masks into tensor
			candidate_masks = torch.tensor(candidate_masks).to(device)

			# print(f"candidate_masks: {candidate_masks} {candidate_masks.shape}")

			# print("Thêm chiều mới vào last_hidden_states + Repeat nó số lần bằng số candidate") 
			last_hidden_states_masks = last_hidden_states.last_hidden_state.unsqueeze(1).repeat(1, max_num_candidate, 1, 1) # batch_size * num_candidate * 512 * 768

			# print("Nhân candidate mask với last_hidden_states mask")
			span_masks = candidate_masks.view(batch_size, max_num_candidate, 512, 1).repeat(1, 1, 1, 768) * last_hidden_states_masks # d_model = 768, max_sequence_len = 512

			# print("Đưa vào mô hình dự đoán") 
			logits = model.forward(span_masks)

			document_labels = torch.tensor(document_labels).type(torch.FloatTensor).to(device)
			# print(f"document_labels: {document_labels} {document_labels.shape}")
			# print(f"logits shape: {logits.shape}")
			# Tính loss giữa dự đoán và label
			output = loss(logits.view(batch_size, 500).type(torch.FloatTensor).to(device), document_labels)


			# Backpropagation
			output.backward()

			print(f"Epoch: {epoch} \t Iter: {i} \t Loss: {output}")

			# Adjust learning weights based on current gradient
			optim.step()


	model.save_checkpoint()

	# Thời điểm hoàn tất training
	t2 = time()

	elapsed = t2 - t1
	print('Elapsed time for training %f seconds.' % elapsed)
