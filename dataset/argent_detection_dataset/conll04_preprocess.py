import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from KGRAPH.entities import Token, Conll04Dataset, Span_candidate, Document
import KGRAPH.entities 
import json

def conll04_processe(batch_size):
	# Tải xuống Dataset
	# !wget https://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04/conll04_train.json
	# !wget https://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04/conll04_dev.json
	# !wget https://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04/conll04_test.json

	# Mở dataset
	# Điền đường dẫn thư mục chứa dataset
	with open("/content/conll04_train.json") as f: 
		train_data = json.load(f)
	# with open("/content/conll04_dev.json") as f:
	# dev_data = json.load(f)
	# with open("/content/conll04_test.json") as f:
	# 	test_data = json.load(f)

	# Dataset sẽ có 3 trường:
	#	- Start và End của từng token sau khi qua Tokenizer, Ids của text qua token
	#	- Agent mask: Start và End
	#	- Label mask: [0, 0, 0, 1, 1, 0, 0, ...]

	# Load tokenizer
	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # BERT

	# Lấy ra Ids của text qua tokenizer, start và end của token
	dataset_ids = []
	for document in train_data:
		start = 0
		end = 0
		document_tokens = []
		tokenized = tokenizer(document["tokens"], add_special_tokens=False)
		for i, token in enumerate(tokenized["input_ids"]):
			end = end + len(token)
			temp = Token(start=start, end=end, ids=token)
			start = start + len(token)
			document_tokens.append(temp)
		document_length = document_tokens[-1].end
		document_processed = Document(document_tokens, document_length)
		dataset_ids.append(document_processed)

	# Build candidate_mask and label_mask
	# Max len = 5
	dataset_candidates = []
	dataset_labels = []
	max_len = 5
	# Generate candidate_mask
	for i, document in enumerate(train_data):
		document_candidates = []
		label = []
		document_length = dataset_ids[i].length
		for length in range(1, max_len+1):
			for start in range(0, len(document["tokens"]) - length + 1):
				add = 0
				candidate_mask = Span_candidate(start, start + length)
				# print(f'{start} {start + length} {document["entities"][0]["start"]} {document["entities"][0]["end"]}')
				# Check điều kiện để add vào label mask
				# Nếu candidate trùng với label trong dataset -> 1
				# Nếu candidate không trùng với label trong dataset -> 0
				for enity in document["entities"]:
					if (start ==  enity["start"]) and ((start + length) ==  enity["end"]):
						label.append(1)
						add += 1
				if (add == 0):
					label.append(0)
				# print(candidate_mask)
				document_candidates.append(candidate_mask)
		dataset_candidates.append(document_candidates)
		dataset_labels.append(label)
	
	train_data = Conll04Dataset(dataset_ids=dataset_ids, dataset_candidates=dataset_candidates, dataset_labels=dataset_labels)

	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  	# dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)
  	# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	return train_dataloader