# Trong Graph sẽ chỉ có 2 kiểu Node đó là Relation và Argent 
class Relation:
	def __init__(self, start: int, end: int):
		self.start = start
		self.end = end

class Agent:
	def __init__(self, start: int, end: int):
		self.start = start
		self.end = end


# Do sự không đồng nhất giữa cách chia token trong dataset và cách chia token trong tokenizer 
class Token:
	def __init__(self, start: int, end: int, ids: list, token: str):
		self.start = start
		self.end = end
		self.ids = ids

# Kiểu dành cho Document
class Document:
  def __init__(self, tokens, length):
    self.tokens = tokens
    self.length = length



# Kiểu dành cho Candidate 
class Span_candidate:
  def _init__(self, start, end):
    self.start = start
    self.end = end


# Kiểu Dataset cho Conll04
class Conll04Dataset(Dataset):
	def __init__(self, dataset_ids, dataset_candidates, dataset_labels):
		self.dataset_ids = dataset_ids
		self.dataset_candidates = dataset_candidates
		self.dataset_labels = dataset_labels

	def __len__(self):
		return len(self.dataset_ids)

	def __getitem__(self, idx):
		return self.dataset_ids[idx], self.dataset_candidates[idx], self.dataset_labels[idx]