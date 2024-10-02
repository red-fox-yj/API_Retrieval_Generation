import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
# nvidia/NV-Embed-v2
class ModelHandlerV1:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name_or_path=model_name, trust_remote_code=True)
        model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
        print(f"Model {model_name} loaded.")
        return model.to('cuda')

    def get_batch_embeddings(self, texts, batch_size=32, prefix=None):
        all_embeddings = []
        if prefix:
            texts = [prefix + text for text in texts]
        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(batch_texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
            all_embeddings.append(embeddings.cpu().numpy())
        print("Embeddings generated.")
        return np.vstack(all_embeddings)

# BAAI/bge-en-icl
class ModelHandlerV2:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval().to('cuda')

    def get_batch_embeddings(self, texts, batch_size=32, prefix=None):
        all_embeddings = []
        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            batch_tokenized = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_tokenized = {k: v.to('cuda') for k, v in batch_tokenized.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_tokenized)
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_tokenized['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        
        print("Embeddings generated.")
        return np.vstack(all_embeddings)

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# dunzhang/stella_en_1.5B_v5
class ModelHandlerV3:
    def __init__(self, model_name, query_prompt_name="s2p_query"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
        self.query_prompt_name = query_prompt_name

    def get_batch_embeddings(self, texts, batch_size=32, prefix=None):
        all_embeddings = []
        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size} using prompt: {self.query_prompt_name}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(batch_texts, prompt_name=self.query_prompt_name)
            all_embeddings.append(embeddings)
        print("Embeddings generated.")
        return np.vstack(all_embeddings)
