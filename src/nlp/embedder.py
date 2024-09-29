import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch


# Define a simple dataset with triplets (anchor, positive, negative text)
class TripletTextDataset(Dataset):
    def __init__(self, anchor_texts, positive_texts, negative_texts, tokenizer, max_len=32):
        self.anchor_texts = anchor_texts
        self.positive_texts = positive_texts
        self.negative_texts = negative_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor_texts)

    def __getitem__(self, idx):
        anchor_text = self.anchor_texts[idx]
        positive_text = self.positive_texts[idx]
        negative_text = self.negative_texts[idx]

        anchor = self.tokenizer(anchor_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        positive = self.tokenizer(positive_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        negative = self.tokenizer(negative_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")

        return anchor, positive, negative
    

# Pretrained BERT model for embeddings
class BertEmbeddingNet(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(BertEmbeddingNet, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token's output as the sentence embedding
        return output.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]


# Triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)
    

class EmbedderPipeline():
    
    def __init__(self) -> None:
        self.anchor_texts = ["king", "queen", "paris"]
        self.positive_texts = ["monarch", "monarch", "france"]
        self.negative_texts = ["dog", "cat", "germany"]
        
    def _opt_setup(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = TripletTextDataset(self.anchor_texts, self.positive_texts, self.negative_texts, self.tokenizer, max_len=8)
        self.dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        self.embedding_net = BertEmbeddingNet(pretrained_model_name='bert-base-uncased')
        self.triplet_loss_fn = TripletLoss(margin=1.0)
        self.optimizer = optim.Adam(self.embedding_net.parameters(), lr=1e-5)
        
    def _train(self, num_epochs: int = 5) -> None:
        for epoch in range(num_epochs):
            self.embedding_net.train()
            running_loss = 0.0
            for anchors, positives, negatives in self.dataloader:
                self.optimizer.zero_grad()

                # Process anchors
                anchor_input_ids = anchors['input_ids'].squeeze(1)  # Remove extra batch dimension
                anchor_attention_mask = anchors['attention_mask'].squeeze(1)
                anchor_embeddings = self.embedding_net(anchor_input_ids, anchor_attention_mask)

                # Process positives
                positive_input_ids = positives['input_ids'].squeeze(1)
                positive_attention_mask = positives['attention_mask'].squeeze(1)
                positive_embeddings = self.embedding_net(positive_input_ids, positive_attention_mask)

                # Process negatives
                negative_input_ids = negatives['input_ids'].squeeze(1)
                negative_attention_mask = negatives['attention_mask'].squeeze(1)
                negative_embeddings = self.embedding_net(negative_input_ids, negative_attention_mask)

                # Compute the triplet loss
                loss = self.triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Update loss
                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.dataloader):.4f}")
            
    def _get_embedding(self, text: str, max_len: int = 32):
        self.embedding_net.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            embedding = self.embedding_net(input_ids, attention_mask)
            return embedding.cpu().numpy()  # convert embedding tensor to a numpy array