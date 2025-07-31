import torch
import torch.nn as nn
class SentimentTransformer(nn.Module):
    def __init__(self, sentiment_feat_dim, transformer_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(sentiment_feat_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        # x: (batch, seq=4, features)
        x = self.input_proj(x)  # Converts input features to the Transformer's size
        x = x.permute(1, 0, 2) # Rearranges dimensions for Transformer [seq, batch, features]
        x = self.transformer(x)
        x = x.permute(1, 2, 0) # Change back to [batch, features, seq]
        x = self.pool(x).squeeze(-1) # Pools across weeks; output: [batch, features]
        return x
        
class PriceLSTM(nn.Module):
    def __init__(self, price_feat_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(price_feat_dim, hidden_dim, batch_first=True, bidirectional=True)
    def forward(self, x):
        # x: (batch, seq=4, features)
        _, (h, _) = self.lstm(x) 
        h = torch.cat([h[0], h[1]], dim=-1) # Combines results from both directions (forward/backward)
        return h

class ImpactScoreModel(nn.Module):
    def __init__(self, sentiment_feat_dim, price_feat_dim):
        super().__init__()
        self.sent_encoder = SentimentTransformer(sentiment_feat_dim)
        self.price_encoder = PriceLSTM(price_feat_dim)
        self.fusion = nn.Linear(64 + 64, 128)
        self.output = nn.Linear(128, 1)
    def forward(self, sent_x, price_x):
        sent_out = self.sent_encoder(sent_x)      # Encode the sentiment block
        price_out = self.price_encoder(price_x)   # Encode the price block
        x = torch.cat([sent_out, price_out], dim=1) # Concatenate both summaries
        x = torch.relu(self.fusion(x))            # Nonlinear activation for learning complex combinations
        impact_score = self.output(x)             # Linear output; predicts impact score
        return impact_score.squeeze(-1)           # Remove unnecessary dimensions

sentiment_feat_dim = 5 # according to our data set, this matches the sample data
price_feat_dim    = 4 # according to our data set, this matches the sample data


model = ImpactScoreModel(sentiment_feat_dim=sentiment_feat_dim, price_feat_dim=price_feat_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.HuberLoss()

num_epochs = 10  # Pick as we wish

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for sent_x, price_x, target_y in train_loader:
        optimizer.zero_grad()                   # Clear gradients
        preds = model(sent_x, price_x)          # 4-week feature blocks go in here!
        loss = loss_fn(preds, target_y)         # Compare predictions and target
        loss.backward()                         # Compute error gradients
        optimizer.step()                        # Update model weights
        epoch_loss += loss.item() * len(sent_x)
    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
