import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoInt(nn.Module):
    """
    A PyTorch implementation of AutoInt.
    
    Reference:
        AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    """
    def __init__(self, feature_sizes, embedding_size=16, embedding_dropout=0.0,
                 att_layer_num=3, att_head_num=2, att_res=True, att_dropout=0.0, 
                 dnn_hidden_units=[32, 32], dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, 
                 dnn_dropout=0, init_std=0.0001, seed=1024, 
                 use_cuda=True, device='cpu', seq_vocab_size=None):
        super(AutoInt, self).__init__()
        self.feature_sizes = feature_sizes # 각 feature의 길이
        self.field_size = len(feature_sizes) # 한 data에서 feature의 종류 개수
        self.embedding_size = embedding_size
        self.embedding_dropout = embedding_dropout
        self.att_layer_num = att_layer_num
        self.att_head_num = att_head_num
        self.att_res = att_res
        self.att_dropout = att_dropout
        self.dnn_hidden_units = dnn_hidden_units # for AutoInt+ : set to None if no DNN is desired
        self.dnn_activation = dnn_activation
        self.l2_reg_dnn = l2_reg_dnn
        self.l2_reg_embedding = l2_reg_embedding
        self.dnn_use_bn = dnn_use_bn
        self.dnn_dropout = dnn_dropout
        self.init_std = init_std
        self.seed = seed
        self.device = device
        
        # Embeddings : 모든 feature_size에 대해 feature_size -> embedding_size 차원으로 변환
        self.embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        
        # Multi-head Self-Attention Layers
        self.att_layers = nn.ModuleList()
        for _ in range(self.att_layer_num):
            self.att_layers.append(
                nn.MultiheadAttention(embed_dim=self.embedding_size, 
                                      num_heads=self.att_head_num, 
                                      dropout=self.att_dropout,
                                      batch_first=True)
            )
        
        # DNN Part (optional - only create if dnn_hidden_units is not None)
        if self.dnn_hidden_units is not None:
            self.dnn_layers = nn.ModuleList()
            input_dim = self.field_size * self.embedding_size # feature 종류만큼 embedding size로 변환되고, 그게 모두 들어오니까...
            for hidden_unit in self.dnn_hidden_units:
                self.dnn_layers.append(nn.Linear(input_dim, hidden_unit)) # 주어진 hidden unit 개수, 차원만큼 hidden layer 추가
                if self.dnn_use_bn:
                    self.dnn_layers.append(nn.BatchNorm1d(hidden_unit))
                self.dnn_layers.append(nn.ReLU()) # Assuming ReLU for now
                if self.dnn_dropout > 0:
                    self.dnn_layers.append(nn.Dropout(self.dnn_dropout))
                input_dim = hidden_unit # hidden unit 차원으로 input dim 변경
                
            self.dnn_output_layer = nn.Linear(input_dim, 1)
        
        if self.att_res is True:
            self.res_layer =  nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.ReLU()
            )

        # Layer Normalization (applied after each attention block)
        self.layer_norm = nn.LayerNorm(self.embedding_size)

        # Final Output Layer (combining Attention output and DNN output)
        # Attention output dimension: field_size * embedding_size
        self.att_output_dim = self.field_size * self.embedding_size 
        self.final_linear = nn.Linear(self.att_output_dim, 1)

        """
            init GRU part (if seq_vocab_size provided)
        """
        self.seq_vocab_size = seq_vocab_size
        if self.seq_vocab_size is not None:
            self.gru_embedding = nn.Embedding(self.seq_vocab_size, self.embedding_size)
            self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.embedding_size, batch_first=True)
            self.gru_linear = nn.Linear(self.embedding_size, 1)

        self.to(self.device)

    def forward(self, Xi, Xv, seq=None):
        """
        Xi: (batch_size, field_size, 1) - Indices
        Xv: (batch_size, field_size, 1) - Values
        """
        # 1. Embedding Layer
        # Xi shape: [batch_size, field_size, 1] -> [batch_size, field_size]
        Xi = Xi.squeeze(-1)
        
        # Look up embeddings
        # embeddings list of [batch_size, embedding_size]
        emb_list = [emb(Xi[:, i]) for i, emb in enumerate(self.embeddings)]
        
        # Stack to get [batch_size, field_size, embedding_size]
        embeddings = torch.stack(emb_list, dim=1)
        
        # Multiply by values (if Xv is not all 1s)
        # Xv shape: [batch_size, field_size] -> [batch_size, field_size, 1]
        if Xv.dim() == 2:
            Xv = Xv.unsqueeze(-1)
        embeddings = embeddings * Xv
        
        # added : apply dropout as the rate previously set to embeddings_dropout
        embeddings = F.dropout(embeddings, p=self.embedding_dropout)

        # 2. Multi-head Self-Attention
        att_input = embeddings
        for att_layer in self.att_layers:
            # MultiheadAttention expects (batch, seq, feature) if batch_first=True
            # query, key, value are the same for self-attention
            att_output, _ = att_layer(att_input, att_input, att_input)
            
            if self.att_res:
                res = self.res_layer(att_input)
                att_output += res
            
            att_output = F.relu(att_output)
            att_output = self.layer_norm(att_output)  # Apply LayerNorm
            att_input = att_output
            
        # Flatten attention output
        att_flat = att_input.reshape(att_input.size(0), -1)
        att_logit = self.final_linear(att_flat)
        
        # 3. DNN Part (optional - only compute if DNN was initialized)
        if self.dnn_hidden_units is not None:
            dnn_input = embeddings.reshape(embeddings.size(0), -1)
            for layer in self.dnn_layers:
                dnn_input = layer(dnn_input)
            dnn_logit = self.dnn_output_layer(dnn_input)
        
        # 4. Combine
        y_pred = att_logit
        if self.dnn_hidden_units is not None:
            y_pred = y_pred + dnn_logit
        
        """
            GRU part
        """
        if self.seq_vocab_size is not None and seq is not None:
            # seq: (N, SeqLen)
            gru_emb = self.gru_embedding(seq) # (N, SeqLen, EmbSize)
            _, h_n = self.gru(gru_emb) # h_n: (1, N, EmbSize)
            gru_out = h_n.squeeze(0)   # (N, EmbSize)
            gru_logit = self.gru_linear(gru_out) # (N, 1)
            y_pred = y_pred + gru_logit
        
        return y_pred

    def fit(self, loader_train, loader_val, optimizer, epochs=100, verbose=False, print_every=100):
        """
        Training a model and valid accuracy.
        """
        model = self.train().to(device=self.device)
        criterion = F.binary_cross_entropy_with_logits

        for epoch in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=torch.long)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float).unsqueeze(-1)
                
                total = model(xi, xv)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    print('Epoch %d, Iteration %d, loss = %.4f' % (epoch, t, loss.item()))
                    if loader_val:
                        self.check_accuracy(loader_val, model)
                    print()
    
    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=torch.long)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.bool).unsqueeze(-1)
                total = model(xi, xv)
                preds = (torch.sigmoid(total) > 0.5)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
        model.train()
