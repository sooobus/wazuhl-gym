class BaseModel(nn.Module):
    def __init__(self, vocab, actions_vocab, hid_size=256, emb_size=128, batch_size=64):
        super(BaseModel, self).__init__()
        self.batch_size = batch_size
        self.hid_size = hid_size
        self.voc_len = len(vocab)
        self.embedding = nn.Embedding(
            num_embeddings=self.voc_len,
            embedding_dim=emb_size,
            padding_idx=0)
        self.actions_embedding = nn.Embedding(
            num_embeddings=len(actions_vocab),
            embedding_dim=emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
        self.actions_lstm = nn.LSTM(emb_size, hid_size // 2, batch_first=True)
        self.linear = nn.Linear(hid_size, hid_size // 2)
        self.linear_actions = nn.Linear(hid_size // 2, hid_size // 4)
        self.length_enc = nn.Linear(1, hid_size // 4)
        self.output = nn.Linear(hid_size // 2 + hid_size // 4 + hid_size // 4, 1)



    def forward(self, x, actions, lengths):
        self.hidden = (torch.randn(1, self.batch_size, self.hid_size),
                       torch.randn(1, self.batch_size, self.hid_size))
        self.hidden_actions = (torch.randn(1, self.batch_size, self.hid_size // 2),
                               torch.randn(1, self.batch_size, self.hid_size // 2))

        x = self.embedding(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.view(len(lengths)),
                                                    batch_first=True, enforce_sorted=False)
        x, self.hidden = self.lstm(x, self.hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[:, -1, :]
        x = F.relu(self.linear(x))

        lens = F.relu(self.length_enc(lengths))

        actions = self.actions_embedding(actions)
        actions, self.hidden_actions = self.actions_lstm(actions, self.hidden_actions)
        actions = actions[:, -1, :]
        actions = F.relu(self.linear_actions(actions))

        x = self.output(torch.cat((x, actions, lens), dim=1))
        return x.view(x.shape[0])