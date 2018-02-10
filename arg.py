class ARG():
    def __init__(self):
        self.P_MAX_LEN = 50
        self.Q_MAX_LEN = 20

        self.hidden_dim = 100
        self.embedding_dim = 100

        self.q_rnn_layers = 2
        self.p_rnn_layers = 2

        self.batch_size = 1

        self.grad_clipping = 20