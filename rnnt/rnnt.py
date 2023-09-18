import torch
import numpy as np
from typing import List

from . import melfbank
from typing import Optional, Tuple

class LstmDrop(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, forget_bias=None):
        super(LstmDrop, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)

        if forget_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2 * hidden_size].fill_(forget_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2 * hidden_size].fill_(0)

        self.inplace_dropout = torch.nn.Dropout(dropout, inplace=True)

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        self.inplace_dropout(x.data)
        return x, h


class StackTime(torch.nn.Module):
    __constants__ = ["factor"]

    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x, x_lens):
        # x: T, B, U
        r = torch.transpose(x, 0, 1)

        # First, pad along the T dimension to the nearest multiple of factor.
        # This is so we can just apply a simple reshape for this op.
        [B, T, U] = r.shape # B, T, U
        zeros = torch.zeros(B, (-T) % self.factor, U, dtype=r.dtype, device=r.device)
        r = torch.cat([r, zeros], 1)

        # Take a factor of `factor` from the T dim into the U dim (this is the
        # actual stack time operation)
        s = r.shape
        rs = [s[0], s[1] // self.factor, s[2] * self.factor]
        r = torch.reshape(r, rs)

        # Transpose back to T, B, U
        rt = torch.transpose(r, 0, 1)

        # Transform the seq lengths
        x_lens = torch.ceil(x_lens.float() / self.factor).int()
        return rt, x_lens


class Encoder(torch.nn.Module):
    def __init__(
        self,
        in_features,
        num_hidden,
        enc_pre_rnn_layers,
        enc_post_rnn_layers,
        forget_bias,
        stack_time,
        dropout
    ):
        super().__init__()
        self.pre_lstms = LstmDrop(
            in_features, num_hidden, enc_pre_rnn_layers, dropout, forget_bias)

        self.stack_time = StackTime(factor=stack_time)

        self.post_lstms = LstmDrop(
            stack_time * num_hidden, num_hidden, enc_post_rnn_layers, dropout, forget_bias)

    def forward(self, x_padded, x_lens):
        x_padded, _ = self.pre_lstms(x_padded, None)
        x_padded, x_lens = self.stack_time(x_padded, x_lens)
        x_padded, _ = self.post_lstms(x_padded, None)
        return x_padded.transpose(0, 1), x_lens

class Prediction(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        n_hidden,
        pred_rnn_layers,
        forget_gate_bias,
        dropout
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size - 1, n_hidden)
        self.n_hidden = n_hidden
        self.dec_rnn = LstmDrop(n_hidden, n_hidden, pred_rnn_layers, dropout, forget_gate_bias)

    def forward(
        self,
        y: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2

        Args:
            y: (B, U)

        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """

        if self.training:
            assert state is None
            y_embed = self.embed(y)
        else:
            if y is None:
                assert state is None
                # N.B. Batch size is always 1 in inference mode as a result of
                # the data-dependent control flow in RNN-T's inference.
                y_embed = torch.zeros(
                    (1, 1, self.n_hidden),
                    device=self.embed.weight.device,
                    dtype=self.embed.weight.dtype)
            else:
                y_embed = self.embed(y)

        pred, new_state = self.dec_rnn(y_embed.transpose(0, 1), state)
        return pred.transpose(0, 1), new_state

class Joint(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        pred_n_hidden,
        enc_n_hidden,
        joint_n_hidden,
        dropout
    ):
        super().__init__()
        self.net = torch.nn.Sequential(*[
            torch.nn.Linear(pred_n_hidden + enc_n_hidden, joint_n_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(joint_n_hidden, vocab_size)
        ])

    def forward(self, f: torch.Tensor, g: torch.Tensor):
        """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)

        returns:
            logits of shape (B, T, U, K + 1)
        """
        # Combine the input states and the output states
        B, T, H = f.shape
        B, U_, H2 = g.shape

        f = f.unsqueeze(dim=2)   # (B, T, 1, H)
        f = f.expand((B, T, U_, H))

        g = g.unsqueeze(dim=1)   # (B, 1, U + 1, H)
        g = g.expand((B, T, U_, H2))

        inp = torch.cat([f, g], dim=3) # (B, T, U, 2H)
        res = self.net(inp)
        # del f, g, inp
        return res

def label_collate(labels):
    """Collates the label inputs for the rnn-t prediction network.

    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(
            f"`labels` should be a list or tensor not {type(labels)}"
        )

    batch_size = len(labels)
    max_len = max(len(l) for l in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, :len(l)] = l
    labels = torch.LongTensor(cat_labels)

    return labels

class Rnnt(torch.nn.Module):
    labels = [
        " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'",
        "<BLANK>"
    ]

    features = 80
    frame_splicing = 3
    in_features = features * frame_splicing
    num_classes = len(labels)
    blank_id = labels.index("<BLANK>")
    max_symbols_per_step = 30

    rnn_type = "lstm"
    enc_n_hidden = 1024
    enc_pre_rnn_layers = 2
    enc_stack_time_factor = 2
    enc_post_rnn_layers = 3
    pred_n_hidden = 320
    pred_rnn_layers = 2
    forget_gate_bias = 1.0
    norm = None
    joint_n_hidden = 512
    dropout = 0.0 #0.32

    def __init__(self):
        super().__init__()

        self.preprocessor = melfbank.MelFilterBanks.rnnt_mel_filter_banks()

        self.encoder = Encoder(
            self.in_features,
            self.enc_n_hidden,
            self.enc_pre_rnn_layers,
            self.enc_post_rnn_layers,
            self.forget_gate_bias,
            self.enc_stack_time_factor,
            self.dropout,
        )

        self.prediction = Prediction(
            self.num_classes,
            self.pred_n_hidden,
            self.pred_rnn_layers,
            self.forget_gate_bias,
            self.dropout,
        )

        self.joint = Joint(
            self.num_classes,
            self.pred_n_hidden,
            self.enc_n_hidden,
            self.joint_n_hidden,
            self.dropout,
        )

    def _train(self, x_padded: torch.Tensor, x_lens: torch.Tensor, y : torch.Tensor):
        # print(f'x_padded = {x_padded.shape}')
        # print(f'x_lens = {x_lens.shape}')
        # print(f'y = {y.shape}')
        x_padded, x_lens = self.preprocessor(x_padded, x_lens)
        x_padded = x_padded.permute(2, 0, 1)
        # print(f'il = {x_padded.shape[0]}')
        enc_logits, enc_logits_lens = self.encoder(x_padded, x_lens)
        pred, new_state = self.prediction(y)
        return self.joint(enc_logits, pred)



    def _decode(self, x: torch.Tensor, out_len: torch.Tensor) -> List[int]:
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        label: List[int] = []
        last_sym = None
        for ti in range(int(out_len.item())):
            enc = x[ti, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and symbols_added < self.max_symbols_per_step:
                pred, new_state = self.prediction(
                    torch.tensor([[last_sym]], dtype=torch.int64, device=x.device) if last_sym is not None else None,
                    state)

                logp = self.joint(enc, pred)[0, 0, 0, :]
                v, k = logp.max(0)
                k = k.item()

                if k == self.blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    last_sym = k
                    state = new_state
                symbols_added += 1

        return label

    def _infer(self, x_padded: torch.Tensor, x_lens: torch.Tensor):
        x_padded, x_lens = self.preprocessor(x_padded, x_lens)
        x_padded = x_padded.permute(2, 0, 1)
        # print(f'il = {x_padded.shape[0]}')
        enc_logits, enc_logits_lens = self.encoder(x_padded, x_lens)

        output: List[List[int]] = []
        for batch_idx in range(enc_logits.size(0)):
            inseq = enc_logits[batch_idx, :, :].unsqueeze(1)
            # inseq: TxBxF
            logitlen = enc_logits_lens[batch_idx]
            sentence = self._decode(inseq, logitlen)
            output.append(sentence)

        # print(f'ol = {len(output[0])}')
        return output

    def forward(self, x_padded: torch.Tensor, x_lens: torch.Tensor, y : torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training: return self._train(x_padded, x_lens, y)
        else: return self._infer(x_padded, x_lens)

