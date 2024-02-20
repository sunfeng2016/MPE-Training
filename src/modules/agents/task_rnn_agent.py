import torch.nn as nn
import torch.nn.functional as F
import torch as th

from torch.nn import LayerNorm
from torch.distributions import Categorical
from torch.nn.functional import one_hot

class TaskRNNAgent(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super(TaskRNNAgent, self).__init__()
        self.args = args

        activation_func = nn.LeakyReLU()

        self.task_net = nn.Sequential(
            nn.Linear(input_shape, args.mlp_hidden_dim),
            activation_func,
            nn.Linear(args.mlp_hidden_dim, args.task_num),
            nn.Softmax(dim=-1)
        )

        self.embed_net = nn.Sequential(
            nn.Linear(args.task_num, args.task_emb_dim, bias=False),
            nn.LayerNorm(args.task_emb_dim),
        )

        self.fc1 = nn.Linear(input_shape + args.task_emb_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, task_embeddings=None, mask=None):
        b, a, e = inputs.size()

        # Assignment a task for each agent
        task_inputs = inputs.view(-1, e)    # [ba, e]
        task_probs = self.task_net(task_inputs)  # [ba, m]
        task_preds = Categorical(task_probs).sample().long()    # [ba, 1]
        task_one_hot = one_hot(task_preds, self.args.task_num).float()  # [ba, m]

        # Compute_entropy
        entropy = Categorical(task_probs).entropy().view(b, -1)

        # Generate an embedding for each task
        task_embeddings_new = self.embed_net(task_one_hot + task_probs - task_probs.detach()).view(b, a, -1)    # [b, a, d]

        if task_embeddings is None:
            task_embeddings = task_embeddings_new
        else:
            task_embeddings = th.where(mask, task_embeddings_new, task_embeddings.detach())
        
        # Concat the embedding and the rnn input
        q_inputs = th.cat((inputs, task_embeddings), dim=-1)    # [b, a, e+d]
        q_inputs = q_inputs.view(b*a, -1)   #[ba, e+d]
        x = F.relu(self.fc1(q_inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # [ba, h]
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1), task_embeddings, entropy