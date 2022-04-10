import torch
import torch.nn as nn


# Build a classifier and get predictions, pooler_outputs, and hidden_states of inputs.
# The pooler_output uses the last layer hidden-state of the specific token, then further processed by a linear
# layer and a Tanh activation function.
class Classifier(nn.Module):
    def __init__(self, num_labels, **kwargs):
        """Initialize the components of the classifier."""
        super(Classifier, self).__init__()
        self.cls_pos = kwargs['cls_pos']

        self.model = kwargs['model_class'].from_pretrained(kwargs['pretrained_model_name'])
        self.dense = nn.Linear(in_features=768, out_features=768, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(in_features=768, out_features=num_labels, bias=True)

    def forward(self, input_ids, attention_mask):
        """Define the computation performed at every cell."""
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        pooler_output = output.last_hidden_state[:, self.cls_pos, :]
        pooler_output = torch.tanh(self.dense(pooler_output))
        pooler_output = self.dropout(pooler_output)

        logits = self.out_proj(pooler_output)
        hidden_states = output.hidden_states if hasattr(output, 'hidden_states') else None

        return logits, pooler_output, hidden_states
