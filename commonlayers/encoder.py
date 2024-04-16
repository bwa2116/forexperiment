from torch import nn
from .attention import Block


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config, perfrelu=False, randomfeatures=False, m=1):
        # Pass the parameter m here since thi class calls on
        # the Block class, which in turn calls on the
        # MultiHeadAttention class
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        self.perfrelu = perfrelu
        self.randomfeatures = randomfeatures
        self.m = m
        for _ in range(config["num_hidden_layers"]):
            block = Block(config, perfrelu=self.perfrelu, randomfeatures=self.randomfeatures, m=self.m)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block

        all_attentions = []
        for block in self.blocks:
            # print(f'shape of x in encoder for loop {x.shape}')
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)
