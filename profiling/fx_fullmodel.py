from typing import List
import torch
import torch
from model import SimpleTransformer
from torch import _dynamo as torchdynamo

class ToyModel(torch.nn.Module):
    def __init__(self, seq_len, dim) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, 2*dim)
        self.linear2 = torch.nn.Linear(2*dim, 2*dim)
        self.linear3 = torch.nn.Linear(2*dim, 2*dim)
        self.linear4 = torch.nn.Linear(2*dim, dim)
        self.linear5 = torch.nn.Linear(dim, 2)
        self.linear6 = torch.nn.Linear(seq_len*2, 1)

    def forward(self, x, condition):
        if condition == 0:
            out = self.linear1(x)
            out = self.linear2(out)
        if condition == 1:
            out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
        out =out.view(out.size(0), -1)
        logits = self.linear6(out)
        return logits

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    print(f">>> Code:\n{gm.code}")
    return gm.forward  # return a python callable

@torch.compile(backend=my_compiler)
def forward(model, x, condition):
    return model(x)

@torchdynamo.optimize(my_compiler)
def d_forward(model, x, condition):
    return model(x, condition)

if __name__ == "__main__":
    batch_size = 16
    seq_len = 10000
    dim = 768
    Toy = ToyModel(seq_len, dim)
    input = torch.randn(batch_size, seq_len, dim)
    conditon = 0
    # c_output = d_forward(Toy, input, conditon)
    d_output=d_forward(Toy, input, conditon)

    model = SimpleTransformer(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    input_tensor = torch.randn(1, 3, 256, 256)

    preds = forward(model, input_tensor,  conditon)