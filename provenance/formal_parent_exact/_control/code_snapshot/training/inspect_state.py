import torch

def tree_view(obj, depth=0):
    if depth > 10:
        return
    for key, value in obj.items():
        if isinstance(value, dict):
            print("  " * depth + f"{key}[{type(value)}]:")
            tree_view(value, depth+1)
            continue
        value_repr = "?"
        if isinstance(value, torch.Tensor):
            value_repr = str(value.size())
        elif isinstance(value, torch.Size):
            value_repr = str(value)
        print("  " * depth + f"{key}[{type(value)}] = {value_repr}")

# state_dict = torch.load("results/917/checkpoint_1.pt")
# print("Idenpendent:")
# tree_view(state_dict)
# print("")
state_dict = torch.load("results/932/checkpoint_1.pt")
# print("Shared:")
tree_view(state_dict)