import numpy as np
import torch

# Applies first the softmax, then minmax scaling to the given tensor, keeping
# the dimensions in tact.
#
def softmax_minmax_scaled(t: torch.Tensor):
    return minmax_on_fm(softmax_on_fm(t))

# Applies the Softmax (2D) to the given Tensor
#
# Assume the given tensor has any shape, where the final two dimensions represent some
# kind of image or feature map. Dimensions are kept identical.
#
def softmax_on_fm(t: torch.Tensor):
    # lehl@2022-02-08: To avoid overflow issues, subtract the maximum value 
    # of t from the whole tensor and then apply softmax as usual
    #
    # see https://stackoverflow.com/questions/42599498/numercially-stable-softmax
    #
    # lehl@2022-02-10: Needs to be run through numpy, as torch is much more
    # sensible to over- and underflow issues, rounding up to infinity or down to 0
    #
    t_np = t.cpu().numpy()

    z = t_np - np.max(t_np)
    y_exp = np.exp(z)
    y_sum = np.sum(y_exp, axis=(-2, -1), keepdims=True)

    if (y_sum == 0).any():
        # import code; code.interact(local=dict(globals(), **locals()))
        y_sum[y_sum == 0] = 1.0

    res = y_exp / y_sum

    return torch.from_numpy(res).to(t.device)

def softmax_torch(t: torch.Tensor):
    z = t - t.max()
    y_exp = torch.exp(z)
    y_sum = torch.sum(y_exp, dim=(-2, -1), keepdim=True)
    
    if (y_sum == 0).any():
        # import code; code.interact(local=dict(globals(), **locals()))
        y_sum[y_sum == 0] = 1.0

    return y_exp / y_sum

# Applies MinMax Scaling (2D) to the given Tensor
#
# Assume the given tensor has any shape, where the final two dimensions represent some
# kind of image or feature map. Dimensions are kept identical.
#
def minmax_on_fm(t: torch.Tensor):
    t_min = torch.min(torch.min(t, dim=-2, keepdim=True).values, dim=-1, keepdim=True).values
    t_max = torch.max(torch.max(t, dim=-2, keepdim=True).values, dim=-1, keepdim=True).values
    
    t_diff = (t_max - t_min)

    # Avoid division by zero
    t_diff[t_diff == 0] = 1.0

    t2 = t - t_min
    return t2 / t_diff
