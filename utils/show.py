import torchvision
import torch

def get_grid_image(X):
    # if X.max() <= 1: X = X * 255
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X

# 输出展示
# def make_image(Xs, Xt):
#     Xs = get_grid_image(Xs)
#     Xt = get_grid_image(Xt)
#     # Y = get_grid_image(Y)
# #     # Yt = get_grid_image(Yt)
# #     # Yst = get_grid_image(Yst)
#     return torch.cat((Xs, Xt), dim=1).numpy()
def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    # Yt = get_grid_image(Yt)
    # Yst = get_grid_image(Yst)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()