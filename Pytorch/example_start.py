import torch


if __name__ == '__main__':
    N, D_in, H, D_out = 64, 1000, 100, 10

    dtype = torch.float
    device = torch.device("cuda:0")

    x = torch.randn(N, D_in, device=device, dtype=dtype).requires_grad_()
    y = torch.randn(N, D_out, device=device, dtype=dtype).requires_grad_()

    w1 = torch.randn(D_in, H, device=device, dtype=dtype).requires_grad_()
    w2 = torch.randn(H, D_out, device=device, dtype=dtype).requires_grad_()

    learning_rate = 1e-6
    for t in range(500):
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        loss = (y_pred - y).pow(2).sum()

        if t % 100 == 99:
            print(t, loss.item())

        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            w1.grad.zero_()
            w2.grad.zero_()

        # grad_y_pred = 2.0 * (y_pred - y)
        # grad_w2 = h_relu.t().mm(grad_y_pred)
        # grad_h_relu = grad_y_pred.mm(w2.t())
        # grad_h = grad_h_relu.clone()
        # grad_h[h < 0] = 0
        # grad_w1 = x.t().mm(grad_h)
        #
        # w1 -= learning_rate * grad_w1
        # w2 -= learning_rate * grad_w2
