import torch


class GradLoss(torch.nn.Module):
    def __init__(self, lambda_grad=0.999, delta=1.0):
        super(GradLoss, self).__init__()
        self.lambda_grad = lambda_grad
        self.mse_loss = torch.nn.MSELoss()
        self.delta = delta

    def forward(self, prediction, target):
        mse_loss = self.mse_loss(prediction, target)
        grad_loss = self.compute_gradient_loss(prediction, target)
        loss = (1 - self.lambda_grad) * mse_loss + self.lambda_grad * grad_loss
        return loss

    def compute_gradient_loss(self, prediction, target):
        grad_pred_x = torch.gradient(prediction, spacing=self.delta, dim=2)[0]
        grad_pred_y = torch.gradient(prediction, spacing=self.delta, dim=3)[0]
        grad_target_x = torch.gradient(target, spacing=self.delta, dim=2)[0]
        grad_target_y = torch.gradient(target, spacing=self.delta, dim=3)[0]

        grad_loss = self.mse_loss(grad_pred_x, grad_target_x) + self.mse_loss(
            grad_pred_y, grad_target_y
        )

        return grad_loss
