from typing import Callable

import torch

class SimpleNet(torch.nn.Module):
    def __init__(self,
                 in_h: int = 897,
                 k: int = 5,
                 filters: int = 8,
                 activation: Callable = torch.nn.Tanh,
                 ) -> None:
        """
        Args:
            in_h: input height
            k: length of history steps
            filters: number of filters
            activation: activation function
        Returns:
            None
        """
        super(SimpleNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Unflatten(1,torch.Size([1,k])),
            torch.nn.Conv2d(1,filters,k,padding=(0,k//2),padding_mode="circular"),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_h*filters,in_h),
            activation(),
            )

    def forward(self, p: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: pose (x, y, theta) [batch, k, 3]
            x: scan history [batch, k, in_h]
            y: past label estimates [batch, k, in_h]
        Returns:
            y_hat: label estimates [batch, in_h]
        """
        return self.net(x)

class LabelNet(torch.nn.Module):
    def __init__(self,
                 in_h: int = 897,
                 k: int = 5,
                 filters: int = 8,
                 activation: Callable = torch.nn.Tanh,
                 ) -> None:
        """
        Args:
            in_h: input height
            k: kernel size
            filters: number of filters
            activation: activation function
        Returns:
            None
        """
        super(LabelNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Unflatten(1,torch.Size([1,k])),
            torch.nn.Conv2d(1,filters,k,padding=(0,k//2),padding_mode="circular"),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_h*filters,in_h),
            activation(),
            )
        self.label_enc = torch.nn.Sequential(
            torch.nn.Unflatten(1,torch.Size([1,k])),
            torch.nn.Conv2d(1,filters,k,padding=(0,k//2),padding_mode="circular"),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_h*filters,in_h),
            activation(),
            )

    def forward(self, p: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: pose (x, y, theta) [batch, k, 3]
            x: scan history [batch, k, in_h]
            y: past label estimates [batch, k, in_h]
        Returns:
            y_hat: label estimates [batch, in_h]
        """
        return self.net(x) * self.label_enc(y)

class LabelPoseNet(torch.nn.Module):
    def __init__(self,
                 in_h: int = 897,
                 k: int = 5,
                 filters: int = 8,
                 activation: Callable = torch.nn.Tanh,
                 ) -> None:
        """
        Args:
            in_h: input height
            k: kernel size
            filters: number of filters
            activation: activation function
        Returns:
            None
        """
        super(LabelPoseNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Unflatten(1,torch.Size([1,k])),
            torch.nn.Conv2d(1,filters,k,padding=(0,k//2),padding_mode="circular"),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_h*filters,in_h),
            activation(),
            )
        self.label_enc = torch.nn.Sequential(
            torch.nn.Unflatten(1,torch.Size([1,k])),
            torch.nn.Conv2d(1,filters,k,padding=(0,k//2),padding_mode="circular"),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_h*filters,in_h),
            activation(),
            )
        self.pose_enc = torch.nn.Sequential(
            torch.nn.Unflatten(1,torch.Size([1,k])),
            torch.nn.Conv2d(1,k,(1,3)),
            activation(),
            torch.nn.Flatten(),
            torch.nn.Linear(k*k,in_h),
            activation(),
            )
        self.internal_activation = activation()

    def forward(self, p: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: pose (x, y, theta) [batch, k, 3]
            x: scan history [batch, k, in_h]
            y: past label estimates [batch, k, in_h]
        Returns:
            y_hat: label estimates [batch, in_h]
        """
        return self.internal_activation(self.net(x) + self.pose_enc(p)) * self.label_enc(y)