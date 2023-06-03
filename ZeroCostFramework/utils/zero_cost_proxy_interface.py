import torch
from torch import nn 
from typing import Union

class ZeroCostProxyInterface:
    def calculate_proxy(
        self,
        net: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: str,
        loss_function: Union[
            nn.L1Loss,
            nn.MSELoss,
            nn.CrossEntropyLoss,
            nn.CTCLoss,
            nn.NLLLoss,
            nn.PoissonNLLLoss,
            nn.GaussianNLLLoss,
            nn.KLDivLoss,
            nn.BCELoss,
            nn.BCEWithLogitsLoss,
            nn.MarginRankingLoss,
            nn.HingeEmbeddingLoss,
            nn.MultiLabelMarginLoss,
            nn.HuberLoss,
            nn.SmoothL1Loss,
            nn.SoftMarginLoss,
            nn.MultiLabelSoftMarginLoss,
            nn.CosineEmbeddingLoss,
            nn.MultiMarginLoss,
            nn.TripletMarginLoss,
            nn.TripletMarginWithDistanceLoss
        ],
        eval: bool = False,
        train: bool = True,
        single_batch: bool = True,
        bn: bool = False,
    ) -> float:
        pass
