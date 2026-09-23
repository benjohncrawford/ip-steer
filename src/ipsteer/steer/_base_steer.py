from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Steer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def steer(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def vector_field(self) -> Tensor:
        raise NotImplementedError
    
    
class VecSteer(Steer):
    def __init__(self):
        super().__init__()
        self.steer_vec = None
    
    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError
    
    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        return X + T * self.steer_vec.to(X.device)
    
    def vector_field(
        self,
        X: Tensor,
    ) -> Tensor:
        return self.steer_vec.broadcast_to(X.shape)
    
class MultiStepVecSteer(Steer):
    def __init__(self):
        super().__init__()
        self.steer_vecs = []
    
    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError
    
    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        for vec in self.steer_vecs:
            ret = X + T * vec.to(X.device)
        return ret
    
    def vector_field(
        self,
        X: Tensor,
    ) -> Tensor:
        return self.steer_vecs[-1].broadcast_to(X.shape)