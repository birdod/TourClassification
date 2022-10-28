from transformers import AutoModel,ViTModel,ViTFeatureExtractor
import torch.nn as nn


class TwoLayerCls(nn.Module):
    def __init__(
      self,
      emd_dim: int, 
      output_dim: int
    ):
      super().__init__()
      self.cls = nn.Sequential(
          nn.Linear(emd_dim, emd_dim),
          nn.LayerNorm(emd_dim),
          nn.Dropout(0.1),
          nn.ReLU(),
          nn.Linear(emd_dim, output_dim)
      )
    def forward(self, x):
        return self.cls(x)

class EmbeddingClassifier(nn.Module):
  def __init__(
      self, 
      emd_dim: int, 
      output_dim: int,
    ):
    super().__init__()
    self.cls = nn.Linear(emd_dim, output_dim)

  def forward(self, x):
    return self.cls(x)
