from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]

def debug_output_gaussians(gaussians: Gaussians):
    print('-'*100)
    print('means shape: ', gaussians.means.shape)
    print('covariances shape: ', gaussians.covariances.shape)
    print('harmonics shape: ', gaussians.harmonics.shape)
    print('opacities shape: ', gaussians.opacities.shape)
    print('-'*100)