from .random_context_provider import RandomContextProvider, RandomContextProviderCfg
from .context_provider import ContextProvider, debug_output_context
from .dataset_context_provider import DatasetContextProvider, DatasetContextProviderCfg

CONTEXT_PROVIDERS: dict[str, ContextProvider] = {
    "random": RandomContextProvider,
    "dataset": DatasetContextProvider,
}

ContextProviderCfg = RandomContextProviderCfg | DatasetContextProviderCfg


def get_context_provider(cfg: ContextProviderCfg) -> ContextProvider:
    return CONTEXT_PROVIDERS[cfg.name](cfg)