from .random_context_provider import RandomContextProvider, RandomContextProviderCfg
from .context_provider import ContextProvider

CONTEXT_PROVIDERS: dict[str, ContextProvider] = {
    "random": RandomContextProvider,
}

ContextProviderCfg = RandomContextProviderCfg


def get_context_provider(cfg: ContextProviderCfg) -> ContextProvider:
    return CONTEXT_PROVIDERS[cfg.name](cfg)