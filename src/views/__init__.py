from .views_provider import BatchedRenderViews, ViewsProvider
from .views_provider_fixed import ViewsProviderFixed, ViewsProviderFixedCfg

__all__ = ["BatchedRenderViews", "ViewsProvider", "ViewsProviderCfg", "get_views_provider"]


VIEWS_PROVIDERS: dict[str, ViewsProvider] = {
    "fixed": ViewsProviderFixed,
}

ViewsProviderCfg = ViewsProviderFixedCfg


def get_views_provider(cfg: ViewsProviderCfg) -> ViewsProvider:
    return VIEWS_PROVIDERS[cfg.name](cfg)