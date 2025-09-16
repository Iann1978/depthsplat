from .views_provider import BatchedRenderViews, ViewsProvider
from .views_provider_fixed import ViewsProviderFixed, ViewsProviderFixedCfg
from .views_provider_gradio import ViewsProviderGradio, ViewsProviderGradioCfg

__all__ = ["BatchedRenderViews", "ViewsProvider", "ViewsProviderCfg", "get_views_provider"]


VIEWS_PROVIDERS: dict[str, ViewsProvider] = {
    "fixed": ViewsProviderFixed,
    "gradio": ViewsProviderGradio,
}

ViewsProviderCfg = ViewsProviderGradioCfg | ViewsProviderFixedCfg


def get_views_provider(cfg: ViewsProviderCfg, **kwargs) -> ViewsProvider:
    return VIEWS_PROVIDERS[cfg.name](cfg, **kwargs)