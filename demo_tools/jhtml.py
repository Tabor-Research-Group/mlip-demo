"""JHTML callouts shared by the demo notebooks.

Recent McUtils releases expose ``JHTML.JAlert`` and ``JHTML.JCard`` directly.
The workshop environment may have an older release with the equivalent
Bootstrap components. Keep one notebook-facing interface for both versions.
"""

from McUtils.Jupyter import JHTML as _JHTML


class JHTML(_JHTML):
    if not hasattr(_JHTML, "JAlert"):
        @staticmethod
        def JAlert(*contents, kind="info"):
            variant = "danger" if kind == "error" else kind
            return _JHTML.Bootstrap.Alert(*contents, variant=variant)

    if not hasattr(_JHTML, "JCard"):
        @staticmethod
        def JCard(*contents, title=None, kind="info"):
            variant = "danger" if kind == "error" else kind
            return _JHTML.Bootstrap.Card(
                *contents,
                header=title,
                cls=f"border-{variant}",
            )
