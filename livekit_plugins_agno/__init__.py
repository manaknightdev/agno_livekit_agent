"""
LiveKit Agents Agno Plugin

Wraps Agno Agents as LiveKit-compatible LLMs for voice pipelines.

Example:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from livekit_plugins_agno import LLMAdapter

    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), tools=[...])
    livekit_llm = LLMAdapter(agent)
"""

from .agno import AgnoStream, LLMAdapter
from .version import __version__

__all__ = ["__version__", "LLMAdapter", "AgnoStream"]


from livekit.agents import Plugin


class AgnoPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)

    def download_files(self) -> None:
        pass


Plugin.register_plugin(AgnoPlugin())


__all__ = ["LLMAdapter", "AgnoStream", "__version__"]
