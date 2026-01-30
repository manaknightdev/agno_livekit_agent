"""
LiveKit Voice Agent with Agno Integration

This example demonstrates how to build a voice agent using LiveKit's
VoicePipelineAgent with Agno's powerful agentic capabilities including
tool calling, knowledge bases, and memory.

Requirements:
    - LIVEKIT_URL and LIVEKIT_API_KEY/LIVEKIT_API_SECRET environment variables
    - OPENAI_API_KEY for the Agno agent
    - DEEPGRAM_API_KEY for STT/TTS (or use other providers)

Run with:
    python main.py dev
"""

import logging
from typing import Annotated

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, silero

from livekit_plugins_agno import LLMAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Define tools for the Agno agent
# =============================================================================


@tool
def get_current_time() -> str:
    """Get the current time. Use this when the user asks what time it is."""
    from datetime import datetime

    return f"The current time is {datetime.now().strftime('%I:%M %p')}"


@tool
def get_weather(city: Annotated[str, "The city to get weather for"]) -> str:
    """Get the weather for a specific city. Use this when the user asks about weather."""
    # In a real application, this would call a weather API
    return f"The weather in {city} is sunny and 72°F (22°C)."


@tool
def calculate(
    expression: Annotated[str, "A mathematical expression to evaluate, e.g., '2 + 2'"],
) -> str:
    """Calculate a mathematical expression. Use this when the user asks for calculations."""
    try:
        # WARNING: In production, use a safe math parser instead of eval
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"I couldn't calculate that expression: {str(e)}"


# =============================================================================
# Create the Agno agent
# =============================================================================


def create_agno_agent() -> Agent:
    """Create and configure the Agno agent with tools and instructions."""

    agent = Agent(
        # Use OpenAI's GPT-4o-mini for fast responses
        model=OpenAIChat(id="gpt-4o-mini"),
        # Add tools for the agent to use
        tools=[get_current_time, get_weather, calculate],
        # System instructions for the voice assistant
        instructions="""You are a helpful voice assistant. 

Key behaviors:
- Keep responses concise and conversational - you're speaking, not writing
- Use natural speech patterns and contractions
- When using tools, briefly explain what you're doing
- If you don't know something, say so honestly
- Be friendly and helpful

You have access to tools for:
- Getting the current time
- Checking weather for any city  
- Performing calculations

Remember: Your responses will be spoken aloud, so avoid long lists, 
markdown formatting, or complex technical jargon.""",
        # Enable markdown for text display (TTS will handle the actual speech)
        markdown=False,
        # Keep responses focused
        add_datetime_to_context=True,
    )

    return agent


# =============================================================================
# LiveKit agent entrypoint
# =============================================================================


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit voice agent.

    This function is called when a new room is created and the agent joins.
    It sets up the voice pipeline with Agno as the LLM backend.
    """
    logger.info(f"Connecting to room: {ctx.room.name}")

    # Wait for the first participant to connect
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for a participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Create the Agno agent
    agno_agent = create_agno_agent()

    # Wrap it with the LiveKit adapter
    livekit_llm = LLMAdapter(
        agno_agent,
        session_id=ctx.room.name,
        user_id=participant.identity,
    )

    # Create the voice pipeline agent
    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=livekit_llm,
        tts=deepgram.TTS(),
    )

    # Set up event handlers
    @assistant.on("user_speech_committed")
    def on_user_speech(msg: llm.ChatMessage):
        logger.info(f"User said: {msg.content}")

    @assistant.on("agent_speech_committed")
    def on_agent_speech(msg: llm.ChatMessage):
        logger.info(f"Agent said: {msg.content}")

    # Start the assistant
    assistant.start(ctx.room, participant)

    # Greet the user
    await assistant.say(
        "Hello! I'm your voice assistant powered by Agno. "
        "I can help you with the time, weather, calculations, and more. "
        "How can I help you today?",
        allow_interruptions=True,
    )


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Run the LiveKit agent
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
