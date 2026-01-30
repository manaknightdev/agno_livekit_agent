import logging
from typing import Annotated

from agno.agent import Agent as AgnoAgent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit_plugins_agno.agno import LLMAdapter
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

prompt = """
You are a helpful voice assistant. 

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
markdown formatting, or complex technical jargon.
"""


def create_agno_agent() -> AgnoAgent:
    """Create and configure the Agno agent with tools and instructions."""

    agent = AgnoAgent(
        # Use OpenAI's GPT-4o-mini for fast responses
        model=OpenAIChat(id="gpt-4o-mini"),
        # Add tools for the agent to use
        tools=[get_current_time, get_weather, calculate],
        # System instructions for the voice assistant
        instructions=prompt,
        # Enable markdown for text display (TTS will handle the actual speech)
        markdown=False,
        # Keep responses focused
        add_datetime_to_context=True,
    )

    return agent


# =============================================================================
# LiveKit agent entrypoint
# =============================================================================
server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=LLMAdapter(agent=create_agno_agent()),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Agent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
