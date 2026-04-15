#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""pipecat-quickstart - Pipecat Voice Agent

This bot uses a cascade pipeline: Speech-to-Text → LLM → Text-to-Speech

Required AI services:
- Whisper (Speech-to-Text, local)
- OpenAI (LLM)
- Kokoro (Text-to-Speech, local)

Run the bot using::

    uv run bot.py
"""

from datetime import datetime

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import (
    RunnerArguments,
    SmallWebRTCRunnerArguments,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService, VALID_VOICES
from pipecat.services.whisper.stt import WhisperSTTService, Model as WhisperModel
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

# Allow custom voice names for non-OpenAI TTS servers
VALID_VOICES["af_heart"] = "af_heart"


async def get_current_time(params: FunctionCallParams):
    """Get the current local time."""
    now = datetime.now().astimezone()
    await params.result_callback({"time": now.strftime("%H:%M")})


async def run_bot(transport: BaseTransport):
    """Main bot logic."""
    logger.info("Starting bot")

    # Speech-to-Text service (local Whisper)
    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(
            model=WhisperModel.TINY.value,
            language="en",
        ),
        device="cpu",
    )

    # Text-to-Speech service (via vllm-omni OpenAI-compatible API)
    tts = OpenAITTSService(
        base_url="http://127.0.0.1:8001/v1",
        api_key="dummy",
        settings=OpenAITTSService.Settings(
            voice="af_heart",
        ),
    )

    # LLM service (via OpenRouter)
    llm = OpenAILLMService(
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        settings=OpenAILLMService.Settings(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way. When you need information you don't have, always use the available tools instead of guessing or making up answers.",
        ),
    )

    llm.register_direct_function(get_current_time)

    tools = ToolsSchema(standard_tools=[get_current_time])

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Pipeline - assembled from reusable components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation
        context.add_message({"role": "user", "content": "Please introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport = None

    match runner_args:
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection

            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
