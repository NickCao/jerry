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

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)
from pipecat.turns.user_start import WakePhraseUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_start_strategies,
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Active timers: name -> (task, total_seconds, start_time)
_active_timers: dict[str, tuple[asyncio.Task, int, float]] = {}


async def get_current_time(params: FunctionCallParams):
    """Get the current local time."""
    now = datetime.now().astimezone()
    await params.result_callback(f"The current time is {now.strftime('%I:%M %p')}.")


async def get_current_date(params: FunctionCallParams):
    """Get the current date and day of the week."""
    now = datetime.now().astimezone()
    await params.result_callback(
        f"Today is {now.strftime('%A')}, {now.strftime('%B %d, %Y')}."
    )


async def set_timer(params: FunctionCallParams, seconds: int, label: str):
    """Set a timer that fires after a given number of seconds.

    Args:
        seconds: Duration in seconds, must be between 1 and 3600.
        label: A short name for this timer, e.g. "pasta" or "5 minute timer".
    """
    if seconds <= 0 or seconds > 3600:
        await params.result_callback(
            "Timer duration must be between 1 and 3600 seconds."
        )
        return

    if label in _active_timers:
        await params.result_callback(f"A timer called '{label}' already exists.")
        return

    async def _timer_task():
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            _active_timers.pop(label, None)
            return
        _active_timers.pop(label, None)
        # Speak directly via TTS, bypassing the LLM entirely.
        await params.llm.push_frame(
            TTSSpeakFrame(text=f"Hey, your timer {label} just went off!")
        )

    task = asyncio.create_task(_timer_task())
    _active_timers[label] = (task, seconds, time.monotonic())
    await params.result_callback(f"Timer '{label}' set for {seconds} seconds.")


async def get_timers(params: FunctionCallParams):
    """List all active timers that have not yet fired."""
    now = time.monotonic()
    lines = []
    for name, (task, total, start) in _active_timers.items():
        if not task.done():
            remaining = max(0, int(total - (now - start)))
            mins, secs = divmod(remaining, 60)
            if mins > 0:
                lines.append(f"{name}: {mins} minutes and {secs} seconds remaining")
            else:
                lines.append(f"{name}: {secs} seconds remaining")
    if lines:
        await params.result_callback("; ".join(lines))
    else:
        await params.result_callback("No active timers.")


async def get_weather(params: FunctionCallParams):
    """Get current weather for Austin, Texas."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=30.2672&longitude=-97.7431"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        "&temperature_unit=fahrenheit&wind_speed_unit=mph"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                c = data["current"]
                await params.result_callback(
                    f"Weather in Austin, Texas: {c['temperature_2m']} degrees Fahrenheit, "
                    f"humidity {c['relative_humidity_2m']}%, "
                    f"wind {c['wind_speed_10m']} miles per hour."
                )
    except Exception as e:
        logger.debug(f"Weather API error: {e}")
        await params.result_callback("Unable to fetch weather data right now.")


def _parse_cpu_stat() -> tuple[int, int]:
    """Parse /proc/stat and return (idle, total) jiffies."""
    stat = Path("/proc/stat").read_text()
    cpu_line = stat.splitlines()[0].split()
    # user, nice, system, idle, iowait, irq, softirq
    vals = [int(v) for v in cpu_line[1:8]]
    idle = vals[3] + vals[4]
    return idle, sum(vals)


async def get_system_status(params: FunctionCallParams):
    """Get system status: CPU/GPU utilization, memory usage, and temperatures."""
    parts = []

    # Memory
    try:
        meminfo = Path("/proc/meminfo").read_text()
        mem = {}
        for line in meminfo.splitlines():
            fields = line.split()
            if fields[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                mem[fields[0].rstrip(":")] = int(fields[1])  # kB
        total_mb = mem.get("MemTotal", 0) // 1024
        avail_mb = mem.get("MemAvailable", 0) // 1024
        used_mb = total_mb - avail_mb
        parts.append(f"Memory: {used_mb} MB used out of {total_mb} MB")
    except Exception as e:
        logger.debug(f"Failed to read memory info: {e}")

    # Temperatures (Jetson thermal zones)
    try:
        thermal = Path("/sys/class/thermal")
        temps = []
        for zone in sorted(thermal.glob("thermal_zone*")):
            name = (zone / "type").read_text().strip()
            temp_mc = int((zone / "temp").read_text().strip())
            temps.append(f"{name} {round(temp_mc / 1000, 1)} C")
        if temps:
            parts.append(f"Temperatures: {', '.join(temps)}")
    except Exception as e:
        logger.debug(f"Failed to read temperatures: {e}")

    # GPU utilization (Jetson — path is JetPack-version-dependent)
    try:
        gpu_load = Path("/sys/devices/gpu.0/load")
        if gpu_load.exists():
            load = int(gpu_load.read_text().strip())
            parts.append(f"GPU utilization: {round(load / 10, 1)}%")
    except Exception as e:
        logger.debug(f"Failed to read GPU load: {e}")

    # CPU utilization (two-sample delta over 100ms for current usage)
    try:
        idle1, total1 = _parse_cpu_stat()
        await asyncio.sleep(0.1)
        idle2, total2 = _parse_cpu_stat()
        d_idle = idle2 - idle1
        d_total = total2 - total1
        if d_total > 0:
            parts.append(f"CPU utilization: {round((1 - d_idle / d_total) * 100, 1)}%")
    except Exception as e:
        logger.debug(f"Failed to read CPU stats: {e}")

    await params.result_callback(". ".join(parts) if parts else "Unable to read system status.")


SYSTEM_INSTRUCTION = (
    "You are Jerry, a helpful voice assistant. Your responses will be spoken "
    "aloud, so avoid emojis, bullet points, or other formatting that can't be "
    "spoken. Respond in a creative, helpful, and brief way. The user activates "
    'you by saying "Hey Jerry" — ignore this wake phrase and respond only to '
    'what follows it. Never start your response with "Hey Jerry". When you '
    "need information you don't have, always use the available tools instead "
    "of guessing or making up answers."
)

TEST_MODE = os.environ.get("JERRY_TEST", "").lower() in ("1", "true", "yes")


async def run_bot(transport: BaseTransport):
    """Main bot logic."""
    logger.info(f"Starting bot (test_mode={TEST_MODE})")

    # Speech-to-Text service (always local Whisper)
    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(
            model=WhisperModel.TINY.value,
            language="en",
        ),
        device="cpu",
    )

    llm = OpenAILLMService(
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        settings=OpenAILLMService.Settings(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    if TEST_MODE:
        from pipecat.services.kokoro.tts import KokoroTTSService

        tts = KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice="af_heart",
                language="en",
            ),
        )
    else:
        tts = OpenAITTSService(
            base_url="http://127.0.0.1:8001/v1",
            api_key="dummy",
            settings=OpenAITTSService.Settings(
                voice="af_heart",
            ),
        )

    all_tools = [
        get_current_time,
        get_current_date,
        set_timer,
        get_timers,
        get_weather,
        get_system_status,
    ]
    for fn in all_tools:
        llm.register_direct_function(fn)

    tools = ToolsSchema(standard_tools=all_tools)

    wake = WakePhraseUserTurnStartStrategy(
        phrases=["hey jerry"],
        timeout=5.0,
    )

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                start=[wake, *default_user_turn_start_strategies()],
            ),
        ),
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=True,
            auto_context_summarization_config=LLMAutoContextSummarizationConfig(
                max_context_tokens=1500,
                max_unsummarized_messages=20,
                summary_config=LLMContextSummaryConfig(
                    target_context_tokens=500,
                    min_messages_after_summary=2,
                ),
            ),
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
