# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file provides a disaggregated prefilling proxy demo to demonstrate an
example usage of XpYd disaggregated prefilling.
We can launch multiple vllm instances (2 for prefill and 2 for decode), and
launch this proxy demo through:
  python3 examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py  \
       --model $model_name  \
       --prefill localhost:8100 localhost:8101   \
       --decode localhost:8200 localhost:8201   \
       --port 8000

Note: This demo will be removed once the PDController implemented in PR 15343
(https://github.com/vllm-project/vllm/pull/15343) supports XpYd.
"""

import argparse
import ipaddress
import itertools
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable

import aiohttp
import requests
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def ensure_request_seed(request: dict) -> None:
    if request.get("seed") is None or request.get("seed") == -1:
        request["seed"] = random.randint(INT64_MIN, INT64_MAX)


def _find_sse_separator(buffer: bytes) -> tuple[int, int] | None:
    separators = [
        (idx, len(separator))
        for separator in (b"\n\n", b"\r\n\r\n")
        if (idx := buffer.find(separator)) >= 0
    ]
    if not separators:
        return None
    return min(separators, key=lambda separator: separator[0])


async def iter_sse_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while separator := _find_sse_separator(buffer):
            idx, separator_len = separator
            event_end = idx + separator_len
            yield buffer[:event_end]
            buffer = buffer[event_end:]
    if buffer:
        yield buffer


def get_sse_data(event: bytes) -> str | None:
    try:
        event_str = event.decode("utf-8")
    except UnicodeDecodeError:
        return None

    data_lines = []
    for line in event_str.splitlines():
        line = line.rstrip("\r")
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())
    if not data_lines:
        return None
    return "\n".join(data_lines)


def get_sse_json(event: bytes) -> dict | None:
    data = get_sse_data(event)
    if data is None or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def extract_generated_text(event: bytes, is_chat: bool) -> str | None:
    data = get_sse_json(event)
    if data is None:
        return None

    generated_texts = []
    for choice in data.get("choices", []):
        if is_chat:
            text = choice.get("delta", {}).get("content")
        else:
            text = choice.get("text")
        if text:
            generated_texts.append(text)

    if not generated_texts:
        return None
    return "".join(generated_texts)


def should_forward_prefill_preamble(event: bytes, is_chat: bool) -> bool:
    if not is_chat:
        return False

    data = get_sse_json(event)
    if data is None:
        return False

    for choice in data.get("choices", []):
        delta = choice.get("delta", {})
        if delta.get("role") or delta.get("content") == "":
            return True
    return False


def rewrite_prefill_event(event: bytes) -> bytes:
    data = get_sse_json(event)
    if data is None:
        return event

    data.pop("usage", None)
    for choice in data.get("choices", []):
        choice["finish_reason"] = None
        choice.pop("stop_reason", None)

    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def rewrite_decode_terminal_event(event: bytes, is_chat: bool) -> bytes | None:
    data = get_sse_json(event)
    if data is None:
        return None

    has_finish_reason = False
    for choice in data.get("choices", []):
        has_finish_reason = (
            has_finish_reason or choice.get("finish_reason") is not None
        )
        if is_chat:
            choice.setdefault("delta", {})["content"] = ""
        else:
            choice["text"] = ""
        choice.pop("token_ids", None)
        choice.pop("logprobs", None)

    if not has_finish_reason:
        return None

    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def supports_stream_prefill_token(request: dict, is_chat: bool) -> bool:
    if request.get("stream") is not True:
        return False
    if request.get("n", 1) != 1:
        return False

    max_tokens = request.get("max_completion_tokens" if is_chat else "max_tokens")
    if max_tokens == 0:
        return False

    unsupported_keys = {"logprobs", "top_logprobs", "return_token_ids"}
    if is_chat:
        unsupported_keys.update({"tools", "tool_choice", "response_format"})
    return not any(request.get(key) for key in unsupported_keys)


class SchedulingPolicy(ABC):
    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError("Scheduling Proxy is not set.")


class Proxy:
    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy,
        stream_prefill_token: bool = False,
        custom_create_completion: Callable[[Request], StreamingResponse] | None = None,
        custom_create_chat_completion: Callable[[Request], StreamingResponse]
        | None = None,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.stream_prefill_token = stream_prefill_token
        self.custom_create_completion = custom_create_completion
        self.custom_create_chat_completion = custom_create_chat_completion
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        self.router.post(
            "/v1/completions", dependencies=[Depends(self.validate_json_request)]
        )(
            self.custom_create_completion
            if self.custom_create_completion
            else self.create_completion
        )
        self.router.post(
            "/v1/chat/completions", dependencies=[Depends(self.validate_json_request)]
        )(
            self.custom_create_chat_completion
            if self.custom_create_chat_completion
            else self.create_chat_completion
        )
        self.router.get("/status", response_class=JSONResponse)(self.get_status)
        self.router.post(
            "/instances/add", dependencies=[Depends(self.api_key_authenticate)]
        )(self.add_instance_endpoint)

    async def validate_json_request(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type: Only 'application/json' is allowed",
            )

    def api_key_authenticate(self, x_api_key: str = Header(...)):
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt with API Key: %s", x_api_key)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )

    async def validate_instance(self, instance: str) -> bool:
        url = f"http://{instance}/v1/models"
        try:
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as client:
                logger.info("Verifying %s ...", instance)
                async with client.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            if model_cur == self.model:
                                logger.info("Instance: %s could be added.", instance)
                                return True
                            else:
                                logger.warning(
                                    "Mismatch model %s : %s != %s",
                                    instance,
                                    model_cur,
                                    self.model,
                                )
                                return False
                        else:
                            return False
                    else:
                        return False
        except aiohttp.ClientError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(str(e))
            return False

    async def add_instance_endpoint(self, request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in ["prefill", "decode"]:
                raise HTTPException(status_code=400, detail="Invalid instance type.")
            if not instance or ":" not in instance:
                raise HTTPException(status_code=400, detail="Invalid instance format.")
            host, port_str = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    raise HTTPException(status_code=400, detail="Invalid port number.")
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail="Invalid instance address."
                ) from e

            is_valid = await self.validate_instance(instance)
            if not is_valid:
                raise HTTPException(
                    status_code=400, detail="Instance validation failed."
                )

            if instance_type == "prefill":
                if instance not in self.prefill_instances:
                    self.prefill_instances.append(instance)
                    self.prefill_cycler = itertools.cycle(self.prefill_instances)
                else:
                    raise HTTPException(
                        status_code=400, detail="Instance already exists."
                    )
            else:
                if instance not in self.decode_instances:
                    self.decode_instances.append(instance)
                    self.decode_cycler = itertools.cycle(self.decode_instances)
                else:
                    raise HTTPException(
                        status_code=400, detail="Instance already exists."
                    )

            return JSONResponse(
                content={"message": f"Added {instance} to {instance_type}_instances."}
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def forward_request(self, url, data, use_chunked=True):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
            try:
                async with session.post(
                    url=url, json=data, headers=headers
                ) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(
                                1024
                            ):
                                yield chunk_bytes
                        else:
                            content = await response.read()
                            yield content
                    else:
                        error_content = await response.text()
                        try:
                            error_content = json.loads(error_content)
                        except json.JSONDecodeError:
                            error_content = error_content
                        logger.error(
                            "Request failed with status %s: %s",
                            response.status,
                            error_content,
                        )
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                raise HTTPException(
                    status_code=502,
                    detail="Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    def schedule(self, cycler: itertools.cycle) -> str:
        return self.scheduling_policy.schedule(cycler)

    async def get_status(self):
        status = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }
        return status

    async def stream_prefill_then_decode(
        self,
        endpoint: str,
        request: dict,
        is_chat: bool,
    ) -> AsyncIterator[bytes]:
        try:
            ensure_request_seed(request)

            kv_prepare_request = request.copy()
            kv_prepare_request["stream"] = True
            kv_prepare_request["max_tokens"] = 1
            if is_chat and "max_completion_tokens" in kv_prepare_request:
                kv_prepare_request["max_completion_tokens"] = 1

            prefill_instance = self.schedule(self.prefill_cycler)
            prefill_text = None
            try:
                prefill_chunks = self.forward_request(
                    f"http://{prefill_instance}{endpoint}", kv_prepare_request
                )
                async for event in iter_sse_events(prefill_chunks):
                    generated_text = extract_generated_text(event, is_chat)
                    if generated_text is not None and prefill_text is None:
                        prefill_text = generated_text
                        yield rewrite_prefill_event(event)
                    elif prefill_text is None and should_forward_prefill_preamble(
                        event, is_chat
                    ):
                        yield rewrite_prefill_event(event)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc

            decode_instance = self.schedule(self.decode_cycler)
            try:
                decode_chunks = self.forward_request(
                    f"http://{decode_instance}{endpoint}", request
                )
                if prefill_text is None:
                    logger.warning(
                        "Prefill stream did not produce a generated text chunk; "
                        "forwarding decode stream unchanged."
                    )
                    async for chunk in decode_chunks:
                        yield chunk
                    return

                skipped_decode_text = None
                async for event in iter_sse_events(decode_chunks):
                    generated_text = extract_generated_text(event, is_chat)
                    if generated_text is not None and skipped_decode_text is None:
                        skipped_decode_text = generated_text
                        if skipped_decode_text != prefill_text:
                            logger.warning(
                                "Prefill/decode first-token mismatch in demo "
                                "streaming mode: prefill=%r decode=%r",
                                prefill_text,
                                skipped_decode_text,
                            )
                        terminal_event = rewrite_decode_terminal_event(
                            event, is_chat
                        )
                        if terminal_event is not None:
                            yield terminal_event
                        continue
                    if skipped_decode_text is None:
                        # Skip leading decode metadata already emitted from
                        # prefill, e.g. the chat role chunk.
                        continue
                    yield event
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc
        except Exception:
            logger.exception("stream_prefill_then_decode failed")
            raise

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            if self.stream_prefill_token and supports_stream_prefill_token(
                request, is_chat=False
            ):
                generator = self.stream_prefill_then_decode(
                    "/v1/completions", request, is_chat=False
                )
                return StreamingResponse(generator, media_type="text/event-stream")

            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1

            prefill_instance = self.schedule(self.prefill_cycler)
            try:
                async for _ in self.forward_request(
                    f"http://{prefill_instance}/v1/completions", kv_prepare_request
                ):
                    continue
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc

            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler)

            try:
                generator = self.forward_request(
                    f"http://{decode_instance}/v1/completions", request
                )
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc
            response = StreamingResponse(generator)
            return response
        except Exception:
            logger.exception("Error occurred in disagg proxy server")
            raise

    async def create_chat_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            if self.stream_prefill_token and supports_stream_prefill_token(
                request, is_chat=True
            ):
                generator = self.stream_prefill_then_decode(
                    "/v1/chat/completions", request, is_chat=True
                )
                return StreamingResponse(
                    content=generator, media_type="text/event-stream"
                )

            # add params to request
            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1
            if "max_completion_tokens" in kv_prepare_request:
                kv_prepare_request["max_completion_tokens"] = 1

            # prefill stage
            prefill_instance = self.schedule(self.prefill_cycler)
            try:
                async for _ in self.forward_request(
                    f"http://{prefill_instance}/v1/chat/completions", kv_prepare_request
                ):
                    continue
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc
            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler)

            try:
                generator = self.forward_request(
                    "http://" + decode_instance + "/v1/chat/completions", request
                )
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc
            response = StreamingResponse(content=generator)
            return response
        except Exception:
            logger.exception("Error occurred in disagg proxy server")
            raise

    def remove_instance_endpoint(self, instance_type, instance):
        if instance_type == "decode" and instance in self.decode_instances:
            self.decode_instances.remove(instance)
            self.decode_cycler = itertools.cycle(self.decode_instances)
        if instance_type == "prefill" and instance in self.prefill_instances:
            self.prefill_instances.remove(instance)
            self.prefill_cycler = itertools.cycle(self.prefill_instances)


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    def __init__(self):
        super().__init__()

    def schedule(self, cycler: itertools.cycle) -> str:
        return next(cycler)


class ProxyServer:
    def __init__(
        self,
        args: argparse.Namespace,
        scheduling_policy: SchedulingPolicy | None = None,
        create_completion: Callable[[Request], StreamingResponse] | None = None,
        create_chat_completion: Callable[[Request], StreamingResponse] | None = None,
    ):
        self.validate_parsed_serve_args(args)
        self.port = args.port
        self.proxy_instance = Proxy(
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy=(
                scheduling_policy
                if scheduling_policy is not None
                else RoundRobinSchedulingPolicy()
            ),
            stream_prefill_token=getattr(args, "stream_prefill_token", False),
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
        )

    def validate_parsed_serve_args(self, args: argparse.Namespace):
        if not args.prefill:
            raise ValueError("Please specify at least one prefill node.")
        if not args.decode:
            raise ValueError("Please specify at least one decode node.")
        self.validate_instances(args.prefill)
        self.validate_instances(args.decode)
        self.verify_model_config(args.prefill, args.model)
        self.verify_model_config(args.decode, args.model)

    def validate_instances(self, instances: list):
        for instance in instances:
            if len(instance.split(":")) != 2:
                raise ValueError(f"Invalid instance format: {instance}")
            host, port = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port)
                if not (0 < port < 65536):
                    raise ValueError(f"Invalid port number in instance: {instance}")
            except Exception as e:
                raise ValueError(f"Invalid instance {instance}: {str(e)}") from e

    def verify_model_config(self, instances: list, model: str) -> None:
        model_suffix = model.split("/")[-1]
        for instance in instances:
            try:
                response = requests.get(f"http://{instance}/v1/models")
                if response.status_code == 200:
                    model_cur = response.json()["data"][0]["id"]
                    model_cur_suffix = model_cur.split("/")[-1]
                    if model_cur_suffix != model_suffix:
                        raise ValueError(
                            f"{instance} serves a different model: "
                            f"{model_cur} != {model}"
                        )
                else:
                    raise ValueError(f"Cannot get model id from {instance}!")
            except requests.RequestException as e:
                raise ValueError(
                    f"Error communicating with {instance}: {str(e)}"
                ) from e

    def run_server(self):
        app = FastAPI()
        app.include_router(self.proxy_instance.router)
        config = uvicorn.Config(app, port=self.port, loop="uvloop")
        server = uvicorn.Server(config)
        server.run()


def parse_args():
    # Todo: allow more config
    parser = argparse.ArgumentParser("vLLM disaggregated proxy server.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )
    parser.add_argument(
        "--stream-prefill-token",
        action="store_true",
        help=(
            "Demo mode: stream the first generated prefill token to the "
            "client, then drop the first generated decode token."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    proxy_server = ProxyServer(args=args)
    proxy_server.run_server()
