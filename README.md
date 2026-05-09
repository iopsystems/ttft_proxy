The proxy is copied from vLLM's
[disagg_proxy_demo](https://github.com/vllm-project/vllm/blob/main/examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py)
but adds a flag which allows the proxy to stream the first token that it
received from the prefill node back to the client, rather than discarding it
and waiting for the decode node to regenerate the token. Since the decode node
also generates the first token this is stripped before the response is sent
back to the client.

For models with non-zero temperature, the prefill and decode nodes are not
guaranteed to generate the same first token, even when run with an identical
runtime and model. Since the rest of the decode tokens are generated using
the first token generated at the decode node, this means that the response
to the client may lose semantic meaning due to token mismatch.

The other source of non-determinism is the seed assigned to a request; if no
seed is present, vLLM automatically generates a random seed. The proxy
addresses this by using the same seed for prefill and decode requests, either
by forwarding the client seed or generating a seed at the proxy and using it
for both requests.

Despite these issues, the proxy can try and help estimate the potential
TTFT gains available to a more robust implementation. The correct way to
implement this would require runtime participation: first, the prefill node
should extend its KV cache and sequence state to include both the prompt and
the first generated token, after which the decode node should pull both the KV
cache and the sequence state, unlike now where only the KV cache is pulled.

An intuitive solution might be to convert the generated token back into words
and append it to the request before sending it to the decode node as part of the
prompt. This does not work because tokenization is contextual and dependent on
the surrounding text, which means that the token->text->token conversion may
result in the decode node seeing a different token from that generated at the
prefill node. A workaround could be to move tokenization to the proxy layer and
modify the runtime so prefill and decode nodes expect to receive a token stream,
rather than a text stream.

## Usage

```
  python3 disagg_proxy_ttft.py  \
       --model $model_name  \
       --prefill localhost:8100 localhost:8101   \
       --decode localhost:8200 localhost:8201   \
       --stream-prefill-token \
       --port 8000
```
