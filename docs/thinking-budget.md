# Why agent runs need a thinking budget (measured 2026-08-27/28)

Setup: an agentic coding harness, four hidden-test tasks scored out of 400, temperature 1.0, no per-task time
limit, on two RTX PRO 6000s (Blackwell) unless noted.

| model / runtime | think cap | runs (/400) | mean | wall per run | tokens per point | decode |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next FP8 + MTP4 (vLLM, TP2) | none (native) | 392 / 359 / 389 | **380** | 31 to 56 min | ~820 | 172 t/s under load |
| GLM-5.3-Flash TR3-4bpw (fork runtime + budget proxy) | 8K | 389 / 359 / 369 | **372** | 42 / 47 / 80 min | ~380 | 110 t/s |
| GLM-5.3-Flash MLX 6-bit (Mac Studio M3 Ultra) | 32K | 389 | 389 | 17 h, 5 interrupts | | 24 t/s |
| DeepSeek V4-Flash abliterated (DSpark, no thinking) | | 287 / 197 / 382 | 289 | ~45 min | | 210 t/s |

## Findings

1. **Unlimited reasoning is a failure mode in agent loops.** GLM TR3 without a budget produced 231K
   characters of coherent reasoning, 65,536 output tokens, and zero tool calls on task 1: 4/100. With a 32K
   budget it worked, but the client re-fed prior turns' reasoning and the GLM chat template keeps it for the
   whole tool loop, so prompts grew to 94K tokens and hit context-overflow 400s (run scored 213/400). With
   8K: 389 / 359 / 369. Lower budget, better agent.
2. **The same holds on the Mac.** 72% of Qwen's output on task 1 was thinking, in 19K-token single turns. A
   4K cap fixed the thinking but not the polish loop (77 turns after a passing solution existed). Models
   need a "stop when tests pass" harness rule, not just a budget.
3. **Mac prefill is the wall for agents, not decode.** At 120K to 180K-token contexts each turn re-prefills
   unless the KV hot cache is large; even then about 1,000 t/s prefill times 150K tokens is minutes per turn.
   The Blackwell box does the same turn in seconds.
4. **GLM TR3 is Qwen-class, not better.** Same scores, same weak task (the regex engine: 98 / 68 / 78), five
   times less aggregate throughput, one quant, a closed runtime, a four-stream cap.

## The proxy

`glm53-tr3/think_budget_proxy.py` (stdlib only) sits between the client and vLLM:

- streams phase 1 and counts reasoning characters (3.5 chars per token measured on this model);
- when the budget is exceeded it aborts, then re-issues the request with the reasoning so far as a
  prefilled assistant turn (`continue_final_message: true`, `add_generation_prompt: false`,
  `chat_template_kwargs.clear_thinking: false`) plus a short "I have thought enough, acting now" tail, so
  the model emits content or tool calls. Tool calls survive the cut in both streaming and blocking modes;
- strips `reasoning_content` from assistant history (the template would otherwise re-render the whole
  loop's thinking);
- retries context-length 400s with a fitted `max_tokens`.

Non-streaming phase 2 must be assembled from an upstream stream: the non-stream reasoning parser puts the
continued answer into `reasoning` and drops tool calls.

Some agent clients cache a provider's base URL, so the proxy owns the public port and vLLM moves to an
internal one.
