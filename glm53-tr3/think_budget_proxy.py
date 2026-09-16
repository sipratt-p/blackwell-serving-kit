#!/usr/bin/env python3
"""Thinking-budget proxy for vLLM reasoning models (built for GLM-5.3-Flash TR3 on :8012, 2026-08-28).

Why: the B12X/vLLM runtime has no thinking-budget flag; GLM-5.3-Flash will reason for 65K+ tokens in one
turn and never act (agent benchmark task 1: 231K chars of reasoning, 0 tool calls). On a Mac Studio the same model
scored 389/400 only because mlx's --thinking-budget cut reasoning at 32K and forced an answer. This proxy does the
same via the OpenAI API: stream phase 1; once reasoning exceeds the budget, abort it and re-issue the request
with the reasoning so far as a prefilled assistant message (continue_final_message=true, clear_thinking=false)
plus a "I have thought enough; acting now" tail, so the model emits content / tool calls.

Usage:  GLM_THINK_BUDGET=32768 UPSTREAM=http://127.0.0.1:8012 PORT=8013 python3 think_budget_proxy.py
Point your agent client at http://127.0.0.1:8013/v1. Everything except /v1/chat/completions is passed through.
Stdlib only. Token count is approximated as chars/3.5 (measured on this model: 231,310 chars = 65,536 tokens).
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8012").rstrip("/")
PORT = int(os.environ.get("PORT", "8013"))
BUDGET_TOKENS = int(os.environ.get("GLM_THINK_BUDGET", "32768"))
CHARS_PER_TOKEN = float(os.environ.get("CHARS_PER_TOKEN", "3.5"))
BUDGET_CHARS = int(BUDGET_TOKENS * CHARS_PER_TOKEN)
CUTOFF_TAIL = os.environ.get("CUTOFF_TAIL", "\n\nOkay, I have thought enough. I will stop reasoning here and act now.")
UPSTREAM_TIMEOUT = int(os.environ.get("UPSTREAM_TIMEOUT", "3600"))

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def upstream(path, body=None, headers=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(UPSTREAM + path, data=data, headers=h, method=method or ("POST" if data else "GET"))
    return urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)

def sse_events(resp):
    """Yield (raw_line, parsed_json_or_None) for each data: line of an SSE stream."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            yield line, None
            break
        try:
            yield line, json.loads(payload)
        except json.JSONDecodeError:
            yield line, None

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "131072"))
_CTX_RE = __import__("re").compile(r"prompt contains at least (\d+) input tokens")

def strip_history_reasoning(body):
    """Agent clients send prior turns' thinking back as reasoning_content; GLM's template re-renders all of it for the
    whole tool loop (prompt hit 94K tokens on benchmark task 2 → 400). Drop it: the template renders <think></think>."""
    b = dict(body); msgs = []
    for m in body.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "assistant" and (m.get("reasoning_content") or m.get("reasoning")):
            m = {k: v for k, v in m.items() if k not in ("reasoning_content", "reasoning")}
        msgs.append(m)
    b["messages"] = msgs
    return b

def fit_max_tokens(body, err_bytes):
    """On vLLM's context-length 400, return a copy with max_tokens that fits (or None if not that error)."""
    m = _CTX_RE.search(err_bytes.decode("utf-8", "replace"))
    if not m:
        return None
    prompt = int(m.group(1)); fit = MAX_MODEL_LEN - prompt - 256
    if fit < 1024:
        return None
    b = dict(body); b["max_tokens"] = min(int(body.get("max_tokens") or fit), fit)
    log(f"context fit: prompt≈{prompt} tok → max_tokens={b['max_tokens']}")
    return b

def open_chat(body):
    """upstream chat call with one retry on context-length 400."""
    try:
        return upstream("/v1/chat/completions", body)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        data = e.read()
        fitted = fit_max_tokens(body, data)
        if fitted is None:
            raise urllib.error.HTTPError(e.url, e.code, e.msg, e.headers, __import__("io").BytesIO(data))
        return upstream("/v1/chat/completions", fitted)

def continuation_body(body, reasoning):
    """Same request, plus the truncated reasoning as a prefilled assistant turn."""
    b = dict(body)
    b["messages"] = list(body.get("messages", [])) + [
        {"role": "assistant", "reasoning_content": reasoning + CUTOFF_TAIL, "content": ""}]
    b["continue_final_message"] = True
    b["add_generation_prompt"] = False
    kw = dict(b.get("chat_template_kwargs") or {})
    kw["clear_thinking"] = False
    b["chat_template_kwargs"] = kw
    return b

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _passthrough(self, body_bytes=None):
        try:
            req = urllib.request.Request(UPSTREAM + self.path, data=body_bytes,
                                         headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                                         method=self.command)
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as r:
                data = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_GET(self):
        self._passthrough()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._passthrough(raw)
        try:
            body = json.loads(raw)
        except Exception:
            return self._send_json(400, {"error": "bad json"})
        body = strip_history_reasoning(body)
        log(f"chat request: stream={bool(body.get('stream'))} max_tokens={body.get('max_tokens')} msgs={len(body.get('messages') or [])} tools={len(body.get('tools') or [])}")
        try:
            if body.get("stream"):
                self._chat_stream(body)
            else:
                self._chat_blocking(body)
        except urllib.error.HTTPError as e:
            data = e.read()
            log("upstream HTTP", e.code, data[:300])
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            log("client went away")
        except Exception as e:
            log("proxy error:", repr(e))
            try:
                self._send_json(500, {"error": f"proxy: {e!r}"})
            except Exception:
                pass

    # ---- streaming path -------------------------------------------------------------------------
    def _chat_stream(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(line):
            chunk = (line + "\n\n").encode()
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()

        reasoning, content_seen, cut = [], False, False
        rchars = 0
        t0 = time.time()
        with open_chat(body) as r:
            for line, ev in sse_events(r):
                if ev is None:
                    emit(line)  # [DONE] or unparsable
                    break
                delta = (ev.get("choices") or [{}])[0].get("delta") or {}
                rc = delta.get("reasoning") or delta.get("reasoning_content")
                if rc:
                    reasoning.append(rc)
                    rchars += len(rc)
                if delta.get("content") or delta.get("tool_calls"):
                    content_seen = True
                if not content_seen and rchars >= BUDGET_CHARS:
                    cut = True
                    break
                emit(line)
        if not cut:
            self._end_chunked()
            return
        # phase 2: force the answer
        rtext = "".join(reasoning)
        log(f"budget hit: {rchars} reasoning chars (~{int(rchars / CHARS_PER_TOKEN)} tok) after {time.time() - t0:.0f}s -> forcing answer")
        cont = continuation_body(body, rtext)
        with open_chat(cont) as r2:
            for line, ev in sse_events(r2):
                emit(line)
                if ev is None:
                    break
        self._end_chunked()

    def _end_chunked(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    # ---- blocking path --------------------------------------------------------------------------
    # Phase 2 is always done via an upstream *stream* and re-assembled: the non-streaming reasoning
    # parser treats a continued turn as still-thinking (answer lands in `reasoning`, tool calls lost),
    # while the streaming parser handles it correctly (verified 2026-08-28).
    def _chat_blocking(self, body):
        b = dict(body)
        req_max = int(b.get("max_tokens") or 0) or None
        b["max_tokens"] = min(req_max, BUDGET_TOKENS) if req_max else BUDGET_TOKENS
        with open_chat(b) as r:
            d = json.loads(r.read())
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        rtext = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if ch.get("finish_reason") == "length" and not msg.get("content") and not msg.get("tool_calls") and rtext:
            log(f"budget hit (blocking): {len(rtext)} reasoning chars -> forcing answer")
            cont = continuation_body(body, rtext)
            cont["stream"] = True
            cont["stream_options"] = {"include_usage": True}
            if req_max:
                cont["max_tokens"] = max(1024, req_max - BUDGET_TOKENS)
            content, tools, finish, usage2, extra_r = [], {}, None, None, []
            with open_chat(cont) as r2:
                for _line, ev in sse_events(r2):
                    if ev is None:
                        break
                    if ev.get("usage"):
                        usage2 = ev["usage"]
                    chs = ev.get("choices") or []
                    if not chs:
                        continue
                    c0 = chs[0]; delta = c0.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning") or delta.get("reasoning_content"):
                        extra_r.append(delta.get("reasoning") or delta.get("reasoning_content"))
                    for t in delta.get("tool_calls") or []:
                        i = t.get("index", 0)
                        slot = tools.setdefault(i, {"id": t.get("id"), "type": "function", "function": {"name": "", "arguments": ""}})
                        if t.get("id"):
                            slot["id"] = t["id"]
                        f = t.get("function") or {}
                        if f.get("name"):
                            slot["function"]["name"] += f["name"]
                        if f.get("arguments"):
                            slot["function"]["arguments"] += f["arguments"]
                    if c0.get("finish_reason"):
                        finish = c0["finish_reason"]
            m2 = {"role": "assistant", "content": "".join(content) or None,
                  "reasoning": rtext + CUTOFF_TAIL + "".join(extra_r),
                  "tool_calls": [tools[i] for i in sorted(tools)] or None}
            d["choices"][0]["message"] = m2
            d["choices"][0]["finish_reason"] = finish or ("tool_calls" if tools else "stop")
            if d.get("usage") and usage2:
                for k in ("completion_tokens", "total_tokens"):
                    d["usage"][k] = (d["usage"].get(k) or 0) + (usage2.get(k) or 0)
        self._send_json(200, d)


if __name__ == "__main__":
    log(f"think-budget proxy :{PORT} -> {UPSTREAM}  budget={BUDGET_TOKENS} tok (~{BUDGET_CHARS} chars)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
