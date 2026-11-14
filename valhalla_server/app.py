"""HTTP API: pipeline jobs in, baked Valhalla models out."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from valhalla_server import internal, runtime
from valhalla_server.config import API_KEY, BASE_MODEL, HOST, PORT
from valhalla_server.inference import _materialize, generate
from valhalla_server.pipeline import job_status, list_models, start_pipeline

# Media ingest carries base64 wav/png items; the engine's own limit is
# 32MB, so the gateway admits the same (chat bodies stay tiny in practice).
MAX_BODY = 33_554_432


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def _body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY:
            self._json(400, {"error": "请求体为空或过大。"})
            return None
        try:
            incoming = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"error": "请求不是 JSON。"})
            return None
        if not isinstance(incoming, dict):
            self._json(400, {"error": "请求不是 JSON。"})
            return None
        return incoming

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/health":
            self._json(200, {"ok": True, "service": "valhalla-server"})
            return
        if not self._authorized():
            self._json(401, {"error": "API Key 不对。"})
            return
        if path == "/v1/runtime":
            self._json(200, runtime.status())
            return
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": list_models()})
            return
        if path == "/v1/decode-heads":
            from valhalla_server import head_server
            from valhalla_server.modelops import list_decode_heads

            self._json(200, {"object": "list", "data": list_decode_heads(),
                             "server": head_server.status()})
            return
        if path.startswith("/v1/jobs/"):
            try:
                self._json(200, job_status(path.rsplit("/", 1)[-1]))
            except KeyError:
                self._json(404, {"error": "没有这个任务。"})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if not self._authorized():
            self._json(401, {"error": "API Key 不对。"})
            return
        incoming = self._body()
        if incoming is None:
            return
        if path == "/v1/infer":
            # Raw valhalla-base-v1 passthrough to the internal runtime.
            try:
                folder = BASE_MODEL
                models = list_models()
                if models:
                    folder = str(_materialize(models[0]["id"]))
                if not internal.start(folder):
                    self._json(503, {"error": "ValhallaBase 运行时没有启动。"})
                    return
                self._json(200, internal.infer(incoming))
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"error": str(exc)[-300:]})
            return
        if path == "/v1/modelops":
            # Automated chain: decompose -> Fate -> fusion -> bake -> GGUF
            # -> quantize -> deploy decode head -> verify -> register.
            from valhalla_server.modelops import start_modelops

            head_id = incoming.get("model_id") if isinstance(incoming.get("model_id"), str) and incoming["model_id"].strip() else "mistral24-auto-v1"
            base = incoming.get("base_model") if isinstance(incoming.get("base_model"), str) else ""
            if not base:
                self._json(400, {"error": "base_model 必填（HF 模型目录）。"})
                return
            try:
                strength = float(incoming.get("strength", 0.08))
            except (TypeError, ValueError):
                self._json(400, {"error": "strength 需要是数字。"})
                return
            quant = incoming.get("quant") if isinstance(incoming.get("quant"), str) and incoming["quant"].strip() else "Q4_K_M"
            samples = incoming.get("samples") if isinstance(incoming.get("samples"), list) else []
            job = start_modelops(head_id.strip(), base.strip(), strength,
                                 quant.strip(), samples,
                                 bool(incoming.get("deploy", True)))
            self._json(202, job)
            return
        if path.startswith("/v1/decode-heads/") and path.endswith("/deploy"):
            from valhalla_server.modelops import deploy_decode_head

            head_id = path[len("/v1/decode-heads/"):-len("/deploy")].strip("/")
            try:
                self._json(200, deploy_decode_head(head_id))
            except KeyError:
                self._json(404, {"error": "没有这个解码头。先跑 /v1/modelops。"})
            return
        if path == "/v1/media/ingest":
            # Multimodal fix (official bridge contract): an item's text +
            # audio + image are CO-INGESTED into the same fragment; the
            # engine binds by co-occurrence, not cross-space similarity.
            # Media bytes are encoded by the repo's DL-front encoder into
            # the 384-d family the engine expects.
            import base64

            from valhalla_server import media

            model_id = incoming.get("model") if isinstance(incoming.get("model"), str) and incoming["model"].strip() else ""
            items_in = incoming.get("items") if isinstance(incoming.get("items"), list) else []
            if not model_id or not items_in:
                self._json(400, {"error": "model 与 items 必填。"})
                return
            try:
                folder = _materialize(model_id)
                if not internal.start(str(folder)):
                    self._json(503, {"error": "ValhallaBase 运行时没有启动。"})
                    return
                ctx_items = []
                for it in items_in:
                    if not isinstance(it, dict):
                        continue
                    item: dict = {}
                    if isinstance(it.get("text"), str) and it["text"].strip():
                        item["text"] = it["text"][:500]
                    af = it.get("audio_features")
                    if af is None and it.get("audio_b64"):
                        af = media.encode_bytes(base64.b64decode(it["audio_b64"]), "audio", ".wav")
                    if isinstance(af, list) and af:
                        item["audio_features"] = [float(x) for x in af]
                    imf = it.get("image_features")
                    if imf is None and it.get("image_b64"):
                        imf = media.encode_bytes(base64.b64decode(it["image_b64"]), "image", ".png")
                    if isinstance(imf, list) and imf:
                        item["image_features"] = [float(x) for x in imf]
                    if item:
                        ctx_items.append(item)
                cfg = internal.chat_config(str(folder))
                session = internal.session_for(model_id)
                if session:
                    payload = {"questions": [], "session_extend_only": True,
                               "session": session, "append_items": ctx_items,
                               "config": cfg}
                else:
                    payload = {"questions": [], "fate_warmup_only": True,
                               "context_items": ctx_items, "config": cfg}
                body = internal.infer(payload)
                internal.keep_session(model_id, body.get("session"))
                sess = body.get("session") or {}
                internal.keep_corpus(model_id, list(sess.get("corpus") or []))
                registry = internal.media_items(model_id) if session else []
                line_no = sum(1 for e in registry if e.get("line_index") is not None)
                for item in ctx_items:
                    entry = {"text": item.get("text"),
                             "audio_features": item.get("audio_features"),
                             "image_features": item.get("image_features"),
                             "line_index": None}
                    if item.get("text"):
                        entry["line_index"] = line_no
                        line_no += 1
                    registry.append(entry)
                internal.keep_media_items(model_id, registry)
                self._json(200, {"ok": True, "model": model_id,
                                 "items": len(ctx_items),
                                 "session_lines": len(sess.get("corpus") or []),
                                 "features": body.get("features")})
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"error": str(exc)[-300:]})
            return
        if path == "/v1/media/ask":
            import base64

            from valhalla_server import hybrid, media

            model_id = incoming.get("model") if isinstance(incoming.get("model"), str) and incoming["model"].strip() else ""
            question = incoming.get("question") if isinstance(incoming.get("question"), str) else ""
            if not model_id or not question:
                self._json(400, {"error": "model 与 question 必填。"})
                return
            try:
                folder = _materialize(model_id)
                if not internal.start(str(folder)):
                    self._json(503, {"error": "ValhallaBase 运行时没有启动。"})
                    return
                q: dict = {"id": "mq", "text": question[:2000], "score_type": "open"}
                af = incoming.get("audio_features")
                if af is None and incoming.get("audio_b64"):
                    af = media.encode_bytes(base64.b64decode(incoming["audio_b64"]), "audio", ".wav")
                if isinstance(af, list) and af:
                    q["audio_features"] = [float(x) for x in af]
                imf = incoming.get("image_features")
                if imf is None and incoming.get("image_b64"):
                    imf = media.encode_bytes(base64.b64decode(incoming["image_b64"]), "image", ".png")
                if isinstance(imf, list) and imf:
                    q["image_features"] = [float(x) for x in imf]
                # Feature grasp: the engine has no feature->line
                # retrieval (fusion forms, line anchor stays textual),
                # so a clear encoder-space winner supplies the evidence
                # line and the decode head answers from its caption.
                registry = internal.media_items(model_id)
                corpus_now = internal.corpus_lines(model_id)
                for modality in ("image", "audio"):
                    feats = q.get(f"{modality}_features")
                    if not feats:
                        continue
                    hit = media.grasp(registry, modality, feats)
                    if hit is None:
                        continue
                    item_no, score = hit
                    line_idx = registry[item_no].get("line_index")
                    if not isinstance(line_idx, int) or not (0 <= line_idx < len(corpus_now)):
                        continue
                    from valhalla_server import mistral_head

                    decoded = mistral_head.decode(
                        f"About the media item described in the evidence: {question}",
                        [corpus_now[line_idx]])
                    if decoded:
                        self._json(200, {"model": model_id, "answer": decoded,
                                         "mode": "feature_grasp",
                                         "grasp": {"item": item_no, "score": round(score, 4),
                                                   "line_index": line_idx},
                                         "fog_decision": None,
                                         "retrieval_corpus_idx": line_idx})
                        return
                session = internal.session_for(model_id)
                payload = {"questions": [q], "config": internal.chat_config(str(folder)),
                           "context": []}
                if session:
                    payload["session"] = session
                body = internal.infer(payload)
                internal.keep_session(model_id, body.get("session"))
                rows = body.get("rows") or []
                row = rows[0] if rows and isinstance(rows[0], dict) else {}
                fog = row.get("fog_decision")
                fog_kind = fog.get("decision", "") if isinstance(fog, dict) else ""
                answer, mode = hybrid.resolve(
                    fog_kind, question, internal.corpus_lines(model_id),
                    row.get("retrieval_corpus_idx"), str(row.get("answer", "")),
                    fog_decision=fog, fog_evidence=row.get("fog_evidence"))
                self._json(200, {"model": model_id, "answer": answer, "mode": mode,
                                 "fog_decision": fog,
                                 "retrieval_corpus_idx": row.get("retrieval_corpus_idx"),
                                 "native_answer": row.get("answer")})
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"error": str(exc)[-300:]})
            return
        if path == "/v1/pipeline":
            model_id = incoming.get("model_id") if isinstance(incoming.get("model_id"), str) and incoming["model_id"].strip() else "unified-core-v1"
            base = incoming.get("base_model") if isinstance(incoming.get("base_model"), str) else ""
            try:
                strength = float(incoming.get("strength", 0.08))
            except (TypeError, ValueError):
                self._json(400, {"error": "strength 需要是数字。"})
                return
            samples = incoming.get("samples") if isinstance(incoming.get("samples"), list) else []
            job = start_pipeline(model_id.strip(), base.strip(), strength, samples)
            self._json(202, job)
            return
        model_id = ""
        if path == "/v1/chat/completions":
            model_id = incoming.get("model") if isinstance(incoming.get("model"), str) else ""
        elif path.startswith("/v1/models/") and path.endswith("/chat/completions"):
            model_id = path[len("/v1/models/"):-len("/chat/completions")].strip("/")
        if not model_id:
            self.send_error(404)
            return
        messages = incoming.get("messages")
        if not isinstance(messages, list) or not messages:
            self._json(400, {"error": "messages 不能为空。"})
            return
        cleaned = []
        for item in messages:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                cleaned.append({"role": str(item.get("role", "user")), "content": item["content"]})
        if not cleaned:
            self._json(400, {"error": "messages 格式不对。"})
            return
        row = {}
        text = ""
        decode_path = ""
        fog_decision = None
        fate_telemetry = None
        try:
            from valhalla_server.pipeline import model_row

            row = model_row(model_id)
        except KeyError:
            self._json(404, {"error": "没有这个模型。先跑 /v1/pipeline。"})
            return
        # Newer pattern: ValhallaBase body first, baked LM as decode head.
        try:
            folder = _materialize(model_id)
            if internal.start(str(folder)):
                question = next((m["content"] for m in reversed(cleaned) if m["role"] == "user"), cleaned[-1]["content"])
                session = internal.session_for(model_id)
                # Session is keyed by model, but the client resends the
                # full history each request: fingerprint the context.
                # Same/appended history -> reuse session, ingest only the
                # delta. Different history -> drop the stale session so a
                # new conversation never silently answers from an old
                # corpus (Fate state follows the session).
                all_lines = [m["content"][:500] for m in cleaned[:-1]][-12:]
                ingested = internal.corpus_lines(model_id)
                if session and all_lines[:len(ingested)] == ingested:
                    context_lines = all_lines[len(ingested):]
                else:
                    if session:
                        internal.drop_session(model_id)
                        session = None
                    context_lines = all_lines
                payload = {
                    "questions": [{"id": "q0", "text": question[:2000], "score_type": "open"}],
                    "config": internal.chat_config(str(folder)),
                    "context": context_lines,
                }
                if session:
                    payload["session"] = session
                body = internal.infer(payload)
                rows = body.get("rows") or []
                if rows and isinstance(rows[0], dict):
                    text = str(rows[0].get("answer", "")).strip()
                    decode_path = str(rows[0].get("decode_path", ""))
                    # Fate is the authority on whether to speak: honor FOG.
                    fog = rows[0].get("fog_decision")
                    fog_decision = fog
                    fog_kind = fog.get("decision", "") if isinstance(fog, dict) else ""
                    # Hybrid resolution: Fate verdict + multi-track evidence +
                    # Mistral-24B decode (weak-emit calibration on abstain).
                    from valhalla_server import hybrid

                    resolved, mode = hybrid.resolve(
                        fog_kind, question, internal.corpus_lines(model_id),
                        rows[0].get("retrieval_corpus_idx"), text,
                        fog_decision=fog, fog_evidence=rows[0].get("fog_evidence"))
                    text = resolved
                    decode_path = f"{decode_path}/{mode}"
                internal.keep_session(model_id, body.get("session"))
                if all_lines:
                    internal.keep_corpus(model_id, all_lines)
                # Pipeline -> bundle: once a session accumulates enough
                # seg fragments, link them into chains (fate_link_only).
                # Measured: fragments -87% at group 8, wire -20%, QA flat.
                sess_now = internal.session_for(model_id)
                frags = ((sess_now or {}).get("snapshot") or {}).get("fragments") or []
                if len(frags) >= internal.LINK_MIN_FRAGMENTS:
                    linked = internal.link_session(
                        sess_now, internal.chat_config(str(folder)))
                    if linked:
                        internal.keep_session(model_id, linked)
                fate_telemetry = body.get("fate_telemetry")
        except Exception:
            text = ""
        if not text:
            try:
                text, row = generate(model_id, cleaned, incoming.get("max_tokens", 256), float(incoming.get("temperature", 0.3)))
                decode_path = "transformers_fallback"
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"error": str(exc)[-300:]})
                return
        self._json(200, {
            "id": f"chatcmpl-{model_id}",
            "object": "chat.completion",
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "valhalla": {
                "format": "valenc1",
                "protocol": "valhalla-base-v1",
                "deployment": "valhalla-unified-core-v1",
                "endpoint": row.get("endpoint"),
                "decode_path": decode_path,
                "fog_decision": fog_decision,
                "fate_telemetry": fate_telemetry,
                "decode_head": "mistral-24b" if "/mistral" in decode_path else "native",
                "model_fate": (row.get("manifest") or {}).get("model_fate"),
                "free_multimodal": (row.get("manifest") or {}).get("free_multimodal"),
            },
        })


def serve() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"valhalla-server http://{HOST}:{PORT}  models=/v1/models  pipeline=POST /v1/pipeline")
    models = list_models()
    if models:
        try:
            folder = _materialize(models[0]["id"])
            ok = internal.start(str(folder))
            print(f"ValhallaBase internal runtime: {'up' if ok else 'unavailable'} (LM head: {models[0]['id']})")
        except Exception as exc:  # noqa: BLE001
            print(f"ValhallaBase internal runtime unavailable: {exc}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()

