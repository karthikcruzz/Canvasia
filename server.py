import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from backend import ArtistBackend


HOST = "0.0.0.0"
PORT = 8001
PUBLIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}


backend = ArtistBackend()
chat_started = False


def read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if not length:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def image_url(path: str | None) -> str | None:
    if not path:
        return None
    return f"/generated/{Path(path).name}"


def serialize_state() -> dict:
    state = backend.to_dict()
    return {
        "chatStarted": chat_started,
        "conversationHistory": backend.conversation_history,
        "state": state,
        "summary": {
            "objects": {
                "order": state["object_contributions"],
                "human": state["human_objects"],
                "ai": state["ai_objects"],
            },
            "style": state["style"],
            "medium": state["medium"],
            "colorPalette": state["color_palette"],
            "layout": state["layout"],
            "composition": state["composition"],
            "livePrompt": state["live_prompt"],
        },
        "generatedImage": image_url(backend.generated_image_path),
        "finalPrompt": backend.final_prompt,
        "aestheticScore": backend.aesthetic_score_average,
        "aestheticScores": backend.aesthetic_scores,
        "lastError": backend.last_error,
    }


class CanvasiaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in PUBLIC_FILES:
            self.serve_file(Path(PUBLIC_FILES[parsed.path]))
            return
        if parsed.path == "/api/state":
            write_json(self, 200, serialize_state())
            return
        if parsed.path.startswith("/generated/"):
            self.serve_generated(parsed.path)
            return
        write_json(self, 404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/start":
                self.start()
                return
            if parsed.path == "/api/reset":
                self.reset()
                return
            if parsed.path == "/api/turn":
                self.turn()
                return
            if parsed.path == "/api/decide":
                self.decide()
                return
            if parsed.path == "/api/generate":
                self.generate()
                return
            if parsed.path in {"/api/sketch", "/api/edit-image"}:
                write_json(self, 200, serialize_state())
                return
            write_json(self, 404, {"error": "Not found"})
        except Exception as exc:
            write_json(self, 500, {"error": str(exc), **serialize_state()})

    def start(self):
        global chat_started
        payload = read_json(self)
        starter = payload.get("starter") or "Human"
        chat_started = True
        backend.start_conversation(starter)
        write_json(self, 200, serialize_state())

    def reset(self):
        global chat_started
        chat_started = False
        backend.reset()
        write_json(self, 200, serialize_state())

    def turn(self):
        payload = read_json(self)
        message = str(payload.get("message", "")).strip()
        if not message:
            write_json(self, 400, {"error": "Message cannot be empty"})
            return
        backend.process_turn(message)
        write_json(self, 200, serialize_state())

    def decide(self):
        backend.canvasia_decides()
        write_json(self, 200, serialize_state())

    def generate(self):
        if backend.state.stage != "Ready":
            write_json(self, 400, {"error": "Generate Image is available after Canvasia says the prompt is ready.", **serialize_state()})
            return
        backend.generate_painting()
        write_json(self, 200, serialize_state())

    def serve_file(self, path: Path):
        if not path.exists():
            write_json(self, 404, {"error": "File not found"})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_generated(self, request_path: str):
        filename = Path(unquote(request_path).replace("/generated/", "", 1)).name
        path = Path("logs") / filename
        if not path.exists() or not path.is_file():
            write_json(self, 404, {"error": "Image not found"})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "image/png"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    server = ThreadingHTTPServer((HOST, PORT), CanvasiaHandler)
    print(f"Canvasia is running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
