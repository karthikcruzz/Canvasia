import base64
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from openai import AzureOpenAI
except ImportError:  # Keeps the app importable before dependencies are installed.
    AzureOpenAI = None


STAGES = [
    "Objects",
    "Style",
    "Medium",
    "Color Palette",
    "Emotion",
    "Lighting",
    "Composition",
    "Ready",
]

OBJECT_ORDER = {
    "AI": ["canvasia", "human", "canvasia", "human"],
    "Human": ["human", "canvasia", "human", "canvasia"],
}


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class PaintingState:
    stage: str = "Objects"
    progress: int = 0
    starter: str | None = None
    object_contributions: list[dict[str, str]] = field(default_factory=list)
    human_objects: list[str] = field(default_factory=list)
    ai_objects: list[str] = field(default_factory=list)
    style: str | None = None
    medium: str | None = None
    color_palette: str | None = None
    emotion: str | None = None
    lighting: str | None = None
    composition: dict[str, list[str]] = field(
        default_factory=lambda: {"foreground": [], "midground": [], "background": []}
    )
    live_prompt: str = ""


class ArtistBackend:
    def __init__(self):
        load_dotenv()
        self.client = self._create_client()
        self.text_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2")
        self.image_model = os.environ.get("AZURE_OPENAI_IMAGE_DEPLOYMENT_NAME", "gpt-image-1.5")
        self.state = PaintingState()
        self.conversation_history: list[dict[str, str]] = []
        self.generated_image_path: str | None = None
        self.final_prompt: str | None = None
        self.last_error: str | None = None
        self.update_live_prompt()

    def _create_client(self):
        if AzureOpenAI is None:
            return None

        api_key = os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            return None

        return AzureOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            api_key=api_key,
            azure_endpoint=endpoint,
        )

    def reset(self):
        self.state = PaintingState()
        self.conversation_history = []
        self.generated_image_path = None
        self.final_prompt = None
        self.last_error = None
        self.update_live_prompt()

    def start_conversation(self, starter: str):
        self.reset()
        self.state.starter = starter if starter in OBJECT_ORDER else "Human"

        if self.state.starter == "AI":
            parsed = self._ask_canvasia(
                extra_instruction=(
                    "Canvasia is starting. Add exactly one vivid object as Canvasia's first idea, "
                    "then ask the human for their first object."
                )
            )
            self._handle_canvasia_response(parsed)
            self._generate_preview_if_ready()
            return self.conversation_history[-1]["content"] if self.conversation_history else ""

        reply = (
            "Let's start with one concrete object from you. Pick anything visual, familiar or strange, "
            "and I'll build from it."
        )
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def process_turn(self, user_message: str):
        self.conversation_history.append({"role": "user", "content": user_message})
        parsed = self._ask_canvasia()
        reply = self._handle_canvasia_response(parsed)
        self._generate_preview_if_ready()
        return reply

    def _ask_canvasia(self, extra_instruction: str | None = None) -> dict[str, Any]:
        if self.client is None:
            return self._fallback_response()

        messages = [{"role": "system", "content": self._build_system_prompt(extra_instruction)}]
        messages.extend(self.conversation_history[-12:])

        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            self.last_error = str(exc)
            return self._fallback_response()

    def _build_system_prompt(self, extra_instruction: str | None = None) -> str:
        object_order = OBJECT_ORDER.get(self.state.starter or "Human", OBJECT_ORDER["Human"])
        next_contributor = self._next_object_contributor(object_order)
        state_json = json.dumps(asdict(self.state), ensure_ascii=False)

        base = f"""
You are Canvasia, a warm AI artist brainstorming a painting with a human.
Your job is to keep the conversation natural, focused on building one artwork, and gently useful.

Tone and guardrails:
- Sound like a collaborative studio partner, not a form or checklist.
- Keep replies to one or two short sentences.
- Ask only one question at a time.
- If the user is unsure, offer two or three concrete art directions and help them choose.
- If the user drifts away from the artwork, briefly acknowledge and guide them back to the painting.
- Do not mention internal stages, rules, schemas, JSON, or extraction.
- Do not use keyword matching language or robotic if/then phrasing.
- Use **bold** only for important visual choices.

Object brainstorming flow:
- Required object contribution order: {object_order}.
- Current next object contributor: {next_contributor or "none"}.
- For object turns, preserve the exact order of object_contributions.
- If Canvasia is the next contributor, invent exactly one concrete visual object and add it.
- If the human is the next contributor and they gave a usable object, add it.
- If the human is the next contributor and they seem unsure, guide them without adding a fake human object.
- After four total object contributions, move naturally into style, then medium, color palette, emotion, lighting, and composition.

Current state:
{state_json}

Return strict JSON with exactly:
{{
  "reply": "message to show the user",
  "state": {{
    "object_contributions": [{{"source": "human|canvasia", "value": "object name"}}],
    "style": null|string,
    "medium": null|string,
    "color_palette": null|string,
    "emotion": null|string,
    "lighting": null|string,
    "composition": {{"foreground": [], "midground": [], "background": []}}
  }}
}}
"""
        if extra_instruction:
            base += f"\nSpecial instruction for this turn: {extra_instruction}\n"
        return base

    def _fallback_response(self) -> dict[str, Any]:
        object_order = OBJECT_ORDER.get(self.state.starter or "Human", OBJECT_ORDER["Human"])
        next_contributor = self._next_object_contributor(object_order)
        state = self._state_payload()

        if next_contributor == "canvasia":
            value = "glass lighthouse" if not self.state.object_contributions else "paper comet"
            state["object_contributions"] = self.state.object_contributions + [
                {"source": "canvasia", "value": value}
            ]
            reply = f"I'll add a **{value}** to the canvas. What object should you add next?"
        elif next_contributor == "human":
            reply = "Give me one concrete object for the painting; ordinary or surreal both work."
        else:
            reply = "Let's keep shaping the painting with one clear visual choice at a time."

        return {"reply": reply, "state": state}

    def _handle_canvasia_response(self, parsed: dict[str, Any]):
        reply = str(parsed.get("reply") or "Let's keep shaping the painting together.")
        state_update = parsed.get("state") or {}
        self._apply_state_update(state_update)
        self.update_live_prompt()
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def _apply_state_update(self, update: dict[str, Any]) -> None:
        contributions = update.get("object_contributions")
        if isinstance(contributions, list):
            self.state.object_contributions = self._normalize_contributions(contributions)

        self.state.human_objects = [
            item["value"] for item in self.state.object_contributions if item["source"] == "human"
        ][:2]
        self.state.ai_objects = [
            item["value"] for item in self.state.object_contributions if item["source"] == "canvasia"
        ][:2]

        for attr in ("style", "medium", "color_palette", "emotion", "lighting"):
            value = update.get(attr)
            if isinstance(value, str) and value.strip():
                setattr(self.state, attr, value.strip())

        composition = update.get("composition")
        if isinstance(composition, dict):
            self.state.composition = {
                "foreground": self._string_list(composition.get("foreground")),
                "midground": self._string_list(composition.get("midground")),
                "background": self._string_list(composition.get("background")),
            }

        self._derive_stage()

    def _normalize_contributions(self, contributions: list[Any]) -> list[dict[str, str]]:
        normalized = []
        required_order = OBJECT_ORDER.get(self.state.starter or "Human", OBJECT_ORDER["Human"])

        for item in contributions:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip().lower()
            if source in {"ai", "assistant", "bot"}:
                source = "canvasia"
            if source not in {"human", "canvasia"}:
                continue
            value = str(item.get("value", "")).strip()
            if value:
                normalized.append({"source": source, "value": value})

        ordered = []
        for index, expected_source in enumerate(required_order):
            if index >= len(normalized):
                break
            item = normalized[index]
            if item["source"] != expected_source:
                item = {"source": expected_source, "value": item["value"]}
            ordered.append(item)
        return ordered[:4]

    def _derive_stage(self) -> None:
        if len(self.state.object_contributions) < 4:
            self.state.stage = "Objects"
        elif not self.state.style:
            self.state.stage = "Style"
        elif not self.state.medium:
            self.state.stage = "Medium"
        elif not self.state.color_palette:
            self.state.stage = "Color Palette"
        elif not self.state.emotion:
            self.state.stage = "Emotion"
        elif not self.state.lighting:
            self.state.stage = "Lighting"
        elif not self._composition_complete():
            self.state.stage = "Composition"
        else:
            self.state.stage = "Ready"
        self.state.progress = STAGES.index(self.state.stage)

    def _composition_complete(self) -> bool:
        placed = []
        for values in self.state.composition.values():
            placed.extend(values)
        object_values = [item["value"] for item in self.state.object_contributions]
        return bool(object_values) and all(value in placed for value in object_values)

    def _next_object_contributor(self, object_order: list[str]) -> str | None:
        count = len(self.state.object_contributions)
        if count >= len(object_order):
            return None
        return object_order[count]

    def _state_payload(self) -> dict[str, Any]:
        return {
            "object_contributions": self.state.object_contributions,
            "style": self.state.style,
            "medium": self.state.medium,
            "color_palette": self.state.color_palette,
            "emotion": self.state.emotion,
            "lighting": self.state.lighting,
            "composition": self.state.composition,
        }

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def update_live_prompt(self):
        parts = []
        if self.state.style and self.state.medium:
            parts.append(f"A {self.state.style} painting on {self.state.medium}")
        elif self.state.style:
            parts.append(f"A {self.state.style} painting")
        elif self.state.medium:
            parts.append(f"A painting on {self.state.medium}")
        else:
            parts.append("A painting")

        comp = self.state.composition
        if any(comp.values()):
            composition_parts = []
            if comp.get("foreground"):
                composition_parts.append(f"{', '.join(comp['foreground'])} in the foreground")
            if comp.get("midground"):
                composition_parts.append(f"{', '.join(comp['midground'])} in the midground")
            if comp.get("background"):
                composition_parts.append(f"{', '.join(comp['background'])} in the background")
            parts.append("depicting " + ", ".join(composition_parts))
        elif self.state.object_contributions:
            objects = [item["value"] for item in self.state.object_contributions]
            parts.append(f"featuring {', '.join(objects)}")

        if self.state.color_palette:
            parts.append(f"with a {self.state.color_palette} color palette")
        if self.state.emotion:
            parts.append(f"evoking {self.state.emotion}")
        if self.state.lighting:
            parts.append(f"under {self.state.lighting} lighting")

        self.state.live_prompt = " ".join(parts).strip() + "."

    def _has_visual_seed(self) -> bool:
        return bool(self.state.object_contributions or self.state.style or self.state.medium)

    def _generate_preview_if_ready(self) -> None:
        if self._has_visual_seed():
            try:
                self.generate_painting()
            except Exception as exc:
                self.last_error = str(exc)

    def generate_painting(self):
        if self.client is None:
            raise RuntimeError("Azure OpenAI client is not configured. Check your .env file.")

        final_prompt = self._build_image_prompt()
        image_res = self.client.images.generate(
            model=self.image_model,
            prompt=final_prompt,
            n=1,
            size=os.environ.get("CANVASIA_IMAGE_SIZE", "1024x1024"),
            response_format="b64_json",
        )
        img_b64 = image_res.data[0].b64_json
        img_data = base64.b64decode(img_b64)

        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = logs_dir / f"painting_{timestamp}.png"
        img_path.write_bytes(img_data)

        self.generated_image_path = str(img_path)
        self.final_prompt = final_prompt
        self._write_log(timestamp)
        return self.generated_image_path, self.final_prompt

    def _build_image_prompt(self) -> str:
        if self.client is None:
            return self.state.live_prompt

        sys_msg = (
            "Turn the current collaborative painting state into one rich image-generation prompt. "
            "Keep it faithful to the selected objects and visual choices. Output only the prompt."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": json.dumps(asdict(self.state), ensure_ascii=False)},
                ],
            )
            prompt = response.choices[0].message.content.strip()
            return prompt or self.state.live_prompt
        except Exception as exc:
            self.last_error = str(exc)
            return self.state.live_prompt

    def _write_log(self, timestamp: str) -> None:
        log_data = {
            "timestamp": timestamp,
            "conversation": self.conversation_history,
            "state": asdict(self.state),
            "final_prompt": self.final_prompt,
            "image_path": self.generated_image_path,
        }
        Path("logs", f"log_{timestamp}.json").write_text(
            json.dumps(log_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.state)
