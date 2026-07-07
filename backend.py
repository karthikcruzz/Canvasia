import base64
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


STAGES = ["Objects", "Style", "Medium", "Color", "Layout", "Ready"]

OBJECT_ORDER = {
    "AI": ["canvasia", "human", "canvasia", "human", "canvasia", "human"],
    "Human": ["human", "canvasia", "human", "canvasia", "human", "canvasia"],
}

OBJECT_POOL = [
    "clockwork pomegranate",
    "velvet telescope",
    "ceramic rain boot",
    "brass jellyfish",
    "folded paper dragon",
    "neon teacup",
    "cracked porcelain mask",
    "floating seed pod",
    "embroidered compass",
    "crystal cassette tape",
    "moonlit greenhouse",
    "silver accordion",
    "glass octopus",
    "paper windmill",
    "striped umbrella",
    "copper violin",
    "marble suitcase",
    "glowing chess knight",
]

DECISION_FALLBACKS = {
    "Style": ["surreal realism", "loose watercolor", "dreamlike impressionism", "graphic folk art"],
    "Medium": ["oil on canvas", "ink and watercolor", "gouache on textured paper", "mixed-media collage"],
    "Color": ["deep teal, ember orange, pearl white", "muted violet, moss green, warm gold", "cobalt blue, blush pink, charcoal"],
    "Layout": [
        "a diagonal arrangement with the largest objects anchoring the lower left and smaller objects drifting upward",
        "a calm central cluster surrounded by smaller objects like orbiting thoughts",
        "a layered scene with the most familiar objects close to the viewer and surreal objects farther back",
    ],
}

AESTHETIC_CRITERIA = [
    "Color Harmony",
    "Visual Style Consistency",
    "Sharpness",
    "Light and Shadow Modeling",
    "Creativity and Originality",
    "Exposure Control",
    "Application of Classical Composition Principles",
    "Depth of Field and Layering",
    "Visual Center Stability",
    "Visual Flow Guidance",
    "Structural Support Stability",
    "Appropriateness of Negative Space",
    "Subject Integrity",
]


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
    layout: str | None = None
    composition: dict[str, list[str]] = field(
        default_factory=lambda: {"foreground": [], "midground": [], "background": []}
    )
    live_prompt: str = ""


class ArtistBackend:
    def __init__(self):
        load_dotenv()
        self.client = self._create_client()
        self.text_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.2")
        self.evaluation_model = (
            os.environ.get("AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME")
            or os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT_NAME")
            or self.text_model
        )
        self.image_model = os.environ.get("AZURE_OPENAI_IMAGE_DEPLOYMENT_NAME", "gpt-image-1.5")
        self.creative_temperature = float(os.environ.get("CANVASIA_CREATIVE_TEMPERATURE", "1.3"))
        self.prompt_temperature = float(os.environ.get("CANVASIA_PROMPT_TEMPERATURE", "1.1"))
        self.state = PaintingState()
        self.conversation_history: list[dict[str, str]] = []
        self.generated_image_path: str | None = None
        self.final_prompt: str | None = None
        self.aesthetic_scores: dict[str, float] = {}
        self.aesthetic_score_average: float | None = None
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

    def _chat_completion(
        self, *, messages: list[dict[str, Any]], response_format=None, temperature=None, model: str | None = None
    ):
        kwargs = {"model": model or self.text_model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if temperature is None or "temperature" not in str(exc).lower():
                raise
            kwargs.pop("temperature", None)
            self.last_error = f"Temperature unsupported by deployment; retried without it. Original error: {exc}"
            return self.client.chat.completions.create(**kwargs)

    def reset(self):
        self.state = PaintingState()
        self.conversation_history = []
        self.generated_image_path = None
        self.final_prompt = None
        self.aesthetic_scores = {}
        self.aesthetic_score_average = None
        self.last_error = None
        self.update_live_prompt()

    def start_conversation(self, starter: str):
        self.reset()
        self.state.starter = starter if starter in OBJECT_ORDER else "Human"

        if self.state.starter == "AI":
            canvasia_object = self._generate_canvasia_object()
            self._add_object("canvasia", canvasia_object)
            self.conversation_history.append({"role": "assistant", "content": canvasia_object})
        else:
            reply = (
                "Let's start with one concrete object from you. Pick anything visual, familiar or strange, "
                "and I'll build from it."
            )
            self.conversation_history.append({"role": "assistant", "content": reply})

        self.update_live_prompt()
        return self.conversation_history[-1]["content"] if self.conversation_history else ""

    def process_turn(self, user_message: str):
        clean_message = str(user_message or "").strip()
        if not clean_message:
            return ""

        if self.state.stage == "Objects":
            return self._process_object_turn(clean_message)

        if self.state.stage in {"Style", "Medium", "Color", "Layout"}:
            return self._process_choice_turn(clean_message)

        reply = "The prompt is ready - click **Generate Image** when you're ready."
        self.conversation_history.append({"role": "user", "content": clean_message})
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def canvasia_decides(self):
        if self.state.stage not in {"Style", "Medium", "Color", "Layout"}:
            return ""

        stage = self.state.stage
        value = self._decide_stage_value(stage)
        self._set_stage_value(stage, value)
        self._derive_stage()
        self._infer_composition_from_layout()
        self.update_live_prompt()
        reply = self._choice_reply(stage, value, decided_by_canvasia=True)
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def _process_object_turn(self, user_message: str):
        display_message = self._yes_and_user_message(user_message)
        human_object = self._strip_yes_and(user_message)
        self.conversation_history.append({"role": "user", "content": display_message})

        self._add_object("human", human_object)

        if len(self.state.object_contributions) < 6 and self._next_object_contributor() == "canvasia":
            canvasia_object = self._generate_canvasia_object()
            self._add_object("canvasia", canvasia_object)
            self.conversation_history.append({"role": "assistant", "content": f"yes and {canvasia_object}"})

        if len(self.state.object_contributions) >= 6:
            self._derive_stage()
            self.update_live_prompt()
            self.conversation_history.append({"role": "assistant", "content": "What style should guide the artwork?"})
        else:
            self.update_live_prompt()

        return self.conversation_history[-1]["content"]

    def _process_choice_turn(self, user_message: str):
        stage = self.state.stage
        self.conversation_history.append({"role": "user", "content": user_message})
        self._set_stage_value(stage, user_message)
        self._derive_stage()
        self._infer_composition_from_layout()
        self.update_live_prompt()
        reply = self._choice_reply(stage, user_message, decided_by_canvasia=False)
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def _choice_reply(self, stage: str, value: str, decided_by_canvasia: bool):
        prefix = "I'll choose" if decided_by_canvasia else "Got it"
        if stage == "Style":
            return f"{prefix}: **{value}**. What medium should carry it?"
        if stage == "Medium":
            return f"{prefix}: **{value}**. What color palette should shape it?"
        if stage == "Color":
            return f"{prefix}: **{value}**. Describe the layout in one go."
        return f"{prefix}: **{value}**. The prompt is ready - click **Generate Image** when you're ready."

    def _yes_and_user_message(self, user_message: str):
        clean = self._strip_yes_and(user_message)
        if self.state.object_contributions:
            return f"yes and {clean}"
        return clean

    def _strip_yes_and(self, value: str):
        text = str(value or "").strip()
        lowered = text.lower()
        for prefix in ("yes and ", "yes, and "):
            if lowered.startswith(prefix):
                return text[len(prefix):].strip()
        return text

    def _add_object(self, source: str, value: str):
        if len(self.state.object_contributions) >= 6:
            return
        clean_value = self._strip_yes_and(value)
        expected = self._next_object_contributor()
        source = source if source in {"human", "canvasia"} else expected
        if expected and source != expected:
            source = expected
        self.state.object_contributions.append({"source": source, "value": clean_value})
        self.state.human_objects = [
            item["value"] for item in self.state.object_contributions if item["source"] == "human"
        ]
        self.state.ai_objects = [
            item["value"] for item in self.state.object_contributions if item["source"] == "canvasia"
        ]

    def _next_object_contributor(self):
        order = OBJECT_ORDER.get(self.state.starter or "Human", OBJECT_ORDER["Human"])
        count = len(self.state.object_contributions)
        if count >= len(order):
            return None
        return order[count]

    def _generate_canvasia_object(self):
        existing = {item["value"].lower() for item in self.state.object_contributions}
        if self.client is None:
            return self._random_object(existing)

        prompt = f"""
Invent one concrete visual object for a yes-and painting game.
Return only JSON: {{"object": "short object noun phrase"}}

Rules:
- One object only.
- 1 to 4 words.
- Fresh, visual, and paintable.
- Do not use weathered lighthouse unless explicitly requested.
- Avoid these existing objects: {sorted(existing)}
"""
        try:
            response = self._chat_completion(
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.creative_temperature,
            )
            parsed = json.loads(response.choices[0].message.content)
            value = str(parsed.get("object", "")).strip()
            if value and value.lower() not in existing:
                return self._strip_yes_and(value)
        except Exception as exc:
            self.last_error = str(exc)
        return self._random_object(existing)

    def _random_object(self, existing: set[str]):
        choices = [item for item in OBJECT_POOL if item.lower() not in existing]
        return random.choice(choices or OBJECT_POOL)

    def _decide_stage_value(self, stage: str):
        if self.client is None:
            return random.choice(DECISION_FALLBACKS[stage])

        prompt = f"""
Choose the {stage.lower()} for this collaborative painting.
Return only JSON: {{"value": "choice"}}

Current objects:
{json.dumps(self.state.object_contributions, ensure_ascii=False)}

Current state:
{json.dumps(asdict(self.state), ensure_ascii=False)}

Rules:
- Be specific.
- Keep it short enough to display in one UI box.
- For Layout, write one concise sentence describing the full arrangement.
"""
        try:
            response = self._chat_completion(
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.creative_temperature,
            )
            parsed = json.loads(response.choices[0].message.content)
            value = str(parsed.get("value", "")).strip()
            return value or random.choice(DECISION_FALLBACKS[stage])
        except Exception as exc:
            self.last_error = str(exc)
            return random.choice(DECISION_FALLBACKS[stage])

    def _set_stage_value(self, stage: str, value: str):
        clean_value = str(value or "").strip()
        if stage == "Style":
            self.state.style = clean_value
        elif stage == "Medium":
            self.state.medium = clean_value
        elif stage == "Color":
            self.state.color_palette = clean_value
        elif stage == "Layout":
            self.state.layout = clean_value

    def _derive_stage(self):
        if len(self.state.object_contributions) < 6:
            self.state.stage = "Objects"
        elif not self.state.style:
            self.state.stage = "Style"
        elif not self.state.medium:
            self.state.stage = "Medium"
        elif not self.state.color_palette:
            self.state.stage = "Color"
        elif not self.state.layout:
            self.state.stage = "Layout"
        else:
            self.state.stage = "Ready"
        self.state.progress = STAGES.index(self.state.stage)

    def _infer_composition_from_layout(self):
        if not self.state.layout or self.client is None:
            return

        objects = [item["value"] for item in self.state.object_contributions]
        prompt = f"""
Infer foreground, midground, and background placement from this layout description.
Do not ask the user anything.
Return only JSON with keys foreground, midground, background, each a list of strings.

Objects: {objects}
Layout: {self.state.layout}
"""
        try:
            response = self._chat_completion(
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            parsed = json.loads(response.choices[0].message.content)
            self.state.composition = {
                "foreground": self._string_list(parsed.get("foreground")),
                "midground": self._string_list(parsed.get("midground")),
                "background": self._string_list(parsed.get("background")),
            }
        except Exception as exc:
            self.last_error = str(exc)

    def _string_list(self, value: Any):
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

        objects = [item["value"] for item in self.state.object_contributions]
        if objects:
            parts.append(f"featuring {', '.join(objects)}")
        if self.state.color_palette:
            parts.append(f"with a {self.state.color_palette} color palette")
        if self.state.layout:
            parts.append(f"arranged as {self.state.layout}")

        self.state.live_prompt = " ".join(parts).strip() + "."

    def generate_painting(self):
        if self.client is None:
            raise RuntimeError("Azure OpenAI client is not configured. Check your .env file.")

        final_prompt = self._build_image_prompt()
        image_res = self.client.images.generate(
            model=self.image_model,
            prompt=final_prompt,
            n=1,
            size=os.environ.get("CANVASIA_IMAGE_SIZE", "1024x1024"),
            quality=os.environ.get("CANVASIA_IMAGE_QUALITY", "high"),
        )

        img_b64 = image_res.data[0].b64_json
        if not img_b64:
            raise RuntimeError("Azure image generation returned no base64 image data.")
        img_data = base64.b64decode(img_b64)

        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = logs_dir / f"painting_{timestamp}.png"
        img_path.write_bytes(img_data)

        self.generated_image_path = str(img_path)
        self.final_prompt = final_prompt
        self.aesthetic_scores = {}
        self.aesthetic_score_average = None
        try:
            self._evaluate_aesthetic_score(img_b64)
        except Exception as exc:
            message = f"Aesthetic scoring failed: {exc}"
            self.last_error = f"{self.last_error}; {message}" if self.last_error else message
        self._write_log(timestamp)
        return self.generated_image_path, self.final_prompt

    def _build_image_prompt(self):
        if self.client is None:
            return self.state.live_prompt

        sys_msg = (
            "Turn the collaborative painting state into one rich image-generation prompt. "
            "Keep all six yes-and objects. Use the user-facing layout as the main composition guide. "
            "You may decide mood and lighting yourself. Output only the prompt."
        )
        try:
            response = self._chat_completion(
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": json.dumps(asdict(self.state), ensure_ascii=False)},
                ],
                temperature=self.prompt_temperature,
            )
            prompt = response.choices[0].message.content.strip()
            return prompt or self.state.live_prompt
        except Exception as exc:
            self.last_error = str(exc)
            return self.state.live_prompt

    def _evaluate_aesthetic_score(self, img_b64: str):
        criteria_lines = "\n".join(
            f"{index}. {criterion}" for index, criterion in enumerate(AESTHETIC_CRITERIA, start=1)
        )
        prompt = (
            "Evaluate the generated artwork image against exactly these 13 aesthetic criteria. "
            "Assign each criterion a numeric score from 0 to 10, where 0 is poor and 10 is excellent. "
            "Use the full 0-10 range when justified. Return only valid JSON in this exact shape: "
            '{"scores":[{"criterion":"Color Harmony","score":7.5}]}. '
            "Do not add markdown, prose, or extra keys.\n\n"
            f"Criteria:\n{criteria_lines}"
        )
        response = self._chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict visual art evaluator. Score only what is visible in the image, "
                        "and produce machine-readable JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            model=self.evaluation_model,
        )
        raw_content = response.choices[0].message.content or "{}"
        data = self._parse_json_object(raw_content)
        normalized_scores = self._normalize_aesthetic_scores(data)

        self.aesthetic_scores = normalized_scores
        self.aesthetic_score_average = round(sum(normalized_scores.values()) / len(AESTHETIC_CRITERIA), 2)

    def _parse_json_object(self, raw_content: str) -> dict[str, Any]:
        content = raw_content.strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
        return json.loads(content)

    def _normalize_aesthetic_scores(self, data: dict[str, Any]) -> dict[str, float]:
        scores_data = data.get("scores", data)
        score_lookup: dict[str, Any] = {}

        if isinstance(scores_data, list):
            for item in scores_data:
                if not isinstance(item, dict):
                    continue
                criterion = item.get("criterion")
                if criterion is not None and "score" in item:
                    score_lookup[self._aesthetic_score_key(str(criterion))] = item["score"]
        elif isinstance(scores_data, dict):
            score_lookup = {self._aesthetic_score_key(str(key)): value for key, value in scores_data.items()}

        normalized: dict[str, float] = {}
        missing = []
        for criterion in AESTHETIC_CRITERIA:
            value = score_lookup.get(self._aesthetic_score_key(criterion))
            if value is None:
                missing.append(criterion)
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                missing.append(criterion)
                continue
            normalized[criterion] = max(0.0, min(10.0, numeric))

        if missing:
            raise ValueError(f"Missing or invalid aesthetic scores for: {', '.join(missing)}")
        return normalized

    def _aesthetic_score_key(self, criterion: str) -> str:
        parts = criterion.lower().replace("#", " ").replace(".", " ").split()
        if parts and parts[0].isdigit():
            parts = parts[1:]
        return " ".join(parts)

    def _write_log(self, timestamp: str):
        log_data = {
            "timestamp": timestamp,
            "conversation": self.conversation_history,
            "state": asdict(self.state),
            "final_prompt": self.final_prompt,
            "image_path": self.generated_image_path,
            "aesthetic_scores": self.aesthetic_scores,
            "aesthetic_score_average": self.aesthetic_score_average,
        }
        Path("logs", f"log_{timestamp}.json").write_text(
            json.dumps(log_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def to_dict(self):
        return asdict(self.state)
