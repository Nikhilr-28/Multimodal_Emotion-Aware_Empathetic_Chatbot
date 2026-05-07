"""
inference.py
Multimodal Emotion-Aware Empathetic Chatbot — Inference Pipeline

Dialogue conditioned on Clara Hill's 3-stage empathy framework:
    Stage 1 — Exploration  (turns 1-3) : reflect, validate, open questions
    Stage 2 — Insight      (turns 4-6) : reframe, connect, deeper understanding
    Stage 3 — Action       (turns 7+)  : gentle encouragement, forward-looking

Modes:
    Text only       : python inference.py --text "I feel overwhelmed lately"
    Image only      : python inference.py --image path/to/face.jpg
    Text + Image    : python inference.py --text "I feel overwhelmed" --image face.jpg
    Interactive     : python inference.py --interactive

Commands in interactive mode:
    reset           : clear conversation history and start fresh
    quit / exit / q : exit the program
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dotenv import load_dotenv
load_dotenv()

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from groq import Groq

from preprocess import EMOTION_LABELS, EMOTION_TO_IDX, NUM_CLASSES


# ── Configuration ──────────────────────────────────────────────────────────────

GROQ_API_KEY             = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL               = "openai/gpt-oss-120b"
CKPT_DIR                 = os.path.join(os.path.dirname(__file__), "checkpoints")
VISION_MODEL             = "convnext_base"
VISION_TIMM              = "convnext_base.fb_in22k_ft_in1k"
FUSION_WEIGHT            = 0.6
IMAGE_SIZE               = 224
RAG_K                    = 3
RAG_INDEX_DIR            = os.path.join(os.path.dirname(__file__), "vector_store")
TRAIN_PATH               = os.path.join(os.path.dirname(__file__), "empchat_train_examples.json")
MAX_HISTORY_TURNS        = 10
EMOTION_UPDATE_THRESHOLD = 0.75     # single-turn confidence required to update immediately
EVIDENCE_TURNS_REQUIRED  = 2        # consecutive low-conf turns needed to confirm a shift
NEGATIVE_EMOTIONS        = {"sadness", "anger", "fear"}


# ── Clara Hill Stage Prompts ───────────────────────────────────────────────────

STAGE_PROMPTS = {
    "exploration": """You are a warm, emotionally intelligent conversational companion.
You are in the Exploration stage — your goal is to understand what the person is feeling.

Guidelines:
- Reflect the user's emotion back to them with genuine warmth
- Ask one open, caring follow-up question to understand more
- Validate that their feelings make complete sense
- Do not give advice or jump to solutions
- Keep it brief and human — 2 sentences maximum
""",
    "insight": """You are a warm, emotionally intelligent conversational companion.
You are in the Insight stage — your goal is to help the person understand their feelings more deeply.

Guidelines:
- Connect what they have shared across the conversation — show you have been listening
- Offer a gentle observation or reframe that helps them see their situation more clearly
- Validate their experience without minimising it
- You may skip the follow-up question if a reassuring statement feels more natural
- Do not repeat "I'm sorry" — empathy is not just apology
- Keep it brief and human — 2 sentences maximum
""",
    "action": """You are a warm, emotionally intelligent conversational companion.
You are in the Action stage — your goal is to gently encourage the person forward.

Guidelines:
- Acknowledge how much they have shared and how understandable their situation is
- Offer one small, gentle forward-looking thought or encouragement
- Frame it as a possibility, never a prescription or direct advice
- Be warm and hopeful without being dismissive of their struggle
- Keep it brief and human — 2 sentences maximum
"""
}


# ── Emotion label maps ─────────────────────────────────────────────────────────

EMOTION_MAP = {
    "admiration": "happiness", "amusement": "happiness", "approval": "happiness",
    "desire": "happiness",     "excitement": "happiness", "gratitude": "happiness",
    "joy": "happiness",        "love": "happiness",       "optimism": "happiness",
    "pride": "happiness",      "relief": "happiness",
    "sadness": "sadness",      "grief": "sadness",        "disappointment": "sadness",
    "embarrassment": "sadness","remorse": "sadness",
    "anger": "anger",          "annoyance": "anger",      "disapproval": "anger",
    "fear": "fear",            "nervousness": "fear",
    "caring": None,            "confusion": None,         "curiosity": None,
    "realization": None,       "neutral": None,           "disgust": None,
    "surprise": None,
}


# ── Transform ──────────────────────────────────────────────────────────────────

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE + 32),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_vision_model(device):
    ckpt_path = os.path.join(CKPT_DIR, f"{VISION_MODEL}_best.pth")
    model = timm.create_model(VISION_TIMM, pretrained=False,
                               num_classes=NUM_CLASSES, drop_rate=0.3)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    print(f"[Vision]  Loaded {VISION_MODEL} "
          f"(val_acc={checkpoint['val_acc']:.2f}% @ epoch {checkpoint['epoch']})")
    return model


def load_text_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name      = "SamLowe/roberta-base-go_emotions"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    print("[Text]    Loaded RoBERTa GoEmotions")
    return tokenizer, model


def load_rag_retriever():
    if not os.path.exists(TRAIN_PATH):
        print("[RAG]     empchat_train_examples.json not found — RAG disabled")
        return None
    from rag_retriever import EmotionFaissRetriever
    retriever = EmotionFaissRetriever(
        train_examples_path=TRAIN_PATH,
        index_dir=RAG_INDEX_DIR,
    )
    print("[RAG]     Retriever ready")
    return retriever


# ── Emotion prediction ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict_image_emotion(model, image_path, device):
    """Returns softmax probability array [happiness, sadness, anger, fear]."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Vision]  Could not open image: {e}")
        return None
    tensor = TEST_TRANSFORM(image)
    tensor = tensor.unsqueeze(0).to(device)  # type: ignore[union-attr]
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        logits = model(tensor)
    probs = F.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
    label = EMOTION_LABELS[int(np.argmax(probs))]
    conf  = float(np.max(probs))
    print(f"[Vision]  Detected: {label} (conf={conf:.2f})")
    return probs


def predict_text_emotion(tokenizer, model, text):
    """Returns softmax probability array [happiness, sadness, anger, fear]."""
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs_raw = torch.softmax(logits, dim=-1).numpy()

    grouped = {e: 0.0 for e in EMOTION_LABELS.values()}
    for idx, prob in enumerate(probs_raw):
        raw_label    = model.config.id2label[idx].lower()
        mapped_label = EMOTION_MAP.get(raw_label)
        if mapped_label and mapped_label in grouped:
            grouped[mapped_label] += float(prob)

    total = sum(grouped.values()) or 1.0
    probs = np.array([grouped[EMOTION_LABELS[i]] / total
                      for i in range(NUM_CLASSES)], dtype=np.float32)
    label = EMOTION_LABELS[int(np.argmax(probs))]
    conf  = float(np.max(probs))
    print(f"[Text]    Detected: {label} (conf={conf:.2f})")
    return probs


def fuse_emotions(image_probs, text_probs):
    """Weighted fusion calibrated at x=0.6 image, 0.4 text."""
    if image_probs is None and text_probs is None:
        return None, "happiness"
    if image_probs is None:
        fused = text_probs
    elif text_probs is None:
        fused = image_probs
    else:
        fused = FUSION_WEIGHT * image_probs + (1 - FUSION_WEIGHT) * text_probs
    label = EMOTION_LABELS[int(np.argmax(fused))]
    print(f"[Fusion]  Final emotion: {label}")
    return fused, label


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_message(emotion, text, retrieved=None):
    parts = []
    if retrieved:
        lines = [
            "Retrieved examples from similar conversations:",
            "Use these as style guidance for empathy and follow-up quality.",
        ]
        for i, ex in enumerate(retrieved, 1):
            lines += [
                f"Example {i}:",
                f"- Emotion: {ex.get('emotion', '')}",
                f"- Context: {ex.get('context', '')}",
                f"- Ideal response: {ex.get('target', '')}",
            ]
        parts.append("\n".join(lines))
        parts.append("Now respond to the current user input.")
    parts.append(f"[Emotion: {emotion}]")
    parts.append(text)
    return "\n\n".join(parts)


# ── Main pipeline ──────────────────────────────────────────────────────────────

class EmpatheticPipeline:
    def __init__(self, api_key):
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Init]    Device: {self.device}")
        self.vision_model = load_vision_model(self.device)
        self.text_tokenizer, self.text_model = load_text_model()
        self.retriever    = load_rag_retriever()
        self.groq_client  = Groq(api_key=api_key)
        self.history           = []
        self.last_emotion      = None
        self.pending_emotion   = None   # emotion candidate building evidence
        self.pending_count     = 0      # consecutive turns it has appeared
        print("[Init]    Pipeline ready\n")

    def reset(self):
        self.history           = []
        self.last_emotion      = None
        self.pending_emotion   = None
        self.pending_count     = 0
        print("[History] Conversation cleared.\n")

    def _get_hill_stage(self):
        turn = len(self.history) // 2
        if turn <= 2:
            return "exploration"
        elif turn <= 5:
            return "insight"
        else:
            return "action"

    def _resolve_session_emotion(self, image_probs, text_probs, is_first_turn):
        """
        Session emotion logic:
          - First turn or image provided : detect fresh via full fusion
          - High confidence (>= threshold) : update immediately if not a suspicious flip
          - Low confidence, same candidate 2 turns in a row : accumulate and confirm
          - Otherwise : hold current session emotion
        """
        if is_first_turn or image_probs is not None:
            self.pending_emotion = None
            self.pending_count   = 0
            _, emotion = fuse_emotions(image_probs, text_probs)
            return emotion

        if text_probs is None:
            return self.last_emotion or "sadness"

        conf      = float(np.max(text_probs))
        new_label = EMOTION_LABELS[int(np.argmax(text_probs))]

        # ── High confidence path ───────────────────────────────────────────────
        if conf >= EMOTION_UPDATE_THRESHOLD:

            if new_label == self.last_emotion:
                return self.last_emotion

            # Allow genuine resolution after 2 consecutive high-confidence happiness detections
            if self.last_emotion in NEGATIVE_EMOTIONS and new_label == "happiness":
                if self.pending_emotion == "happiness":
                    self.pending_count += 1
                else:
                    self.pending_emotion = "happiness"
                    self.pending_count   = 1

                if self.pending_count >= 2:
                    print(f"[Emotion] Genuine resolution confirmed — updating: "
                        f"{self.last_emotion} → happiness")
                    self.pending_emotion = None
                    self.pending_count   = 0
                    return "happiness"

                print(f"[Emotion] Suppressed positive flip "
                    f"({self.last_emotion} → {new_label}) — "
                    f"evidence {self.pending_count}/2 — holding: {self.last_emotion}")
                return self.last_emotion

            print(f"[Emotion] State updated: {self.last_emotion} → {new_label}")
            self.pending_emotion = None
            self.pending_count   = 0
            return new_label

        # ── Low confidence path — accumulate evidence ─────────────────────────
        if new_label == self.pending_emotion:
            self.pending_count += 1
        else:
            self.pending_emotion = new_label
            self.pending_count   = 1

        if (self.pending_count >= EVIDENCE_TURNS_REQUIRED
                and new_label != self.last_emotion):
            # Suppress positive flip even via accumulation
            if self.last_emotion in NEGATIVE_EMOTIONS and new_label == "happiness":
                print(f"[Emotion] Accumulated evidence suppressed "
                      f"({self.last_emotion} → {new_label}) — holding: {self.last_emotion}")
                self.pending_emotion = None
                self.pending_count   = 0
                return self.last_emotion

            print(f"[Emotion] Accumulated evidence — updating: "
                  f"{self.last_emotion} → {new_label}")
            self.pending_emotion = None
            self.pending_count   = 0
            return new_label

        print(f"[Emotion] Low confidence ({conf:.2f}) — "
              f"evidence {self.pending_count}/{EVIDENCE_TURNS_REQUIRED} "
              f"for {new_label} — holding: {self.last_emotion}")
        return self.last_emotion

    def _generate_response(self, emotion, text, is_first_turn):
        retrieved = []
        if is_first_turn and self.retriever and text:
            try:
                retrieved = self.retriever.retrieve_examples(
                    query=text, emotion=emotion, k=RAG_K
                )
            except Exception:
                pass

        user_message = build_user_message(emotion, text, retrieved)
        self.history.append({"role": "user", "content": user_message})

        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        stage      = self._get_hill_stage()
        system_msg = STAGE_PROMPTS[stage]
        print(f"[Stage]   Clara Hill — {stage}")

        completion = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_msg}] + self.history,  # type: ignore[arg-type]
            temperature=1,
            max_completion_tokens=300,
            top_p=1,
            reasoning_effort="medium",  # type: ignore[arg-type]
            stream=True,
            stop=None,
        )

        print("\n[Response]")
        response_text = ""
        try:
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                response_text += content
        except Exception as e:
            print(f"\n[Warning] Stream interrupted — retrying with fallback model...")
            try:
                fallback = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": system_msg}] + self.history,  # type: ignore[arg-type]
                    temperature=1,
                    max_completion_tokens=300,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
                response_text = fallback.choices[0].message.content or ""
                print(response_text)
            except Exception as e2:
                print(f"[Error] Retry failed: {e2}")
                response_text = "I'm here with you — could you tell me more about what you're feeling?"
        print("\n")

        self.history.append({"role": "assistant", "content": response_text})
        return response_text

    def run(self, text=None, image_path=None):
        if text is None and image_path is None:
            print("[Error]   Provide at least text or image input.")
            return None

        is_first_turn = len(self.history) == 0

        image_probs = predict_image_emotion(
            self.vision_model, image_path, self.device
        ) if image_path else None

        text_probs = predict_text_emotion(
            self.text_tokenizer, self.text_model, text
        ) if text else None

        emotion           = self._resolve_session_emotion(image_probs, text_probs, is_first_turn)
        self.last_emotion = emotion

        input_text = text or "..."
        return self._generate_response(emotion, input_text, is_first_turn)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Emotion-Aware Empathetic Chatbot"
    )
    parser.add_argument("--text",        type=str, help="User text input")
    parser.add_argument("--image",       type=str, help="Path to face image")
    parser.add_argument("--api-key",     type=str, default=GROQ_API_KEY,
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive loop mode")
    args = parser.parse_args()

    if not args.api_key:
        print("[Error]   No Groq API key provided. "
              "Use --api-key or set GROQ_API_KEY environment variable.")
        return

    pipeline = EmpatheticPipeline(api_key=args.api_key)

    if args.interactive:
        print("=== Interactive Mode ===")
        print("Commands: 'reset' to clear history, 'quit' to exit\n")
        while True:
            text = input("You: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text.lower() == "reset":
                pipeline.reset()
                continue
            image_path = input("Image path (press Enter to skip): ").strip() or None
            pipeline.run(text=text, image_path=image_path)
    else:
        pipeline.run(text=args.text, image_path=args.image)


if __name__ == "__main__":
    main()