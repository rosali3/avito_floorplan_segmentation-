"""
Общий шаг детекции боксов через GroundingDINO по текстовым промптам классов —
используется и zero_shot_grounded_sam.py (vanilla SAM), и
infer_and_eval_finetuned.py (дообученный decoder), чтобы разница между ними
была ИСКЛЮЧИТЕЛЬНО в качестве маски, а не в разных боксах (иначе сравнение
"zero-shot vs finetuned" было бы нечестным).

Ванильный SAM НЕ умеет принимать текстовые промпты/названия классов — поэтому
для zero-shot class-aware instance segmentation обязательно нужен
детектор-по-тексту (GroundingDINO) впереди SAM (см. обсуждение в progress.md).

Установка GroundingDINO (см. models/sam/SETUP.md для деталей):
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO && pip install -e .
    # веса + конфиг:
    #   GroundingDINO_SwinT_OGC.py            (конфиг, лежит в репозитории)
    #   groundingdino_swint_ogc.pth           (веса, скачать с релизов репозитория)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Синонимы расширяют текстовый промпт и дают больше шансов, что GroundingDINO
# найдёт объект, даже если его внутренний токенайзер предпочитает другое слово
# для того же понятия. canonical class остаётся одним из 7 из classes.yaml.
CLASS_SYNONYMS = {
    "living": ["living room", "living"],
    "bedroom": ["bedroom"],
    "bathroom": ["bathroom", "restroom", "toilet", "wc", "washroom"],
    "kitchen": ["kitchen"],
    "balcony": ["balcony", "loggia"],
    "wall": ["wall"],
    "opening": ["door", "window", "entrance", "opening"],
}


def build_text_prompt(canonical_class_names: list[str]) -> tuple[str, dict[str, str]]:
    """Возвращает (text_prompt, synonym_to_canonical) — GroundingDINO промпт
    формата "phrase1 . phrase2 . ..." и обратный маппинг фразы -> canonical name."""
    phrases = []
    synonym_to_canonical = {}
    for canon in canonical_class_names:
        for syn in CLASS_SYNONYMS.get(canon, [canon]):
            phrases.append(syn)
            synonym_to_canonical[syn.lower()] = canon
    return " . ".join(phrases), synonym_to_canonical


def match_phrase_to_canonical(phrase: str, synonym_to_canonical: dict[str, str]) -> str | None:
    phrase = phrase.lower().strip()
    if phrase in synonym_to_canonical:
        return synonym_to_canonical[phrase]
    for syn, canon in synonym_to_canonical.items():
        if syn in phrase or phrase in syn:
            return canon
    return None


class GroundingBoxDetector:
    def __init__(self, config_path: str, checkpoint_path: str, canonical_class_names: list[str],
                 canon_name_to_id: dict[str, int], device: str = "cuda",
                 box_threshold: float = 0.30, text_threshold: float = 0.25):
        from groundingdino.util.inference import load_model

        self.model = load_model(config_path, checkpoint_path)
        self.model = self.model.to(device)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.text_prompt, self.synonym_to_canonical = build_text_prompt(canonical_class_names)
        self.canon_name_to_id = canon_name_to_id
        print(f"[grounding_boxes] text_prompt = '{self.text_prompt}'")

    def detect(self, image_path: str) -> list[dict]:
        """Возвращает список {category_id, bbox_xyxy (в пикселях исходного
        изображения), score} для одной картинки."""
        from groundingdino.util.inference import load_image, predict

        image_source, image_tensor = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        h, w = image_source.shape[:2]
        # groundingdino возвращает боксы в cxcywh, нормализованные [0,1]
        boxes_xyxy = _cxcywh_norm_to_xyxy_px(boxes.cpu().numpy(), w, h)

        out = []
        for box, score, phrase in zip(boxes_xyxy, logits.cpu().numpy(), phrases):
            canon = match_phrase_to_canonical(phrase, self.synonym_to_canonical)
            if canon is None:
                continue
            out.append({
                "category_id": self.canon_name_to_id[canon],
                "bbox_xyxy": box.tolist(),
                "score": float(score),
            })
        return out


def _cxcywh_norm_to_xyxy_px(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0 = (cx - bw / 2) * w
    y0 = (cy - bh / 2) * h
    x1 = (cx + bw / 2) * w
    y1 = (cy + bh / 2) * h
    return np.stack([x0, y0, x1, y1], axis=1)
