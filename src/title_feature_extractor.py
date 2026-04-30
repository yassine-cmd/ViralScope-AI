"""TitleFeatureExtractor — Power words, caps, punctuation features."""

import re

import numpy as np


class TitleFeatureExtractor:
    """Extract handcrafted title style features."""

    POWER_WORDS = [
        "secret", "truth", "never", "always", "destroyed", "exposed",
        "banned", "leaked", "shocking", "insane", "crazy", "unbelievable",
        "impossible", "viral", "hack", "revealed", "must", "warning",
        "danger", "emergency", "hidden", "real", "caught",
    ]

    def __init__(self, config):
        self.config = config

    def extract(self, titles):
        """Extract title features for a list of titles."""
        features = []
        for title in titles:
            title_str = str(title)

            all_caps = sum(1 for c in title_str if c.isupper()) / max(len(title_str), 1)
            num_count = len(re.findall(r"\d+", title_str))
            punct_count = len(re.findall(r"[!?.,;:]", title_str))
            title_lower = title_str.lower()
            power_count = sum(1 for w in self.POWER_WORDS if w in title_lower)
            direct_address_count = len(re.findall(r"\b(you|your|yours)\b", title_str, re.IGNORECASE))
            has_bracket_pipe = 1 if re.search(r"[\[|()]+", title_str) else 0
            length = len(title_str)
            starts_num = 1 if title_str and title_str[0].isdigit() else 0
            has_excl = 1 if "!" in title_str else 0
            word_count = len(title_str.split())

            features.append([
                all_caps, num_count, punct_count, power_count,
                length, starts_num, has_excl, word_count,
                num_count + power_count + has_excl,
                direct_address_count,
                has_bracket_pipe,
            ])

        return np.array(features, dtype=np.float32)

    def get_feature_names(self):
        return [
            "all_caps_ratio", "number_count", "punct_count", "power_word_count",
            "title_length", "starts_with_number", "has_exclamation",
            "word_count", "clickbait_score", "direct_address_count", "has_bracket_pipe",
        ]
