"""Licence policy for third-party training data.

The values intentionally mirror the policy in README.md.  New source modules must
call :func:`is_acceptable_licence` before an item is written.
"""

from __future__ import annotations

import re

ALLOWED_LICENCES = (
    "CC0",
    "Public domain",
    "CC BY (all versions)",
    "Etalab Licence Ouverte 2.0",
    "dl-de/zero-2.0",
    "dl-de/by-2.0",
    "swisstopo Open Government Data (free use with attribution)",
    "Norwegian Licence for Open Government Data (NLOD) 2.0",
)


def normalise_licence(value: str | None) -> str:
    """Return a stable identifier for an accepted or candidate licence string."""
    raw = (value or "").strip().lower()
    text = raw
    text = text.replace("etalab licence ouverte", "etalab").replace("licence ouverte", "etalab")
    text = text.replace("creative commons", "cc").replace("creative-commons", "cc")
    text = re.sub(r"[_\s]+", "-", text)
    text = re.sub(r"[^a-z0-9./-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    text = re.sub(r"^(?:license|licence)-for-this-object-", "", text)
    if re.search(r"(?:^|[-_\s])cc0(?:$|[-_\s])", raw) or text in {
        "cc0",
        "cc-0",
        "cc0-1-0",
        "cc0-1.0",
        "cc-0-1-0",
        "cc-0-1.0",
        "cc-zero",
    }:
        return "cc0"
    if "public domain" in raw or text in {
        "public-domain",
        "publicdomain",
        "pd",
        "pd-self",
        "pd-old",
        "pd-us",
    }:
        return "public-domain"
    if "swisstopo" in text and ("ogd" in text or "free-use-with-attribution" in text):
        return "swisstopo-ogd"
    if "opendata.swiss" in raw and "free use with attribution" in raw:
        return "swisstopo-ogd"
    if text in {"swisstopo-ogd", "swisstopo"}:
        return "swisstopo-ogd"
    if text in {"etalab-2-0", "etalab-2.0"}:
        return "etalab-2.0"
    if text in {"dl-de-zero-2-0", "dl-de-zero-2.0", "dl-de/zero-2.0"}:
        return "dl-de/zero-2.0"
    if text in {"dl-de-by-2-0", "dl-de-by-2.0", "dl-de/by-2.0"}:
        return "dl-de/by-2.0"
    if text in {"nlod", "nlod-2-0", "nlod-2.0", "norsk-lisens-for-offentlige-data"}:
        return "nlod-2.0"
    if "norwegian-licence-for-open-government-data" in text and "nlod" in text and ("2-0" in text or "2.0" in text):
        return "nlod-2.0"
    return text


_SHARE_ALIKE_TOKENS: tuple[str, ...] = ("by-sa", "by_sa", "bysa", "sharealike", "share-alike")


def is_acceptable_licence(value: str | None, overrides: tuple[str, ...] = ()) -> bool:
    """Apply the model-compatible data licence allowlist.

    Overrides are explicit normalised identifiers in the YAML and are deliberately
    evaluated only after the hard NC/ND/ODbL/SA exclusions.
    """
    raw = (value or "").strip().lower()
    normalised = normalise_licence(value)
    if not raw or any(
        token in raw for token in ("odbl", "non-commercial", "noncommercial", "no-derivatives", "fair use")
    ):
        return False
    if re.search(r"(?:^|[-_/])(?:nc|nd)(?:[-_/]|$)", normalised):
        return False
    if any(token in raw or token in normalised for token in _SHARE_ALIKE_TOKENS):
        return False
    if normalised in {normalise_licence(item) for item in overrides}:
        return True
    for allowed in ALLOWED_LICENCES:
        if allowed == "CC BY (all versions)":
            if normalised == "cc-by" or normalised.startswith("cc-by-"):
                return True
        elif normalised == normalise_licence(allowed):
            return True
    return False
