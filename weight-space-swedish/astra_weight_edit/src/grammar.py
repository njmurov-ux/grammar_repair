"""Synthetic Swedish adjective-inflection task.

The label is an inflection exponent, not a noun-phrase embedding. Surface forms
are a deterministic function of the stem and the exponent. Nouns are carrier
phrases only: bil (uter) and hus (neuter). They do not enter the feature vector,
because two nouns that call for the same exponent are not independent evidence.

Lexical classes do not share one paradigm:

- regular / ends_ig: stor, stort, stora
- geminating: ny, nytt, nya; röd, rött, röda; vit, vitt, vita
- syncopating: vacker, vackert, vackra; gammal, gammalt, gamla
- ends_en: mogen, moget, mogna
- invariant: bra in every cell
- liten: liten, litet, lilla, små

Gender is informative in singular indefinite and is neutralized under plural or
definiteness, except that liten keeps number and definiteness apart (lilla vs små).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import N_STEM_BITS, SEED, SPLIT_SIZES, TARGET_DEFINITENESS, TARGET_GENDER, TARGET_NUMBER

EXPONENTS = (
    "BASE",
    "NEUT_T",
    "NEUT_TT",
    "NEUT_ET",
    "WEAK_A",
    "WEAK_DROP_E",
    "WEAK_NA",
    "LILLA",
    "SMA",
)

GRAMMAR_FEATURES = (
    "gender_uter",
    "gender_neuter",
    "number_sg",
    "number_pl",
    "definiteness_indef",
    "definiteness_def",
)

CLASS_FEATURES = (
    "syncopating",
    "ends_en",
    "geminating",
    "invariant",
    "irregular_liten",
    "ends_ig",
    "gem_vowel",
    "gem_d",
    "gem_t",
    "ends_er",
    "ends_el",
)

STEM_FEATURES = tuple(f"stem_{i}" for i in range(N_STEM_BITS))
FEATURE_NAMES = GRAMMAR_FEATURES + CLASS_FEATURES + STEM_FEATURES

GENDERS = ("uter", "neuter")
NUMBERS = ("sg", "pl")
DEFINITENESS = ("indef", "def")
CONDITIONS = tuple(f"{g}|{n}|{d}" for g in GENDERS for n in NUMBERS for d in DEFINITENESS)

VOWELS = set("aeiouyåäö")


@dataclass(frozen=True)
class Lexeme:
    stem: str
    lexical_class: str
    spelling: str
    flags: dict[str, int]


def _flags(**overrides: int) -> dict[str, int]:
    base = {name: 0 for name in CLASS_FEATURES}
    base.update(overrides)
    unknown = set(overrides) - set(CLASS_FEATURES)
    if unknown:
        raise ValueError(f"unknown class flags: {sorted(unknown)}")
    return base


def _build_lexicon() -> list[Lexeme]:
    items: list[Lexeme] = []

    regular_add_t = [
        "stor", "fin", "grön", "dum", "ful", "varm", "lång", "hög", "låg", "ren",
        "mjuk", "snabb", "stark", "svag", "ung", "kall", "tjock", "smal", "grov", "stum",
    ]
    regular_d_to_t = ["hård", "rund", "vild", "ond"]
    for stem in regular_add_t:
        items.append(Lexeme(stem, "regular", "add_t", _flags()))
    for stem in regular_d_to_t:
        items.append(Lexeme(stem, "regular", "d_to_t", _flags()))

    for stem in [
        "rolig", "tråkig", "dålig", "trevlig", "konstig", "viktig", "duktig", "nyttig",
        "ledig", "billig", "härlig", "farlig", "tydlig", "möjlig", "vanlig",
    ]:
        items.append(Lexeme(stem, "ends_ig", "add_t", _flags(ends_ig=1)))

    for stem in ["ny", "blå", "grå", "fri", "rå", "slö"]:
        items.append(Lexeme(stem, "geminating", "add_tt", _flags(geminating=1, gem_vowel=1)))
    for stem in ["röd", "god", "glad", "bred", "vred", "död", "vid", "pryd"]:
        items.append(Lexeme(stem, "geminating", "d_to_tt", _flags(geminating=1, gem_d=1)))
    for stem in ["vit", "fet", "söt", "lat", "het", "våt", "slät", "flat"]:
        items.append(Lexeme(stem, "geminating", "t_add_t", _flags(geminating=1, gem_t=1)))

    for stem in ["vacker", "säker", "tapper", "bitter", "läcker", "mager", "fager", "dyster"]:
        items.append(Lexeme(stem, "syncopating", "add_t", _flags(syncopating=1, ends_er=1)))
    for stem in ["gammal", "enkel", "ädel", "usel"]:
        items.append(Lexeme(stem, "syncopating", "add_t", _flags(syncopating=1, ends_el=1)))

    for stem in [
        "mogen", "galen", "vaken", "öppen", "egen", "ledsen", "sliten", "sprucken",
        "bruten", "nyfiken", "vilsen", "belåten",
    ]:
        items.append(Lexeme(stem, "ends_en", "en_to_et", _flags(ends_en=1)))

    for stem in [
        "bra", "extra", "kul", "gyllene", "äkta", "stilla", "orange", "gratis", "udda", "nuvarande",
    ]:
        items.append(Lexeme(stem, "invariant", "invariant", _flags(invariant=1)))

    items.append(Lexeme("liten", "liten", "en_to_et", _flags(irregular_liten=1, ends_en=1)))
    return items


LEXICON = _build_lexicon()
LEXEME_BY_STEM = {item.stem: item for item in LEXICON}


def exponent(item: Lexeme, gender: str, number: str, definiteness: str) -> str:
    """Map grammatical features and lexical class onto an inflection exponent."""
    if item.flags["invariant"]:
        return "BASE"
    if item.flags["irregular_liten"]:
        if number == "pl":
            return "SMA"
        if definiteness == "def":
            return "LILLA"
        if gender == "neuter":
            return "NEUT_ET"
        return "BASE"
    weak = number == "pl" or definiteness == "def"
    if weak:
        if item.flags["ends_en"]:
            return "WEAK_NA"
        if item.flags["syncopating"]:
            return "WEAK_DROP_E"
        return "WEAK_A"
    if gender == "uter":
        return "BASE"
    if item.flags["ends_en"]:
        return "NEUT_ET"
    if item.flags["geminating"]:
        return "NEUT_TT"
    return "NEUT_T"


def surface_form(stem: str, exponent_name: str) -> str:
    item = LEXEME_BY_STEM[stem]
    if exponent_name == "BASE":
        return stem
    if exponent_name == "LILLA":
        return "lilla"
    if exponent_name == "SMA":
        return "små"
    if exponent_name == "NEUT_T":
        if item.spelling == "d_to_t":
            return stem[:-1] + "t"
        if item.spelling != "add_t":
            raise ValueError(f"{stem} has spelling {item.spelling}, not a NEUT_T spelling")
        return stem + "t"
    if exponent_name == "NEUT_TT":
        if item.spelling == "add_tt":
            return stem + "tt"
        if item.spelling == "d_to_tt":
            return stem[:-1] + "tt"
        if item.spelling == "t_add_t":
            return stem + "t"
        raise ValueError(f"{stem} has spelling {item.spelling}, not a NEUT_TT spelling")
    if exponent_name == "NEUT_ET":
        if not stem.endswith("en"):
            raise ValueError(f"{stem} does not end in -en")
        return stem[:-2] + "et"
    if exponent_name == "WEAK_A":
        return stem + "a"
    if exponent_name == "WEAK_DROP_E":
        # Drop the unstressed vowel: vacker -> vackra, enkel -> enkla.
        # gammal also simplifies mm, because the doubling only closes the syllable
        # that disappears: gammal -> gamla, not gammla. gammalt keeps the doubling.
        if stem == "gammal":
            return "gamla"
        return stem[:-2] + stem[-1] + "a"
    if exponent_name == "WEAK_NA":
        if not stem.endswith("en"):
            raise ValueError(f"{stem} does not end in -en")
        return stem[:-2] + "na"
    raise ValueError(f"unknown exponent {exponent_name}")


def phrase(surface: str, gender: str, number: str, definiteness: str) -> str:
    if number == "sg" and definiteness == "indef" and gender == "uter":
        return f"en {surface} bil"
    if number == "sg" and definiteness == "indef" and gender == "neuter":
        return f"ett {surface} hus"
    if number == "pl" and definiteness == "indef" and gender == "uter":
        return f"{surface} bilar"
    if number == "pl" and definiteness == "indef" and gender == "neuter":
        return f"{surface} hus"
    if number == "sg" and definiteness == "def" and gender == "uter":
        return f"den {surface} bilen"
    if number == "sg" and definiteness == "def" and gender == "neuter":
        return f"det {surface} huset"
    if number == "pl" and definiteness == "def" and gender == "uter":
        return f"de {surface} bilarna"
    if number == "pl" and definiteness == "def" and gender == "neuter":
        return f"de {surface} husen"
    raise ValueError(f"bad condition {(gender, number, definiteness)}")


def _stem_code(text: str) -> list[float]:
    digest = hashlib.sha256(f"grammar-repair-v1:{text}".encode("utf-8")).digest()
    bits = []
    for i in range(N_STEM_BITS):
        # Small on purpose: these bits keep held-out stems from being exact
        # duplicates, but they must not overpower the grammatical flags.
        bits.append(0.1 if (digest[i] >> (i % 8)) & 1 else 0.0)
    return bits


def unique_stem_codes(stems: list[str]) -> dict[str, list[float]]:
    codes: dict[str, list[float]] = {}
    used: set[tuple[float, ...]] = set()
    for stem in stems:
        salt = 0
        code = _stem_code(stem)
        while tuple(code) in used:
            salt += 1
            code = _stem_code(f"{stem}#{salt}")
        used.add(tuple(code))
        codes[stem] = code
    return codes


def _assign_splits(lexicon: list[Lexeme], seed: int) -> dict[str, str]:
    rng = np.random.RandomState(seed)
    by_class: dict[str, list[str]] = {}
    for item in lexicon:
        by_class.setdefault(item.lexical_class, []).append(item.stem)
    assignment: dict[str, str] = {}
    for lexical_class, stems in by_class.items():
        spec = SPLIT_SIZES[lexical_class]
        if lexical_class == "liten":
            if spec != {"inlexicon": 1} or stems != ["liten"]:
                raise AssertionError("liten is a single in-lexicon irregular")
            assignment["liten"] = "inlexicon"
            continue
        order = list(stems)
        rng.shuffle(order)
        expected = sum(spec.values())
        if expected != len(order):
            raise AssertionError(f"{lexical_class}: split sizes {spec} sum to {expected}, lexicon has {len(order)}")
        cursor = 0
        for split_name in ("train", "dev_diag", "dev_gate", "test"):
            take = spec[split_name]
            for stem in order[cursor : cursor + take]:
                assignment[stem] = split_name
            cursor += take
    return assignment


def feature_vector(item: Lexeme, gender: str, number: str, definiteness: str, stem_code: list[float]) -> np.ndarray:
    grammar = [
        1.0 if gender == "uter" else 0.0,
        1.0 if gender == "neuter" else 0.0,
        1.0 if number == "sg" else 0.0,
        1.0 if number == "pl" else 0.0,
        1.0 if definiteness == "indef" else 0.0,
        1.0 if definiteness == "def" else 0.0,
    ]
    flags = [float(item.flags[name]) for name in CLASS_FEATURES]
    vec = np.asarray(grammar + flags + list(stem_code), dtype=np.float32)
    if vec.shape != (len(FEATURE_NAMES),):
        raise AssertionError(f"feature width {vec.shape} != {len(FEATURE_NAMES)}")
    return vec


def is_target_condition(lexical_class: str, gender: str, number: str, definiteness: str) -> bool:
    """Strong neuter cell: the only place gender changes the exponent.

    Invariant adjectives are not included. Their neuter form is the base form,
    so a gender collapse does not change the label.
    """
    return (
        lexical_class != "invariant"
        and gender == TARGET_GENDER
        and number == TARGET_NUMBER
        and definiteness == TARGET_DEFINITENESS
    )


def build_dataset(seed: int = SEED) -> pd.DataFrame:
    codes = unique_stem_codes([item.stem for item in LEXICON])
    splits = _assign_splits(LEXICON, seed)
    rows = []
    for item in LEXICON:
        for gender in GENDERS:
            for number in NUMBERS:
                for definiteness in DEFINITENESS:
                    exp = exponent(item, gender, number, definiteness)
                    surface = surface_form(item.stem, exp)
                    rows.append(
                        {
                            "stem": item.stem,
                            "lexical_class": item.lexical_class,
                            "spelling": item.spelling,
                            "gender": gender,
                            "number": number,
                            "definiteness": definiteness,
                            "condition": f"{gender}|{number}|{definiteness}",
                            "exponent": exp,
                            "label": EXPONENTS.index(exp),
                            "surface": surface,
                            "phrase": phrase(surface, gender, number, definiteness),
                            "split": splits[item.stem],
                            "is_target": is_target_condition(item.lexical_class, gender, number, definiteness),
                            "features": feature_vector(item, gender, number, definiteness, codes[item.stem]),
                        }
                    )
    frame = pd.DataFrame(rows)
    validate_dataset(frame)
    return frame


def matrix_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(frame["features"].to_list()).astype(np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    return x, y


def _assert_ending(stem: str, suffix: str) -> None:
    if not stem.endswith(suffix):
        raise AssertionError(f"{stem} should end with {suffix}")


def validate_dataset(frame: pd.DataFrame) -> None:
    stems = [item.stem for item in LEXICON]
    if len(stems) != len(set(stems)):
        raise AssertionError("duplicate stems in the lexicon")
    if set(frame["split"].unique()) != {"train", "dev_diag", "dev_gate", "test", "inlexicon"}:
        raise AssertionError(f"unexpected splits: {sorted(frame['split'].unique())}")

    for item in LEXICON:
        if item.lexical_class == "geminating":
            if item.spelling == "add_tt":
                if item.stem[-1] not in VOWELS:
                    raise AssertionError(f"{item.stem} should end in a vowel")
            elif item.spelling == "d_to_tt":
                _assert_ending(item.stem, "d")
            elif item.spelling == "t_add_t":
                _assert_ending(item.stem, "t")
            else:
                raise AssertionError(item.spelling)
        if item.lexical_class == "syncopating":
            if item.stem[-2] not in {"e", "a"}:
                raise AssertionError(f"{item.stem} has no syncopated vowel")
        if item.flags["ends_en"] and not item.stem.endswith("en"):
            raise AssertionError(item.stem)
        if item.flags["ends_ig"] and not item.stem.endswith("ig"):
            raise AssertionError(item.stem)
        if item.flags["ends_er"] and not item.stem.endswith("er"):
            raise AssertionError(item.stem)
        if item.flags["ends_el"] and not item.stem.endswith(("el", "al")):
            raise AssertionError(item.stem)

    # Gender is neutralized in the plural and in the definite, except for liten,
    # where definiteness and number stay distinct and gender is still neutralized.
    for (stem, number, definiteness), group in frame.groupby(["stem", "number", "definiteness"]):
        labels = set(group["exponent"])
        item = LEXEME_BY_STEM[stem]
        if item.lexical_class == "liten":
            if number == "pl":
                if labels != {"SMA"}:
                    raise AssertionError("liten plural must be små for both genders")
            elif definiteness == "def":
                if labels != {"LILLA"}:
                    raise AssertionError("liten definite singular must be lilla")
            continue
        if number == "pl" or definiteness == "def" or item.lexical_class == "invariant":
            if len(labels) != 1:
                raise AssertionError(f"{stem} {number} {definiteness} changes with gender: {labels}")

    # Held-out stems of one class must not duplicate a training input exactly.
    for lexical_class, group in frame.groupby("lexical_class"):
        if lexical_class == "liten":
            continue
        train_keys = set()
        for row in group[group["split"] == "train"].itertuples(index=False):
            train_keys.add((row.condition, tuple(row.features.tolist())))
        held = group[group["split"].isin(["dev_diag", "dev_gate", "test"])]
        for row in held.itertuples(index=False):
            key = (row.condition, tuple(row.features.tolist()))
            if key in train_keys:
                raise AssertionError(f"held-out row duplicates a training input: {lexical_class} {row.stem}")

    split_stems = {
        split: set(frame.loc[frame["split"] == split, "stem"])
        for split in ("train", "dev_diag", "dev_gate", "test", "inlexicon")
    }
    if split_stems["train"] & split_stems["test"]:
        raise AssertionError("train/test stem overlap")
    if split_stems["train"] & split_stems["dev_diag"] or split_stems["train"] & split_stems["dev_gate"]:
        raise AssertionError("train/dev stem overlap")
    if split_stems["test"] & split_stems["dev_diag"] or split_stems["test"] & split_stems["dev_gate"]:
        raise AssertionError("test/dev stem overlap")
    if split_stems["inlexicon"] != {"liten"}:
        raise AssertionError("inlexicon split should contain only liten")

    for stem, gender, number, definiteness, surface, text in HAND_CHECKED:
        match = frame[
            (frame["stem"] == stem)
            & (frame["gender"] == gender)
            & (frame["number"] == number)
            & (frame["definiteness"] == definiteness)
        ]
        if len(match) != 1:
            raise AssertionError(f"missing hand check row {stem} {gender} {number} {definiteness}")
        got_surface = match.iloc[0]["surface"]
        got_phrase = match.iloc[0]["phrase"]
        if got_surface != surface or got_phrase != text:
            raise AssertionError(
                f"{stem} {gender} {number} {definiteness}: got {got_surface!r} / {got_phrase!r}, "
                f"expected {surface!r} / {text!r}"
            )

    # The target cell is exactly "neuter singular indefinite, not invariant".
    # It uses more than one exponent: NEUT_T, NEUT_TT and NEUT_ET.
    target = frame[frame["is_target"]]
    if set(target["exponent"]) != {"NEUT_T", "NEUT_TT", "NEUT_ET"}:
        raise AssertionError(set(target["exponent"]))
    if set(target["gender"]) != {"neuter"} or set(target["number"]) != {"sg"} or set(target["definiteness"]) != {"indef"}:
        raise AssertionError("target cell is not confined to neuter singular indefinite")
    if "invariant" in set(target["lexical_class"]):
        raise AssertionError("invariant adjectives are not a gender-agreement target")
    if not bool(frame.loc[frame["exponent"] == "NEUT_TT", "is_target"].all()):
        raise AssertionError("NEUT_TT leaked outside the target cell")
    if frame["label"].min() < 0 or frame["label"].max() >= len(EXPONENTS):
        raise AssertionError("label index out of range")


HAND_CHECKED = [
    ("stor", "uter", "sg", "indef", "stor", "en stor bil"),
    ("stor", "neuter", "sg", "indef", "stort", "ett stort hus"),
    ("stor", "uter", "pl", "indef", "stora", "stora bilar"),
    ("stor", "neuter", "sg", "def", "stora", "det stora huset"),
    ("stor", "neuter", "pl", "def", "stora", "de stora husen"),
    ("ny", "uter", "sg", "indef", "ny", "en ny bil"),
    ("ny", "neuter", "sg", "indef", "nytt", "ett nytt hus"),
    ("ny", "uter", "pl", "def", "nya", "de nya bilarna"),
    ("blå", "neuter", "sg", "indef", "blått", "ett blått hus"),
    ("blå", "uter", "pl", "indef", "blåa", "blåa bilar"),
    ("grå", "neuter", "sg", "def", "gråa", "det gråa huset"),
    ("fri", "neuter", "sg", "indef", "fritt", "ett fritt hus"),
    ("rå", "uter", "pl", "indef", "råa", "råa bilar"),
    ("slö", "neuter", "sg", "indef", "slött", "ett slött hus"),
    ("röd", "neuter", "sg", "indef", "rött", "ett rött hus"),
    ("röd", "uter", "sg", "def", "röda", "den röda bilen"),
    ("god", "neuter", "sg", "indef", "gott", "ett gott hus"),
    ("god", "neuter", "pl", "indef", "goda", "goda hus"),
    ("glad", "uter", "sg", "def", "glada", "den glada bilen"),
    ("bred", "neuter", "sg", "indef", "brett", "ett brett hus"),
    ("vid", "neuter", "sg", "indef", "vitt", "ett vitt hus"),
    ("vid", "uter", "pl", "indef", "vida", "vida bilar"),
    ("död", "neuter", "sg", "indef", "dött", "ett dött hus"),
    ("pryd", "neuter", "sg", "indef", "prytt", "ett prytt hus"),
    ("vit", "neuter", "sg", "indef", "vitt", "ett vitt hus"),
    ("vit", "uter", "sg", "indef", "vit", "en vit bil"),
    ("fet", "neuter", "sg", "indef", "fett", "ett fett hus"),
    ("söt", "uter", "pl", "def", "söta", "de söta bilarna"),
    ("lat", "neuter", "sg", "indef", "latt", "ett latt hus"),
    ("het", "neuter", "sg", "def", "heta", "det heta huset"),
    ("våt", "neuter", "sg", "indef", "vått", "ett vått hus"),
    ("slät", "uter", "pl", "indef", "släta", "släta bilar"),
    ("flat", "neuter", "sg", "indef", "flatt", "ett flatt hus"),
    ("hård", "neuter", "sg", "indef", "hårt", "ett hårt hus"),
    ("hård", "uter", "pl", "indef", "hårda", "hårda bilar"),
    ("rund", "neuter", "sg", "indef", "runt", "ett runt hus"),
    ("vild", "neuter", "sg", "indef", "vilt", "ett vilt hus"),
    ("ond", "neuter", "sg", "indef", "ont", "ett ont hus"),
    ("vacker", "neuter", "sg", "indef", "vackert", "ett vackert hus"),
    ("vacker", "uter", "pl", "indef", "vackra", "vackra bilar"),
    ("vacker", "neuter", "sg", "def", "vackra", "det vackra huset"),
    ("säker", "uter", "pl", "indef", "säkra", "säkra bilar"),
    ("dyster", "neuter", "sg", "indef", "dystert", "ett dystert hus"),
    ("tapper", "uter", "sg", "def", "tappra", "den tappra bilen"),
    ("gammal", "neuter", "sg", "indef", "gammalt", "ett gammalt hus"),
    ("gammal", "uter", "sg", "def", "gamla", "den gamla bilen"),
    ("enkel", "neuter", "pl", "def", "enkla", "de enkla husen"),
    ("ädel", "uter", "sg", "def", "ädla", "den ädla bilen"),
    ("usel", "neuter", "sg", "indef", "uselt", "ett uselt hus"),
    ("mogen", "neuter", "sg", "indef", "moget", "ett moget hus"),
    ("mogen", "uter", "pl", "indef", "mogna", "mogna bilar"),
    ("öppen", "neuter", "sg", "def", "öppna", "det öppna huset"),
    ("egen", "neuter", "sg", "indef", "eget", "ett eget hus"),
    ("egen", "uter", "pl", "def", "egna", "de egna bilarna"),
    ("ledsen", "uter", "sg", "def", "ledsna", "den ledsna bilen"),
    ("ledsen", "neuter", "sg", "indef", "ledset", "ett ledset hus"),
    ("nyfiken", "neuter", "sg", "indef", "nyfiket", "ett nyfiket hus"),
    ("nyfiken", "uter", "sg", "def", "nyfikna", "den nyfikna bilen"),
    ("vilsen", "neuter", "pl", "indef", "vilsna", "vilsna hus"),
    ("belåten", "neuter", "sg", "indef", "belåtet", "ett belåtet hus"),
    ("sliten", "uter", "pl", "def", "slitna", "de slitna bilarna"),
    ("sprucken", "neuter", "sg", "indef", "sprucket", "ett sprucket hus"),
    ("bruten", "uter", "sg", "def", "brutna", "den brutna bilen"),
    ("liten", "uter", "sg", "indef", "liten", "en liten bil"),
    ("liten", "neuter", "sg", "indef", "litet", "ett litet hus"),
    ("liten", "uter", "sg", "def", "lilla", "den lilla bilen"),
    ("liten", "neuter", "sg", "def", "lilla", "det lilla huset"),
    ("liten", "uter", "pl", "indef", "små", "små bilar"),
    ("liten", "neuter", "pl", "indef", "små", "små hus"),
    ("liten", "neuter", "pl", "def", "små", "de små husen"),
    ("liten", "uter", "pl", "def", "små", "de små bilarna"),
    ("bra", "neuter", "sg", "indef", "bra", "ett bra hus"),
    ("bra", "uter", "sg", "def", "bra", "den bra bilen"),
    ("bra", "neuter", "pl", "def", "bra", "de bra husen"),
    ("kul", "uter", "pl", "indef", "kul", "kul bilar"),
    ("gyllene", "neuter", "sg", "def", "gyllene", "det gyllene huset"),
    ("orange", "neuter", "sg", "indef", "orange", "ett orange hus"),
    ("gratis", "neuter", "sg", "def", "gratis", "det gratis huset"),
    ("nuvarande", "uter", "pl", "indef", "nuvarande", "nuvarande bilar"),
    ("äkta", "uter", "sg", "indef", "äkta", "en äkta bil"),
    ("udda", "neuter", "pl", "def", "udda", "de udda husen"),
    ("rolig", "neuter", "sg", "indef", "roligt", "ett roligt hus"),
    ("rolig", "uter", "pl", "def", "roliga", "de roliga bilarna"),
    ("dålig", "neuter", "sg", "def", "dåliga", "det dåliga huset"),
    ("vanlig", "uter", "sg", "indef", "vanlig", "en vanlig bil"),
    ("möjlig", "neuter", "sg", "indef", "möjligt", "ett möjligt hus"),
    ("fin", "neuter", "sg", "indef", "fint", "ett fint hus"),
    ("lång", "uter", "sg", "def", "långa", "den långa bilen"),
    ("kall", "neuter", "sg", "indef", "kallt", "ett kallt hus"),
    ("ung", "neuter", "pl", "indef", "unga", "unga hus"),
]


def dataset_summary(frame: pd.DataFrame) -> dict:
    stems = frame.groupby(["lexical_class", "split"])["stem"].nunique().unstack(fill_value=0)
    return {
        "n_rows": int(len(frame)),
        "n_stems": int(frame["stem"].nunique()),
        "n_exponents": len(EXPONENTS),
        "n_features": len(FEATURE_NAMES),
        "hand_checked": len(HAND_CHECKED),
        "stems_by_class_and_split": stems.to_dict(),
        "target_rows_by_split": frame.groupby("split")["is_target"].sum().astype(int).to_dict(),
        "rows_by_split": frame.groupby("split").size().astype(int).to_dict(),
    }
