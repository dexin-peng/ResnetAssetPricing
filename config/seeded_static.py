from __future__ import annotations


INPUT_FEATURES = 161
MAX_HIDDEN_LAYER_COUNT = 20
SEED_WIDTH_TEMPLATES: dict[int, list[int]] = {
    1: [32],
    2: [32, 16],
    3: [32, 16, 8],
    4: [32, 16, 8, 4],
}


def expand_widths(*, seed: int, depth: int) -> list[int]:
    if seed not in SEED_WIDTH_TEMPLATES:
        raise ValueError(f"Unsupported seed width template: s{seed}")
    if depth < seed or depth > MAX_HIDDEN_LAYER_COUNT:
        raise ValueError(f"Invalid seeded depth combination: s{seed}d{depth}")

    base = SEED_WIDTH_TEMPLATES[seed]
    repeats, remainder = divmod(depth, seed)
    counts = [repeats] * seed
    for index in range(remainder):
        counts[seed - 1 - index] += 1

    widths: list[int] = []
    for width, count in zip(base, counts, strict=True):
        widths.extend([width] * count)
    return widths


def pair_hidden_sizes(widths: list[int], *, input_features: int = INPUT_FEATURES) -> list[list[int]]:
    if not widths:
        raise ValueError("widths cannot be empty")

    pairs: list[list[int]] = []
    previous = input_features
    for width in widths:
        pairs.append([previous, width])
        previous = width
    return pairs


def expand_pair_hidden_sizes(*, seed: int, depth: int, input_features: int = INPUT_FEATURES) -> list[list[int]]:
    if seed not in SEED_WIDTH_TEMPLATES:
        raise ValueError(f"Unsupported seed width template: s{seed}")
    if depth < seed or depth > MAX_HIDDEN_LAYER_COUNT:
        raise ValueError(f"Invalid seeded depth combination: s{seed}d{depth}")

    widths = SEED_WIDTH_TEMPLATES[seed]
    stages: list[list[int]] = []
    previous = input_features
    for width in widths:
        stages.append([previous, width])
        stages.append([width, width])
        previous = width

    repeats, remainder = divmod(depth, len(stages))
    counts = [repeats] * len(stages)
    transition_order = list(range(len(stages) - 2, -1, -2))
    identity_order = list(range(len(stages) - 1, 0, -2))
    for index in (transition_order + identity_order)[:remainder]:
        counts[index] += 1

    pairs: list[list[int]] = []
    for pair, count in zip(stages, counts, strict=True):
        pairs.extend([pair.copy() for _ in range(count)])
    return pairs
