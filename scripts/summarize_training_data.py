"""Print training-only preprocessing constants and patch statistics."""

import json

from tessscope.data.bbbc039 import (
    global_training_percentiles,
    patch_origins,
    prepare_patch,
    records_for_split,
)


def main() -> None:
    normalization = global_training_percentiles()
    counts: list[int] = []
    for record in records_for_split("training"):
        for origin in patch_origins("training"):
            counts.append(int(prepare_patch(record, origin, normalization).valid_objects.sum()))
    result = {
        "object_percentiles": {
            "lower_percentile": 0.1,
            "upper_percentile": 99.9,
            "lower_value": normalization.lower,
            "upper_value": normalization.upper,
            "source": "75 decontaminated BBBC039 training images only",
        },
        "training_patches": len(counts),
        "valid_objects_per_patch": {
            "minimum": min(counts),
            "maximum": max(counts),
            "mean": sum(counts) / len(counts),
            "at_cap_32": sum(count == 32 for count in counts),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
