CATEGORIES = ["timing", "content"]

EVALUATION_REGISTRY = [
    # timing evaluation
    {
        "dimension": "smooth_turn_taking",
        "category": "timing",
        "event_types": ["TURN_SHIFT"],
        "domains": ["chat", "task"],
        "languages": ["en", "ja"],
    },
    {
        "dimension": "pause_handling",
        "category": "timing",
        "event_types": ["TURN_HOLD", "SHORTPAUSE"],
        "domains": ["chat", "task"],
        "languages": ["en", "ja"],
    },
    {
        "dimension": "user_backchanneling",
        "category": "timing",
        "event_types": ["BC"],
        "domains": ["chat", "task"],
        "languages": ["en", "ja"],
    },
    {
        "dimension": "user_barge_in",
        "category": "timing",
        "event_types": ["BARGE_IN"],
        "domains": ["chat", "task"],
        "languages": ["en", "ja"],
    },
    # content evaluation
    #{
    #    "dimension": "instruction_following",
    #    "category": "content",
    #    "event_types": ["TURN_SHIFT"],
    #    "domains": ["task"], # only task-oriented dialogue
    #    "languages": ["en", "ja"],
    #},
    {
        "dimension": "response_relevance",
        "category": "content",
        "event_types": ["TURN_SHIFT"],
        "domains": ["chat", "task"],
        "languages": ["en", "ja"],
    },
    {
        "dimension": "contextual_consistency",
        "category": "content",
        "event_types": ["TURN_SHIFT"],
        "domains": ["chat", "task"],
        "languages": ["en", "ja"],
    },
    {
        "dimension": "qa_accuracy",
        "category": "content",
        "event_types": ["TURN_SHIFT"],
        "domains": ["task"], # only task-oriented dialogue
        "languages": ["en", "ja"],
    },
]

def get_dimensions_for_event(
    event_type: str,
    domain: str,
    lang: str,
    categories: list[str],
) -> list[str]:
    dimensions = []

    for item in EVALUATION_REGISTRY:
        if event_type not in item["event_types"]:
            continue
        if item["category"] not in categories:
            continue
        if domain not in item["domains"]:
            continue
        if lang not in item["languages"]:
            continue

        dimensions.append(item["dimension"])

    assert len(dimensions) == len(set(dimensions)),\
    "Same dimension has been registered multiple times."

    return dimensions
