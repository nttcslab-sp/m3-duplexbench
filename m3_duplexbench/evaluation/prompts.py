RESPONSE_RELEVANCE_SYSTEM_PROMPT = (
    "Judge whether the model response is relevant to the current user utterance. "
    "Output only one integer: 0, 1, or 2.\n"
    "2: relevant and responsive.\n"
    "1: partially relevant, vague, generic, or incomplete.\n"
    "0: irrelevant, unrelated, or not a meaningful response.\n"
    "Judge only relevance to the current user utterance, not factual correctness."
)
QA_ACCURACY_SYSTEM_PROMPT = (
    "Judge whether the model response is correct with respect to the reference answer. "
    "Output only one integer: 0, 1, or 2.\n"
    "2: correct or semantically equivalent to the reference answer.\n"
    "1: partially correct, incomplete, vague, or contains some correct information.\n"
    "0: incorrect, unsupported by the reference answer, or not an answer to the user utterance.\n"
    "Judge factual accuracy against the reference answer, not style or fluency."
)

CONTEXTUAL_CONSISTENCY_SYSTEM_PROMPT = (
    "Judge whether the model response is consistent with its own previous "
    "statements in the dialogue context. Output only one integer: 0, 1, or 2.\n"
    "2: consistent and grounded in the previous dialogue context.\n"
    "1: not clearly grounded in the context, but does not contradict it.\n"
    "0: contradicts the model's previous statements or established context.\n"
    "Judge only consistency with the dialogue context, not factual accuracy "
    "or response relevance to the current user utterance."
)

INSTRUCTION_FOLLOWING_SYSTEM_PROMPT_BINARY = (
    "You are tasked to judge whether the model answer attempts to follow the "
    "user instruction . Given a user question and a model answer , output 1 "
    "if the model answer attempts to follow the user instruction , even if "
    "the response is incomplete or only partially generated . Output 0 if "
    "the model answer does not attempt to follow the instruction at all . "
    "Do not output anything else ."
)

QA_ACCURACY_SYSTEM_PROMPT_BINARY = (
    "You are tasked to judge whether the model answer is correct with respect "
    "to the reference transcript. Given a user question, a model answer, and "
    "a reference answer, output 1 if the model answer is correct or mostly "
    "correct. Output 0 if the model answer is incorrect, unsupported, or does "
    "not answer the question. Do not output anything else."
)

