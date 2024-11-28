import numpy as np
from typing_extensions import List, Literal, Tuple, TypedDict, Union

# ======================================
# Typings
# ======================================


class Entity(TypedDict):
    text: str
    label: str


# ======================================
# Helper functions
# ======================================


def compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.

    Returns:
        The precision, recall, and F1 score.
    """
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    return p, r, f1


def longest_common_substring(text1: str, text2: str) -> float:
    """Find the longest common substring between two strings.

    Args:
        text1: The first string.
        text2: The second string.

    Returns:
        The length of the longest common substring.
    """

    m = len(text1)
    n = len(text2)
    lcsuff = np.zeros((m + 1, n + 1))
    result = 0  # To store length of the longest common substring

    # Building the lcsuff table in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                lcsuff[i][j] = 0
            elif text1[i - 1] == text2[j - 1]:
                lcsuff[i][j] = lcsuff[i - 1][j - 1] + 1
                result = max(result, lcsuff[i][j])
            else:
                lcsuff[i][j] = 0

    return result


# ======================================
# NER evaluation functions
# ======================================


def _exact_ner_evaluation(
    true_ents: List[Entity], pred_ents: List[Entity]
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using exact matching.

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.

    Returns:
        The precision, recall, and F1 score.
    """
    true_ents_set = set((ent["text"], ent["label"]) for ent in true_ents)
    pred_ents_set = set((ent["text"], ent["label"]) for ent in pred_ents)

    tp = len(true_ents_set & pred_ents_set)
    fp = len(pred_ents_set - true_ents_set)
    fn = len(true_ents_set - pred_ents_set)

    p, r, f1 = compute_metrics(tp, fp, fn)
    return p, r, f1


def _relaxed_ner_evaluation(
    true_ents: List[Entity], pred_ents: List[Entity]
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using relaxed matching.

    When using relaxed matching, the algorithm considers an entity as correct if the
    predicted entity contains the true entity. The labels of both entities must also match.

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.

    Returns:
        The precision, recall, and F1 score.
    """
    true_ents_set = set((ent["text"], ent["label"]) for ent in true_ents)
    pred_ents_set = set((ent["text"], ent["label"]) for ent in pred_ents)

    tp = 0
    for true_ent in true_ents_set:
        for pred_ent in pred_ents_set:
            if true_ent[1] == pred_ent[1] and true_ent[0] in pred_ent[0]:
                tp += 1
                break

    fp = len(pred_ents_set) - tp
    fn = len(true_ents_set) - tp
    p, r, f1 = compute_metrics(tp, fp, fn)
    return p, r, f1


def _overlap_ner_evaluation(
    true_ents: List[Entity], pred_ents: List[Entity]
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance using overlap matching.

    It is based on the longest common substring between the true and predicted entities.
    Furthermore, the label of the entity is also considered during the evaluation. The
    inspiration for this algorithm comes from the following paper:

    @inproceedings{bert-score,
        title={BERTScore: Evaluating Text Generation with BERT},
        author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
        booktitle={International Conference on Learning Representations},
        year={2020},
        url={https://openreview.net/forum?id=SkeHuCVFDr}
    }

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.

    Returns:
        The precision, recall, and F1 score.
    """

    if len(true_ents) == 0 or len(pred_ents) == 0:
        return 0.0, 0.0, 0.0

    text_matrix = np.zeros((len(true_ents), len(pred_ents)))
    label_matrix = np.zeros((len(true_ents), len(pred_ents)))

    for i, true_ent in enumerate(true_ents):
        for j, pred_ent in enumerate(pred_ents):
            label_matrix[i, j] = true_ent["label"] == pred_ent["label"]
            text_matrix[i, j] = longest_common_substring(
                true_ent["text"], pred_ent["text"]
            ) / len(true_ent["text"])

    matrix = text_matrix * label_matrix
    p = np.mean(np.max(matrix, axis=0))
    r = np.mean(np.max(matrix, axis=1))
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

    return p, r, f1


def evaluate_ner_performance(
    true_ents: List[Entity],
    pred_ents: List[Entity],
    match_type: Union[
        Literal["exact"], Literal["relaxed"], Literal["overlap"]
    ] = "exact",
) -> Tuple[float, float, float]:
    """Evaluate named entity recognition performance.

    Args:
        true_ents: List of true entities.
        pred_ents: List of predicted entities.
        match_type: The evaluation method to use, either "exact", "relaxed" or "overlap".

    Returns:
        The precision, recall, and F1 score.
    """
    if match_type == "exact":
        p, r, f1 = _exact_ner_evaluation(true_ents, pred_ents)
    elif match_type == "relaxed":
        p, r, f1 = _relaxed_ner_evaluation(true_ents, pred_ents)
    elif match_type == "overlap":
        p, r, f1 = _overlap_ner_evaluation(true_ents, pred_ents)
    else:
        raise ValueError(f"Unknown match_type method: {match_type}")
    return p, r, f1
