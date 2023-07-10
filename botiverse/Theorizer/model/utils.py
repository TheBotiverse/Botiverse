def get_overlap_position(para_ids, ans_ids, ans_prefix_ids):
    """
    Get the position (start and end indices) of the overlapping region between a paragraph and an answer after the answer prefix.

    Args:
        para_ids (list): The paragraph token IDs.
        ans_ids (list): The answer token IDs.
        ans_prefix_ids (list): The prefix token IDs of the answer.

    Returns:
        tuple: A tuple representing the start and end indices of the overlapping region.
    """
    # Find the first index where the paragraph and answer prefix differ
    for i, (para_id, ans_prefix_id) in enumerate(zip(para_ids, ans_prefix_ids)):
        if para_id != ans_prefix_id:
            first_diff_index = i
            break
    else:
        first_diff_index = min(len(ans_prefix_ids), len(para_ids))

    # Calculate the end index of the overlapping region
    overlap_end_index = min(first_diff_index + len(ans_ids), len(para_ids))

    return (first_diff_index, overlap_end_index)


def pad_dataset(dataset, padding=0):
    """Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler."""
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in MODEL_INPUTS:
        dataset[name] = [
            x + [padding if name != "lm_labels" else -100] * (max_l - len(x))
            for x in dataset[name]
        ]
    return dataset
