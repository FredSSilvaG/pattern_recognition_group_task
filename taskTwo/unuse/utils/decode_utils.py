
SPECIAL_MAP = {
    "s_pt": ".",
    "s_cm": ",",
    "s_mi": "-",
    "s_sq": ";",
    "s_qo": ":",
    "s_qt": "'",
    "s_s": "ſ",
    **{f"s_{i}": str(i) for i in range(10)}
}


def decode_transcription(trans):
    chars = trans.split('-')
    return ''.join(SPECIAL_MAP.get(c, c) for c in chars)


# Split the document ID
def split_doc_id(doc_id):
    doc_number, line_number, word_number = doc_id.split('-')
    return doc_number, line_number, word_number
