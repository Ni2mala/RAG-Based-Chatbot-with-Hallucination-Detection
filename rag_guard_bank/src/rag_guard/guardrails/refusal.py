def refusal_message(unsupported_claims: list[str] | None = None) -> str:
    # Never repeat the model's own claims here: they failed verification.
    # (They stay available in the response's verification.unsupported_claims for the UI/debugging.)
    if unsupported_claims:
        return (
            "I could not verify that my answer is supported by the approved banking "
            "documents, so I cannot share it. Please contact customer support to "
            "confirm this information."
        )
    return (
        "I could not find enough information in the approved banking documents "
        "to answer that reliably. Please rephrase your question or contact our "
        "customer support."
    )
