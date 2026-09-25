def refusal_message(unsupported_claims: list[str] | None = None) -> str:
    if unsupported_claims:
        preview = "; ".join(unsupported_claims[:2])
        return (
            "I could not verify that my answer is fully supported by the approved "
            f"banking documents (claims such as: \"{preview}\"). For safety, I "
            "cannot stand behind this answer. Please contact customer support "
            "to confirm this information."
        )
    return (
        "I could not find enough information in the approved banking documents "
        "to answer that reliably. Please rephrase your question or contact our "
        "customer support."
    )
