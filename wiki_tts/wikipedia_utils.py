BLOCKLIST = ['see also', 'references', 'external links', 'further reading', 'notes']


def get_valid_sections(sections) -> list:
    """
    Recursively extracts all sections and subsections (h2, h3, h4...),
    skipping blocklisted metadata sections and their children.
    """
    valid = []
    for s in sections:
        if s.title.lower() in BLOCKLIST:
            continue
        valid.append(s)
        valid.extend(get_valid_sections(s.sections))
    return valid


def find_section_by_title(sections, target_title: str):
    """
    Recursively searches the section tree for a specific title.
    Returns the section object or None.
    """
    for s in sections:
        if s.title == target_title:
            return s
        found = find_section_by_title(s.sections, target_title)
        if found:
            return found
    return None
