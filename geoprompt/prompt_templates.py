"""
GeoPrompt: Satellite-specific prompt template library.
60+ domain-specific templates organized by perspective, scale, and scene type.
"""

from typing import Dict, List


def geographic_templates(cls: str) -> List[str]:
    return [
        f"satellite view of {cls}",
        f"overhead imagery showing {cls}",
        f"aerial photograph of {cls}",
        f"top-down view of {cls}",
        f"bird's eye view of {cls}",
        f"remote sensing image of {cls}",
        f"earth observation showing {cls}",
    ]


def technical_templates(cls: str) -> List[str]:
    return [
        f"high-resolution aerial {cls}",
        f"multispectral imagery of {cls}",
        f"VHR remote sensing {cls}",
        f"optical satellite image of {cls}",
        f"RGB aerial photo of {cls}",
    ]


def contextual_templates(cls: str) -> List[str]:
    return [
        f"{cls} visible from space",
        f"geospatial feature: {cls}",
        f"land cover class {cls}",
        f"Earth surface {cls}",
        f"{cls} in a satellite scene",
        f"remotely sensed {cls}",
    ]


def descriptive_templates(cls: str) -> List[str]:
    return [
        f"bird's eye view of {cls} structure",
        f"remote sensing photography of {cls}",
        f"overhead view showing {cls} texture",
        f"aerial mapping of {cls}",
    ]


def get_all_templates(cls: str) -> List[str]:
    templates = (
        geographic_templates(cls)
        + technical_templates(cls)
        + contextual_templates(cls)
        + descriptive_templates(cls)
    )
    seen, out = set(), []
    for t in templates:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


HIERARCHICAL_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "building": {
        "scene":    ["urban landscape with buildings overhead",
                     "city block seen from satellite"],
        "object":   ["individual building structure in satellite image",
                     "building footprint overhead"],
        "subclass": ["single-family house aerial view",
                     "commercial building complex overhead"],
        "pixel":    ["building rooftop texture satellite",
                     "building boundary and edges overhead"],
    },
    "road": {
        "scene":    ["road network from satellite",
                     "transportation infrastructure overhead"],
        "object":   ["individual road segment overhead",
                     "highway in satellite image"],
        "subclass": ["paved road aerial view",
                     "highway with lanes from satellite"],
        "pixel":    ["road surface texture overhead",
                     "road boundary in aerial image"],
    },
    "water": {
        "scene":    ["water body from satellite",
                     "river or lake from above"],
        "object":   ["river segment in satellite image",
                     "lake overhead view"],
        "subclass": ["flowing river aerial",
                     "still lake from satellite"],
        "pixel":    ["water surface texture overhead",
                     "water boundary in satellite"],
    },
    "forest": {
        "scene":    ["forest area from satellite",
                     "tree canopy overhead"],
        "object":   ["dense forest in satellite image",
                     "tree cluster overhead"],
        "subclass": ["deciduous forest aerial",
                     "coniferous forest from satellite"],
        "pixel":    ["tree canopy texture satellite",
                     "forest boundary aerial"],
    },
    "agricultural": {
        "scene":    ["agricultural land from satellite",
                     "crop fields from above"],
        "object":   ["individual crop field aerial",
                     "farm plot in satellite image"],
        "subclass": ["paddy field aerial",
                     "wheat field from satellite"],
        "pixel":    ["crop row texture satellite",
                     "field boundary overhead"],
    },
    "barren": {
        "scene":    ["bare land from satellite",
                     "barren area overhead"],
        "object":   ["barren ground in satellite image",
                     "bare earth aerial"],
        "subclass": ["desert aerial",
                     "construction site overhead"],
        "pixel":    ["bare soil texture overhead",
                     "barren surface satellite"],
    },
}


def get_hierarchical_templates(cls: str, level: str = "all") -> List[str]:
    hier = HIERARCHICAL_TEMPLATES.get(cls.lower(), {})
    if not hier:
        return get_all_templates(cls)
    if level == "all":
        out = []
        for lvl_templates in hier.values():
            out.extend(lvl_templates)
        return out
    return hier.get(level, get_all_templates(cls))


LOVEDA_CLASSES = ["background", "building", "road", "water", "barren", "forest", "agricultural"]
ISAID_CLASSES  = ["plane", "ship", "storage_tank", "baseball_diamond", "tennis_court",
                  "basketball_court", "ground_track_field", "harbor", "bridge",
                  "large_vehicle", "small_vehicle", "helicopter", "roundabout",
                  "soccer_ball_field", "swimming_pool"]
DOTA_CLASSES   = ISAID_CLASSES + ["container_crane", "airport", "helipad"]
