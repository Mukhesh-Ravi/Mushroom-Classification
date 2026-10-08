"""Human-readable names for the mushroom dataset's single-letter codes."""

TARGET = "class"
CLASS_LABELS = {"e": "Edible", "p": "Poisonous"}

# column -> (display name, {code: meaning})
FEATURES = {
    "cap-shape": ("Cap shape", {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"}),
    "cap-surface": ("Cap surface", {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"}),
    "cap-color": ("Cap color", {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"}),
    "bruises": ("Bruises", {"t": "yes", "f": "no"}),
    "odor": ("Odor", {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul", "m": "musty", "n": "none", "p": "pungent", "s": "spicy"}),
    "gill-attachment": ("Gill attachment", {"a": "attached", "d": "descending", "f": "free", "n": "notched"}),
    "gill-spacing": ("Gill spacing", {"c": "close", "w": "crowded", "d": "distant"}),
    "gill-size": ("Gill size", {"b": "broad", "n": "narrow"}),
    "gill-color": ("Gill color", {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"}),
    "stalk-shape": ("Stalk shape", {"e": "enlarging", "t": "tapering"}),
    "stalk-root": ("Stalk root", {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted"}),
    "stalk-surface-above-ring": ("Stalk surface (above ring)", {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"}),
    "stalk-surface-below-ring": ("Stalk surface (below ring)", {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"}),
    "stalk-color-above-ring": ("Stalk color (above ring)", {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"}),
    "stalk-color-below-ring": ("Stalk color (below ring)", {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange", "p": "pink", "e": "red", "w": "white", "y": "yellow"}),
    "veil-type": ("Veil type", {"p": "partial", "u": "universal"}),
    "veil-color": ("Veil color", {"n": "brown", "o": "orange", "w": "white", "y": "yellow"}),
    "ring-number": ("Ring number", {"n": "none", "o": "one", "t": "two"}),
    "ring-type": ("Ring type", {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large", "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"}),
    "spore-print-color": ("Spore print color", {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green", "o": "orange", "u": "purple", "w": "white", "y": "yellow"}),
    "population": ("Population", {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several", "y": "solitary"}),
    "habitat": ("Habitat", {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste", "d": "woods"}),
}
