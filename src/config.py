# Central place for "ground truth" / labels you compare against.
NOTES = """
They are all "gettable" - e.g. not completley unheard, ones like "an island in french polynesia". Maybe we should incorporate into the model with known 0 listens on some of these


"""


NUMBER_OF_COUNTRIES_WITH_LISTENS = 95
TOTAL_NUMBER_OF_COUNTRIES_WITH_ONE_LISTEN = 6
# None means we don't know, we just know it is incorrect
COUNTRIES_LISTENS = {
    "Madagascar": 73,  # ep22
    "Namibia": 0,  # ep22: likely 0? madagascar only mentioned for having listens
    "Costa Rica": 183,  # ep23: although this was actually a week later
    "Uganda": 59,  # ep24: 2 weeks after the start
    "North Korea": None,  # 25: they don't say
    "Guyana": None,  # 26: Tuvalu mentioned but not guessed. No listens specifically mentioned
    "Northern Mariana Islands": 4,  # 27 not sure what happened here TODO listen
    "Bhutan": 2,  # 28
    "Brunei": 23,  # 29
    "Nepal": 7,  # 30
    "Eswatini": 4,  # 31
    "Virgin Islands, U.S.": 20,  # 32
    "Equatorial Guinea": 0,  # 33
    "San Marino": 1,  # 34
    "Liechtenstein": 225,  # 35: winner stays on. "over 200"
    "Turkmenistan": 0,  # 36
    "Seychelles": 84,  # 37
    "Mauritius": 8,  # 38
    "Georgia": None,  # 30 more than 1 listener but it was not mentioned on pod
    "Vatican City": 0,  # 40, TODO lsten and check
    "Oman": 29,  # 41
    "Fiji": 1,  # 42
    "Vanuatu": 0,  # 43
    "Bolivia": 64,  # 44
    "Faroe Islands": 1,  # 45
    "Belarus": 73,  # 46
    "Palau": 0,  # 47
    "Aruba": 66,  # ep 48
    "Ecuador": 294,  # ep49
    "Iraq": 234,  # ep50, at the time the game started they said
    "Gabon": 1,  # ep51
}

#
# Optional supervision: CENSORED (lower-bounded) listen counts
# -----------------------------------------------------------
# Use this when you know a country is *definitely* not a single-listen country, but you
# do NOT trust (or do not have) an exact count.
#
# Example:
# - "USA": 2   means "USA has at least 2 listens" (Y >= 2)
# - "Spain": 50 means "Spain has at least 50 listens" (Y >= 50)
#
# This is generally safer than inventing exact huge counts (e.g. 5000/20000), because
# exact counts dominate a Poisson likelihood and can distort coefficients.
#
# Tokens can be alpha-3 codes ("USA") or names ("United States"). They are resolved via
# the same resolver as COUNTRIES_LISTENS.
#
# Keep these conservative by default (>=2) unless you truly know a meaningful lower bound.
CENSORED_COUNTRIES_LOWER_BOUNDS: dict[str, int] = {
    # Known/assumed "definitely multi" (>=2). Adjust upward only if you have strong evidence.
    "USA": 2,
    "Canada": 2,
    "Australia": 2,
    "New Zealand": 2,
    "Ireland": 2,
    "Spain": 2,
    "France": 2,
    "Germany": 2,
    "Netherlands": 2,
    "Sweden": 2,
    # 10
    "Denmark": 2,
    "Turkey": 2,
    "India": 2,
    "Indonesia": 2,
    "Japan": 2,
    "China": 2,
    "Brazil": 2,
    "Austria": 2,
    "Norway": 2,
    "Hong Kong": 2,
    # 20
    "Vietnam": 2,
    "Belgium": 2,
    "Singapore": 2,
    "Greece": 2,
    "Portugal": 2,
    "Mexico": 2,
    "Italy": 2,
}
