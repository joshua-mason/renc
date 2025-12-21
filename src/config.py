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
    "Northern Marianas Islands": 4,  # 27 not sure what happened here TODO listen
    "Bhutan": 2,  # 28
    "Brunei": 23,  # 29
    "Nepal": 7,  # 30
    "Eswatini": 4,  # 31
    "US Virgin Islands": 20,  # 32
    "Equitorial Guinnea": 0,  # 33
    "San Marino": 1,  # 34
    "Lichtenstein": 225,  # 35: winner stays on. "over 200"
    "Turkmenistan": 0,  # 36
    "Seycehlles": 84,  # 37
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
