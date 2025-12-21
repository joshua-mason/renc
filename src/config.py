# Central place for "ground truth" / labels you compare against.

# CORRECT_COUNTRIES = ["San Marino", "Fiji", "Faroe Islands", "Gabon"]
# INCORRECT_COUNTRIES = [
#     "Madagascar",
#     "Namibia",
#     "North Korea",
#     "Costa Rica",
#     "Uganda",
#     "Northern Mariana Islands",
#     "Guyana",
#     "Bhutan",
#     "Lichtenstein",
#     "Brunei",
#     "Nepal",
#     "Turkmenistan",
#     "Belarus",
#     "Aruba",
#     "Ecuador",
#     "Iraq",
#     "Eswatini",
#     "US Virgin Islands",
#     "Seychelles",
#     "Mauritius",
#     "Georgia",
#     "Vatican City",
#     "Oman",
#     "Vanuatu",
#     "Bolivia",
#     "Palau",
# ]


# COUNTRIES_WITH_MORE_THAN_ONE_LISTEN = [
#     "united kingdom",
#     "usa",
#     "spain",
#     "canada",
#     "australia",
#     "germany",
#     "france",
#     "italy",
#     "ireland",
#     "netherlands",
#     "sweden",
#     "new zealand",
#     "south africa",
#     "norway",
#     "denmark",
#     "singapore",
#     "switzerland",
#     "belgium",
#     "india",
#     "japan",
# ]


NOTES = """
They are all "gettable" - e.g. not completley unheard, ones like "an island in french polynesia". Maybe we should incorporate into the model with known 0 listens on some of these


"""


NUMBER_OF_COUNTRIES_WITH_LISTENS = 95
TOTAL_NUMBER_OF_COUNTRIES_WITH_ONE_LISTEN = 6
# None means we don't know, we just know it is incorrect
COUNTRIES_LISTENS = {
    "Madagascar": 73,  # ep22
    "Namibia": None,  # likely 0? madagascar only mentioned for having listens
    "Costa Rica": 183,  # ep23: although this was actually a week later
    "Uganda": 59,  # ep24: 2 weeks after the start
    "North Korea": None,  # 25: Need to listen again
    "Tuvalu": None,  # 26 TODO listen again I don't understand the transcript
    "Northern Marianas Island": None,  # 27 not sure what happened here TODO listen
    "Bhutan": 2,  # 28
    "Brunei": 23,  # 29
    # TODO Missing transcript ep 30
    "Eswatini": 4,  # 31
    "US Virgin Islands": 20,  # 32
    "Equitorial Guinnea": 0,  # 33
    "San Marino": 1,  # 34
    "Lichtenstein": 200,  # 35 (TODO cehck as the transcript says >200)
    "Turkmenistan": 0,  # 36
    "Seycehlles": 84,  # 37
    "Mauritius": 8,  # 38
    "Georgia": None,  # 39 - but not suer now many, think omre than one.. confused
    "Vatican City": 0,  # 40, TODO lsten and check
    # TODO no transcript, 41
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
