"""Location eligibility -- is a job workable from the owner's home country?

Ported from open-jobs' ``tools/locparse.py`` (CC0-1.0,
github.com/elliottdehn/open-jobs) -- turns free-text job locations (+ JD text)
into structured fields, and answers "is this job eligible for someone with
location preference X?" deterministically, no network, no LLM.

Adapted for Hiredra:

* Dropped the ``LOC_TABLE`` / ``location-countries.json`` fallback -- that is
  a nearest-neighbour table open-jobs *publishes* from its own model and would
  mean depending on their live service for an edge case. ``parse()`` here
  never returns a ``country_est`` guess; a location with no parseable
  country/city just has no countries, same as upstream when the table isn't
  loaded.
* Local ``city_countries.json`` (the ~5k bare-city-name table, same CC0
  source) IS kept -- it is a static file, already on disk, no network.
* Everything else (country/US-state/Indian-city dictionaries, the remote/
  hybrid/onsite heuristics, ``RESTRICT_RE``/``MACRO``, ``eligibility()``) is
  the same logic, reformatted to this file's style.

Recognition rules added 2026-10-27 from freehire's ``internal/dict/location``
(MIT), each measured over every job in the database before adoption
(docs/freehire-merge/04-verification.md sec 7): 2-letter country codes and the
US-state clash rule ("KA, IN" and "Bangalore, IN" were read as Indiana), the
GeoNames city table as a last resort (``geonames_cities.tsv``), a spaced dash
as a separator, UK postcodes, region and subdivision names, Cyrillic city
prefixes, and a restriction stated in the title.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

US_STATES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}
STATE_BY_NAME = {v.lower(): k for k, v in US_STATES.items()}
CA_PROV = {
    "ON": "Ontario",
    "QC": "Quebec",
    "BC": "British Columbia",
    "AB": "Alberta",
    "MB": "Manitoba",
    "SK": "Saskatchewan",
    "NS": "Nova Scotia",
    "NB": "New Brunswick",
    # Hiredra addition: the rest of Canada (upstream had eight)
    "NL": "Newfoundland and Labrador",
    "PE": "Prince Edward Island",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "YT": "Yukon",
}
PROV_BY_NAME = {v.lower(): k for k, v in CA_PROV.items()}

COUNTRIES = {
    "united states": "US",
    "usa": "US",
    "u.s.": "US",
    "u.s": "US",
    "u.s.a.": "US",
    "us": "US",
    "america": "US",
    "united states of america": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "great britain": "GB",
    "northern ireland": "GB",
    "canada": "CA",
    "germany": "DE",
    "deutschland": "DE",
    "france": "FR",
    "spain": "ES",
    "españa": "ES",
    "italy": "IT",
    "italia": "IT",
    "netherlands": "NL",
    "the netherlands": "NL",
    "belgium": "BE",
    "switzerland": "CH",
    "austria": "AT",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "ireland": "IE",
    "poland": "PL",
    "polska": "PL",
    "portugal": "PT",
    "czech republic": "CZ",
    "czechia": "CZ",
    "hungary": "HU",
    "romania": "RO",
    "greece": "GR",
    "ukraine": "UA",
    "türkiye": "TR",
    "turkey": "TR",
    "india": "IN",
    "china": "CN",
    "japan": "JP",
    "south korea": "KR",
    "korea": "KR",
    "singapore": "SG",
    "hong kong": "HK",
    "taiwan": "TW",
    "australia": "AU",
    "new zealand": "NZ",
    "philippines": "PH",
    "indonesia": "ID",
    "malaysia": "MY",
    "thailand": "TH",
    "vietnam": "VN",
    "pakistan": "PK",
    "bangladesh": "BD",
    "sri lanka": "LK",
    "mexico": "MX",
    "méxico": "MX",
    "brazil": "BR",
    "brasil": "BR",
    "argentina": "AR",
    "colombia": "CO",
    "chile": "CL",
    "peru": "PE",
    "costa rica": "CR",
    "uruguay": "UY",
    "israel": "IL",
    "united arab emirates": "AE",
    "uae": "AE",
    "dubai": "AE",
    "saudi arabia": "SA",
    "qatar": "QA",
    "egypt": "EG",
    "south africa": "ZA",
    "nigeria": "NG",
    "kenya": "KE",
    "morocco": "MA",
}
CITY_COUNTRY = {
    "london": "GB",
    "manchester": "GB",
    "birmingham": "GB",
    "edinburgh": "GB",
    "dublin": "IE",
    "paris": "FR",
    "berlin": "DE",
    "munich": "DE",
    "münchen": "DE",
    "hamburg": "DE",
    "frankfurt": "DE",
    "amsterdam": "NL",
    "rotterdam": "NL",
    "brussels": "BE",
    "zurich": "CH",
    "zürich": "CH",
    "geneva": "CH",
    "vienna": "AT",
    "wien": "AT",
    "stockholm": "SE",
    "oslo": "NO",
    "copenhagen": "DK",
    "helsinki": "FI",
    "warsaw": "PL",
    "warszawa": "PL",
    "krakow": "PL",
    "kraków": "PL",
    "prague": "CZ",
    "praha": "CZ",
    "lisbon": "PT",
    "lisboa": "PT",
    "madrid": "ES",
    "barcelona": "ES",
    "milan": "IT",
    "milano": "IT",
    "rome": "IT",
    "athens": "GR",
    "tel aviv": "IL",
    "bangalore": "IN",
    "bengaluru": "IN",
    "hyderabad": "IN",
    "pune": "IN",
    "mumbai": "IN",
    "chennai": "IN",
    "gurgaon": "IN",
    "gurugram": "IN",
    "noida": "IN",
    "delhi": "IN",
    "new delhi": "IN",
    "singapore": "SG",
    "tokyo": "JP",
    "seoul": "KR",
    "sydney": "AU",
    "melbourne": "AU",
    "toronto": "CA",
    "vancouver": "CA",
    "montreal": "CA",
    "montréal": "CA",
    "ottawa": "CA",
    "calgary": "CA",
    "mexico city": "MX",
    "são paulo": "BR",
    "sao paulo": "BR",
    "buenos aires": "AR",
    "bogotá": "CO",
    "bogota": "CO",
    "santiago": "CL",
    "lima": "PE",
    "dubai": "AE",
    "hong kong": "HK",
    "taipei": "TW",
    "manila": "PH",
    "jakarta": "ID",
    "kuala lumpur": "MY",
    "bangkok": "TH",
    "ho chi minh city": "VN",
    "hanoi": "VN",
    "cape town": "ZA",
    "johannesburg": "ZA",
    "lagos": "NG",
    "nairobi": "KE",
    "new york": "US",
    "new york city": "US",
    "nyc": "US",
    "san francisco": "US",
    "sf": "US",
    "los angeles": "US",
    "seattle": "US",
    "austin": "US",
    "boston": "US",
    "chicago": "US",
    "denver": "US",
    "atlanta": "US",
    "miami": "US",
    "dallas": "US",
    "houston": "US",
    "washington": "US",
    "san jose": "US",
    "san diego": "US",
    "portland": "US",
    "phoenix": "US",
    "philadelphia": "US",
    "minneapolis": "US",
    "salt lake city": "US",
    "raleigh": "US",
    "charlotte": "US",
    "nashville": "US",
    "pittsburgh": "US",
    "detroit": "US",
    "columbus": "US",
    "san francisco bay area": "US",
    "bay area": "US",
    "silicon valley": "US",
    "remote - usa": "US",
}
REMOTE_RE = re.compile(r"\b(remote|work from home|wfh|distributed|anywhere)\b", re.I)
HYBRID_RE = re.compile(r"\bhybrid\b", re.I)
ONSITE_RE = re.compile(r"\b(on-?site|in-?office|in person)\b", re.I)

ISO3 = {
    "usa": "US",
    "gbr": "GB",
    "ind": "IN",
    "can": "CA",
    "deu": "DE",
    "fra": "FR",
    "aus": "AU",
    "nld": "NL",
    "esp": "ES",
    "ita": "IT",
    "pol": "PL",
    "prt": "PT",
    "irl": "IE",
    "swe": "SE",
    "che": "CH",
    "sgp": "SG",
    "jpn": "JP",
    "bra": "BR",
    "mex": "MX",
    "isr": "IL",
    "are": "AE",
    "phl": "PH",
    "idn": "ID",
    "mys": "MY",
    "vnm": "VN",
    "kor": "KR",
    "chn": "CN",
    "hkg": "HK",
    "twn": "TW",
    "nzl": "NZ",
    "arg": "AR",
    "col": "CO",
    "chl": "CL",
    "zaf": "ZA",
    "nga": "NG",
    "ken": "KE",
    "ukr": "UA",
    "tur": "TR",
    "rou": "RO",
    "hun": "HU",
    "cze": "CZ",
    "grc": "GR",
    "aut": "AT",
    "bel": "BE",
    "dnk": "DK",
    "nor": "NO",
    "fin": "FI",
}
COUNTRIES.update(ISO3)
ISO_NAMES = {
    "afghanistan": "AF",
    "albania": "AL",
    "algeria": "DZ",
    "american samoa": "AS",
    "andorra": "AD",
    "angola": "AO",
    "anguilla": "AI",
    "antarctica": "AQ",
    "antigua and barbuda": "AG",
    "arab republic of egypt": "EG",
    "argentina": "AR",
    "argentine republic": "AR",
    "armenia": "AM",
    "aruba": "AW",
    "ascension and tristan da cunha saint helena": "SH",
    "australia": "AU",
    "austria": "AT",
    "azerbaijan": "AZ",
    "bahamas": "BS",
    "bahrain": "BH",
    "bangladesh": "BD",
    "barbados": "BB",
    "belarus": "BY",
    "belgium": "BE",
    "belize": "BZ",
    "benin": "BJ",
    "bermuda": "BM",
    "bhutan": "BT",
    "bolivarian republic of venezuela": "VE",
    "bolivia": "BO",
    "bolivia, plurinational state of": "BO",
    "bonaire": "BQ",
    "bonaire, sint eustatius and saba": "BQ",
    "bosnia and herzegovina": "BA",
    "botswana": "BW",
    "bouvet island": "BV",
    "brazil": "BR",
    "british indian ocean territory": "IO",
    "british virgin islands": "VG",
    "brunei darussalam": "BN",
    "bulgaria": "BG",
    "burkina faso": "BF",
    "burundi": "BI",
    "cabo verde": "CV",
    "cambodia": "KH",
    "cameroon": "CM",
    "canada": "CA",
    "cayman islands": "KY",
    "central african republic": "CF",
    "chad": "TD",
    "chile": "CL",
    "china": "CN",
    "christmas island": "CX",
    "cocos (keeling) islands": "CC",
    "colombia": "CO",
    "commonwealth of dominica": "DM",
    "commonwealth of the bahamas": "BS",
    "commonwealth of the northern mariana islands": "MP",
    "comoros": "KM",
    "congo, the democratic republic of the": "CD",
    "cook islands": "CK",
    "costa rica": "CR",
    "croatia": "HR",
    "cuba": "CU",
    "curaçao": "CW",
    "cyprus": "CY",
    "czech republic": "CZ",
    "czechia": "CZ",
    "côte d'ivoire": "CI",
    "democratic people's republic of korea": "KP",
    "democratic republic of sao tome and principe": "ST",
    "democratic republic of timor-leste": "TL",
    "democratic socialist republic of sri lanka": "LK",
    "denmark": "DK",
    "djibouti": "DJ",
    "dominica": "DM",
    "dominican republic": "DO",
    "eastern republic of uruguay": "UY",
    "ecuador": "EC",
    "egypt": "EG",
    "el salvador": "SV",
    "equatorial guinea": "GQ",
    "eritrea": "ER",
    "estonia": "EE",
    "eswatini": "SZ",
    "ethiopia": "ET",
    "falkland islands (malvinas)": "FK",
    "faroe islands": "FO",
    "federal democratic republic of ethiopia": "ET",
    "federal democratic republic of nepal": "NP",
    "federal republic of germany": "DE",
    "federal republic of nigeria": "NG",
    "federal republic of somalia": "SO",
    "federated states of micronesia": "FM",
    "federative republic of brazil": "BR",
    "fiji": "FJ",
    "finland": "FI",
    "france": "FR",
    "french guiana": "GF",
    "french polynesia": "PF",
    "french republic": "FR",
    "french southern territories": "TF",
    "gabon": "GA",
    "gabonese republic": "GA",
    "gambia": "GM",
    "germany": "DE",
    "ghana": "GH",
    "gibraltar": "GI",
    "grand duchy of luxembourg": "LU",
    "greece": "GR",
    "greenland": "GL",
    "grenada": "GD",
    "guadeloupe": "GP",
    "guam": "GU",
    "guatemala": "GT",
    "guernsey": "GG",
    "guinea": "GN",
    "guinea-bissau": "GW",
    "guyana": "GY",
    "haiti": "HT",
    "hashemite kingdom of jordan": "JO",
    "heard island and mcdonald islands": "HM",
    "hellenic republic": "GR",
    "holy see (vatican city state)": "VA",
    "honduras": "HN",
    "hong kong": "HK",
    "hong kong special administrative region of china": "HK",
    "hungary": "HU",
    "iceland": "IS",
    "independent state of papua new guinea": "PG",
    "independent state of samoa": "WS",
    "india": "IN",
    "indonesia": "ID",
    "iran": "IR",
    "iran, islamic republic of": "IR",
    "iraq": "IQ",
    "ireland": "IE",
    "islamic republic of afghanistan": "AF",
    "islamic republic of iran": "IR",
    "islamic republic of mauritania": "MR",
    "islamic republic of pakistan": "PK",
    "isle of man": "IM",
    "israel": "IL",
    "italian republic": "IT",
    "italy": "IT",
    "jamaica": "JM",
    "japan": "JP",
    "jersey": "JE",
    "jordan": "JO",
    "kazakhstan": "KZ",
    "kenya": "KE",
    "kingdom of bahrain": "BH",
    "kingdom of belgium": "BE",
    "kingdom of bhutan": "BT",
    "kingdom of cambodia": "KH",
    "kingdom of denmark": "DK",
    "kingdom of eswatini": "SZ",
    "kingdom of lesotho": "LS",
    "kingdom of morocco": "MA",
    "kingdom of norway": "NO",
    "kingdom of saudi arabia": "SA",
    "kingdom of spain": "ES",
    "kingdom of sweden": "SE",
    "kingdom of thailand": "TH",
    "kingdom of the netherlands": "NL",
    "kingdom of tonga": "TO",
    "kiribati": "KI",
    "korea, democratic people's republic of": "KP",
    "korea, republic of": "KR",
    "kuwait": "KW",
    "kyrgyz republic": "KG",
    "kyrgyzstan": "KG",
    "lao people's democratic republic": "LA",
    "laos": "LA",
    "latvia": "LV",
    "lebanese republic": "LB",
    "lebanon": "LB",
    "lesotho": "LS",
    "liberia": "LR",
    "libya": "LY",
    "liechtenstein": "LI",
    "lithuania": "LT",
    "luxembourg": "LU",
    "macao": "MO",
    "macao special administrative region of china": "MO",
    "madagascar": "MG",
    "malawi": "MW",
    "malaysia": "MY",
    "maldives": "MV",
    "mali": "ML",
    "malta": "MT",
    "marshall islands": "MH",
    "martinique": "MQ",
    "mauritania": "MR",
    "mauritius": "MU",
    "mayotte": "YT",
    "mexico": "MX",
    "micronesia, federated states of": "FM",
    "moldova": "MD",
    "moldova, republic of": "MD",
    "monaco": "MC",
    "mongolia": "MN",
    "montenegro": "ME",
    "montserrat": "MS",
    "morocco": "MA",
    "mozambique": "MZ",
    "myanmar": "MM",
    "namibia": "NA",
    "nauru": "NR",
    "nepal": "NP",
    "netherlands": "NL",
    "new caledonia": "NC",
    "new zealand": "NZ",
    "nicaragua": "NI",
    "niger": "NE",
    "nigeria": "NG",
    "niue": "NU",
    "norfolk island": "NF",
    "north korea": "KP",
    "north macedonia": "MK",
    "northern mariana islands": "MP",
    "norway": "NO",
    "oman": "OM",
    "pakistan": "PK",
    "palau": "PW",
    "palestine": "PS",
    "palestine, state of": "PS",
    "panama": "PA",
    "papua new guinea": "PG",
    "paraguay": "PY",
    "people's democratic republic of algeria": "DZ",
    "people's republic of bangladesh": "BD",
    "people's republic of china": "CN",
    "peru": "PE",
    "philippines": "PH",
    "pitcairn": "PN",
    "plurinational state of bolivia": "BO",
    "poland": "PL",
    "portugal": "PT",
    "portuguese republic": "PT",
    "principality of andorra": "AD",
    "principality of liechtenstein": "LI",
    "principality of monaco": "MC",
    "province of china taiwan": "TW",
    "puerto rico": "PR",
    "qatar": "QA",
    "republic of albania": "AL",
    "republic of angola": "AO",
    "republic of armenia": "AM",
    "republic of austria": "AT",
    "republic of azerbaijan": "AZ",
    "republic of belarus": "BY",
    "republic of benin": "BJ",
    "republic of bosnia and herzegovina": "BA",
    "republic of botswana": "BW",
    "republic of bulgaria": "BG",
    "republic of burundi": "BI",
    "republic of cabo verde": "CV",
    "republic of cameroon": "CM",
    "republic of chad": "TD",
    "republic of chile": "CL",
    "republic of colombia": "CO",
    "republic of costa rica": "CR",
    "republic of croatia": "HR",
    "republic of cuba": "CU",
    "republic of cyprus": "CY",
    "republic of côte d'ivoire": "CI",
    "republic of djibouti": "DJ",
    "republic of ecuador": "EC",
    "republic of el salvador": "SV",
    "republic of equatorial guinea": "GQ",
    "republic of estonia": "EE",
    "republic of fiji": "FJ",
    "republic of finland": "FI",
    "republic of ghana": "GH",
    "republic of guatemala": "GT",
    "republic of guinea": "GN",
    "republic of guinea-bissau": "GW",
    "republic of guyana": "GY",
    "republic of haiti": "HT",
    "republic of honduras": "HN",
    "republic of iceland": "IS",
    "republic of india": "IN",
    "republic of indonesia": "ID",
    "republic of iraq": "IQ",
    "republic of kazakhstan": "KZ",
    "republic of kenya": "KE",
    "republic of kiribati": "KI",
    "republic of korea": "KR",
    "republic of latvia": "LV",
    "republic of liberia": "LR",
    "republic of lithuania": "LT",
    "republic of madagascar": "MG",
    "republic of malawi": "MW",
    "republic of maldives": "MV",
    "republic of mali": "ML",
    "republic of malta": "MT",
    "republic of mauritius": "MU",
    "republic of moldova": "MD",
    "republic of mozambique": "MZ",
    "republic of myanmar": "MM",
    "republic of namibia": "NA",
    "republic of nauru": "NR",
    "republic of nicaragua": "NI",
    "republic of north macedonia": "MK",
    "republic of palau": "PW",
    "republic of panama": "PA",
    "republic of paraguay": "PY",
    "republic of peru": "PE",
    "republic of poland": "PL",
    "republic of san marino": "SM",
    "republic of senegal": "SN",
    "republic of serbia": "RS",
    "republic of seychelles": "SC",
    "republic of sierra leone": "SL",
    "republic of singapore": "SG",
    "republic of slovenia": "SI",
    "republic of south africa": "ZA",
    "republic of south sudan": "SS",
    "republic of suriname": "SR",
    "republic of tajikistan": "TJ",
    "republic of the congo": "CG",
    "republic of the gambia": "GM",
    "republic of the marshall islands": "MH",
    "republic of the niger": "NE",
    "republic of the philippines": "PH",
    "republic of the sudan": "SD",
    "republic of trinidad and tobago": "TT",
    "republic of tunisia": "TN",
    "republic of türkiye": "TR",
    "republic of uganda": "UG",
    "republic of uzbekistan": "UZ",
    "republic of vanuatu": "VU",
    "republic of yemen": "YE",
    "republic of zambia": "ZM",
    "republic of zimbabwe": "ZW",
    "romania": "RO",
    "russian federation": "RU",
    "rwanda": "RW",
    "rwandese republic": "RW",
    "réunion": "RE",
    "saint barthélemy": "BL",
    "saint helena": "SH",
    "saint helena, ascension and tristan da cunha": "SH",
    "saint kitts and nevis": "KN",
    "saint lucia": "LC",
    "saint martin (french part)": "MF",
    "saint pierre and miquelon": "PM",
    "saint vincent and the grenadines": "VC",
    "samoa": "WS",
    "san marino": "SM",
    "sao tome and principe": "ST",
    "saudi arabia": "SA",
    "senegal": "SN",
    "serbia": "RS",
    "seychelles": "SC",
    "sierra leone": "SL",
    "singapore": "SG",
    "sint eustatius and saba bonaire": "BQ",
    "sint maarten (dutch part)": "SX",
    "slovak republic": "SK",
    "slovakia": "SK",
    "slovenia": "SI",
    "socialist republic of viet nam": "VN",
    "solomon islands": "SB",
    "somalia": "SO",
    "south africa": "ZA",
    "south georgia and the south sandwich islands": "GS",
    "south korea": "KR",
    "south sudan": "SS",
    "spain": "ES",
    "sri lanka": "LK",
    "state of israel": "IL",
    "state of kuwait": "KW",
    "state of palestine": "PS",
    "state of qatar": "QA",
    "sudan": "SD",
    "sultanate of oman": "OM",
    "suriname": "SR",
    "svalbard and jan mayen": "SJ",
    "sweden": "SE",
    "swiss confederation": "CH",
    "switzerland": "CH",
    "syria": "SY",
    "syrian arab republic": "SY",
    "taiwan": "TW",
    "taiwan, province of china": "TW",
    "tajikistan": "TJ",
    "tanzania": "TZ",
    "tanzania, united republic of": "TZ",
    "thailand": "TH",
    "the democratic republic of the congo": "CD",
    "the state of eritrea": "ER",
    "the state of palestine": "PS",
    "timor-leste": "TL",
    "togo": "TG",
    "togolese republic": "TG",
    "tokelau": "TK",
    "tonga": "TO",
    "trinidad and tobago": "TT",
    "tunisia": "TN",
    "turkmenistan": "TM",
    "turks and caicos islands": "TC",
    "tuvalu": "TV",
    "türkiye": "TR",
    "u.s. virgin islands": "VI",
    "uganda": "UG",
    "ukraine": "UA",
    "union of the comoros": "KM",
    "united arab emirates": "AE",
    "united kingdom": "GB",
    "united kingdom of great britain and northern ireland": "GB",
    "united mexican states": "MX",
    "united republic of tanzania": "TZ",
    "united states": "US",
    "united states minor outlying islands": "UM",
    "united states of america": "US",
    "uruguay": "UY",
    "uzbekistan": "UZ",
    "vanuatu": "VU",
    "venezuela": "VE",
    "venezuela, bolivarian republic of": "VE",
    "viet nam": "VN",
    "vietnam": "VN",
    "virgin islands of the united states": "VI",
    "virgin islands, british": "VG",
    "virgin islands, u.s.": "VI",
    "wallis and futuna": "WF",
    "western sahara": "EH",
    "yemen": "YE",
    "zambia": "ZM",
    "zimbabwe": "ZW",
    "åland islands": "AX",
}
ISO3_ALL = {
    "abw": "AW",
    "afg": "AF",
    "ago": "AO",
    "aia": "AI",
    "ala": "AX",
    "alb": "AL",
    "and": "AD",
    "are": "AE",
    "arg": "AR",
    "arm": "AM",
    "asm": "AS",
    "ata": "AQ",
    "atf": "TF",
    "atg": "AG",
    "aus": "AU",
    "aut": "AT",
    "aze": "AZ",
    "bdi": "BI",
    "bel": "BE",
    "ben": "BJ",
    "bes": "BQ",
    "bfa": "BF",
    "bgd": "BD",
    "bgr": "BG",
    "bhr": "BH",
    "bhs": "BS",
    "bih": "BA",
    "blm": "BL",
    "blr": "BY",
    "blz": "BZ",
    "bmu": "BM",
    "bol": "BO",
    "bra": "BR",
    "brb": "BB",
    "brn": "BN",
    "btn": "BT",
    "bvt": "BV",
    "bwa": "BW",
    "caf": "CF",
    "can": "CA",
    "cck": "CC",
    "che": "CH",
    "chl": "CL",
    "chn": "CN",
    "civ": "CI",
    "cmr": "CM",
    "cod": "CD",
    "cog": "CG",
    "cok": "CK",
    "col": "CO",
    "com": "KM",
    "cpv": "CV",
    "cri": "CR",
    "cub": "CU",
    "cuw": "CW",
    "cxr": "CX",
    "cym": "KY",
    "cyp": "CY",
    "cze": "CZ",
    "deu": "DE",
    "dji": "DJ",
    "dma": "DM",
    "dnk": "DK",
    "dom": "DO",
    "dza": "DZ",
    "ecu": "EC",
    "egy": "EG",
    "eri": "ER",
    "esh": "EH",
    "esp": "ES",
    "est": "EE",
    "eth": "ET",
    "fin": "FI",
    "fji": "FJ",
    "flk": "FK",
    "fra": "FR",
    "fro": "FO",
    "fsm": "FM",
    "gab": "GA",
    "gbr": "GB",
    "geo": "GE",
    "ggy": "GG",
    "gha": "GH",
    "gib": "GI",
    "gin": "GN",
    "glp": "GP",
    "gmb": "GM",
    "gnb": "GW",
    "gnq": "GQ",
    "grc": "GR",
    "grd": "GD",
    "grl": "GL",
    "gtm": "GT",
    "guf": "GF",
    "gum": "GU",
    "guy": "GY",
    "hkg": "HK",
    "hmd": "HM",
    "hnd": "HN",
    "hrv": "HR",
    "hti": "HT",
    "hun": "HU",
    "idn": "ID",
    "imn": "IM",
    "ind": "IN",
    "iot": "IO",
    "irl": "IE",
    "irn": "IR",
    "irq": "IQ",
    "isl": "IS",
    "isr": "IL",
    "ita": "IT",
    "jam": "JM",
    "jey": "JE",
    "jor": "JO",
    "jpn": "JP",
    "kaz": "KZ",
    "ken": "KE",
    "kgz": "KG",
    "khm": "KH",
    "kir": "KI",
    "kna": "KN",
    "kor": "KR",
    "kwt": "KW",
    "lao": "LA",
    "lbn": "LB",
    "lbr": "LR",
    "lby": "LY",
    "lca": "LC",
    "lie": "LI",
    "lka": "LK",
    "lso": "LS",
    "ltu": "LT",
    "lux": "LU",
    "lva": "LV",
    "mac": "MO",
    "maf": "MF",
    "mar": "MA",
    "mco": "MC",
    "mda": "MD",
    "mdg": "MG",
    "mdv": "MV",
    "mex": "MX",
    "mhl": "MH",
    "mkd": "MK",
    "mli": "ML",
    "mlt": "MT",
    "mmr": "MM",
    "mne": "ME",
    "mng": "MN",
    "mnp": "MP",
    "moz": "MZ",
    "mrt": "MR",
    "msr": "MS",
    "mtq": "MQ",
    "mus": "MU",
    "mwi": "MW",
    "mys": "MY",
    "myt": "YT",
    "nam": "NA",
    "ncl": "NC",
    "ner": "NE",
    "nfk": "NF",
    "nga": "NG",
    "nic": "NI",
    "niu": "NU",
    "nld": "NL",
    "nor": "NO",
    "npl": "NP",
    "nru": "NR",
    "nzl": "NZ",
    "omn": "OM",
    "pak": "PK",
    "pan": "PA",
    "pcn": "PN",
    "per": "PE",
    "phl": "PH",
    "plw": "PW",
    "png": "PG",
    "pol": "PL",
    "pri": "PR",
    "prk": "KP",
    "prt": "PT",
    "pry": "PY",
    "pse": "PS",
    "pyf": "PF",
    "qat": "QA",
    "reu": "RE",
    "rou": "RO",
    "rus": "RU",
    "rwa": "RW",
    "sau": "SA",
    "sdn": "SD",
    "sen": "SN",
    "sgp": "SG",
    "sgs": "GS",
    "shn": "SH",
    "sjm": "SJ",
    "slb": "SB",
    "sle": "SL",
    "slv": "SV",
    "smr": "SM",
    "som": "SO",
    "spm": "PM",
    "srb": "RS",
    "ssd": "SS",
    "stp": "ST",
    "sur": "SR",
    "svk": "SK",
    "svn": "SI",
    "swe": "SE",
    "swz": "SZ",
    "sxm": "SX",
    "syc": "SC",
    "syr": "SY",
    "tca": "TC",
    "tcd": "TD",
    "tgo": "TG",
    "tha": "TH",
    "tjk": "TJ",
    "tkl": "TK",
    "tkm": "TM",
    "tls": "TL",
    "ton": "TO",
    "tto": "TT",
    "tun": "TN",
    "tur": "TR",
    "tuv": "TV",
    "twn": "TW",
    "tza": "TZ",
    "uga": "UG",
    "ukr": "UA",
    "umi": "UM",
    "ury": "UY",
    "usa": "US",
    "uzb": "UZ",
    "vat": "VA",
    "vct": "VC",
    "ven": "VE",
    "vgb": "VG",
    "vir": "VI",
    "vnm": "VN",
    "vut": "VU",
    "wlf": "WF",
    "wsm": "WS",
    "yem": "YE",
    "zaf": "ZA",
    "zmb": "ZM",
    "zwe": "ZW",
}
for _k, _v in ISO_NAMES.items():
    COUNTRIES.setdefault(_k, _v)
for _k, _v in ISO3_ALL.items():
    COUNTRIES.setdefault(_k, _v)
# "Georgia" is both a US state and a country: bare "Georgia" stays the state (far more common in job
# postings); these forms, or a Georgian city in the same segment, mean the country.
COUNTRIES.update(
    {
        "nederland": "NL",
        "österreich": "AT",
        "oesterreich": "AT",
        "schweiz": "CH",
        "suisse": "CH",
        "svizzera": "CH",
        "polska": "PL",
        "česko": "CZ",
        "cesko": "CZ",
        "česká republika": "CZ",
        "sverige": "SE",
        "norge": "NO",
        "danmark": "DK",
        "suomi": "FI",
        "magyarország": "HU",
        "românia": "RO",
        "belgië": "BE",
        "belgique": "BE",
        "éire": "IE",
        "lietuva": "LT",
        "latvija": "LV",
        "eesti": "EE",
        "hrvatska": "HR",
        "srbija": "RS",
        "slovensko": "SK",
        "slovenija": "SI",
        "българия": "BG",
        "ísland": "IS",
        "perú": "PE",
        "日本": "JP",
        "中国": "CN",
        "대한민국": "KR",
        "한국": "KR",
        "việt nam": "VN",
        "भारत": "IN",
        "россия": "RU",
        "україна": "UA",
        "ελλάδα": "GR",
        "台灣": "TW",
        "香港": "HK",
        "المملكة العربية السعودية": "SA",
        "الإمارات": "AE",
        "मुंबई": "IN",
        "united states of america (usa)": "US",
    }
)
COUNTRIES.update(
    {
        "democratic republic of the congo": "CD",
        "dr congo": "CD",
        "drc": "CD",
        "congo": "CG",
        "republic of the congo": "CG",
        "congo-brazzaville": "CG",
        "congo-kinshasa": "CD",
        "georgia (country)": "GE",
        "country of georgia": "GE",
        "republic of georgia": "GE",
        "sakartvelo": "GE",
        "georgia country": "GE",
    }
)
CITY_COUNTRY.update({"tbilisi": "GE", "batumi": "GE", "kutaisi": "GE"})

try:  # ~5k bare city names that resolve to one country >=90% of the time (open-jobs' build-city-table.py)
    _city_table = json.loads(Path(__file__).with_name("city_countries.json").read_text(encoding="utf-8"))
    # Hiredra (2026-10-27): the table was built from job-location strings, so it also maps state/country
    # codes and plain words to the country they were mostly seen with ("in"/"field"/"home" -> US). Codes are
    # read by the subdivision/country rules in _one; the words are not places. Both are skipped.
    _codes = {c.lower() for c in (*US_STATES, *CA_PROV, *ISO_NAMES.values())}
    _not_places = {
        "all", "commerce", "country", "countryside", "down", "early", "export", "field", "friendly", "harmony",
        "hazard", "headquarters", "home", "hotel", "king", "mena", "nationwide", "null", "ocean", "pacific",
        "page", "price", "purchase", "rehab", "reserve", "shape", "star", "statewide", "supply", "unavailable",
        "united", "unity", "valley", "various", "welcome",
    }  # fmt: skip
    for _k, _v in _city_table.items():
        if _k not in _codes and _k not in _not_places:
            CITY_COUNTRY.setdefault(_k, _v)
except Exception:
    pass

GEORGIAN_CITIES = {"tbilisi", "batumi", "kutaisi", "rustavi"}

# Hiredra addition (2026-10-27): the names freehire's hand-curated nameToCountry pins and ours lacked
# (github.com/strelov1/freehire internal/dict/location/dictionaries.go, MIT): country names in other
# languages and high-frequency cities ("Bristol", "Belgrade", "Malaga", "Stuttgart", "Pangyo", "Coimbatore").
_FREEHIRE_COUNTRY_NAMES = {
    "u.k.": "GB", "britain": "GB", "russia": "RU", "рф": "RU", "беларусь": "BY", "узбекистан": "UZ",
    "казахстан": "KZ", "украина": "UA", "ivory coast": "CI", "cote d'ivoire": "CI", "burma": "MM",
    "palestinian territories": "PS", "alemania": "DE", "méjico": "MX", "francia": "FR", "países bajos": "NL",
    "reino unido": "GB", "estados unidos": "US", "suiza": "CH", "suecia": "SE", "polonia": "PL", "rumanía": "RO",
    "hungría": "HU", "turquía": "TR", "japón": "JP", "filipinas": "PH", "malasia": "MY", "corea del sur": "KR",
    "emiratos árabes unidos": "AE", "arabia saudita": "SA", "egipto": "EG", "espanha": "ES", "alemanha": "DE",
    "frança": "FR", "grécia": "GR", "países baixos": "NL", "suíça": "CH", "polónia": "PL", "roménia": "RO",
    "hungria": "HU", "turquia": "TR", "coreia do sul": "KR", "emirados árabes unidos": "AE", "arábia saudita": "SA",
    "egito": "EG", "frankreich": "FR", "spanien": "ES", "italien": "IT", "niederlande": "NL",
    "vereinigtes königreich": "GB", "vereinigte staaten": "US", "schweden": "SE", "polen": "PL", "rumänien": "RO",
    "ungarn": "HU", "türkei": "TR", "griechenland": "GR", "philippinen": "PH", "indonesien": "ID", "südkorea": "KR",
    "ägypten": "EG", "belgien": "BE", "dänemark": "DK", "finnland": "FI", "irland": "IE", "norwegen": "NO",
    "tschechien": "CZ", "bulgarien": "BG", "saudi-arabien": "SA", "vereinigte arabische emirate": "AE",
    
}  # fmt: skip
_FREEHIRE_CITIES = {
    "méxico city": "MX", "pangyo": "KR", "valletta": "MT",
    "bristol": "GB", "liverpool": "GB", "leeds": "GB", "brisbane": "AU", "perth": "AU", "wellington": "NZ",
    "moscow": "RU", "lvov": "UA", "kharkiv": "UA", "kharkov": "UA", "dnipro": "UA", "dnepr": "UA",
    "dnepropetrovsk": "UA", "vinnytsia": "UA", "ivano-frankivsk": "UA", "zaporizhzhia": "UA", "zaporozhye": "UA",
    "chernivtsi": "UA", "chernovtsy": "UA", "ternopil": "UA", "ternopol": "UA", "uzhhorod": "UA", "uzhgorod": "UA",
    "rivne": "UA", "rovno": "UA", "poltava": "UA", "khmelnytskyi": "UA", "zhytomyr": "UA", "zhitomir": "UA",
    "sumy": "UA", "mykolaiv": "UA", "chernihiv": "UA", "chernigov": "UA", "kryvyi rih": "UA", "krivoy rog": "UA",
    "kherson": "UA", "kropyvnytskyi": "UA", "luhansk": "UA", "toshkent": "UZ", "samarkand": "UZ", "nur-sultan": "KZ",
    "bishkek": "KG", "dushanbe": "TJ", "ashgabat": "TM", "chisinau": "MD", "baku": "AZ", "спб": "RU", "питер": "RU",
    "казань": "RU", "самара": "RU", "краснодар": "RU", "ростов-на-дону": "RU", "воронеж": "RU", "уфа": "RU",
    "пермь": "RU", "челябинск": "RU", "волгоград": "RU", "красноярск": "RU", "тюмень": "RU", "саратов": "RU",
    "тольятти": "RU", "ижевск": "RU", "ульяновск": "RU", "барнаул": "RU", "владивосток": "RU", "хабаровск": "RU",
    "иркутск": "RU", "ярославль": "RU", "томск": "RU", "оренбург": "RU", "кемерово": "RU", "рязань": "RU",
    "набережные челны": "RU", "пенза": "RU", "липецк": "RU", "тула": "RU", "киров": "RU", "чебоксары": "RU",
    "калининград": "RU", "ставрополь": "RU", "сочи": "RU", "иваново": "RU", "брянск": "RU", "белгород": "RU",
    "сургут": "RU", "владимир": "RU", "архангельск": "RU", "калуга": "RU", "смоленск": "RU", "волжский": "RU",
    "курск": "RU", "орёл": "RU", "череповец": "RU", "вологда": "RU", "магнитогорск": "RU", "тамбов": "RU",
    "мурманск": "RU", "тверь": "RU", "новокузнецк": "RU", "астрахань": "RU", "великий новгород": "RU", "псков": "RU",
    "чита": "RU", "улан-удэ": "RU", "якутск": "RU", "норильск": "RU", "новороссийск": "RU", "таганрог": "RU",
    "сарапул": "RU", "майкоп": "RU", "подольск": "RU", "химки": "RU", "мытищи": "RU", "балашиха": "RU",
    "курган": "RU", "саранск": "RU", "йошкар-ола": "RU", "благовещенск": "RU", "кисловодск": "RU",
    "петропавловск-камчатский": "RU", "комсомольск-на-амуре": "RU", "новый уренгой": "RU", "минск": "BY",
    "ташкент": "UZ", "алматы": "KZ", "астана": "KZ", "ереван": "AM", "баку": "AZ", "киев": "UA", "київ": "UA",
    "львів": "UA", "львов": "UA", "харків": "UA", "харьков": "UA", "одеса": "UA", "одесса": "UA", "дніпро": "UA",
    "днепр": "UA", "вінниця": "UA", "винница": "UA", "івано-франківськ": "UA", "ивано-франковск": "UA",
    "запоріжжя": "UA", "запорожье": "UA", "чернівці": "UA", "черновцы": "UA", "тернопіль": "UA", "тернополь": "UA",
    "ужгород": "UA", "рівне": "UA", "ровно": "UA", "полтава": "UA", "хмельницький": "UA", "хмельницкий": "UA",
    "житомир": "UA", "черкаси": "UA", "суми": "UA", "сумы": "UA", "миколаїв": "UA", "чернігів": "UA",
    "чернигов": "UA", "кривий ріг": "UA", "кривой рог": "UA", "херсон": "UA", "кропивницький": "UA",
    "луганськ": "UA", "kosovo": "XK", "brunei": "BN", "macau": "MO", "tailandia": "TH", "japão": "JP",
    "florianópolis": "BR", "medellín": "CO", "medellin": "CO", "montevideo": "UY", "busan": "KR", "coimbatore": "IN",
    "köln": "DE", "stuttgart": "DE", "düsseldorf": "DE", "dusseldorf": "DE", "leipzig": "DE", "dresden": "DE",
    "geneve": "CH", "valencia": "ES", "seville": "ES", "malaga": "ES", "bergen": "NO", "belgrade": "RS",
    "bratislava": "SK", "reykjavik": "IS", "reykjavík": "IS", "sharjah": "AE",
}  # fmt: skip
for _k, _v in _FREEHIRE_COUNTRY_NAMES.items():
    COUNTRIES.setdefault(_k, _v)
for _k, _v in _FREEHIRE_CITIES.items():
    CITY_COUNTRY.setdefault(_k, _v)
CITY_COUNTRY["бишкек"] = "KG"  # the open-jobs table had US

# Hiredra addition (2026-10-27): region names below country level state their country ("Kerala",
# "Schleswig-Holstein", "New South Wales"). freehire's foreignSubdivisionNames (foreign_subdivisions.go, MIT;
# names only, "Punjab"/"Victoria" withheld there as ambiguous) plus every Indian state and union territory
# and the misspellings seen in our postings ("Gujrat" was read as Gujrat, Pakistan).
_SUBDIVISION_NAMES = {
    "республика адыгея": "RU", "адыгея": "RU", "республика алтай": "RU", "республика башкортостан": "RU",
    "башкортостан": "RU", "башкирия": "RU", "республика бурятия": "RU", "бурятия": "RU", "республика дагестан": "RU",
    "дагестан": "RU", "республика ингушетия": "RU", "ингушетия": "RU", "кабардино-балкарская республика": "RU",
    "кабардино-балкария": "RU", "республика калмыкия": "RU", "калмыкия": "RU",
    "карачаево-черкесская республика": "RU", "карачаево-черкесия": "RU", "республика карелия": "RU", "карелия": "RU",
    "республика коми": "RU", "республика марий эл": "RU", "марий эл": "RU", "республика мордовия": "RU",
    "мордовия": "RU", "республика саха": "RU", "якутия": "RU", "республика северная осетия": "RU",
    "северная осетия": "RU", "республика татарстан": "RU", "татарстан": "RU", "республика тыва": "RU", "тыва": "RU",
    "удмуртская республика": "RU", "удмуртия": "RU", "республика хакасия": "RU", "хакасия": "RU",
    "чеченская республика": "RU", "чечня": "RU", "чувашская республика": "RU", "чувашия": "RU",
    "алтайский край": "RU", "забайкальский край": "RU", "камчатский край": "RU", "краснодарский край": "RU",
    "красноярский край": "RU", "пермский край": "RU", "приморский край": "RU", "ставропольский край": "RU",
    "хабаровский край": "RU", "амурская область": "RU", "архангельская область": "RU", "астраханская область": "RU",
    "белгородская область": "RU", "брянская область": "RU", "владимирская область": "RU",
    "волгоградская область": "RU", "вологодская область": "RU", "воронежская область": "RU",
    "ивановская область": "RU", "иркутская область": "RU", "калининградская область": "RU",
    "калужская область": "RU", "кемеровская область": "RU", "кузбасс": "RU", "кировская область": "RU",
    "костромская область": "RU", "курганская область": "RU", "курская область": "RU", "ленинградская область": "RU",
    "липецкая область": "RU", "магаданская область": "RU", "московская область": "RU", "мурманская область": "RU",
    "нижегородская область": "RU", "новгородская область": "RU", "новосибирская область": "RU",
    "омская область": "RU", "оренбургская область": "RU", "орловская область": "RU", "пензенская область": "RU",
    "псковская область": "RU", "ростовская область": "RU", "рязанская область": "RU", "самарская область": "RU",
    "саратовская область": "RU", "сахалинская область": "RU", "свердловская область": "RU",
    "смоленская область": "RU", "тамбовская область": "RU", "тверская область": "RU", "томская область": "RU",
    "тульская область": "RU", "тюменская область": "RU", "ульяновская область": "RU", "челябинская область": "RU",
    "ярославская область": "RU", "еврейская автономная область": "RU", "ненецкий автономный округ": "RU",
    "ханты-мансийский автономный округ": "RU", "югра": "RU", "чукотский автономный округ": "RU",
    "ямало-ненецкий автономный округ": "RU", "moscow oblast": "RU", "moscow region": "RU", "leningrad oblast": "RU",
    "sverdlovsk oblast": "RU", "tatarstan": "RU", "bashkortostan": "RU", "tyne and wear": "GB",
    "north yorkshire": "GB", "north east england": "GB", "north west england": "GB",
    "yorkshire and the humber": "GB", "east of england": "GB", "south east england": "GB",
    "south west england": "GB", "qld": "AU", "south australia": "AU", "tasmania": "AU", "northern territory": "AU",
    "australian capital territory": "AU", "bayern": "DE", "bavaria": "DE", "baden-württemberg": "DE",
    "baden-wurttemberg": "DE", "baden-wuerttemberg": "DE", "north rhine-westphalia": "DE", "hessen": "DE",
    "hesse": "DE", "niedersachsen": "DE", "lower saxony": "DE", "sachsen": "DE", "saxony": "DE",
    "sachsen-anhalt": "DE", "saxony-anhalt": "DE", "rheinland-pfalz": "DE", "rhineland-palatinate": "DE",
    "schleswig-holstein": "DE", "thüringen": "DE", "thuringia": "DE", "mecklenburg-vorpommern": "DE",
    "saarland": "DE", "brandenburg": "DE", "kerala": "IN", "rajasthan": "IN", "west bengal": "IN",
    "andhra pradesh": "IN", "odisha": "IN", "bihar": "IN", "assam": "IN", "jharkhand": "IN", "chhattisgarh": "IN",
    "uttarakhand": "IN", "himachal pradesh": "IN", "bengaluru urban": "IN", "arunachal pradesh": "IN",
    "manipur": "IN", "meghalaya": "IN", "mizoram": "IN", "nagaland": "IN", "sikkim": "IN", "tripura": "IN",
    "andaman and nicobar islands": "IN", "dadra and nagar haveli": "IN", "daman and diu": "IN",
    "jammu and kashmir": "IN", "ladakh": "IN", "lakshadweep": "IN", "puducherry": "IN", "delhi ncr": "IN",
    "gujrat": "IN", "tamilnadu": "IN", "uttrakhand": "IN", "telengana": "IN", "orissa": "IN", "pondicherry": "IN",
    "chattisgarh": "IN", "maharastra": "IN",
}  # fmt: skip
for _k, _v in _SUBDIVISION_NAMES.items():
    COUNTRIES.setdefault(_k, _v)


# Hiredra addition: region spellings seen in our postings ("Home Based - Americas", "Latin America").
_REGION_ALIASES = {
    "americas": "Americas",
    "the americas": "Americas",
    "latin america": "Latam",
    "latinoamerica": "Latam",
    "latinoamérica": "Latam",
    "mena": "Mena",
    "middle east": "Mena",
    "cis": "Cis",
    "central asia": "Cis",
}


def _load_geonames() -> dict[str, tuple[str, bool]]:
    """alias -> (ISO-2 country, contested) from geonames_cities.tsv (freehire's cities1000 export).

    Rows are population-sorted, so the first claimant of an alias wins; a trailing '*' marks an alias that
    more than one country claims (counted over the whole GeoNames dump), which states no country.
    """
    out: dict[str, tuple[str, bool]] = {}
    try:
        text = Path(__file__).with_name("geonames_cities.tsv").read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        for alias in parts[2].split("|"):
            out.setdefault(alias.rstrip("*"), (parts[1].upper(), alias.endswith("*")))
    return out


# Hiredra addition (2026-10-27, docs/freehire-merge/04-verification.md sec 7), from freehire's
# internal/dict/location (MIT). ~34k GeoNames places: the evidence that a city is North American (to read a
# clashing 2-letter code) and the last-resort country of a line that states none.
_GEONAMES = _load_geonames()

# A 2-letter token that is both a US state / Canadian province and an ISO country code ("IN" Indiana or
# India, "DE", "CO", "IL", "PE"...) is read as the subdivision only when the city before it is North
# American. For these codes the state reading dominates job postings, so it is the default after any city
# not placed in the clashing country itself -- freehire's measured stateFirstSubdivisions (IN, DE, CO, ID,
# AR, TN, MT measured the other way: the country wins).
_STATE_FIRST = {"PA", "MA", "MD", "IL", "MN", "AL", "MO", "LA", "ME"}
_ISO2 = set(ISO_NAMES.values())
# Indian state / union-territory codes as some ATSs write them before the country: "KA, IN", "TN, IN".
_INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "CT", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA",
    "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TG", "TN", "TR",
    "TS", "UK", "UL", "UP", "UT", "WB",
}  # fmt: skip


def _city_country(name: str) -> str | None:
    """The country a city name points to (curated table first, then GeoNames' most populous claimant)."""
    n = name.lower().strip(" .")
    if n in CITY_COUNTRY:
        return CITY_COUNTRY[n]
    hit = _GEONAMES.get(n)
    return hit[0] if hit else None


def _subdivision_wins(code: str, prev: str, us_context: bool, us_state: bool = True) -> bool:
    """Read a clashing 2-letter code as the US/CA subdivision (True) or as the country (False)?

    A city the dictionaries place in the clashing country wins ("Bangalore, IN", "Tel Aviv, IL", "Lima, PE").
    A US-state code after a city of some THIRD country is the state: US towns reuse foreign names ("Athens,
    GA", "Munster, IN"). ``us_context``: the line names the US elsewhere ("Vienna, VA, USA").
    """
    if code not in _ISO2 or code == "CA":  # CA stays California, as upstream (freehire too)
        return True
    place = _city_country(prev) if prev else None
    if place in ("US", "CA"):
        return True
    if place == code:
        return False
    return us_context or (us_state and place is not None) or (code in _STATE_FIRST and bool(prev))


_US_CONTEXT_RE = re.compile(
    r"\b(?:united states|usa|u\.s\.a?\.?)(?![\w.])|,\s*(?:"
    + "|".join(sorted(s for s in US_STATES if s not in _ISO2))
    + r")\b|\b(?:"
    + "|".join(sorted((re.escape(n) for n in STATE_BY_NAME if n != "georgia"), key=len, reverse=True))
    + r")\b",
    re.I,
)

# Hiredra additions (M0.4 finding, not in upstream locparse): Ashby/Greenhouse boards often
# name the WORKPLACE, not just the city ("San Francisco Office", "New York HQ", "London office").
# A trailing workplace word is dropped before the lookup, so the city underneath resolves; a bare
# "HQ" stays unknown. Never a guess: only names already in the tables resolve.
_WORKPLACE_RE = re.compile(
    r"(?:\s+|\s*[-–]\s*)(?:head\s+)?(?:office|offices|hq|headquarters|hub|campus)\.?$", re.I
)  # "-HQ" too: "San Francisco-HQ"
# A full UK postcode states GB, the way a ZIP states US, and is cut so the place before it still resolves
# ("London SW1A 2HB"). Ported from freehire's stripUKPostcode (MIT): the full outward + inward form under
# a real postcode area only, never an ordinal ("B2 2nd floor"); GY/JE/IM are Crown dependencies, not GB.
_UK_POSTCODE_RE = re.compile(r"\b([A-Z]{1,2})\d[A-Z\d]? \d[ABD-HJLNP-UW-Z]{2}\b", re.I)
_UK_POSTCODE_AREAS = set(
    "ab al b ba bb bd bh bl bn br bs bt ca cb cf ch cm co cr ct cv cw da dd de dg dh dl dn dt dy e ec eh en ex "
    "fk fy g gl gu ha hd hg hp hr hs hu hx ig ip iv ka kt kw ky l la ld le ll ln ls lu m me mk ml n ne ng nn np "
    "nr nw ol ox pa pe ph pl po pr rg rh rm s sa se sg sk sl sm sn so sp sr ss st sw sy ta td tf tn tq tr ts tw "
    "ub w wa wd wf wn wr ws wv yo ze".split()
)


# Russian/Ukrainian "city" markers before a name ("г Москва", "город Самара", "м. Львів"); freehire's
# cityMarkerPrefixes (MIT). Each ends in a separator, so "Грозный" and "Мурманск" are untouched.
_CYRILLIC_CITY_PREFIX_RE = re.compile(r"^(?:город|місто|г\.|м\.|г|м)\s+|^(?:г\.|м\.)(?=\S)", re.I)


def _strip_uk_postcode(tok: str) -> tuple[str, bool]:
    for m in _UK_POSTCODE_RE.finditer(tok):
        if m.group(1).lower() in _UK_POSTCODE_AREAS and m.group(0)[-2:].lower() not in ("st", "nd", "rd", "th"):
            return " ".join((tok[: m.start()] + " " + tok[m.end() :]).split()).strip(" ."), True
    return tok, False


# A flag emoji is an explicit country: two regional-indicator letters ("Lyon 🇫🇷" -> FR).
_FLAG_RE = re.compile("([\U0001f1e6-\U0001f1ff])([\U0001f1e6-\U0001f1ff])")
CITY_COUNTRY.update({"sf bay area": "US", "san francisco bay area": "US"})
TZ_CODES = {
    "EST",
    "PST",
    "CST",
    "MST",
    "HST",
    "AKST",
    "EDT",
    "PDT",
    "CDT",
    "MDT",
    "GMT",
    "UTC",
    "CET",
    "IST",
    "BST",
    "AEST",
}


def _split(loc: str) -> list[str]:
    """Hyphenated forms: "Mexico-Remote", "US - Remote (...)", "IND-Pune-Smartworks" -> split on
    hyphens too when a side is a known country/code (keeps "Winston-Salem" intact)."""

    def hy(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        if f"{a}-{b}".lower() in COUNTRIES:
            return m.group(0)  # Guinea-Bissau, Timor-Leste
        if b.isupper() and len(b) == 2 and b in US_STATES and a[:1].isupper():
            return f"{a}, {b}"  # "Boston-MA"
        return (
            f"{a}; {b}"
            if (a.lower() in COUNTRIES or b.lower() in COUNTRIES or REMOTE_RE.fullmatch(a) or REMOTE_RE.fullmatch(b))
            else m.group(0)
        )

    loc = re.sub(
        r"\bgeorgia\s*\(\s*country\s*\)", "georgia country", loc, flags=re.I
    )  # "(country)" must not become its own segment
    loc = re.sub(r"\b([A-Za-z.]+(?: [A-Za-z.]+)?)\s*[-–]\s*([A-Za-z.]+)\b", hy, loc)  # also "Czech Republic-Prague"
    loc = loc.replace(">", ";")  # "Hungary > Budapest"
    loc = re.sub(r"\s+[-–—]\s+", ";", loc)  # a spaced dash always separates (freehire): "Massachusetts - Boston"
    loc = re.sub(r"[()\[\]]", ";", loc)  # "Remote (United States)" -> two segments
    loc = re.sub(
        r"\b(remote|hybrid|on-?site|work from home|wfh)\b\s*[-–:]?\s*", lambda m: m.group(1) + ";", loc, flags=re.I
    )  # "Hybrid - Austin" -> "hybrid; Austin"
    loc = re.sub(r"(\S)\s+(remote|hybrid);", r"\1; \2;", loc, flags=re.I)  # "US Remote" -> "US; remote;"
    # "·" (U+00B7 middle dot) too -- Hiredra's own Ashby provider folds a job's
    # multiple locations with " · " (see [[location-format-not-uniform]]); the
    # upstream "•" (bullet) alone doesn't catch it. Case-insensitive -- "OR"/
    # "AND" (all-caps, common in real postings: "Beijing OR Shanghai") must
    # split the same as lowercase "or"/"and".
    parts = re.split(r"\s*(?:;|\||/|:|\bor\b|\band\b|&|•|·)\s*", loc, flags=re.I)  # ":" too: "US: PST or EST"
    return [p.strip(" -–,") for p in parts if p and p.strip(" -–,")]


def _one(seg: str, out: dict) -> None:
    s = seg.strip().strip("()[]")
    if not s:
        return
    low = s.lower()
    if REMOTE_RE.search(low) and len(low) < 40:
        out["remote_hint"] = True
    # icims style: US-CA-San Jose / US-Remote
    m = re.match(r"^([A-Z]{2})-([A-Z]{2})-(.+)$", s)
    if m and m.group(1) == "US" and m.group(2) in US_STATES:
        out["countries"].add("US")
        out["regions"].add(m.group(2))
        out["cities"].add(m.group(3).strip().title())
        return
    m = re.match(r"^([A-Z]{2,3})-(.+)$", s)
    if m and m.group(1).lower() in COUNTRIES:
        out["countries"].add(COUNTRIES[m.group(1).lower()])
        rest = m.group(2).split("-")[0].strip()
        if rest.lower() != "remote":
            out["cities"].add(rest.title())
            return
        return
    m = re.match(r"^([A-Z]{2})-([A-Za-z][^,]*)$", s)
    if m and (m.group(1) in US_STATES or m.group(1) in CA_PROV or m.group(1) in _ISO2):
        s = f"{m.group(2)}, {m.group(1)}"  # "TX-Houston", "IL-Cupertino", "IN-Pune": judge the code after its city
    toks = [t.strip() for t in re.split(r",", s) if t.strip()]
    toks2 = []
    for t in toks:  # "SD USA" -> "SD", "USA"
        m2 = re.match(r"^([A-Z]{2})\s+(.+)$", t.strip())
        if m2 and m2.group(1) in US_STATES and m2.group(2).lower().strip(". ") in COUNTRIES:
            toks2 += [m2.group(1), m2.group(2)]
        elif m2 and m2.group(1) in US_STATES and re.fullmatch(r"\d{5}(?:-\d{4})?", m2.group(2).strip()):
            toks2.append(m2.group(1))  # "TN 37090"
        elif (m3 := re.match(r"^(.*[a-z].*?)\s+([A-Z]{2})(?:\s+\d{5}(?:-\d{4})?)?$", t.strip())) and (
            m3.group(2) in US_STATES or m3.group(2) in CA_PROV or m3.group(2) in _INDIAN_STATE_CODES
        ):
            toks2 += [m3.group(1), m3.group(2)]  # "Atlanta GA", "Pune IN": the code is judged as its own token
        else:
            toks2.append(t)
    toks = toks2
    if len(toks) == 1 and REMOTE_RE.fullmatch(toks[0].strip().lower()):
        return  # bare "Remote"
    city = None
    explicit: set[str] = set()
    inferred: set[str] = set()
    # "KA, IN" / "TN, India" / "Mumbai, MH": an Indian state code after an Indian city or before the country.
    # Read as India; the code must not fall through to a US state (TN, GA, LA) or a country (MH Marshall Is.).
    for i, tok in enumerate(toks):
        code = tok.strip()
        nxt = toks[i + 1].strip().lower() if i + 1 < len(toks) else ""
        if (
            code in _INDIAN_STATE_CODES
            and code.isupper()
            and (nxt in ("in", "india") or (i and _city_country(toks[i - 1]) == "IN"))
        ):
            explicit.add("IN")
            toks[i] = ""
    lows = [t.lower().strip(". ") for t in toks]
    for i, t in enumerate(toks):
        prev = toks[i - 1].strip() if i else ""
        for a, b in _FLAG_RE.findall(t):
            code = chr(ord(a) - 0x1F1E6 + 65) + chr(ord(b) - 0x1F1E6 + 65)
            explicit.add("GB" if code == "UK" else code)
        t = _FLAG_RE.sub("", t).strip()
        t = _CYRILLIC_CITY_PREFIX_RE.sub("", t)  # "г Москва", "м. Львів"
        t, uk_postcode = _strip_uk_postcode(t)
        if uk_postcode:
            explicit.add("GB")
        if not t:
            continue
        t = _WORKPLACE_RE.sub("", t).strip() or t
        if len(t.rstrip(".")) == 2:
            t = t.rstrip(".")  # "Boston, MA."
        tl = t.lower().strip(" .:")
        if tl == "washington" and any(x in ("dc", "d.c", "district of columbia") for x in lows):
            out["cities"].add("Washington")
            explicit.add("US")  # Hiredra fix: the city was recorded but the country never was
            continue
        if tl == "georgia" and any(x in GEORGIAN_CITIES or "country" in x for x in lows):
            explicit.add("GE")
            continue
        if (
            tl in ISO3_ALL and tl not in ISO3
        ):  # generated alpha-3: only as an upper-case token, never a US time zone (EST = Estonia)
            tok = t.strip(" .:")
            if (
                tok == tok.upper()
                and tok not in TZ_CODES
                and not re.search(r"time ?zone|hours|\b(pst|est|cst|mst)\b", low)
            ):
                explicit.add(ISO3_ALL[tl])
            continue
        if tl in COUNTRIES:
            explicit.add(COUNTRIES[tl])
            continue
        if t.upper() in US_STATES and (len(t) == 2) and _subdivision_wins(t.upper(), prev, out["us_context"]):
            explicit.add("US")
            out["regions"].add(t.upper())
            continue
        if tl in STATE_BY_NAME:
            explicit.add("US")
            out["regions"].add(STATE_BY_NAME[tl])
            continue
        if t.upper() in CA_PROV and len(t) == 2 and _subdivision_wins(t.upper(), prev, out["us_context"], False):
            explicit.add("CA")
            out["regions"].add("CA-" + t.upper())
            continue
        # A bare country code, upper-case only ("Berlin, DE", "Pune, IN"; upstream read only uk/us), so a
        # word ("in office", "na") never reads as one.
        if len(t) == 2 and t.isupper() and t in _ISO2:
            place = _city_country(prev) if prev else None
            if place is None or place == t:
                explicit.add(t)
            # else a local subdivision code after a city of another country ("Belo Horizonte, MG" is Minas
            # Gerais, not Madagascar; "Adelaide, SA"): the city decides
            continue
        if tl in PROV_BY_NAME:
            explicit.add("CA")
            out["regions"].add("CA-" + PROV_BY_NAME[tl])
            continue
        if tl in CITY_COUNTRY:
            inferred.add(CITY_COUNTRY[tl])
            out["cities"].add(t.title())
            continue
        if REMOTE_RE.search(tl) or tl in (
            "worldwide",
            "global",
            "emea",
            "apac",
            "latam",
            "europe",
            "eastern europe",
            "western europe",
            "north america",
            "south america",
            "asia",
            "africa",
            *_REGION_ALIASES,
        ):
            if not REMOTE_RE.search(tl):
                out["regions"].add(_REGION_ALIASES.get(tl, tl.title()))
            continue
        geo = _GEONAMES.get(tl)
        if geo:  # a long-tail city (GeoNames): its country is the line's last resort, see parse()
            out["cities"].add(t.title())
            if not geo[1]:
                out["geo_fallback"].add(geo[0])
            continue
        if (
            city is None
            and len(t) < 40
            and not re.search(r"\d", t)
            and not REMOTE_RE.search(tl)
            and not HYBRID_RE.search(tl)
        ):
            city = t
    if city:
        out["cities"].add(city.title())
    # an explicit country/state wins over a country inferred from a city name (Vienna, VA is not Austria)
    out["countries"] |= explicit if explicit else inferred


# A restriction stated in the title, read only when the location line gave no geography ("Data Engineer -
# US Remote" at location "Remote"). Ported from freehire's RestrictionFromTitle (MIT): a bracketed suffix
# anchored on "remote"/"location" ("[Remote-US]", "(Remote Europe)"); a final " - ... Remote" segment is our
# addition. The anchor keeps a plain "(US)" language/team note from counting.
_TITLE_SUFFIX_RE = re.compile(r"\[([^\[\]]*)\]|\(([^()]*)\)|\s[-–|]\s+([^-–|()\[\]]+)$")
_TITLE_ANCHOR_RE = re.compile(r"\b(?:remote|location)\b", re.I)


def _title_restriction(title: str, out: dict) -> None:
    for m in _TITLE_SUFFIX_RE.finditer(title or ""):
        inner = next(g for g in m.groups() if g is not None).replace("-", " ")
        if not _TITLE_ANCHOR_RE.search(inner):
            continue
        rest = _TITLE_ANCHOR_RE.sub(" ", inner)
        rest = re.sub(r"^\s*(?:in|within|from|only)\s+(?:the\s+)?", "", rest, flags=re.I).strip(" :,")
        tmp: dict = {**out, "countries": set(), "regions": set(), "cities": set(), "geo_fallback": set()}
        for seg in _split(rest):
            _one(seg, tmp)
        out["countries"] |= tmp["countries"]
        out["regions"] |= tmp["regions"]


def parse(location: str | None, jd: str = "", title: str = "") -> dict:
    """``{"remote": "remote|hybrid|onsite|unknown", "countries": [...], "regions": [...], "cities": [...]}``.

    Heuristic, deterministic, no network. Countries are ISO-3166 alpha-2; US
    regions are state codes.
    """
    out: dict = {
        "countries": set(),
        "regions": set(),
        "cities": set(),
        "remote_hint": False,
        "us_context": bool(_US_CONTEXT_RE.search(location or "")),
        "geo_fallback": set(),
    }
    for seg in _split(location or ""):
        _one(seg, out)
    # Nothing in the line stated a country, so the long-tail GeoNames cities may -- with one voice only
    # ("Belgrade" -> RS; "Recife, Benidorm" -> nothing). Ported from freehire's cityFallbackCountries (MIT).
    if not out["countries"] and len(out["geo_fallback"]) == 1:
        out["countries"] |= out["geo_fallback"]
    if not out["countries"] and not out["regions"]:
        _title_restriction(title, out)
    loc = ((location or "") + " | " + (title or "")).lower()
    text = (jd or "").lower()
    head = text[:2500]
    # An explicit in-office requirement anywhere in the JD beats a "Remote" location (many "Remote - US"
    # postings then say "anchor days Mon/Tue/Fri" or "Monday-Thursday onsite"). Requirement-shaped phrases
    # only, so "hybrid cloud", "onsite interview", "for in-office employees" don't count.
    onsite_req = re.compile(
        r"\b(?:\d|one|two|three|four|five)\+? days?(?: a| per| each)? week (?:in|at|from) (?:the |our )?office|anchor days?|"
        r"\b(?:monday|mon|tuesday|tue|wednesday|wed|thursday|thu|friday|fri)\b[^.\n]{0,40}\b(?:on-?site|in[- ]office|in the office)|"
        r"\b(?:on-?site|in[- ]office|in the office)\b[^.\n]{0,30}\b(?:monday|mon|tuesday|tue|wednesday|wed|thursday|thu|friday|fri)\b|"
        r"\b(?:will )?requires? [^.\n]{0,40}\b(?:on-?site|in[- ]office|in the office)|\bability to work on-?site in\b|"
        r"\b(?:this|the) (?:role|position) is (?:a )?(?:hybrid|on-?site|in-?office)|\bhybrid (?:role|position|schedule|work(?:ing)? model|model)\b",
        re.I,
    )
    full_onsite = re.compile(
        r"\b(?:fully|100%|entirely) (?:on-?site|in[- ]office)\b|\b(?:5|five) days?(?: a| per)? week (?:in|at) (?:the )?office\b|\bno remote\b|\bnot (?:a )?remote\b|"
        r"\b(?:work(?:ing)? (?:from|out of|in)|based (?:in|out of)|join (?:us|the team) (?:in|at)) our [^.\n]{0,40}\boffices?\b|\bin[- ]person\b[^.\n]{0,30}\boffice\b",
        re.I,
    )
    if HYBRID_RE.search(loc):
        remote = "hybrid"
    elif full_onsite.search(text) and not HYBRID_RE.search(head):
        remote = "onsite"
    elif onsite_req.search(text):
        remote = "hybrid"
    elif HYBRID_RE.search(head):
        remote = "hybrid"
    elif out["remote_hint"] or REMOTE_RE.search(loc):
        remote = "remote"
    elif ONSITE_RE.search(loc) or re.search(
        r"\b(on-?site|in-?office) (role|position|\d+ days)\b|\bthis (role|position) is (on-?site|in-?office)\b", head
    ):
        remote = "onsite"
    # a role-specific remote statement in the JD counts anywhere; a company blurb ("remote-first", "fully remote team")
    # counts only when the location isn't a specific city
    elif re.search(
        r"\b(?:100%|fully|entirely) remote (?:role|position|job|opportunity)\b|\bremote (role|position|job)\b|\bthis (role|position) is (?:fully |100% )?remote\b|\bwork(?:ing)? remotely from anywhere\b|\bwork location:?\s*(?:fully |100% )?remote\b|\b(?:fully|100%) remote\s*(?:[—–\-:,.!]|\n|we hire|must|you)",
        head,
    ):
        remote = "remote"
    elif not out["cities"] and re.search(r"\b(fully|100%|entirely) remote\b|\bremote[- ]first\b", head):
        remote = "remote"
    else:
        remote = "unknown"
    return {
        "remote": remote,
        "countries": sorted(out["countries"]),
        "regions": sorted(out["regions"]),
        "cities": sorted(out["cities"])[:6],
    }


# ---- eligibility against the owner's location preference ----
RESTRICT_RE = re.compile(
    r"\b(must (?:be|reside|live|be located|be based)[^.]{0,40}?\b(in|within)\b|(?:only|exclusively) (?:for|open to)? ?(?:candidates|applicants|residents)?[^.]{0,20}?(?:in|based in|located in|from)\b|(?:authori[sz]ed|eligible) to work in|(?:based|located|residing) in|(?:us|u\.s\.|united states)[- ]?(?:only|based)|(?:latam|emea|apac|india|canada|uk)[- ]?only)\b[^.\n]{0,60}",
    re.I,
)
MACRO = {
    "Latam": {"MX", "BR", "AR", "CO", "CL", "PE", "CR", "UY"},
    "Emea": {
        "GB",
        "DE",
        "FR",
        "ES",
        "IT",
        "NL",
        "BE",
        "CH",
        "AT",
        "SE",
        "NO",
        "DK",
        "FI",
        "IE",
        "PL",
        "PT",
        "CZ",
        "HU",
        "RO",
        "GR",
        "UA",
        "TR",
        "IL",
        "AE",
        "SA",
        "QA",
        "EG",
        "ZA",
        "NG",
        "KE",
        "MA",
    },
    "Apac": {"IN", "CN", "JP", "KR", "SG", "HK", "TW", "AU", "NZ", "PH", "ID", "MY", "TH", "VN", "PK", "BD", "LK"},
    "Europe": {
        "GB",
        "DE",
        "FR",
        "ES",
        "IT",
        "NL",
        "BE",
        "CH",
        "AT",
        "SE",
        "NO",
        "DK",
        "FI",
        "IE",
        "PL",
        "PT",
        "CZ",
        "HU",
        "RO",
        "GR",
        "UA",
    },
    "Eastern Europe": {"PL", "CZ", "HU", "RO", "UA", "GR"},
    "Western Europe": {"GB", "DE", "FR", "ES", "IT", "NL", "BE", "CH", "AT", "IE", "PT"},
    "North America": {"US", "CA", "MX"},
    "Americas": {"US", "CA", "MX", "BR", "AR", "CO", "CL", "PE", "CR", "UY"},
    # freehire's region vocabulary (internal/dict/location regionCountries, MIT)
    "Mena": {
        "AE",
        "SA",
        "IL",
        "EG",
        "TR",
        "QA",
        "KW",
        "BH",
        "OM",
        "JO",
        "LB",
        "IQ",
        "IR",
        "MA",
        "DZ",
        "TN",
        "LY",
        "YE",
        "PS",
    },
    "Cis": {"RU", "BY", "MD", "AM", "AZ", "GE", "UZ", "KZ", "KG", "TJ", "TM"},
    "South America": {"BR", "AR", "CO", "CL", "PE", "UY"},
    "Asia": {"IN", "CN", "JP", "KR", "SG", "HK", "TW", "PH", "ID", "MY", "TH", "VN", "PK", "BD", "LK"},
    "Africa": {"ZA", "NG", "KE", "EG", "MA"},
}

_PLACE_KEYS = sorted(COUNTRIES.keys(), key=len, reverse=True)
_MACRO_KEYS = sorted(MACRO.keys(), key=len, reverse=True)


def find_places(text: str) -> tuple[set[str], set[str]]:
    """Countries (ISO-2) and macro-regions named anywhere in a phrase, e.g. 'must be located in the
    United States'."""
    t = " " + re.sub(r"[^a-z0-9. ]+", " ", (text or "").lower()) + " "
    cs: set[str] = set()
    rs: set[str] = set()
    # longest names first, and each match is blanked out, so "northern ireland" is GB (not also IE),
    # "papua new guinea" is PG (not GN), "new mexico" is a US state (not MX)
    for st in sorted(STATE_BY_NAME, key=len, reverse=True):
        if f" {st} " in t:
            cs.add("US")
            t = t.replace(f" {st} ", "  ")
    for k in sorted(_PLACE_KEYS, key=len, reverse=True):
        if len(k) < 3 and k not in ("uk", "us"):
            continue  # skip 2-letter noise except uk/us
        if k in ISO3_ALL and k not in ("usa",):
            continue  # alpha-3 codes ("and", "are", "can"...) are words in prose
        if k == "georgia":
            continue  # bare "georgia" in prose is the state far more often; handled by the state pass above
        if f" {k} " in t or f" {k}. " in t:
            cs.add(COUNTRIES[k])
            t = t.replace(f" {k} ", "  ").replace(f" {k}. ", "  ")
    for k in _MACRO_KEYS:
        if f" {k.lower()} " in t:
            rs.add(k)
    return cs, rs


def _clause_elig(p: dict, j: dict, rm: str, jd: str) -> tuple[bool | None, str]:
    """One preference clause against one job. p = parse(clause), j = parse(job), rm = effective arrangement."""
    remote_only = p["remote"] == "remote" and not p["cities"]
    want = set(p["countries"])
    for r in p["regions"]:
        want |= MACRO.get(r, set())
    have = set(j["countries"])
    for r in j["regions"]:
        have |= MACRO.get(r, set())
    have_jd: set[str] = set()
    for m in RESTRICT_RE.finditer((jd or "")[:6000]):
        cs, rs = find_places(m.group(0))
        have_jd |= cs
        for r in rs:
            have_jd |= MACRO.get(r, set())
    # The JD can narrow an explicit location, never widen it: benefits boilerplate ("for US-based full-time
    # employees we offer...") must not make a "Remote - EMEA" posting US-eligible. With no location, the JD decides.
    have = ((have & have_jd) or have) if have else have_jd
    if remote_only:
        # the job must actually be labelled remote (location, title, or an explicit statement). Unknown is not remote.
        if rm in ("onsite", "hybrid"):
            return False, f"{rm} (you asked for remote)"
        if rm != "remote":
            return False, "not labelled remote"
        if not want:
            return None, "no country in preference"
        if have & want:
            return True, "remote, location matches"
        if have:
            return False, f"restricted to {', '.join(sorted(have))[:40]}"
        return True, "remote, no stated region"
    if not want:
        return None, "no country in preference"
    if have and not (have & want):
        return False, f"restricted to {', '.join(sorted(have))[:40]}"
    if p["cities"] or (p["regions"] and not all(r in MACRO for r in p["regions"])):
        # a local clause ("Austin, TX", "Berlin"): the job must be in that city/state, or be remote (workable from there)
        pc = {c.lower() for c in p["cities"]}
        pr = {r for r in p["regions"] if r not in MACRO}
        jc = {c.lower() for c in j["cities"]}
        if pc & jc:
            return True, f"in {sorted(pc & jc)[0].title()}"
        if pr & set(j["regions"]):
            return True, f"in {sorted(pr & set(j['regions']))[0]}"
        if rm == "remote" and (have & want or not have):
            return True, "remote, works from there"
        if not have and not j["cities"] and not j["regions"]:
            return None, "no location info"
        if jc or j["regions"]:
            return False, f"not near {(sorted(p['cities']) + sorted(pr) + ['?'])[0]}"
        return None, "no city info"
    if have & want:
        return True, "location matches"
    if have:
        return False, f"restricted to {', '.join(sorted(have))[:40]}"
    if j["remote"] == "remote":
        return True, "remote, no stated region"
    return None, "no location info"


def eligibility(
    pref: str, location: str | None, jd: str = "", title: str = "", arrangement: str | None = None
) -> tuple[bool | None, str]:
    """Is a job eligible for someone with location preference ``pref``?

    Returns ``(eligible, reason)``. ``pref`` may be a compound OR of clauses --
    "Austin, TX or Remote, US", "Berlin; Remote, DE" -- and the job is eligible
    if ANY clause accepts it. Clause types: a remote clause ("Remote, US")
    requires the job to be labelled remote in a matching country; a local
    clause ("Austin, TX", "Berlin") requires the job to be in that city/state,
    or remote in the country; a bare country ("US or Canada") matches
    country-wide. ``None`` = can't tell. ``arrangement`` overrides the parsed
    one (pass an enrichment's, when one exists).
    """
    clauses = [c.strip() for c in re.split(r"\bor\b|;|\||/", pref or "", flags=re.I) if c.strip()]
    if not clauses:
        return None, "no preference"
    j = parse(location or "", jd, title)
    rm = arrangement if arrangement in ("remote", "hybrid", "onsite") else j["remote"]
    outs = [_clause_elig(parse(c), j, rm, jd) for c in clauses]
    for ok, why in outs:
        if ok is True:
            return True, why
    if any(ok is None for ok, _ in outs):
        return None, next(why for ok, why in outs if ok is None)
    return False, " · ".join(dict.fromkeys(why for _, why in outs))[:80]
