"""Cases for Q1 to Q6. Each one is an item a Grade 7 student could beat
without reading, or a good item that must not be flagged."""
from quizrules import check_item

P = ("The Banaue Rice Terraces are carved into the mountains of Ifugao province in northern "
     "Luzon, Philippines. They were built by the Ifugao people, who shaped the steep slopes by "
     "hand and fed the fields with water from the rainforests above. Local tradition holds that "
     "the terraces are about 2,000 years old, although some studies suggest they are much "
     "younger. In 1995, UNESCO added the Rice Terraces of the Philippine Cordilleras to its "
     "World Heritage List. Today, farmers still plant rice on the terraces.")

CASES = [
    # (label, question, options, answer_index, codes that must appear)
    ("the long answer gives itself away",
     "Where are the Banaue Rice Terraces carved?",
     ["the mountains of Ifugao province in northern Luzon, Philippines",
      "Manila", "Cebu", "Davao"], 0, ["Q1"]),

    ("only the answer is a place",
     "Where are the terraces carved?",
     ["Ifugao province", "2,000 years", "by hand", "rice"], 0, ["Q4"]),

    ("only the answer is a number",
     "How old are the terraces said to be?",
     ["about 2,000 years old", "the Ifugao people", "the rainforests", "World Heritage"], 0, ["Q4"]),

    # There is no absurd-distractor rule, on purpose. "aliens from Mars" gets
    # through, and so does every good distractor, which is the trade made.
    ("an overlapping distractor",
     "Who built the terraces?",
     ["the Ifugao people", "the Ifugao", "the Spanish settlers", "the Ilocano farmers"],
     0, ["Q6"]),

    ("one option sits inside another",
     "What did UNESCO add to its list?",
     ["the Rice Terraces of the Philippine Cordilleras", "the Rice Terraces",
      "the rainforests", "the farmers"], 0, ["Q6"]),

    ("the stem ends in 'an'",
     "The Ifugao people shaped the slopes using an",
     ["ancient hand method", "bulldozer", "machine", "truck"], 0, ["Q3"]),

    # ---- items that must NOT be flagged ----
    ("a clean item, all four are places",
     "Where are the terraces carved?",
     ["Ifugao province", "Benguet province", "Mountain Province", "Kalinga province"], 0, []),

    ("a clean item, all four are years",
     "When did UNESCO add the terraces to its list?",
     ["1995", "1985", "2005", "1975"], 0, []),

    ("a clean item, all four are people",
     "Who built the terraces?",
     ["the Ifugao people", "the Spanish settlers", "the Ilocano farmers", "the American workers"],
     0, []),

    ("a clean item, options similar in length",
     "What fed the fields with water?",
     ["the rainforests above", "the rivers below", "the seasonal rain", "the mountain springs"],
     0, []),

    ("answer not in the first slot and still clean",
     "When did UNESCO add the terraces to its list?",
     ["1975", "1985", "1995", "2005"], 2, []),
]


def main():
    bad = 0
    for label, q, opts, ai, want in CASES:
        got = check_item(q, opts, ai, P)
        got_codes = {g.split(":")[0] for g in got}
        want_codes = set(want)
        ok = want_codes <= got_codes and (want_codes or not got_codes)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {label}")
        if got:
            for g in got:
                print(f"          {g}")
        if not ok:
            print(f"          wanted {sorted(want_codes) or 'nothing'}, got {sorted(got_codes) or 'nothing'}")
    print()
    print(f"items: {'FAILED' if bad else 'all ' + str(len(CASES)) + ' cases pass'}")
    return bad


# ---- Q7, the Literal / Inferential label -----------------------------------
LABEL_CASES = [
    ("Inferential", "Where did the Banaue Rice Terraces have water?", "rainforests above", ["Q7"]),
    ("Inferential", "When did UNESCO add the terraces to its list?", "1995", ["Q7"]),
    ("Literal", "Where are the terraces carved?", "Ifugao province", []),
    ("Inferential", "Why did the Ifugao people shape the slopes by hand?", "they had no machines", []),
    ("Inferential", "Bakit itinayo ng mga Ifugao ang mga payyo?", "wala silang makinarya", []),
    ("Literal", "Why did UNESCO add the terraces to its list?", "their cultural value", ["Q7"]),
    ("Inferential", "What does the age of the terraces suggest about the Ifugao?", "long settlement", []),
]


def labels():
    from quizrules import check_label
    bad = 0
    print()
    for kind, q, opt, want in LABEL_CASES:
        got = check_label(kind, q, opt, P)
        ok = set(want) == {g.split(":")[0] for g in got}
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  [{kind:11}] {q[:50]}")
        if got:
            print(f"          {got[0]}")
    print()
    print("FAILED" if bad else f"all {len(LABEL_CASES)} label cases pass")
    return bad


if __name__ == "__main__":
    failed = main() + labels()
    raise SystemExit(1 if failed else 0)
