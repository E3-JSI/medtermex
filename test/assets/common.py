TRUE_ENTS = [
    {"text": "Barack Obama", "label": "PERSON"},
    {"text": "Washington", "label": "LOCATION"},
    {"text": "US", "label": "COUNTRY"},
]


TEST_CASES_EXACT = [
    ([TRUE_ENTS], [[]], 0.0, 0.0, 0.0),
    (
        [TRUE_ENTS],
        [[{"text": "Barack Obama", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "Barack Obama", "label": "PERSON"},
                {"text": "Washington", "label": "LOCATION"},
            ]
        ],
        1.0,
        0.6666666666666666,
        0.8,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "Barack Obama", "label": "PERSON"},
                {"text": "Washington", "label": "LOCATION"},
                {"text": "US", "label": "COUNTRY"},
            ]
        ],
        1.0,
        1.0,
        1.0,
    ),
]

TEST_CASES_RELAXED = [
    ([TRUE_ENTS], [[]], 0.0, 0.0, 0.0),
    (
        [TRUE_ENTS],
        [[{"text": "Barack Obama", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "Barack Obama", "label": "PERSON"},
                {"text": "Washington", "label": "LOCATION"},
            ]
        ],
        1.0,
        0.6666666666666666,
        0.8,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "Barack Obama", "label": "PERSON"},
                {"text": "Washington", "label": "LOCATION"},
                {"text": "US", "label": "COUNTRY"},
            ]
        ],
        1.0,
        1.0,
        1.0,
    ),
    (
        [TRUE_ENTS],
        [[{"text": "Barack Obama Jr.", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "Barack Obama", "label": "PERSON"},
                {"text": "Washington D.C.", "label": "LOCATION"},
            ]
        ],
        1.0,
        0.6666666666666666,
        0.8,
    ),
]


TEST_CASES_OVERLAP = [
    ([TRUE_ENTS], [[]], 0.0, 0.0, 0.0),
    (
        [TRUE_ENTS],
        [[{"text": "Barack Obama", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "Barack Obama", "label": "PERSON"},
                {"text": "Washington", "label": "LOCATION"},
            ]
        ],
        1.0,
        0.6666666666666666,
        0.8,
    ),
    (
        [TRUE_ENTS],
        [
            [
                {"text": "Barack Obama", "label": "PERSON"},
                {"text": "Washington", "label": "LOCATION"},
                {"text": "US", "label": "COUNTRY"},
            ]
        ],
        1.0,
        1.0,
        1.0,
    ),
    (
        [TRUE_ENTS],
        [[{"text": "Obama", "label": "PERSON"}]],
        5 / 12,
        5 / 36,
        5 / 24,
    ),
    (
        [TRUE_ENTS],
        [[{"text": "Barack Obama", "label": "PERSON"}]],
        1.0,
        0.3333333333333333,
        0.5,
    ),
]
