import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, CategoriesOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version = '2018-11-16',
    iam_apikey = 'X2Vc13XCvB-MNuGjNrK6T5Jb9hp-fiv56bHL3o4-qoi5',
    url = 'https://gateway-lon.watsonplatform.net/natural-language-understanding/api'
)

response = natural_language_understanding.analyze(
    text = ["We will.", "always democracy"],
    features = Features(categories=CategoriesOptions(limit=3)),
    language = 'en'
).get_result()

# print(response['categories'][0]['label'])

print(json.dumps(response, indent=2))
