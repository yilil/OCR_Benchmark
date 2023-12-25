from evaluate import load
import re

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

pattern = r'\s+'


def data_preprocessing(prediction_path, reference_path):
    with open(prediction_path) as predictions, open(reference_path) as references:

        prediction_text = ""
        for line in predictions.readlines():
            line = re.sub(pattern, ' ', line.strip())
            if not len(line) > 0:
                continue
            prediction_text += line + " "
        prediction_text.strip()

        reference_text = ""
        for line in references.readlines():
            line = re.sub(pattern, ' ', line.strip())
            if not len(line) > 0:
                continue
            reference_text += line + " "
        reference_text.strip()
        return prediction_text, reference_text


def ocr_evaluate(metric, predictions, references):
    metric = load(metric)
    prediction_text, reference_text = data_preprocessing(predictions, references)

    return metric.compute(
        predictions=[prediction_text],
        references=[reference_text]
    )


def wfer_evaluate(predictions, references):
    prediction, ground_truth = data_preprocessing(predictions, references)

    vectorizer = CountVectorizer(lowercase=False).fit([ground_truth, prediction])

    ground_truth_bow = vectorizer.transform([ground_truth]).toarray()[0]
    prediction_bow = vectorizer.transform([prediction]).toarray()[0]
    dist = distance.cityblock(ground_truth_bow, prediction_bow)

    total_freq = sum(ground_truth_bow) + sum(prediction_bow)

    return dist / total_freq
