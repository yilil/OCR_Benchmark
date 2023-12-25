import os
from concurrent.futures import ThreadPoolExecutor

from config import *
from models.amazon_textract.amazon_textract import AmazonTextract
from models.paddleocr.paddleocr import Paddle
from models.tesseract.tesseract import Tesseract
from models.googleocr.googleocr import GoogleOCR
from models.microsoft.microsoftOCR import MicroSoftOCR
from utils.evaluation import *
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


def evaluate_model_helper(information):
    i, results, model_name, filename, prediction_path, ground_truth_path = information
    # Calculate errors
    cer_result = ocr_evaluate("cer", prediction_path, ground_truth_path)
    wer_result = ocr_evaluate("wer", prediction_path, ground_truth_path)
    wfer_result = wfer_evaluate(prediction_path, ground_truth_path)

    results[i] = [filename, model_name, cer_result, wer_result, wfer_result]


def evaluate_model(model, dataset):
    ground_truth = config["BENCHMARK_TEXT_GROUND_TRUTH_PATH"].replace("$datasetname$", dataset)

    file_list = []
    for each in os.listdir(model.text_path):
        if model.model_name in each:
            file_list.append(each)

    results = [[]] * len(file_list)
    files = []

    for i in range(len(file_list)):
        file = file_list[i]
        # Exclude the model name and file extension
        filename = file[:file.rfind("_")]
        model_name = model.model_name
        # Get the absolute path of the OCR prediction file
        prediction_path = os.path.join(os.path.abspath(model.text_path), f"{filename}_{model_name}.txt")
        # Get the absolute path of the OCR ground truth file
        ground_truth_path = os.path.join(os.path.abspath(ground_truth), filename + ".txt")
        files.append((i, results, model_name, filename, prediction_path, ground_truth_path))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(evaluate_model_helper, files), total=len(files)))

    return results


# aggregate a model's evaluation, find the mean accuracies (take the means of image result)
def aggregate_evaluation(model_evaluation, dataset):
    mean = np.mean([[cer, wer, wfer] for _, __, cer, wer, wfer in model_evaluation], axis=0)
    mean_formatted = ['{:0.6f}'.format(val) for val in mean]
    return np.append(dataset, mean_formatted)


# display evaluations of 1...N OCR model in a nice format
def format_evaluation(evaluations):
    df = pd.DataFrame(columns=["model", "dataset", "CER", "WER", "WFER"])

    for i, each_key in enumerate(evaluations.keys()):
        for j, each_row in enumerate(evaluations.get(each_key)):
            each_row = np.append(each_key, each_row).tolist()
            df.loc[j * len(evaluations.keys()) + i] = each_row

    df = df.sort_index()
    # Convert to LaTeX table
    latex_table = df.to_latex(index=False, column_format='|c|c|c|c|c|', header=True, bold_rows=True, escape=False)

    # Add required LaTeX packages and modify the table for full borders
    latex_table = latex_table. \
        replace("\\toprule", "\\hline"). \
        replace("\\midrule", "\\hline"). \
        replace("\\bottomrule", "\\hline")

    return df, latex_table


def run_benchmark():
    method = "THREAD"
    # choose models
    model_list = [
        Tesseract(method=method),
        # AmazonTextract(method=method)
    ]

    evaluation_summary = {}
    datasets = ["NoisyOCRDataset", "WillsOCRDataset"]
    for dataset in datasets:
        for model in model_list:
            # set the current dataset path for image input and text output
            model.text_path = config["BENCHMARK_TEXT_PATH"].replace("$datasetname$", dataset)
            model.image_path = config["BENCHMARK_IMAGE_PATH"].replace("$datasetname$", dataset)

            model.img_to_text()
            model_evaluation = evaluate_model(model, dataset)
            if evaluation_summary.get(model.model_name) is None:
                evaluation_summary[model.model_name] = [aggregate_evaluation(model_evaluation, dataset)]
            else:
                evaluation_summary[model.model_name].append(aggregate_evaluation(model_evaluation, dataset))

    # serialise/de-serialise valuation summary
    # pickle.dump(evaluation_summary, open("evaluation_summary.pkl", "wb"))
    # evaluation_summary = pickle.load(open("evaluation_summary.pkl","rb"))
    df, latex_table = format_evaluation(evaluation_summary)
    print(latex_table)


if __name__ == '__main__':
    run_benchmark()
