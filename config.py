folder_config = {
    "PDF_PATH": "resources/pdfs/",
    "IMAGE_PATH": "artifacts/images/",
    "TEXT_PATH": "artifacts/text/",
    "FILTERED_IMAGE_PATH": "artifacts/filtered_images/",
    "LOG_PATH": "log",
    "BENCHMARK_IMAGE_PATH": "dataset/$datasetname$/images/",
    "BENCHMARK_TEXT_PATH": "dataset/$datasetname$/text/",
    "BENCHMARK_TEXT_GROUND_TRUTH_PATH": "dataset/$datasetname$/ground_truth/",
}

image_config = {
    "DPI": 200,
}

config = {**folder_config, **image_config}
