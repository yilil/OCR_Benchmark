import os
from abc import abstractmethod
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from tqdm import tqdm

from config import *
import logging


class Engine:
    def __init__(self, method="DEFAULT"):
        self.method = method
        self.model_name = "Engine"
        self.image_path = config["BENCHMARK_IMAGE_PATH"]
        self.text_path = config["BENCHMARK_TEXT_PATH"]

    def img_to_text(self, skip=True):
        files = []
        # Loop over each file in the directory
        for file in os.listdir(self.image_path):
            if not file.endswith('.jpg'):
                continue

            file_full_path = os.path.join(os.path.abspath(self.image_path), file)
            text_file = os.path.join(os.path.abspath(self.text_path), file.rstrip(".jpg")) + f"_{self.model_name}.txt"

            # skip if the file has already been converted
            if skip and os.path.isfile(text_file):
                logging.info("Skipped - {} \t reason: {}".format(text_file, "Already Convert from Image to Text"))
                continue

            files.append(file_full_path)

        if self.method == "DEFAULT":
            for file in tqdm(files):
                self.img_to_text_helper(file)
        elif self.method == "THREAD":
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(tqdm(executor.map(self.img_to_text_helper, files), total=len(files)))
        elif self.method == "PROCESS":
            with Pool(multiprocessing.cpu_count()) as p:
                p.starmap(self.img_to_text_helper, [(file,) for file in files])

    @abstractmethod
    def img_to_text_helper(self, file):
        pass
