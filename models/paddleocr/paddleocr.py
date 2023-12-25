'''
Colab
https://colab.research.google.com/drive/12gTWJoYW1Vpvr22mX2El5N0Dv_cPtJbW#scrollTo=nOywAmid7dCh
'''

import os
import subprocess
import re
from models.engine import Engine
import logging

pattern = r"\[(.*?)\]\sppocr INFO:\s\[(.*?)\]\],\s"


class Paddle(Engine):
    def __init__(self, method="DEFAULT"):
        super().__init__(method)
        self.model_name = "paddle:PP-OCRV3"

    def img_to_text_helper(self, file):
        new_file_name = os.path.splitext(os.path.basename(file))[0] + f"_{self.model_name}"
        new_file_path = os.path.join(os.path.abspath(self.text_path), new_file_name)

        '''
        Run engine by using bash command
        '''

        cmd = [
            "paddleocr",
            "--image_dir",
            new_file_path,
            "--lang",
            "en",
            "--use_gpu",
            "true"
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            logging.error(f'Error occurred while running command: {result.stderr.decode()}')
            return

        lines = result.stdout.decode('utf-8').split('\n')
        res = [line.split(", ('")[-1].split("', ")[0] for line in lines if
               re.search("\[(.*?)\]\sppocr INFO:\s\[(.*?)\]\],\s", line)]

        # Write the result to a file
        with open(new_file_path, "w") as file:
            file.write('\n'.join(res))
