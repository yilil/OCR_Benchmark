import os
from models.engine import Engine
import subprocess
import logging


class Tesseract(Engine):
    def __init__(self, method="DEFAULT"):
        super().__init__(method)
        self.model_name = "tesseract"

    def img_to_text_helper(self, file):
        new_file_name = os.path.splitext(os.path.basename(file))[0] + f"_{self.model_name}"
        new_file_path = os.path.join(os.path.abspath(self.text_path), new_file_name)

        '''
        Run engine by using bash command
        '''
        cmd = [
            'tesseract',
            os.path.join(self.image_path, file),
            new_file_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Check if the command completed successfully
        if result.returncode != 0:
            logging.error(f'Error occurred while running command: {result.stderr.decode()}')
