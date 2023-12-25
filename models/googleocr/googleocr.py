import os
from models.engine import Engine
from google.cloud import vision
import logging


class GoogleOCR(Engine):
    def __init__(self, method="DEFAULT"):
        super().__init__(method)
        self.model_name = "googleocr"
        self.client = vision.ImageAnnotatorClient()

    def img_to_text_helper(self, file):
        new_file_name = os.path.splitext(os.path.basename(file))[0] + f"_{self.model_name}"
        new_file_path = os.path.join(os.path.abspath(self.text_path), new_file_name)

        """Detects text in the file."""

        with open(file, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = self.client.text_detection(image=image)
        text = response.text_annotations[0].description

        with open(new_file_path + ".txt", "w") as file:
            file.write(text)

        if response.error.message:
            logging.error(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )
