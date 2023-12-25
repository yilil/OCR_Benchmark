import os
from models.engine import Engine
import boto3


class AmazonTextract(Engine):
    def __init__(self, method="DEFAULT"):
        super().__init__(method)
        self.model_name = "AmazonTextract"
        self.client = boto3.client('textract', region_name='ap-southeast-2',
                                   aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                                   aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

    def img_to_text_helper(self, file):
        new_file_name = os.path.splitext(os.path.basename(file))[0] + f"_{self.model_name}"
        new_file_path = os.path.join(os.path.abspath(self.text_path), new_file_name)

        """Detects text in the file."""

        with open(file, 'rb') as image:
            img = bytearray(image.read())

        response = self.client.detect_document_text(
            Document={'Bytes': img}
        )

        text = "\n".join([item["Text"] for item in response["Blocks"] if item["BlockType"] == "LINE"])

        with open(new_file_path + ".txt", "w") as file:
            file.write(text)
