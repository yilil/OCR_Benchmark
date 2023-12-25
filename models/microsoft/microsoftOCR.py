from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

import os
import time
from models.engine import Engine


class MicroSoftOCR(Engine):
    def __init__(self, method="DEFAULT"):
        super().__init__(method)
        self.model_name = "microsoftOCR"
        '''
        Authenticates your credentials and creates a client.
        '''
        subscription_key = os.environ["VISION_KEY"]
        endpoint = os.environ["VISION_ENDPOINT"]

        self.computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    def img_to_text_helper(self, file):
        new_file_name = os.path.splitext(os.path.basename(file))[0] + f"_{self.model_name}"
        new_file_path = os.path.join(os.path.abspath(self.text_path), new_file_name)

        # Call API with URL and raw response (allows you to get the operation location)
        read_response = self.computervision_client.read_in_stream(open(file, "rb"), raw=True)

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results
        while True:
            read_result = self.computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        # Print the detected text, line by line
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                res = ""
                for line in text_result.lines:
                    res += (line.text + "\n")

                with open(new_file_path + ".txt", "w") as file:
                    file.write(res.rstrip("\n"))
