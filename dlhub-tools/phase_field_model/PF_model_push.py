from dlhub_sdk.models import BaseMetadataModel
from dlhub_sdk.client import DLHubClient
import json

# Read the model description
with open('PF_model_metadata.json') as fp:
    model = BaseMetadataModel.from_dict(json.load(fp)) 

# Publish the model to DLHub
client = DLHubClient()
print(client.get_username())
result = client.publish_servable(model)
print(result)
print('Model published to DLHub. ID:', result)