from dlhub_sdk.models import BaseMetadataModel
from dlhub_sdk.client import DLHubClient
import json
import argparse

def publish(metadata_file):
    # Read the model description
    with open(metadata_file) as fp:
        model = BaseMetadataModel.from_dict(json.load(fp))

    # Publish the model to DLHub
    client = DLHubClient()
    print("Publishing model as " + client.get_username() + "/" + model['dlhub']['name'])
    result = client.publish_servable(model)
    print('Model published to DLHub. ID:', result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a model to DLHub")

    parser.add_argument('metadata_file', help='Path to a generated file with model metadata')

    args = parser.parse_args()

    publish(args.metadata_file)
    