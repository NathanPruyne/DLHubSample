from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from dlhub_sdk.utils.schemas import validate_against_dlhub_schema
from dlhub_sdk.utils.types import compose_argument_block
import json
import os

#Enter the model and norms file here
model = PythonStaticMethodModel.create_model('dlhub_apply', 'apply', function_kwargs={
    'model_file': '.pth',
    'norms_file': '.txt'
})

model.set_inputs('list', 'List of numpy arrays for image files', item_type = compose_argument_block('ndarray', 'Image', [None, None, 3]))
model.set_outputs('list', 'List of numpy arrays of completed segmentations', item_type = compose_argument_block('ndarray', 'Segmentation', [None, None]))

#This name will be how the model is accessed in DLHub, choose accordingly
model.set_name('dendrite_segmentation')

#Modify these attributes as needed
model.set_title('Dendrite Segmentation Model')
model.set_authors(['Stan, Tiberiu', 'Thompson, Zachary T.', 'Voorhees, Peter W.', 'Pritz, Joshua'], ['Northwestern University', 'Northwestern University', 'Northwestern University', 'Northwestern University'])
model.set_domains(['materials science'])
model.set_abstract("A deep learning model to detect dendrites")

model.add_related_identifier('10.1016/j.matchar.2020.110119', 'DOI', 'IsDescribedBy')

model.add_requirement('torch', '1.6.0')
model.add_requirement('torchvision', '0.7.0')
model.add_requirement('numpy', 'detect')

model.add_file('dlhub_apply.py')
model.add_file('models.py')
#Replace both files below with the needed model and norms files
model.add_file('.pth')
model.add_file('.txt')

metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')

#Modify the json file name here
with open('metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)