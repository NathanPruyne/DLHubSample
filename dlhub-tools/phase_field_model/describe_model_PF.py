from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from dlhub_sdk.utils.schemas import validate_against_dlhub_schema
from dlhub_sdk.utils.types import compose_argument_block
import json
import os

model = PythonStaticMethodModel.create_model('dlhub_apply', 'apply', function_kwargs={
    'model_file': 'PFNetwork.pth',
    'norms_file': 'PFnormsnew.txt'
})

model.set_inputs('list', 'List of numpy arrays for image files', item_type = compose_argument_block('ndarray', 'Image', [None, None, 3]))
model.set_outputs('list', 'List of numpy arrays of completed segmentations', item_type = compose_argument_block('ndarray', 'Segmentation', [None, None]))

model.set_title('Dendrite Segmentation PF SegNet Model')
model.set_name('dendrite_pf_segnet')
model.set_authors(['Stan, Tiberiu', 'Thompson, Zachary T.', 'Voorhees, Peter W.', 'Pritz, Joshua'], ['Northwestern University', 'Northwestern University', 'Northwestern University', 'Northwestern University'])
model.set_domains(['materials science'])
model.set_abstract("A deep learning model, based on SegNet, to detect dendrites in an X-ray tomography image")

model.add_related_identifier('10.1016/j.matchar.2020.110119', 'DOI', 'IsDescribedBy')

model.add_requirement('torch', '1.6.0')
model.add_requirement('torchvision', '0.7.0')
model.add_requirement('numpy', 'detect')

model.add_file('dlhub_apply.py')
model.add_file('models.py')
model.add_file('PFNetwork.pth')
model.add_file('PFnormsnew.txt')

metadata = model.to_dict()
print(json.dumps(metadata, indent=2))
validate_against_dlhub_schema(metadata, 'servable')

with open('PF_model_metadata.json', 'w') as fp:
    json.dump(metadata, fp, indent=2)