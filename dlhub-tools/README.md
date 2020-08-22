# Using DLHub with MatSeg models

Uploading and utilizing models to DLHub revolves around the `dlhub_apply.py` file. This method accepts a list of numpy array representations of images to segment, and outputs numpy arrays representing the outputted segmentations. The full process of uploading a model to DLHub and utilizing it has four steps:

1. Generating model and norms files
2. Describing the model
3. Publishing the model to DLHub
4. Testing and utilizng the model

**IMPORTANT**: Many files in this folder depend on functionalities in the core MatSeg package, and thus should be stored in the same folder as the core MatSeg files in order to function properly.

## Generating model and norms files

The first step to using DLHub with MatSeg is to generating a .pth file that contains all necessary data to create the model and a text file containing the average and standard deviation needed in the batch normalization part of the processing.

### Generating full model definition (`generate_full_model.py`)

After a model is saved using `main.py`, its saved state must be translated into a full model definition (rather than a state dict) using `generate_full_model.py`. This method accepts a model definition using dataset and version number just as other methods, and requires a 3rd positional argument of the file name to save the full model file to (exclude the .pth extension as it is added automatically). For instance, v7 of the tomography model can be saved to `tomography_v7.pth` using

```
python3 generate_full_model.py tomography v7 tomography_v7
```

### Generating norms files (`generate_norms.py`)

The model's normalization preprocessing's average and standard deviation must also be stored in a file for use on DLHub, as the training dataset will not be stored. These normalization characteristics can be saved using `generate_norms.py`. The arguments are the same as storing the full model definition, and the given filename automatically has the .txt extension added to it. Thus, to store the normalization characteristics for v7 of the tomography model in `tomography_v7.txt` use

```
python3 generate_norms.py tomography v7 tomography_v7
```

**IMPORTANT**: Due to limitations of the pickle module, model and norms files must also be saved in the same directory (not a subdirectory) as the `dlhub_apply.py` and `models.py` files, or they will not be properly detected once published to DLHub.

## Describing a model (`describe_model.py`)

The next step is to describe the model with metadata fields, a name, and the required model and norms files. This is done by modifying a copy of `describe_model.py` with details specific to the model. Four changes to this file must be made:

1. Replace the `function_kwargs` in the model definition (line 8) with the appropriate model and norms files. For instance:

```
model = PythonStaticMethodModel.create_model('dlhub_apply', 'apply', function_kwargs={
    'model_file': 'tomography_v7.pth',
    'norms_file': 'tomography_v7.txt'
})
```

2. Set the name in line 17 to the name you would like the model to be called by on DLHub. For instance:

```
model.set_name('dendrite_segmentation_tomography_model')
```

3. Fill in the names of the model and norms files on lines 34 and 35 respectively:

```
model.add_file('tomography_v7.pth')
model.add_file('tomography_v7.txt')
```

4. Fill in the name of the json file to save the metadata to on line 42:

```
with open('tomo_v7_metadata.json', 'w') as fp:
```

Other modifications to the model metadata can be made, including the title (line 20), authors (using 2 arrays, first with author names and second with respective affiliations, on line 21), domains (line 22), a brief description (line 23), and any identifiers (line 25).

Running `describe_model.py` will output a metadata json file to the designated location.

## Publishing a model to DLHub (`publish_model.py`)

After generating a json file with the model metadata, required files, and other attributes, the model can be published to DLHub using `publish_model.py`. This method takes one positional argument, the json file outputted when describing the model:

```
python3 publish_model.py tomo_v7_metadata.json
```

At this point a DLHub client will be initialized, and you may be prompted to login using Globus Auth to publish the model. The method will print out the name that the model is stored under in DLHub and what it should be called with (will be client_username/name where name is defined in the model description), and the ID the model is stored under in DLHub.

## Testing a model

There are two methods that can be used to test a model once it has been published to DLHub:

### `test_model.py`

This method processes all .png image files in a directory and outputs the results as the same filenames in a different directory. It takes 3 positional arguments, a path to the directory containing files to process, the name of the model on DLHub (outputted when the model is published and how it is displayed on DLHub serach), and the path to a directory to store the processed images in. For instance, to test the tomography v7 model on the tomography dataset, storing the results in a results folder, use:

```
python3 test_model.py data/tomography/test/images npruyne_globusid/dendrite_tomography_unet results/
```

All .png files in `data/tomography/test/images` will be processed and have their outputs outputted with the same name as the original image file in the `results/` directory.

### `test_model_one_by_one.py`

This method sends files one by one to DLHub for processing rather than all at once in one list. This can be helpful when sending large files, as it makes it less likely for the data sent to DLHub to exceed the size limit. It takes the same arguments as `test_model.py`.

## Other tools

### `dlhub_apply_test.py`

`dlhub_apply_test.py` provides a way to test the functionality of `dlhub_apply.py` locally to confirm the functionality with certain model and norms files before uploading to DLHub. It takes four positional arguments, the model file, norms file, directory containing images to process, and directory to output resulting segmentation images. For instance, to test the tomography v7 model before uploading:

```
python3 dlhub_apply_test.py tomography_v7.pth tomography_v7.txt data/tomography/test/images results/
```

## Included models

Also in this directory are the static norms files, description, publishing, and testing scripts, and json metadata representations for two models: a SegNet model trained on phase field data and a UNet model trained on tomography data (version 7 in the original configuration). Many methods here are hardcoded, but can allow you to test the functionality of the corresponding models on DLHub ([Phase Field Model](https://petreldata.net/dlhub/detail/https%253A%252F%252Fdlhub.org%252Fservables%252Ff066d1e1-b121-4f8d-803d-ab4eb8659955/) and [Tomography Model](https://petreldata.net/dlhub/detail/https%253A%252F%252Fdlhub.org%252Fservables%252F532d8ade-fad2-438e-b780-572ab68cca76/)). The corresponding .pth model files are too large to store on GitHub, but are available in this [Google Drive folder](https://drive.google.com/drive/folders/1JsjqLznwsWCoU6ah5uH121Ho2a5K3p3c?usp=sharing).

**IMPORTANT**: All files must once again be transferred to the same directory as the core MatSeg files for the proessing to work properly.

## Details on `dlhub_apply.py` functionality

Processing in a DLHub container is done through a copy of `dlhub_apply.py`. This method takes in a list of numpy arrays, each containing one image in 3 RGB channels. It begins by transforming the numpy array into normalized PyTorch tensors, then feeds this tensor into the pretrained model. It then outputs a list of numpy arrays of the same length and dimensions as the input (except with only one channel), which represents the resulting segmentation. 