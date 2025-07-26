# 3D Image Reconstruction with SFCN

## Files Structure

`reconstruct_image.py` - the main file that performs classifier training and image reconstruction.

## Usage

```bash
python reconstruct_image.py
```
No additional arguments required.

## Configuration

`classificator.py` contains the SFCN model class.
`datagenerator_pd.py` contains the DataGenerator class for training and validation.

All classes are imported in `reconstruct_image.py`