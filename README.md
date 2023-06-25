# Dual Level Adaptive Weighting
Implement of paper:[Dual Level Adaptive Weighting for Cloth-changing
Person Re-identification]
## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- torchvision
- scipy
- numpy
- scikit_learn



## Data

#### LTCC
Download from [here](https://naiq.github.io/LTCC_Perosn_ReID.html)

#### PRCC
Download from [here](http://www.isee-ai.cn/%7Eyangqize/clothing.html)

## Train

You can specify more parameters in opt.py. Note that the evaluation protocols differ for the LTCC, PRCC, and Market1501 datasets. As a result, there are three alternative code files available: main.py for the LTCC dataset, main_prcc.py for the PRCC dataset, and main_market1501.py for the Market1501 dataset. Each code file is specifically tailored to the corresponding dataset's evaluation protocol and should be used accordingly.

```
python main.py --mode train --data_path <path/to/LTCC-ReID> 
```

## Evaluate

Use pretrained weight or your trained weight

```
python main.py --mode evaluate --data_path <path/to/LTCC-ReID> --weight <path/to/weight_name.pt> 
```


## Visualize

Visualize rank10 query result of one image(query from bounding_box_test)

Extract features will take a few munutes, or you can save features as .mat file for multiple uses

```
python main.py --mode vis --query_image <path/to/query_image> --weight <path/to/weight_name.pt> 
```


