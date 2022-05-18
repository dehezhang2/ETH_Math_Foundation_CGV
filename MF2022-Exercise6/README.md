# Exercise 6 - DEEP LEARNING

## Code Implemented

* `SRDataset` class: in the `./code/dataset.py` 

* `BasicSRModel` class: in the `./code/models.py` 

* Training loop: in the `./code/train.py`

* Model evaluation: 

  * in `./code/eval.py`
  * visualization is in `./code/eval.ipynb`

* The configurations are in `./configs`

* To run the code, make sure you use the following file structure

  ```
  .
  ├── configs
  │   ├── 1e-2.yaml
  │   ├── 1e-3.yaml
  │   ├── 1e-5.yaml
  │   ├── 1e-6.yaml
  │   ├── baseline.yaml
  │   └── residual.yaml
  ├── runs
  │   ├── baseline
  │   ├── lr1e-2
  │   ├── lr1e-3
  │   ├── lr1e-5
  │   ├── lr1e-6
  │   └── residual
  ├── models
  ├── best_models
  │   ├── model_best_lr_1E-02.pth
  │   ├── model_best_lr_1E-03.pth
  │   ├── model_best_lr_1E-04.pth
  │   ├── model_best_lr_1E-05.pth
  │   ├── model_best_lr_1E-06.pth
  │   └── model_res_best_lr_1E-04.pth
  ├── train
  └── eval
  ```

* Run

  * For the training (baseline model), you should use the command in`./code`:

  ```shell
  python train.py --config ../configs/baseline.yaml
  ```

  * For the evaluation(baseline model), you should use the command in`./code`:

  ```shell
  python eval.py --config ../configs/baseline.yaml
  ```

  * Or you can directly use the `eval.ipynb` file. 

## 1.1. Task 1 - Datasets, Preprocessing and Data loading

* The result of dataset testing is

```
 * Dataset contains 301 image(s).
```

## 1.2. Task 2 - Derivations and deeper understanding

* The result of model testing is:

```
372803
```

## 1.3. Task 3 - Implement the Training Loop

* The L1 loss during training is shown below

![Screen Shot 2022-05-17 at 10.15.59 AM](assets/Screen Shot 2022-05-17 at 10.15.59 AM.png)

* The parameters are saved per 200 epochs, also, the model with minimum training loss is also saved .

## 1.4. Task 4 - Model Evaluation

* Table of evaluation result:

  |                     | PSNR      | SSIM     |
  | ------------------- | --------- | -------- |
  | Bilinear (baseline) | 25.757679 | 0.867994 |
  | Bicubic (baseline)  | 24.444685 | 0.856417 |
  | Nearest (baseline)  | 21.104866 | 0.758209 |
  
* The evolution of the validation loss and training loss (L1, PSNR and SSIM) is shown below:

  | <img src="assets/Screen Shot 2022-05-17 at 10.15.59 AM.png" alt="Screen Shot 2022-05-17 at 10.15.59 AM" style="zoom:100%;" /> |
  | ------------------------------------------------------------ |
  | <img src="assets/Screen Shot 2022-05-17 at 10.37.30 AM.png" alt="Screen Shot 2022-05-17 at 10.37.30 AM" style="zoom:100%;" /> |
  | <img src="assets/Screen Shot 2022-05-17 at 10.37.34 AM.png" alt="Screen Shot 2022-05-17 at 10.37.34 AM" style="zoom:100%;" /> |

## 1.5. Task 5 - Exploration

### A Different Model

* Training: The training loss plot is shown below (the blue one represents the residual, and the orange one represents the baseline model)
  * **Comparison**: Since the residual connections directly add the result of bilinear interpolation to the output, the performance is always better than the baseline model (lower l1 loss and higher PSNR, SSIM). 

| <img src="assets/Screen Shot 2022-05-17 at 8.08.13 PM.png" alt="Screen Shot 2022-05-17 at 8.08.13 PM" style="zoom:100%;" /> |
| ------------------------------------------------------------ |
| <img src="assets/Screen Shot 2022-05-17 at 8.08.28 PM.png" alt="Screen Shot 2022-05-17 at 8.08.28 PM" style="zoom:100%;" /> |
| <img src="assets/Screen Shot 2022-05-17 at 8.08.43 PM.png" alt="Screen Shot 2022-05-17 at 8.08.43 PM" style="zoom:100%;" /> |

* Evaluation: Table of evaluation result:

|                     | PSNR      | SSIM     |
| ------------------- | --------- | -------- |
| Bilinear (baseline) | 25.757679 | 0.867994 |
| Bicubic (baseline)  | 24.444685 | 0.856417 |
| Nearest (baseline)  | 21.104866 | 0.758209 |
| Bilinear (Resnet)   | 27.583456 | 0.898651 |
| Bicubic (Resnet)    | 26.075844 | 0.885841 |
| Nearest (Resnet)    | 22.011930 | 0.780135 |

### Effect of the Learning Rate

* Training: The training loss plot with different learning rate is shown below (the blue one represents the residual, and the orange one represents the baseline model)

  * Without large learning rate (without 1e-2, 1e-3): orange (1e-4), pink (1e-5), green (1e-6)
  * The performance is better as learning rate getting larger in this range

  | <img src="assets/Screen Shot 2022-05-17 at 10.41.48 PM.png" alt="Screen Shot 2022-05-17 at 10.41.48 PM" style="zoom:100%;" /> |
  | ------------------------------------------------------------ |
  | <img src="assets/Screen Shot 2022-05-17 at 10.42.44 PM.png" alt="Screen Shot 2022-05-17 at 10.42.44 PM" style="zoom:100%;" /> |
  | <img src="assets/Screen Shot 2022-05-17 at 10.42.56 PM.png" alt="Screen Shot 2022-05-17 at 10.42.56 PM" style="zoom:100%;" /> |

  * However, for the large learning rate, the performance is not stable (1e-2 red, 1e-3 blue)

  | <img src="assets/Screen Shot 2022-05-17 at 10.50.39 PM.png" alt="Screen Shot 2022-05-17 at 10.50.39 PM" style="zoom:100%;" /> | ![Screen Shot 2022-05-17 at 10.51.31 PM](assets/Screen Shot 2022-05-17 at 10.51.31 PM.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | <img src="assets/Screen Shot 2022-05-17 at 10.51.02 PM.png" alt="Screen Shot 2022-05-17 at 10.51.02 PM" style="zoom:100%;" /> | ![Screen Shot 2022-05-17 at 10.51.41 PM](assets/Screen Shot 2022-05-17 at 10.51.41 PM.png) |
  | <img src="assets/Screen Shot 2022-05-17 at 10.51.16 PM.png" alt="Screen Shot 2022-05-17 at 10.51.16 PM" style="zoom:100%;" /> | ![Screen Shot 2022-05-17 at 10.51.53 PM](assets/Screen Shot 2022-05-17 at 10.51.53 PM.png) |

### Different Downscaling During Inference

*  Table of evaluation result:

|                     | PSNR      | SSIM     |
| ------------------- | --------- | -------- |
| Bilinear (baseline) | 25.757679 | 0.867994 |
| Bicubic (baseline)  | 24.444685 | 0.856417 |
| Nearest (baseline)  | 21.104866 | 0.758209 |
| Bilinear (Resnet)   | 27.583456 | 0.898651 |
| Bicubic (Resnet)    | 26.075844 | 0.885841 |
| Nearest (Resnet)    | 22.011930 | 0.780135 |

* Images of superresolution result with different downscaling methods
  * As shown in the following two examples, the algorithm does have the effect to do superresolution task. However, the generated image is darker than the original one. 
  * Comparison: 
    * Since we use bilinear interpolation to train the model, the result of blinear interpolation is better than the other two (brighter and sharper result). 
    * For the nearest neighbor, the detailed texture is different from the other two methods. The result is not smooth enough because there are some strange veins in the texture.
    * For the bicubic interpolation, the result is over smoothened compared with bilinear result. 

| High resolution (ground truth) | Low resolution                                       |                         |
| ------------------------------ | ---------------------------------------------------- | ----------------------- |
| ![](assets/hr_image.png)       | <img src="assets/lr_image.png" style="zoom:200%;" /> |                         |
| Bilinear                       | Bicubic                                              | Nearest   neighbor      |
| ![](assets/hr_pred.png)        | ![](assets/hr_cubic.png)                             | ![](assets/hr_near.png) |

| High resolution (ground truth) | Low resolution                                         |                           |
| ------------------------------ | ------------------------------------------------------ | ------------------------- |
| ![](assets/hr_image_1.png)     | <img src="assets/lr_image_1.png" style="zoom:200%;" /> |                           |
| Bilinear                       | Bicubic                                                | Nearest   neighbor        |
| ![](assets/hr_pred_1.png)      | ![](assets/hr_cubic_1.png)                             | ![](assets/hr_near_1.png) |

