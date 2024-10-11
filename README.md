# GST

# LightGBM For GST

Youtube Video:





## Methodology

## 1.Handling Datasets



## 2.Model Selection

For this task, I selected LightGBM, a gradient boosting framework that uses decision trees and is highly optimized for speed and efficiency, particularly with large datasets. LightGBM handles imbalanced data effectively by adjusting leaf-wise growth and leveraging advanced sampling techniques.

LightGBM Configuration:

-> Boosting Type: 'gbdt' (Gradient Boosting Decision Trees) was used to optimize the model iteratively.

-> Number of Estimators: 100 estimators were set, allowing the model to refine its predictions over many rounds.

->Learning Rate: A moderate learning rate of 0.1 was used to ensure gradual improvement in performance and avoid overfitting.

->Max Depth: A value of -1 was used to allow LightGBM to determine the optimal depth for each tree.

->Subsample and Colsample: Both were set to 0.8, meaning that LightGBM randomly selects 80% of the rows and features for each tree to reduce overfitting and improve generalization.

-> Missing Values: LightGBM handles missing values automatically, which reduces preprocessing time.

## 3.Training and Validation
The model was trained on the processed training data, and predictions were made on the test set. LightGBM's leaf-wise tree growth allowed it to make precise splits while reducing error in each boosting iteration.
## 4. Evaluation Metrics

Several metrics were employed to evaluate the performance of the LightGBM model:

LightGBM Test Accuracy: 0.9783 

LightGBM Test Precision: 0.8453

LightGBM Test Recall: 0.9421 

LightGBM Test F1 Score: 0.8911 

LightGBM Test Balanced Accuracy: 0.9621

LightGBM Test Log Loss: 0.0491 

LightGBM Test AUC-ROC: 0.9949 

