# Patient Outcome Prediction

## Project Description

The goal of this project is to predict patient outcomes by estimating the probabilities of three possible classes for each identifier (`id`):

- **Status_C**: Alive without a transplant.
- **Status_CL**: Alive with a transplant.
- **Status_D**: Deceased.

For each patient, the output will be a probability distribution across these three classes.

## Log Loss Metric

The model's performance is evaluated using the **log loss** (negative log-likelihood) metric, which quantifies the accuracy of predicted probabilities for each class.

The log loss is defined as:

\[
\text{log loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \log(p_{ij})
\]

Where:
- \( N \) is the number of observations (patients).
- \( M \) is the number of classes (3 classes: `Status_C`, `Status_CL`, `Status_D`).
- \( y_{ij} \) is 1 if the true class for patient \( i \) is class \( j \), and 0 otherwise.
- \( p_{ij} \) is the predicted probability that patient \( i \) belongs to class \( j \).

### Handling Predicted Probabilities

The model outputs predicted classes as discrete labels (0, 1, or 2). These are then converted into probabilities using a one-hot encoding, where the predicted class gets a probability of 1, and the other classes get a probability of 0.

To prevent numerical issues with the logarithm, the predicted probabilities are clipped to the range [1e-15, 1 - 1e-15].

## Usage

1. Prepare your training and testing data.
2. Train a machine learning model to predict patient outcomes.
3. Use the model's predictions to calculate log loss using the provided `log_loss` function.


