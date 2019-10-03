# Predicting-Ferry-Delays
---
This is the repository used for collaboration on the solution for the CANSSI National Case Study Competition 2019. More info can be found at http://www.canssi.ca/news-events/canssi-datathon-2019/

The idea for this project is to take the various inputs surrounding a ferry trip and output a value in [0,1] representing the probability of the ferry being delayed. The scoring is done using Area Under the Curve (AUC) Receiver Operating Characteristics (ROC). This means that the further away from the true value you are, the worse you score. This neural network was created using Tensorflow with the following details:

 - Loss function: Binary Cross Entropy
    - The ideal loss function would be AUC (since this is how the values are scored) however since AUC is not differentiable, it cannot be used as a loss function. Binary cross entrpoy works well for this problem as the true values are binary and the cross entropy loss, or log loss, increases as the differences between the predicted probability and the actual label increase
- Topology: 30, 10, 1
    - Chose to not include a bias term on the input layer and to include a hidden layer of size 10
- Activation function: ReLU
    - Chosen over sigmoid to reduce the liklihood of the gradient vanishing
- Optimizer: RMSProp
    - Chosen over gradient descent because it maintains momentum in the gradient for a faster learning rate