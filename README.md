# deep-learning-challenge

**Overview of the Analysis:**

The purpose of this analysis is to develop a deep learning model that can predict whether applicants will be successful if funded by Alphabet Soup, a venture capital firm. The goal is to create a classification model that will help Alphabet Soup determine which funding applications are likely to be successful, allowing them to allocate their resources more effectively.

**Results:**

*Data Preprocessing:*

- **Target Variable(s):** The target variable for the model is the "IS_SUCCESSFUL" column, which indicates whether the funding application was successful (1) or not (0).

- **Feature Variable(s):** The feature variables include various columns from the dataset, such as "APPLICATION_TYPE," "AFFILIATION," "CLASSIFICATION," and others.

- **Variable(s) Removed:** Some variables like "EIN" and "NAME" are removed from the input data as they are neither targets nor useful features for the model.

**Compiling, Training, and Evaluating the Model:**

- **Neurons, Layers, and Activation Functions:**
  - I experimented with different configurations of neurons, layers, and activation functions to optimize the model's performance.

### Model 1
- **Architecture:**
  - First hidden layer: 
    - Neurons: 1
    - Activation function: relu
    - Input dimension: 44
  - Second hidden layer:
    - Neurons: 1
    - Activation function: relu
  - Output layer:
    - Neurons: 1
    - Activation function: sigmoid
- **Epochs:** 100
- **Performance:**
  - Accuracy: 97.21%
  - Loss: 0.1114

### Model 2
- **Architecture:**
  - First hidden layer: 
    - Neurons: 6
    - Activation function: relu
    - Input dimension: 44
  - Second hidden layer:
    - Neurons: 6
    - Activation function: tanh
  - Third hidden layer:
    - Neurons: 6
    - Activation function: tanh
  - Output layer:
    - Neurons: 1
    - Activation function: sigmoid
- **Epochs:** 50
- **Performance:**
  - Accuracy: 99.97%
  - Loss: 0.0023

### Model 3
- **Architecture:**
  - First hidden layer: 
    - Neurons: 1
    - Activation function: relu
    - Input dimension: 44
- **Epochs:** 200
- **Performance:**
  - Accuracy: 95.49%
  - Loss: 0.6962

### Model 4
- **Architecture:**
  - First hidden layer: 
    - Neurons: 1
    - Activation function: relu
    - Input dimension: 44
  - Second hidden layer:
    - Neurons: 1
    - Activation function: sigmoid
- **Epochs:** 300
- **Performance:**
  - Accuracy: 94.94%
  - Loss: 0.1723

In Model 1, I used the Relu function as it is ideal for modeling positive, nonlinear input data for classification or regression. Followed by the Sigmoid function for the output layer, which normalizes values to a probability between 0 and 1, which is ideal for a binary classification dataset like ours. I began with using 1 neuron for each layer as adding more neurons to a single hidden layer only boosts performance if there are subtle differences between values. The model achieved an accuracy of 97.21%.

To further increase the accuracy of the neural network model, I experimented with more neurons and layers. In model 2, I used 6 neurons and 4 layers with tanh function to increase the complexity of the deep learning model. More hidden layers evaluate higher-order interactions between weighted variables and identify complex, nonlinear relationships. Adding more neurons provides the network with a greater capacity to learn complex patterns and representations from the data. In this model, the tanh activation function was used for hidden layers which drastically improved our accuracy to 99.9%. Tanh has a smooth gradient compared to ReLU. The smoothness of the gradient makes it easier for the optimization algorithm (e.g., gradient descent) to find the global minimum of the loss function, leading to faster convergence during training even with fewer epochs.


**Summary**

- Model 2, with multiple hidden layers, neurons, and the 'tanh' activation function, achieved the highest accuracy of 99.9%. This suggests that deeper networks with specific activation functions can improve predictive performance.
- Tuning hyperparameters, such as the number of layers, neurons, and training epochs, plays a crucial role in achieving the desired model performance.
- Experimentation with different configurations is essential to find the optimal architecture for a given problem.
- Hyperparameter tuning and feature engineering are ongoing processes in deep learning model development and can further enhance model accuracy.

**Recommendation for a Different Model:**

If the deep learning model does not perform satisfactorily, an alternative model to consider is an ensemble model, such as a Random Forest or Gradient Boosting Classifier. These models often perform well in classification tasks and can handle a mix of feature types, including categorical and numerical variables. Additionally, hyperparameter tuning can be applied to improve their performance.

The deep learning model may require a large amount of data and extensive tuning to achieve optimal results, whereas ensemble models are robust and can provide good results with less data and less intensive tuning.

The choice of model ultimately depends on the specific dataset, objectives, and computational resources available to Alphabet Soup.
