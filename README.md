## Deep Learning Assignment 1

### Objective
The primary objective is to familiarize students with the core concepts of deep learning and computational neural networks. Through implementing a Softmax classifier, students will gain insights into vectorized computation, loss function optimization, and the importance of hyperparameter tuning in model performance.

### Key Components

- **Softmax Classifier Implementation:** Students will start by implementing a naive version of the Softmax loss function and its gradient with nested loops. They will then optimize this implementation by vectorizing the computation to improve efficiency.

- **Gradient Checking:** To verify the correctness of the implemented gradient, numerical gradient checking is used. This step is crucial to ensure that the analytical gradients computed during backpropagation are accurate.

- **Hyperparameter Tuning:** Using the CIFAR-10 dataset, students will split the data into training, validation, and test sets. The assignment guides students through the process of tuning learning rates and regularization strengths to find the best model configurations.

- **Optimization with Stochastic Gradient Descent (SGD):** Students will implement SGD to optimize the Softmax loss function. This process includes iterating over minibatches of the data and updating the model weights based on the gradient of the loss with respect to the weights.

- **Model Evaluation and Analysis:** The final model is evaluated on a test dataset to gauge its generalization performance. Additionally, students are tasked with visualizing the weights learned by the model for each class, providing insights into what features the model finds most discriminative.

### Tools and Technologies
- **Python** as the programming language.
- **NumPy** for efficient numerical computations.
- **Matplotlib** for visualization of datasets and learned weights.
- **Jupyter Notebooks** for interactive development and testing.

### Learning Outcomes
- Understand the workflow of building and optimizing a machine learning model for image classification tasks.
- Gain proficiency in vectorized coding, significantly reducing computational time.
- Learn how to split data into training, validation, and test sets to ensure robust model evaluation.
- Explore the effects of hyperparameter tuning on model performance.
- Visualize how a model perceives different classes based on the weights it has learned during training.

This assignment is a step towards building foundational skills in deep learning, preparing students for more advanced topics such as convolutional neural networks, recurrent neural networks, and deep reinforcement learning.

## Deep Learning Assignment 2
Deep Learning Assignment 2 builds upon the foundational knowledge established in the first assignment, advancing into more complex areas of deep learning, such as convolutional neural networks (CNNs), dropout, and normalization techniques. This assignment is designed to provide students with hands-on experience in implementing and optimizing CNNs for image classification tasks using the CIFAR-10 dataset. Here’s a detailed overview of the assignment:

### Objective

The main goal is to deepen students' understanding of convolutional neural networks and their application in image classification tasks. By implementing various components of CNNs, including convolutional layers, pooling layers, and normalization techniques, students will learn how to build efficient and effective image recognition systems.

### Key Components

- **Convolutional Networks Implementation:** Students will implement convolutional layers and integrate them into CNN models, exploring how these layers help in capturing spatial hierarchies in images.

- **Dropout:** The assignment explores dropout as a regularization technique to prevent overfitting in neural networks. Students will implement dropout layers and observe their impact on model performance.

- **Normalization Techniques:** Students will implement batch normalization and spatial batch normalization, learning how these techniques can accelerate training and improve model accuracy.

- **Fully Connected Networks:** Building on the first assignment, students will further explore fully connected networks, incorporating dropout and normalization techniques to enhance model performance.

- **Practical Experiments:** Using the CIFAR-10 dataset, students will conduct experiments to compare different network architectures, regularization methods, and normalization techniques, gaining insights into their effects on model accuracy and performance.

### Tools and Technologies

- **Python** for programming, with a focus on **NumPy** for numerical computations and **Matplotlib** for data visualization.
- **Jupyter Notebooks** for an interactive development environment, enabling efficient experimentation and analysis.

### Learning Outcomes

- **Advanced Understanding of CNNs:** Students will gain a comprehensive understanding of how CNNs work, including the implementation and optimization of convolutional and pooling layers.

- **Regularization Techniques:** Through the implementation of dropout, students will learn how to design neural networks that generalize well to unseen data.

- **Normalization Methods:** By implementing and experimenting with batch normalization and spatial batch normalization, students will understand these techniques' roles in stabilizing and accelerating neural network training.

- **Hands-on Skills:** The assignment emphasizes practical skills in building and optimizing CNN models for image classification, preparing students for real-world applications of deep learning.

This assignment aims to equip students with the skills and knowledge to develop advanced neural network models for image classification and beyond, laying the groundwork for exploring more sophisticated deep learning models and techniques.

## Deep Learning Assignment 3

Deep Learning Assignment 3 elevates the complexity and application of neural networks by introducing Generative Adversarial Networks (GANs), network visualization techniques, and Recurrent Neural Networks (RNNs) for image captioning. This assignment is structured to deepen understanding and hands-on experience with these advanced topics, utilizing TensorFlow for implementation. Here’s a detailed overview for a README summary:

### Objective

This assignment aims to explore and implement advanced neural network models and techniques including GANs for generating images, visualization techniques to understand network decisions, and RNNs for generating captions for images. Through practical exercises, students will gain a deeper understanding of how neural networks can be used for image generation, visualization, and natural language processing tasks.

### Key Components

- **Generative Adversarial Networks (GANs):** Students will implement GANs to generate new images that mimic the distribution of a given dataset. This involves building and training both generator and discriminator networks in a competitive setting where the generator learns to produce increasingly realistic images.

- **Network Visualization:** The assignment explores techniques for visualizing and interpreting what convolutional neural networks learn. Students will implement methods to visualize the activations and features that networks focus on for classification tasks, enhancing understanding of model behavior.

- **Recurrent Neural Networks (RNNs) for Image Captioning:** Focusing on the intersection of computer vision and natural language processing, students will build and train RNNs to generate descriptive captions for images. This involves processing image features and sequential text data to produce coherent and relevant captions.

- **Experimentation and Analysis:** Each component requires students to experiment with different architectures, parameters, and techniques. Students will analyze the effects of their design choices on model performance and the quality of generated content.

### Tools and Technologies

- **TensorFlow:** Implementation and experimentation are conducted using TensorFlow, emphasizing its capabilities for building and optimizing complex models.
- **Python:** The primary programming language used for the assignment.
- **Jupyter Notebooks:** Interactive notebooks are used for code development, experimentation, and analysis, providing a platform for iterative exploration and visualization.

### Learning Outcomes

- **Advanced Model Implementation:** Students will gain hands-on experience implementing complex models like GANs and RNNs, advancing beyond basic neural network architectures.
- **Deep Understanding of Neural Network Applications:** Through building models for image generation and captioning, students will understand the practical applications and challenges of using neural networks in diverse domains.
- **Visualization and Interpretation Skills:** Implementing network visualization techniques will equip students with the ability to interpret model decisions, a critical skill for model evaluation and improvement.

This assignment represents a comprehensive exploration of advanced neural network concepts and applications, providing students with the skills and knowledge to tackle complex problems in deep learning. Through practical implementation and experimentation, students will appreciate the power and challenges of current deep learning techniques.
