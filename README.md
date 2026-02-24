# X-Ray Pneumonia Detection Using Explainable AI (LIME)

## Introduction

Medical image classification is one of the most impactful applications of deep learning. Pneumonia is a serious lung infection that can be detected using chest X-ray images. However, traditional deep learning models often act as “black boxes,” making predictions without explaining the reasoning behind them. In medical applications, interpretability is as important as accuracy.

This project focuses on building a pneumonia detection model using Transfer Learning and enhancing it with Explainable AI using LIME (Local Interpretable Model-Agnostic Explanations).

---

## Dataset Description

The dataset used for this project was the **Chest X-Ray Pneumonia dataset from Kaggle**. It contains labeled X-ray images categorized into:

- **NORMAL**
- **PNEUMONIA**

The dataset was divided into training, validation, and test sets to ensure proper evaluation of model performance.

---

## Data Preprocessing

Before training the model, the following preprocessing steps were applied:

- Resized all images to **224 × 224 pixels**
- Normalized pixel values using rescaling (**1/255**)
- Applied data augmentation techniques:
  - Rotation
  - Zoom
  - Horizontal flipping

Data augmentation helped improve generalization and reduce overfitting.

---

## Model Architecture

To build an efficient classifier, **ResNet50**, a pre-trained Convolutional Neural Network trained on ImageNet, was used.

### Steps Followed:

- Loaded ResNet50 without the top classification layer
- Froze the base model layers to retain learned features
- Added:
  - Global Average Pooling layer
  - Dense layer (128 neurons, ReLU activation)
  - Output layer (1 neuron, Sigmoid activation for binary classification)

### Model Compilation

- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metric:** Accuracy  

The model was trained for **3 epochs** and evaluated on unseen test data.

---

## Model Prediction

The model outputs a probability score between **0 and 1**:

- **0 → Normal**
- **1 → Pneumonia**

A sigmoid activation function was used for binary classification.

---

## Explainable AI Using LIME

Deep learning models often provide high accuracy but lack transparency. To address this issue, LIME (LimeImageExplainer) was integrated to interpret individual predictions.

### Why LIME?

In healthcare applications, it is essential to understand which regions of an X-ray influenced the model’s decision. Doctors and healthcare professionals must trust AI predictions before adopting them.

### How LIME Works

- The image is segmented into superpixels.
- LIME perturbs different parts of the image by masking regions.
- It observes how prediction probability changes.
- It builds a simple interpretable model locally around that prediction.
- It highlights the most influential regions.

### Output Interpretation

The output visualization highlights important areas in the X-ray image that contributed positively or negatively to the pneumonia classification.

This ensures the model focuses on lung infection regions rather than irrelevant areas such as image borders or background.

---

## Results and Impact

- Successfully implemented transfer learning for medical image classification.
- Built a binary pneumonia detection system.
- Integrated Explainable AI for model transparency.
- Demonstrated how AI can support medical diagnosis while improving interpretability and trust.

---

## Conclusion

This project combines deep learning, medical imaging, and explainable AI to create a transparent pneumonia detection system. By integrating LIME, the model moves beyond a black-box classifier and becomes a more interpretable and reliable decision-support tool.

The project highlights the importance of combining accuracy with explainability, especially in sensitive domains like healthcare.
