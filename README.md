# TranslateAI: Ancient Japanese Cursive Character Recognition

This repository contains a PyTorch-based deep learning pipeline designed to recognize and classify ancient Japanese cursive characters (Kuzushiji) using the **KMNIST (Kuzushiji-MNIST)** dataset. 

This project was developed as part of a final examination/project evaluating deep neural network design, data scaling, advanced optimizations (Adam), and batch normalization using PyTorch.

## ———————————————————————

## 📖 Project Overview

Digitizing historical manuscripts is a significant challenge in cultural preservation. **TranslateAI** aims to automate the transcription of Japanese cursive texts. The repository demonstrates a complete workflow:
1. **Data Loading & Preprocessing:** MinMax normalization, splitting, and PyTorch tensor preparation.
2. **Visualizations:** Inspecting individual class samples and validating class distributions.
3. **Neural Network Architecture:** Multi-layer feedforward network equipped with `BatchNorm1d` to improve convergence.
4. **Optimization Pipeline:** Adam optimizer combined with CrossEntropyLoss.
5. **Analytics & Diagnostics:** Evaluation metrics, learning curves, and detailed confusion matrix analysis.

## 📊 Dataset

We utilize **KMNIST**, which comprises 10 cursive characters (*o, ki, su, tsu, na, ha, ma, ya, re, wo*) with 28x28 grayscale images.
*   **Total Training Images:** 60,000
*   **Validation Images:** 10,000 (split from training set)
*   **Total Test Images:** 10,000

## 🧠 Model Architecture

The classifier is built using PyTorch's `nn.Sequential` with the following configuration:

| Layer | Type | Input Dimension | Output Dimension | Activation / Features |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Flatten | 28x28 | 784 | - |
| 2 | Linear | 784 | 512 | BatchNorm1d, ReLU |
| 3 | Linear | 512 | 256 | BatchNorm1d, ReLU |
| 4 | Linear | 256 | 128 | BatchNorm1d, ReLU |
| 5 | Linear | 128 | 10 | Output Logits |

## 📈 Results & Evaluation

### Model Performance
*   **Final Training Accuracy:** 99.72%
*   **Final Validation Accuracy:** 96.24%
*   **Test Accuracy:** 91.74%

### Overfitting Diagnostics
As shown by the learning curves, the training loss steadily approaches zero while the validation loss begins to diverge slightly after the 3rd epoch. This indicates the model is **overfitting** to the training data.

### Error Analysis (Top Misclassifications)
Based on the confusion matrix, the model struggles most with visually similar ancient cursive characters:
1. **'su'** misclassified as **'tsu'**
2. **'ki'** misclassified as **'ma'**
3. **'na'** misclassified as **'re'**

---

## 🛠️ Mitigating Overfitting (Next Steps)
To improve generalizability on the test set and mitigate overfitting, the following optimizations are recommended:
*   **Dropout Regularization:** Incorporating `nn.Dropout(p=0.3)` layers after each ReLU activation to randomly deactivate neurons during training.
*   **Weight Decay (L2 Regularization):** Adding penalty to the optimizer to prevent weight values from exploding:
    ```python
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    ```
*   **Early Stopping:** Halting training early when the validation loss stops improving.
*   **Data Augmentation:** Utilizing `torchvision.transforms` (such as random rotations, translation, or scaling) to synthetically expand the variety of training samples.

## 🚀 How to Run
1. Install dependencies: `pip install torch torchvision scikit-learn matplotlib numpy`
2. Clone and open the notebook inside a Jupyter environment or Google Colab.
3. Run all cells sequentially.
