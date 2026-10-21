# Parkinson's Disease Detection from Hand Drawings – Deep Learning

A deep learning project where I use **transfer learning** to classify whether a person has Parkinson's disease from images of **spiral and wave drawings**.

Parkinson's affects movement, and one early sign is tremor and difficulty with fine motor control, which shows up when people draw simple shapes like spirals and waves. The aim was to see how well pre-trained image models can pick this up from a very small dataset.

> The notebook was written in French, but the code and outputs are standard Python/Keras.

---

## Dataset

- **Source:** [Parkinson's Drawings](https://www.kaggle.com/datasets/kmader/parkinsons-drawings) (Kaggle)
- **204 images** of spiral and wave drawings, labelled **Healthy** or **Parkinson**
- Split into: **128 training**, **16 validation**, **60 test** images

## Tools

- Python, TensorFlow / Keras
- Google Colab (GPU)
- Matplotlib, NumPy, scikit-learn

---

## Approach

**1. Data preparation**
- Resized all images to 224 × 224
- Rescaled pixel values to 0–1
- Used **data augmentation** on the training set (shear, zoom, horizontal flip) to make the most of the small dataset

**2. Transfer learning**

With only 128 training images, training a model from scratch wasn't realistic, so I used two models pre-trained on ImageNet (millions of images) and added my own layers on top:

| Model | Layers added on top |
|-------|---------------------|
| **VGG16** | Flatten → Dense (512, ReLU) → Dropout (0.5) → Dense (1, sigmoid) |
| **InceptionV3** | Global Average Pooling → Dense (512, ReLU) → Dense (1, sigmoid) |

Each model was trained in two stages:
1. **Feature extraction:** the pre-trained layers frozen, only my new layers trained (50 epochs)
2. **Fine-tuning:** some pre-trained layers unfrozen and retrained with a lower learning rate (20 epochs)

---

## Results (test set, 60 images)

| Model | Feature extraction | After fine-tuning |
|-------|-------------------|-------------------|
| **VGG16** | **88.3%** | 73.3% |
| **InceptionV3** | 85.0% | 70.0% |

**Best model: VGG16 with frozen layers, 88.3% test accuracy.**

### What the results show

- **Transfer learning worked well on a tiny dataset:** both models reached 85%+ accuracy with only 128 training images.
- **Fine-tuning made both models worse.** Training accuracy went up to 100% while test accuracy dropped. That's a clear sign of **overfitting**: with this little data, unfreezing more layers let the models memorise the training images instead of learning general patterns.
- **The validation set was too small to rely on.** With only 16 images, validation accuracy jumped between 81% and 100%, so the test set gives a much better picture of real performance.

## What I'd do differently

- Use **k-fold cross-validation** instead of a fixed split, since the dataset is so small
- Train **separate models for spirals and waves**, as they're quite different shapes
- Report **precision, recall and a confusion matrix**, not just accuracy. In a medical setting, missing someone with Parkinson's (a false negative) matters more than a false alarm
- Use **early stopping** to prevent overfitting during fine-tuning

## How to run it

1. Download the dataset from Kaggle and upload it to Google Drive in a folder called `parkinson`, with `training`, `validating` and `testing` subfolders.
2. Open the notebook in Google Colab and run the cells in order. It mounts Google Drive to read the images.
