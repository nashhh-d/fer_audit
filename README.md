# fer_audit
# Explainable & Bias-Audited CNN Classifier

An explainable facial-expression classification system built using
FER2013 and a pretrained ResNet18 CNN.

The project combines image classification, Grad-CAM explainability,
condition-based reliability analysis, and human-review routing.

## Features

- 7-class facial expression classification
- ResNet18 transfer learning
- FER2013 dataset
- Data augmentation
- Confusion matrix and classification metrics
- Grad-CAM visual explanations
- Brightness, contrast, and sharpness analysis
- Failure-pattern analysis
- Confidence-based human-review routing
- Streamlit interactive interface

## Classes

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model

The classifier uses an ImageNet-pretrained ResNet18 with a modified
fully connected classification layer for the seven FER2013 classes.

## Reliability Audit

The model was evaluated under different observable image conditions.

### Brightness

| Condition | Accuracy |
|---|---:|
| Low | 66.5% |
| Medium | 69.9% |
| High | 73.2% |

The largest observed condition-based difference was associated with
brightness.

### Contrast

| Condition | Accuracy |
|---|---:|
| Low | 70.5% |
| Medium | 69.4% |
| High | 69.7% |

Performance remained relatively stable across contrast groups.

### Sharpness

| Condition | Accuracy |
|---|---:|
| Low | 68.2% |
| Medium | 71.2% |
| High | 70.2% |

Low-sharpness images showed somewhat lower performance.

## Expression Performance

| Expression | Accuracy |
|---|---:|
| Angry | 66.0% |
| Disgust | 73.0% |
| Fear | 51.0% |
| Happy | 86.6% |
| Neutral | 69.5% |
| Sad | 56.1% |
| Surprise | 82.8% |

The largest observed confusion was Sad → Neutral.

## Explainability

Grad-CAM is used to visualize the image regions contributing to
the model's prediction.

This helps inspect whether predictions are based on meaningful
facial regions and investigate incorrect predictions.

## Human Review

A proposed review rule flags predictions when:

- confidence < 0.60
- brightness < 70
- sharpness < 50

This is intended as a demonstration routing rule rather than an
optimized production threshold.

## Streamlit Demo

The application allows a user to upload an image and view:

- predicted expression
- prediction confidence
- image brightness
- image contrast
- image sharpness
- Grad-CAM visualization
- human-review recommendation

## Limitations

FER2013 does not provide reliable demographic attributes, so this
project focuses on condition-based reliability and robustness rather
than demographic fairness.

Facial-expression classification should not be interpreted as a
definitive measurement of a person's actual emotional or psychological
state.

## Project Structure

```text
fer2013-explainable-cnn/
├── app.py
├── README.md
├── requirements.txt
├── src/
├── models/
└── outputs/
