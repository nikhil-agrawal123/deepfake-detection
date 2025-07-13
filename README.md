# 🔍 Deepfake Image Detection with SqueezeNet

This project uses **PyTorch** and **SqueezeNet** to train a deepfake image classifier. It distinguishes between real and fake (deepfake) face images using transfer learning and mixed-precision training for efficiency.

---

## 📁 Project Structure

```
.
├── data/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── val/
│       ├── real/
│       └── fake/
├── squeezenet_model.pth        # Saved trained model
├── main.py                     # Main training & inference script
└── README.md
```

---

## 🚀 Features

- Uses **SqueezeNet** for lightweight performance
- **Transfer learning** with pre-trained weights
- **Mixed Precision Training** via `torch.amp` for faster computation
- Modular structure for easy use in training and inference
- Automatically saves the best-performing model based on validation accuracy

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- tqdm
- PIL

Install dependencies:
```bash
pip install torch torchvision tqdm pillow
```

---

## 🧠 Dataset Format

Organize the dataset in this format:

```
data/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

Each folder should contain `.jpg` or `.png` image files.

---

## 🏋️‍♂️ Training

To train the model:

```bash
python main.py
```

- The best model is saved as `squeezenet_model.pth`
- Default settings:
  - Epochs: 25
  - Batch size: 32
  - Optimizer: Adam
  - Learning rate: 1e-5

---

## 🔍 Inference

Update the image path in `main.py`:

```python
image_path = 'data/val/real/real_1.jpg'
```

After training, the model will predict the class of this image and print it:

```bash
Prediction: real
```

---

## ⚙️ Customization

You can modify:
- The model architecture (e.g., ResNet, MobileNet)
- Image size or augmentations
- Data directories and batch sizes
- Learning rate and optimizer in `main.py`

---

## 💡 Notes

- Uses **SqueezeNet 1.0**, a compact CNN architecture (~5MB)
- Fine-tuned last layer (`Conv2d`) for binary classification
- Uses **AMP (Automatic Mixed Precision)** for performance on GPU

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
