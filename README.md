
# 🧠 Breast Cancer Prediction using EfficientNetB0 U-Net

This project uses a **U-Net architecture** with **EfficientNetB0** as the encoder backbone to segment and predict breast cancer regions from ultrasound images. The goal is to perform **semantic segmentation** to highlight tumor areas using medical imaging data.

---

## 📂 Project Structure

breast\_cancer\_gub/
│
├── main.py                # Main script: model building, training & visualization
├── data/                  # Folder containing dataset (organized into class folders)
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── unet\_model.keras       # Saved trained model (optional if exported)
├── README.md              # This file
└── .gitignore             # Ignored files (optional)


---

## 📊 Dataset

This project uses the **BUSI dataset** (Breast Ultrasound Images), which contains:

- **3 categories**: benign, malignant, and normal
- Each image has a corresponding **ground truth mask**

> 📁 Folder structure should be:
> Dataset_BUSI_with_GT/
> ├── benign/
> ├── malignant/
> └── normal/

Each category contains:
- Ultrasound images
- Mask images (with "mask" in the filename)

---

## 🧪 Key Features

- ✅ **EfficientNetB0** used as encoder
- ✅ U-Net decoder with transposed convolutions
- ✅ Mixed precision training for performance
- ✅ Early stopping for generalization
- ✅ Visualization of predicted vs actual tumor segmentation

### 2. Install Requirements

Make sure you have Python 3.7+ and install dependencies:

### 3. Place Your Dataset

Download and place the **BUSI Dataset** inside the project folder, like this:

## 📈 Training Details

* **Input Image Size**: 128x128
* **Batch Size**: 32
* **Epochs**: 50
* **Optimizer**: Adadelta
* **Loss**: Binary Crossentropy
* **Metrics**: Accuracy

---

## 📷 Sample Outputs

The script displays side-by-side:

* Original image
* Predicted mask overlay
* Actual mask overlay

## 📌 Notes

* Mixed precision training is used for faster computation (recommended if GPU is available).
* Make sure dataset folder names and paths match what is used in the code.

---

## 📚 References

* [U-Net Paper](https://arxiv.org/abs/1505.04597)
* [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
* [BUSI Dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)

---

## 🧑‍💻 Author

**Ashok Kumar Reddy P**
AI/ML enthusiast focused on medical imaging and deep learning applications.

## 🪪 License

This project is for educational and research purposes only. Dataset may require proper attribution as per its original source.

