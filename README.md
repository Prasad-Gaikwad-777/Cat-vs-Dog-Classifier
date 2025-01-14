# 🐱 Cat vs Dog Classifier 🐶

A deep learning project that classifies images into two categories: **Cats** and **Dogs**. This project uses a **Convolutional Neural Network (CNN)** built with TensorFlow and deployed using **Streamlit**.

---

## 📂 Dataset

**Source:** [Dogs vs Cats Dataset on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats/data)

### 📊 Dataset Structure:
- **Folders:**
  - `train`: Contains training images.
  - `test`: Contains testing images.
- **Subfolders in each:**
  - `cats`: Images of cats.
  - `dogs`: Images of dogs.

### 🖼️ Image Counts:
- **Training Set**:
  - `cats`: 10,000 images
  - `dogs`: 10,000 images
- **Testing Set**:
  - `cats`: 2,500 images
  - `dogs`: 2,500 images

---

## 🚀 Features

- **Model**: A CNN trained to classify cats and dogs.
- **Framework**: Built using TensorFlow.
- **Deployment**: Interactive web application using Streamlit.
- **Code**: Includes a Jupyter Notebook for training and app code for deployment.
- **Requirements**: A `requirements.txt` file is provided to set up the environment.

---

## ⚙️ How to Use

### Prerequisites
- Python 3.10 or above
- Required Python libraries listed in `requirements.txt`

### 📊 Download the Dataset
- Download the [Dogs vs Cats Dataset on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats/data) from **kaggle**
- Place the dataset in the appropriate folder structure (`train` and `test`).

### Train the Model
- Open `cat_vs_dog_classifier.ipynb` in Jupyter Notebook or Google Colab.
- Follow the steps to preprocess data, build the model, and train it.

### Run the Streamlit App
- Add the `model path` of yours in the file
- Open and Run the `App.py` file

### Open the URL provided by Streamlit to access the app.
-  use command `streamlit run app.py` in terminal
-  and it will open browser in localhost


---

## 📝 Project Workflow 


### Data Preprocessing:
- Images resized to 256x256.
- Normalized pixel values to a range of 0-1.

### Model Architecture:
- Convolutional layers with max-pooling.
- Fully connected dense layers.
- Binary classification using softmax activation.

### Training:
- Trained on 20,000 images (10,000 cats and 10,000 dogs).
- Validation performed on 5,000 images (2,500 cats and 2,500 dogs).

### Deployment:
- Streamlit app accepts user-uploaded images and predicts whether it’s a cat or a dog.

---

## 🌟 Example Usage

### Upload an Image:
- Drag and drop an image of a cat or dog in the Streamlit app.
- Get Predictions:

### Example Output: "Result: Dog!"

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow/Keras:** For building the CNN model.
- **Streamlit:** For deployment as a web app.
- **OpenCV & Pillow:** For image preprocessing.

---

## 🙏 Acknowledgments

- ***Kaggle*** for providing the Dogs vs Cats Dataset.
- ***TensorFlow/Keras*** for their robust machine learning framework.
- ***Streamlit** for its simplicity in creating interactive applications.

---

![streamlit app interface](https://github.com/user-attachments/assets/792e8080-c70d-45d7-b9ee-ffb47a7649ed)

---
---

![Streamlit app output](https://github.com/user-attachments/assets/83d38f93-21f0-4b8b-8518-a5493be937b8)



# 💡 Feel free to fork and star ⭐, Let’s Connect and Build Amazing Projects Together!
