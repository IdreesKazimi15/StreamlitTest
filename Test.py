import streamlit as st
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = load_digits()

# Preprocessing
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=2018)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Instantiate and train the KNN classifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_std, Y_train)

# Load the uploaded image
uploaded_file = st.file_uploader("Upload an image of a digit (jpg, jpeg, or png)", type=["jpg", "jpeg", "png"])

# Display instructions
st.write("Please upload an image containing a single digit (0-9).")

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize the image to 8x8
    resized_img = img.resize((8, 8))

    # Convert the image to grayscale
    img_gray = resized_img.convert('L')

    # Convert the PIL Image object to a NumPy array
    img_array = np.array(img_gray)

    # Flatten the image array
    flat_img = img_array.flatten()

    # Display a placeholder while the prediction is being calculated
    with st.spinner('Predicting...'):
        # Predict the digit using the KNN classifier
        prediction = knn_clf.predict([flat_img])[0]  # Get the prediction

    # Display the prediction
    st.success(f"Prediction: {prediction}")

# Add a footer with attribution
st.sidebar.markdown("---")
st.sidebar.markdown("DIGIT RECOGNITION APP")

# Add a footer with attribution
st.markdown("---")
st.markdown("DIGIT RECOGNITION APP")