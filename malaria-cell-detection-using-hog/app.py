import streamlit as st
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
import os
import base64
import shap
from lime.lime_tabular import LimeTabularExplainer
import io

# Page configuration
st.set_page_config(
    page_title="Malaria Cell Detection",
    page_icon="🔬",
    layout="wide"
)

# Function to download example images
def get_binary_file_downloader_html(bin_file, file_label):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

# Load the model and label encoder
@st.cache_resource
def load_model():
    model_path = 'malaria_svm_hog_model.joblib'
    le_path = 'label_encoder.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        st.error("Model files not found. Please run the notebook to export the model first.")
        st.stop()
        
    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
    return model, label_encoder

# Extract HOG features from image
def extract_hog_features(image):
    IMG_DIMS = (128, 64)  # Height x Width format for skimage
    image = resize(image, IMG_DIMS)
    features, hog_image = hog(image, orientations=9, 
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), 
                              visualize=True, 
                              channel_axis=-1)
    return features, hog_image

# Cache image decoding and HOG to reduce rerender flicker
@st.cache_data(show_spinner=False)
def process_image_bytes(file_bytes: bytes):
    img = imread(io.BytesIO(file_bytes))
    feats, hog_img = extract_hog_features(img)
    return img, feats, hog_img

def normalize_to_uint8(image_like: np.ndarray) -> np.ndarray:
    """Normalize array to 0-255 uint8 for stable st.image rendering."""
    arr = np.asarray(image_like).astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    denom = (max_v - min_v) if (max_v - min_v) > 1e-8 else 1.0
    arr = (arr - min_v) / denom
    return (arr * 255.0).astype(np.uint8)

# Background feature generator for explainability methods
def build_background_from_features(instance_features: np.ndarray, num_samples: int = 50, noise_scale: float = 0.01) -> np.ndarray:
    """Create a lightweight background distribution around the instance.

    Kernel SHAP and LIME both need a reference dataset. Since we don't ship the
    full training set with the app, we synthesize a small local neighborhood
    around the current image's HOG features by adding low-variance Gaussian noise.
    """
    rng = np.random.default_rng(42)
    tiled = np.tile(instance_features, (num_samples, 1))
    noise = rng.normal(loc=0.0, scale=noise_scale, size=tiled.shape)
    background = tiled + noise
    return background.astype(np.float64)

# Probability wrapper built on top of SVM decision function
def predict_proba_from_features(model, X: np.ndarray) -> np.ndarray:
    """Return pseudo-probabilities for LIME using sigmoid(decision_function)."""
    raw = model.decision_function(X)
    # Ensure 2D
    raw = np.array(raw).reshape(-1)
    p1 = 1.0 / (1.0 + np.exp(-raw))
    p0 = 1.0 - p1
    return np.vstack([p0, p1]).T

# Main function
def main():
    # Load model
    model, label_encoder = load_model()
    
    # Global style tweaks
    st.markdown(
        """
        <style>
            :root {
                --brand: #5b8def;
                --accent: #7bdcb5;
                --text-strong: #111827;
                --muted: #6b7280;
            }
            .main > div { padding-top: 0.5rem; }
            h1, h2, h3, h4 { color: var(--text-strong); }
            .stAlert { border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
            .block-container { padding-top: 1rem; padding-bottom: 2.5rem; }
            .section-title { margin: 0 0 6px 0; font-weight: 700; }
            .footer-card { background: linear-gradient(135deg, rgba(91,141,239,0.08), rgba(123,220,181,0.08)); border: 1px solid rgba(0,0,0,0.06); border-radius: 14px; padding: 18px 20px; text-align: center; }
            .footer-title { font-weight: 600; font-size: 16px; margin: 0 0 8px 0; }
            .footer-sub { font-size: 13px; color: var(--muted); margin: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar with key info
    with st.sidebar:
        st.title("🔬 Malaria Cell Detection")
        st.markdown("Model: SVM (RBF) on HOG features")
        st.markdown("Made for education. Not for clinical use.")
        st.divider()
        st.markdown("### Quick Tips")
        st.markdown("- Upload a cell image below to get a prediction\n- Open the SHAP/LIME tabs in results for explanations")
    
    # Model information
    st.info("""
    ### Model Performance
    - **Training Accuracy**: 92.2%
    - **Validation Accuracy**: 80.4%
    
    ⚠️ **Disclaimer**: This model is for educational purposes and experimentation only. Please use with caution and do not use for actual medical diagnosis.
    """)
    
    # Documentation section
    with st.expander("📚 Documentation"):
        st.markdown("""
        ## Problem Statement
        
        Malaria remains one of the world's deadliest infectious diseases, affecting millions of people worldwide. Early and accurate diagnosis is crucial for effective treatment and reducing mortality rates. Traditional diagnosis involves manual microscopic examination of blood smears by trained professionals, which is time-consuming, requires expertise, and can be subject to human error.
        
        ## Project Approach
        
        This project uses machine learning to automate the detection of malaria parasites in blood smear images:
        
        1. **Data Collection**: Used a dataset of labeled blood cell images (parasitized and uninfected)
        2. **Feature Extraction**: Applied Histogram of Oriented Gradients (HOG) to extract meaningful features from the cell images
        3. **Model Training**: Trained a Support Vector Machine (SVM) classifier on the extracted features
        4. **Evaluation**: Validated the model on a separate test set to ensure generalizability
        
        ## How It Works
        
        The application follows these steps to analyze a blood cell image:
        
        1. **Image Upload**: User uploads a microscopic blood cell image
        2. **Preprocessing**: The image is resized and normalized
        3. **Feature Extraction**: HOG features are extracted to capture the cell's morphological characteristics
        4. **Classification**: The SVM model predicts whether the cell is infected with malaria parasites
        5. **Result Display**: The prediction result and confidence score are displayed along with visual aids
        
        ## Technical Details
        
        - **Feature Extraction**: HOG with 9 orientations, 8×8 pixels per cell, and 2×2 cells per block
        - **Classifier**: SVM with RBF kernel
        - **Image Size**: Processed at 128×64 pixels
        """)
    
    # Example Images (no tabs)
    st.markdown("### Example Images")
    example_folder = "examples"
    if os.path.exists(example_folder):
        st.success("Example images are available in the examples folder. You can use them to test the model.")
        col_examples1, col_examples2 = st.columns(2)

        with col_examples1:
            st.markdown("#### Parasitized (Malaria) Examples:")
            malaria_files = [
                "C50P11thinF_IMG_20150724_114951_cell_148.png",
                "C59P20thinF_IMG_20150803_111333_cell_144.png"
            ]
            for file in malaria_files:
                file_path = os.path.join(example_folder, file)
                if os.path.exists(file_path):
                    try:
                        img = imread(file_path)
                        st.image(img, caption=file, width=250)
                        st.markdown(get_binary_file_downloader_html(file_path, f"Download {file}"), unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not display {file}: {e}")

        with col_examples2:
            st.markdown("#### Uninfected (Healthy) Examples:")
            healthy_files = [
                "C112P73ThinF_IMG_20150930_131659_cell_94.png",
                "C130P91ThinF_IMG_20151004_142709_cell_110.png"
            ]
            for file in healthy_files:
                file_path = os.path.join(example_folder, file)
                if os.path.exists(file_path):
                    try:
                        img = imread(file_path)
                        st.image(img, caption=file, width=250)
                        st.markdown(get_binary_file_downloader_html(file_path, f"Download {file}"), unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not display {file}: {e}")
    else:
        st.warning("Examples folder not found. Add an 'examples' folder with a few sample images.")
    
    # Upload section directly on the page (no tabs)
    st.markdown("### Upload Blood Cell Image")
    st.write("Upload an image of a blood cell to detect whether it's infected with malaria parasites.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Initialize prediction variables
    prediction_result = None
    confidence_score = None
    hog_img = None
    original_image = None
    
    # Process the uploaded image
    if uploaded_file is not None:
        try:
            # Read and process using cache to avoid flicker on reruns
            file_bytes = uploaded_file.getvalue()
            original_image, features, hog_img = process_image_bytes(file_bytes)
            
            # Make prediction
            prediction = model.predict([features])[0]
            prediction_proba = model.decision_function([features])[0]
            confidence_score = abs(prediction_proba)
            prediction_result = label_encoder.inverse_transform([prediction])[0]
            
            # Horizontal separator
            st.markdown("---")
            
            # Results section
            st.header("Prediction Results")
            
            # Create columns for results display
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("### Uploaded Image")
                st.image(original_image, caption="Original Blood Cell Image", width=300)
                
                # Image summary
                st.markdown("### Image Analysis")
                st.markdown(f"""
                - **Image Size**: {original_image.shape[0]} × {original_image.shape[1]} pixels
                - **Color Channels**: {original_image.shape[2] if len(original_image.shape) > 2 else "Grayscale"}
                - **Processing**: HOG feature extraction with 9 orientations
                """)
            
            with res_col2:
                # Display prediction outcome with styling
                st.markdown("### Detection Result")
                
                if prediction_result.lower() in ["parasitized", "malaria"]:
                    st.error(f"## Prediction: {prediction_result}")
                    st.markdown("⚠️ **Cell appears to be infected with malaria parasites**")
                else:
                    st.success(f"## Prediction: {prediction_result}")
                    st.markdown("✅ **Cell appears to be uninfected**")
                # Confidence score
                st.metric("Confidence Score", f"{confidence_score:.2f}")
                
                # HOG Feature Visualization (stable image render)
                st.markdown("### HOG Feature Visualization")
                hog_disp = normalize_to_uint8(hog_img)
                st.session_state["hog_disp"] = hog_disp
                st.image(st.session_state["hog_disp"], caption="HOG Features", use_column_width=True, clamp=True)

            # Explainability sections (no tabs)
            st.markdown("---")
            st.subheader("SHAP Explanation (Kernel SHAP on HOG features)")
            try:
                background = build_background_from_features(features, num_samples=80, noise_scale=0.02)
                # Use faster linear explainer (model is LinearSVC per artifacts)
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(features.reshape(1, -1))
                # For binary decision_function, shap returns a 1D array
                if isinstance(shap_values, list):
                    shap_1d = np.array(shap_values).reshape(-1)
                else:
                    shap_1d = np.array(shap_values).reshape(-1)
                top_k = 20
                top_idx = np.argsort(np.abs(shap_1d))[-top_k:][::-1]
                fig_shap, ax_shap = plt.subplots(figsize=(7, 4))
                ax_shap.bar(range(len(top_idx)), shap_1d[top_idx], color=["#d62728" if v < 0 else "#2ca02c" for v in shap_1d[top_idx]])
                ax_shap.set_xticks(range(len(top_idx)))
                ax_shap.set_xticklabels([f"f{int(i)}" for i in top_idx], rotation=45, ha="right")
                ax_shap.set_ylabel("SHAP value")
                ax_shap.set_title("Top HOG features driving the decision")
                st.pyplot(fig_shap)
                plt.close(fig_shap)
                with st.expander("What does SHAP mean?"):
                    st.markdown("""
                    SHAP shows how each input feature pushes the model toward malaria (positive) or healthy (negative). Bars to the right (green) increase the malaria score; bars to the left (red) decrease it. Length = strength of influence.
                    """)
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

            st.subheader("LIME Explanation (on HOG features)")
            try:
                background = build_background_from_features(features, num_samples=200, noise_scale=0.03)
                feature_names = [f"hog_{i}" for i in range(features.shape[0])]
                class_names = list(label_encoder.classes_)
                explainer = LimeTabularExplainer(
                    training_data=background,
                    feature_names=feature_names,
                    class_names=class_names,
                    mode="classification",
                    discretize_continuous=True,
                    random_state=42,
                )
                exp = explainer.explain_instance(
                    data_row=features,
                    predict_fn=lambda x: predict_proba_from_features(model, x),
                    num_features=15,
                    num_samples=1000,
                )
                fig_lime = exp.as_pyplot_figure()
                st.pyplot(fig_lime)
                plt.close(fig_lime)
            except Exception as e:
                st.warning(f"LIME explanation unavailable: {e}")
                
            # Technical explanation
            st.markdown("---")
            st.markdown("""
            ### Understanding the Results
            
            - **HOG Features**: The visualization shows gradient directions that help the model identify the presence of malaria parasites
            - **Confidence Score**: Higher values indicate stronger model certainty
            - **Explainability**: SHAP estimates feature attributions using a game-theoretic approach; LIME approximates the model locally with a simple surrogate
            - **Limitations**: This model has approximately 80% accuracy on validation data and should not be used for clinical diagnosis
            """)
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    # Footer
    footer = st.container()
    with footer:
        st.markdown("---")
        st.markdown("""
        ### Educational Purpose Only
        
        This application is designed for educational and experimental purposes only. It demonstrates the potential of machine learning in medical image analysis but is not intended for clinical use.
        
        **Important Notice**: Do not use this tool for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.
        """)
        st.markdown("""
        <div class="footer-card">
            <p class="footer-title">Malaria Detection Model · Educational Use</p>
            <p class="footer-sub">Designed & Developed by <b>Aradhya Pavan</b> · SVM with HOG Features</p>
        </div>
        """, unsafe_allow_html=True)


    

if __name__ == "__main__":
    main()