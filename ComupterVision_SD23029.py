import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ---------------------------------------------------------
# Step 1: Configure Streamlit Page
# ---------------------------------------------------------
st.set_page_config(
    page_title="{ @˘̩̩̩ꈊ˘̩̩̩@ } Here is Lab 5",
    layout="centered"
)

st.title("( Φ ω Φ )Here is all about AI Computer Vision")
st.markdown("""
**Course:** BSD3513 - Introduction to AI ʕ๑•ɷ•๑｀ʔ 
**Lab Report:** Classify images using a pre-trained ResNet18 model (CPU-based).
""")

# ---------------------------------------------------------
# Step 2: Import Libraries (Done above)
# ---------------------------------------------------------

# ---------------------------------------------------------
# Step 3: Configure CPU Settings
# ---------------------------------------------------------
# Explicitly forcing CPU as required by the lab question
device = torch.device("cpu")
st.info(f"✅ System Configuration: Running on **{device}**")

# ---------------------------------------------------------
# Step 4: Load Pre-trained ResNet18 Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Load weights (the "brain" of the model)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Set to evaluation mode (essential for inference)
    model.eval()
    model.to(device)
    
    # Get the specific transforms required by this model
    preprocess_func = weights.transforms()
    
    return model, preprocess_func, weights.meta["categories"]

model, preprocess, class_names = load_model()

# ---------------------------------------------------------
# Step 6: User Interface for Image Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Step 6: Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------------------------------------------------
    # Step 5 & 7: Preprocessing & Inference
    # ---------------------------------------------------------
    st.write("🔍 **Analyzing image...**")

    # Transform image to tensor and add batch dimension (unsqueeze)
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    # Perform inference without calculating gradients (saves memory)
    with torch.no_grad():
        output = model(img_tensor)

    # ---------------------------------------------------------
    # Step 8: Apply Softmax & Get Top-5
    # ---------------------------------------------------------
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Prepare data for display
    results = []
    for i in range(5):
        class_name = class_names[top5_catid[i]]
        score = float(top5_prob[i])
        results.append({"Class": class_name, "Probability": score})

    df = pd.DataFrame(results)

    # Display Table
    st.subheader("Step 8: Top-5 Predictions")
    st.table(df)

    # ---------------------------------------------------------
    # Step 9: Visualization (Bar Chart)
    # ---------------------------------------------------------
    st.subheader("Step 9: Confidence Visualization")
    st.bar_chart(df.set_index("Class"))

    # ---------------------------------------------------------
    # Step 10: Discussion of Results
    # ---------------------------------------------------------
    st.subheader("Step 10: Discussion")
    
    top_class = results[0]['Class']
    top_score = results[0]['Probability']
    
    if top_score > 0.8:
        confidence_desc = "high confidence"
    elif top_score > 0.5:
        confidence_desc = "moderate confidence"
    else:
        confidence_desc = "low confidence (uncertain)"

    discussion_text = f"""
    Based on the analysis, the model predicts this image is a **{top_class}** with **{confidence_desc}** ({top_score:.2%}). 
    The bar chart above visualizes the probability distribution produced by the Softmax function. 
    Since we are running on **CPU**, the inference time depends on the local processor speed, but ResNet18 is lightweight enough for real-time usage here.
    """
    st.success(discussion_text)

else:
    st.info("Please upload an image to start the computer vision analysis.")