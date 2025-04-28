import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv
from tensorflow.keras.applications import VGG16, ResNet101

# Add this near the top of your file, after the imports
def initialize_session_state():
    if 'model_predictions' not in st.session_state:
        st.session_state.model_predictions = {
            'Transfer Learning - Xception': None,
            'Custom CNN': None,
            'VGG16': None,
            'ResNet101': None
        }
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

# Function to create a radar chart comparing vital parameters
def create_radar_chart(vital_params_normal, vital_params_tumorous, tumor_type):
    categories = list(vital_params_normal.keys())
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(vital_params_normal.values()),
        theta=categories,
        fill='toself',
        name='Normal Individual',
        line=dict(color='blue', width=2),
        marker=dict(size=8, symbol='circle')
    ))

    # Set color based on tumor type
    color_map = {
        'Glioma': 'red',
        'Meningioma': 'pink',
        'Pituitary': 'purple',
        'No Tumor': 'green'
    }
    tumor_color = color_map.get(tumor_type, 'red')  # Default to red if not found

    fig.add_trace(go.Scatterpolar(
        r=list(vital_params_tumorous.values()),
        theta=categories,
        fill='toself',
        name=f'{tumor_type} Tumorous Individual',
        line=dict(color=tumor_color, width=2),
        marker=dict(size=8, symbol='circle')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(vital_params_normal.values()), max(vital_params_tumorous.values())) + 10],
                showline=True,
                linewidth=2,
                gridcolor='gray',
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                linewidth=2,
                gridcolor='gray',
                gridwidth=1
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.8,
            y=1.1,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='black',
            borderwidth=1
        ),
        title=dict(
            text='Comparison of Vital Parameters',
            font=dict(size=16)
        )
    )

    return fig

# Streamlit button to show health analysis
def show_health_analysis_button(model_prediction):
    if st.button('Show Health Analysis'):
        # Example vital parameters for demonstration
        vital_params_normal = {
            'Heart Rate': 70, 
            'Temperature': 98.6, 
            'Blood Pressure': 120,
            'Respiratory Rate': 16,
            'Oxygen Saturation': 98
        }
        
        # Dynamic vital parameters based on tumor type
        vital_params_tumorous = {
            'Glioma': {
                'Heart Rate': 85, 
                'Temperature': 99.1, 
                'Blood Pressure': 130,
                'Respiratory Rate': 20,
                'Oxygen Saturation': 95
            },
            'Meningioma': {
                'Heart Rate': 80, 
                'Temperature': 98.9, 
                'Blood Pressure': 125,
                'Respiratory Rate': 18,
                'Oxygen Saturation': 96
            },
            'Pituitary': {
                'Heart Rate': 78, 
                'Temperature': 98.7, 
                'Blood Pressure': 122,
                'Respiratory Rate': 17,
                'Oxygen Saturation': 97
            },
            'No Tumor': {
                'Heart Rate': 70, 
                'Temperature': 98.6, 
                'Blood Pressure': 120,
                'Respiratory Rate': 16,
                'Oxygen Saturation': 98
            }
        }

        radar_chart = create_radar_chart(vital_params_normal, vital_params_tumorous[model_prediction], model_prediction)
        st.plotly_chart(radar_chart)

        # Display vital parameters alongside the radar chart
        st.write("### Vital Parameters Comparison")
        
        # Create a table for vital parameters
        st.write("#### Vital Parameters Table")
        st.table({
            "Parameter": list(vital_params_normal.keys()),
            "Normal Individual": list(vital_params_normal.values()),
            f"{model_prediction} Tumorous Individual": list(vital_params_tumorous[model_prediction].values())
        })

# Streamlit button to show model comparison

def show_model_comparison_button():
    if st.button('Show Model Comparison'):
        valid_predictions = {
            model: accuracy 
            for model, accuracy in st.session_state.model_predictions.items() 
            if accuracy is not None
        }
        
        if not valid_predictions:
            st.warning("No models have been used for prediction yet.")
            return
            
        # Create a bar chart for model comparison
        fig = go.Figure()
        
        for model, accuracy in valid_predictions.items():
            fig.add_trace(go.Bar(
                name=model,
                x=[model],
                y=[accuracy],
                marker_color='blue' if 'Xception' in model else 'orange' if 'Custom CNN' in model else 'green' if 'VGG16' in model else 'purple' if 'ResNet101' in model else 'red'
            ))
        
        fig.update_layout(
            title='Model Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Confidence (%)',
            barmode='group',
            showlegend=True,
            width=600,
            height=400
        )
        
        st.plotly_chart(fig)
        
        # Add a comparison table
        st.write("### Model Comparison Table")
        comparison_data = {
            "Model": list(valid_predictions.keys()),
            "Confidence (%)": [f"{acc:.2f}%" for acc in valid_predictions.values()]
        }
        st.table(comparison_data)


load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

output_dir = 'saliency_maps'
os.makedirs(output_dir , exist_ok=True)


def generate_explanation(img_path , model_prediction, confidence):

  prompt = f"""You are an expert neurologist. You are tasked with explaining a saliency map of a brain tumor MRI scan.
    The saliency map was generated by a deep learning model that was trained to classify brain tumors as either glioma, meningioma, pituitary, or no tumor.
    The deep learning model predicted the image to be of class '{model_prediction}' with a confidence of {confidence * 100}%.

    In your response:
    - Start with "This MRI Scan shows that these person is diagnosed from {model_prediction} due to ...." explain in scientific terms and what it e
    - Explain what regions of the brain the model is focusing on so that you know what it focused on to predict these
      Refer to the regions highlighted in light cyan, those are the regions where the model is focusing on.
    - Explain possible reasons why the model made the prediction it did
    - DO NOT mention "This model focusess on ..." or anything about the model
    - Talk like a neuro scientist using scientific terms
    - Keep your explanations to 5 sentence maximum

    Let's think step by step about this. Verify step by step.

  """

  img = PIL.Image.open(img_path)

  model = genai.GenerativeModel(model_name="gemini-1.5-flash")
  response = model.generate_content([prompt, img])

  return response.text


def generate_saliency_map(model, img_array, class_index, img_size):
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy().squeeze()

    # Resize gradients to match original image size
    gradients = cv2.resize(gradients, img_size)

    # Create a circular mask for the brain area
    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

    # Apply mask to gradients
    gradients = gradients * mask

    # Normalize only the brain area
    brain_gradients = gradients[mask]
    if brain_gradients.max() > brain_gradients.min():
        brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
    gradients[mask] = brain_gradients

    # Apply a higher threshold
    threshold = np.percentile(gradients[mask], 80)
    gradients[gradients < threshold] = 0

    # Apply more aggressive smoothing
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

    # Create a heatmap overlay with enhanced contrast
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, img_size)

    # Superimpose the heatmap on original image with increased opacity
    original_img = image.img_to_array(img)
    superimposed_img = heatmap * 0.7 + original_img * 0.3
    superimposed_img = superimposed_img.astype(np.uint8)

    img_path = os.path.join(output_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    saliency_map_path = f'saliency_maps/{uploaded_file.name}'

    # Save the saliency map
    cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return superimposed_img




# Load VGG16 Model
def load_vgg16_model(model_path):
    return load_model(model_path)

# Load ResNet101 Model
def load_resnet_model(model_path):
    return load_model(model_path)

def load_xception_model(model_path):
  img_shape = (299,299,3)

  base_model = tf.keras.applications.Xception(
      include_top = False,
      weights = 'imagenet',
      input_shape = img_shape,
      pooling = 'max'
  )

  model = Sequential([
      base_model,
      Flatten(),
      Dropout(rate = 0.3),
      Dense(128, activation = 'relu'),
      Dropout(rate = 0.25),
      Dense(4, activation = 'softmax')
  ])

  model.build((None,)+img_shape)

  model.compile(Adamax(learning_rate= 0.001),
              loss = 'categorical_crossentropy',
              metrics = [
                  'accuracy',
                  Precision(),
                  Recall()
              ])
  model.load_weights(model_path)

  return model


st.title('Brain Tumor Classification')
st.write('Upload an MRI scan to classify')

uploaded_file = st.file_uploader("Choose an Image....." , type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Initialize session state
    initialize_session_state()
    
    # Check if a new image was uploaded
    if st.session_state.current_image != uploaded_file.name:
        st.session_state.current_image = uploaded_file.name
        st.session_state.model_predictions = {
            'Transfer Learning - Xception': None,
            'Custom CNN': None,
            'VGG16': None,
            'ResNet101': None
        }
    
    selected_model = st.radio(
        "Select Model",
        ("Transfer Learning - Xception", "Custom CNN" , "VGG16", "ResNet101")
    )
    
    if selected_model == "Transfer Learning - Xception":
        model = load_xception_model('trained_xception_model.weights.h5')
        img_size = (299, 299)
    elif selected_model == "Custom CNN":
        model = load_model('cnn_model.h5')
        img_size = (150, 150)
    elif selected_model == "VGG16":
        model = load_vgg16_model('vgg16_model.h5')
        img_size = (150, 150)  # Make sure your model was trained with this size
    else:  # ResNet101
        model = load_resnet_model('resnet101_model.h5')
        img_size = (150, 150)  # Make sure your model was trained with this size
    
    labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    result = labels[class_index]

    # Store the prediction in session state
    st.session_state.model_predictions[selected_model] = prediction[0][class_index] * 100

    st.write(f'Predicted Class: {result}')
    st.write('Predictions: ')
    for label, prob in zip(labels, prediction[0]):
        st.write(f"{label}: {prob*100:.2f}%")

    saliency_map = generate_saliency_map(model, img_array, class_index, img_size)

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(saliency_map, caption="Saliency Map", use_container_width=True)

    st.write("## Classification Results")

    result_container = st.container()
    result_container.markdown(
        f"""
        Prediction
            {result}

        Confidence
            {prediction[0][class_index]:.4%}
        """,
        unsafe_allow_html=True
    )

    probabilities = prediction[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probabilities = probabilities[sorted_indices]

    fig = go.Figure(go.Bar(
        x=sorted_probabilities,
        y=sorted_labels,
        orientation='h',
        marker_color=['red' if label == result else 'blue' for label in sorted_labels]
    ))

    fig.update_layout(
        title='Probabilities for each class',
        xaxis_title='Probability',
        yaxis_title='Class',
        height=400,
        width=600,
        yaxis=dict(autorange="reversed")
    )

    for i, prob in enumerate(sorted_probabilities):
        fig.add_annotation(
            x=prob,
            y=i,
            text=f'{prob:.4f}',
            showarrow=False,
            xanchor='left',
            xshift=5
        )

    st.plotly_chart(fig)
    
    show_health_analysis_button(result)
    
    # Show model comparison
    show_model_comparison_button()

    saliency_map_path = f'saliency_maps/{uploaded_file.name}'
    explanation = generate_explanation(saliency_map_path, result, prediction[0][class_index])

    st.write("## Explanation")
    st.write(explanation)