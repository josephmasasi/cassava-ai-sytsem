import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

from predictor import load_model


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Cassava AI Disease Detector",
    page_icon="🌿",
    layout="centered"
)


# ==============================
# LOAD CASSAVA MODEL
# ==============================

@st.cache_resource
def get_cassava_model():

    model = load_model()
    model.eval()

    return model


cassava_model = get_cassava_model()



# ==============================
# LOAD LEAF VERIFICATION MODEL
# ==============================

@st.cache_resource
def get_leaf_model():

    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.eval()

    return model


leaf_model = get_leaf_model()



# Load ImageNet labels

with open("imagenet_classes.txt") as f:

    imagenet_classes = [
        line.strip()
        for line in f.readlines()
    ]



# ==============================
# TRANSFORMATIONS
# ==============================

cassava_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


leaf_transform = ResNet18_Weights.DEFAULT.transforms()



# ==============================
# LEAF VERIFICATION FUNCTION
# ==============================

def verify_leaf(image):

    image_tensor = leaf_transform(image)

    image_tensor = image_tensor.unsqueeze(0)


    with torch.no_grad():

        output = leaf_model(image_tensor)


    probabilities = torch.softmax(
        output[0],
        dim=0
    )


    top5 = torch.topk(
        probabilities,
        5
    )


    leaf_keywords = [

        "leaf",
        "plant",
        "tree",
        "flower",
        "grass",
        "vegetable",
        "cabbage",
        "lettuce",
        "spinach",
        "vine"

    ]


    highest_leaf_confidence = 0


    for index, probability in zip(
        top5.indices,
        top5.values
    ):

        label = imagenet_classes[index].lower()


        if any(
            word in label
            for word in leaf_keywords
        ):

            highest_leaf_confidence = (
                probability.item()*100
            )

            return True, highest_leaf_confidence



    return False, 5.0



# ==============================
# DISEASE CLASSES
# ==============================

classes = [

"Cassava Bacterial Blight (CBB)",

"Cassava Brown Streak Disease (CBSD)",

"Cassava Green Mite (CGM)",

"Cassava Mosaic Disease (CMD)",

"Healthy Leaf"

]



# ==============================
# STREAMLIT INTERFACE
# ==============================


st.title(
    "🌿 Cassava Disease Detection AI"
)


st.write(
"""
Upload a cassava leaf image.
"""
)



uploaded_file = st.file_uploader(

    "Upload image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)



if uploaded_file:


    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Show only original image

    st.image(

        image,

        caption="Uploaded Image",

        use_container_width=True

    )



    # First stage: leaf verification

    is_leaf, leaf_confidence = verify_leaf(image)



    if not is_leaf:


        st.error(
        " This is not a cassava leaf."
        )


        st.warning(
        "Leaf verification confidence: 5.00%"
        )


        st.stop()



    else:


        st.success(
        "Starting disease analysis..."
        )



    # ==============================
    # CASSAVA DISEASE PREDICTION
    # ==============================


    img = cassava_transform(image)

    img = img.unsqueeze(0)



    with torch.no_grad():


        output = cassava_model(img)


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )



    disease = classes[
        predicted.item()
    ]


    disease_confidence = (
        confidence.item()*100
    )



    st.subheader(
        "Prediction Result"
    )


    st.success(
        f"Disease: {disease}"
    )


    st.info(
        f"Disease Confidence: {disease_confidence:.2f}%"
    )