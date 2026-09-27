import os
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Expression, classified.",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# HTML HELPER
# ============================================================

def html(content):
    """
    Render HTML directly with Streamlit's HTML renderer.
    This avoids Markdown interpreting HTML as code blocks.
    """
    st.html(content)


# ============================================================
# COLOUR PALETTE
# ============================================================

DEEP_TEAL = "#28616A"
OCEAN_TEAL = "#0B858B"
SOFT_TEAL = "#4E9DA0"
MIST_GREY = "#A8B0AF"
MAUVE = "#AAA5AD"
DUSTY_ROSE = "#D1A5A4"

BACKGROUND = "#E8EEEC"
DARK_TEXT = "#18383D"
SECONDARY_TEXT = "#526A6D"
WHITE = "#F7F9F8"


# ============================================================
# CUSTOM CSS
# ============================================================

html(f"""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap'
);


/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {{
    background-color: {BACKGROUND};
    color: {DARK_TEXT};
}}

.block-container {{
    max-width: 1120px;
    padding-top: 3.5rem;
    padding-bottom: 5rem;
}}


/* ============================================================
   TYPOGRAPHY
   ============================================================ */

p,
div,
span,
label {{
    font-family: 'DM Sans', sans-serif;
}}

h1,
h2,
h3 {{
    color: {DARK_TEXT} !important;
}}


/* ============================================================
   HEADER
   ============================================================ */

.kicker {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.67rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {OCEAN_TEAL};
    margin-bottom: 1.1rem;
}}

.hero-title {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 4rem;
    line-height: 0.98;
    letter-spacing: -0.035em;
    color: {DARK_TEXT};
    margin: 0;
}}

.hero-subtitle {{
    max-width: 720px;
    margin-top: 1.5rem;
    color: {SECONDARY_TEXT};
    font-size: 0.98rem;
    line-height: 1.75;
}}

.hero-accent {{
    width: 70px;
    height: 4px;
    background: {DUSTY_ROSE};
    margin-top: 1.8rem;
}}


/* ============================================================
   SECTION LABELS
   ============================================================ */

.section-label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {OCEAN_TEAL};
    margin-bottom: 0.7rem;
}}

.section-heading {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.85rem;
    line-height: 1.2;
    color: {DARK_TEXT};
    margin-bottom: 0.7rem;
}}

.section-description {{
    color: {SECONDARY_TEXT};
    font-size: 0.88rem;
    line-height: 1.7;
    max-width: 720px;
}}


/* ============================================================
   DIVIDERS
   ============================================================ */

.divider {{
    height: 1px;
    background: {MIST_GREY};
    opacity: 0.7;
    margin: 2.8rem 0;
}}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {{
    background: {WHITE};
    border: 1px dashed {SOFT_TEAL};
    border-radius: 0;
    padding: 0.5rem;
}}

[data-testid="stFileUploader"] section {{
    background: transparent;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: {WHITE};
}}

[data-testid="stFileUploaderDropzoneInstructions"] {{
    color: {SECONDARY_TEXT} !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: {SECONDARY_TEXT} !important;
}}

[data-testid="stFileUploader"] button {{
    background: {OCEAN_TEAL} !important;
    color: white !important;
    border: none !important;
    border-radius: 2px !important;
}}

[data-testid="stFileUploader"] button:hover {{
    background: {DEEP_TEAL} !important;
}}


/* ============================================================
   IMAGE
   ============================================================ */

[data-testid="stImage"] img {{
    border: 1px solid {MIST_GREY};
}}

.image-caption {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SECONDARY_TEXT};
    margin-top: 0.7rem;
}}


/* ============================================================
   PREDICTION
   ============================================================ */

.prediction-container {{
    background: {WHITE};
    border-left: 5px solid {OCEAN_TEAL};
    padding: 1.5rem 1.6rem;
    margin-top: 0.6rem;
}}

.prediction-label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: {SECONDARY_TEXT};
}}

.prediction-name {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3rem;
    line-height: 1;
    text-transform: capitalize;
    color: {DARK_TEXT};
    margin-top: 0.45rem;
}}

.prediction-confidence {{
    font-family: 'DM Mono', monospace !important;
    color: {OCEAN_TEAL};
    font-size: 0.78rem;
    margin-top: 0.7rem;
}}


/* ============================================================
   STATUS BOXES
   ============================================================ */

.status-good {{
    background: #E0EEEC;
    border: 1px solid {SOFT_TEAL};
    color: {DEEP_TEAL};
    padding: 1rem 1.1rem;
    font-size: 0.82rem;
    line-height: 1.55;
    margin-top: 1rem;
}}

.status-warning {{
    background: #F0E2E2;
    border: 1px solid {DUSTY_ROSE};
    color: #704D50;
    padding: 1rem 1.1rem;
    font-size: 0.82rem;
    line-height: 1.55;
    margin-top: 1rem;
}}


/* ============================================================
   METRICS
   ============================================================ */

.metric {{
    border-top: 1px solid {MIST_GREY};
    padding-top: 0.8rem;
}}

.metric-label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {SECONDARY_TEXT};
}}

.metric-value {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: {DARK_TEXT};
    margin-top: 0.3rem;
}}

.metric-note {{
    font-size: 0.72rem;
    color: {SECONDARY_TEXT};
    margin-top: 0.15rem;
}}


/* ============================================================
   GRAD-CAM
   ============================================================ */

.explanation-box {{
    background: {WHITE};
    border: 1px solid {MIST_GREY};
    padding: 0.8rem;
    margin-top: 1.2rem;
}}

.explanation-note {{
    color: {SECONDARY_TEXT};
    font-size: 0.78rem;
    line-height: 1.6;
    margin-top: 0.8rem;
}}


/* ============================================================
   MODEL INFORMATION
   ============================================================ */

.model-strip {{
    background: {DEEP_TEAL};
    color: white;
    padding: 1.4rem 1.5rem;
    min-height: 95px;
}}

.model-strip-label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #A8D4D4;
}}

.model-strip-value {{
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.35rem;
    color: white;
}}


/* ============================================================
   FOOTER
   ============================================================ */

.footer-label {{
    font-family: 'DM Mono', monospace !important;
    color: {OCEAN_TEAL};
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}

.footer-note {{
    color: {SECONDARY_TEXT};
    font-size: 0.72rem;
    line-height: 1.7;
}}


/* ============================================================
   STREAMLIT CLEANUP
   ============================================================ */

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    background: {BACKGROUND} !important;
}}


/* ============================================================
   SCROLLBAR
   ============================================================ */

::-webkit-scrollbar {{
    width: 7px;
}}

::-webkit-scrollbar-track {{
    background: {BACKGROUND};
}}

::-webkit-scrollbar-thumb {{
    background: {MIST_GREY};
}}

::-webkit-scrollbar-thumb:hover {{
    background: {SOFT_TEAL};
}}

</style>
""")


# ============================================================
# CONSTANTS
# ============================================================

CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Model is expected to be in the same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_resnet18_v3.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.fc.in_features,
            len(CLASSES)
        )
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

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


# ============================================================
# IMAGE CONDITION ANALYSIS
# ============================================================

def analyze_image(image):

    image_array = np.array(
        image
    )

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    return (
        brightness,
        contrast,
        sharpness
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = CLASSES[
        prediction.item()
    ]

    confidence_value = confidence.item()

    return (
        predicted_class,
        confidence_value,
        tensor
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    image,
    tensor,
    predicted_class
):

    target_layers = [
        model.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    target_index = CLASSES.index(
        predicted_class
    )

    targets = [
        ClassifierOutputTarget(
            target_index
        )
    ]

    grayscale_cam = cam(
        input_tensor=tensor,
        targets=targets
    )[0]

    image_resized = image.resize(
        (224, 224)
    )

    rgb_image = (
        np.array(
            image_resized
        )
        .astype(np.float32)
        / 255.0
    )

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization


# ============================================================
# HEADER
# ============================================================

html(f"""
<div class="kicker">
    COMPUTER VISION · FER2013
</div>
""")

html(f"""
<div class="hero-title">
    Expression,<br>
    classified.
</div>
""")

html("""
<div class="hero-accent"></div>
""")

html("""
<div class="hero-subtitle">
    A visual study of how a convolutional model
    reads facial expressions — and when that
    reading becomes uncertain.
</div>
""")

html("""
<div class="divider"></div>
""")


# ============================================================
# 01 — INPUT
# ============================================================

html("""
<div class="section-label">
    01 / THE INPUT
</div>
""")

html("""
<div class="section-heading">
    Start with a face.
</div>
""")

html("""
<div class="section-description">
    Upload a facial image and see how the trained
    classifier interprets it. The analysis includes
    the predicted expression, confidence, image
    conditions and a visual explanation.
</div>
""")

st.write("")

uploaded_file = st.file_uploader(
    "Choose a facial image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ========================================================
    # PREDICTION
    # ========================================================

    (
        predicted_class,
        confidence,
        tensor
    ) = predict(image)


    # ========================================================
    # IMAGE CONDITIONS
    # ========================================================

    (
        brightness,
        contrast,
        sharpness
    ) = analyze_image(image)


    # ========================================================
    # 02 — PREDICTION
    # ========================================================

    html("""
    <div class="divider"></div>
    """)

    html("""
    <div class="section-label">
        02 / THE PREDICTION
    </div>
    """)

    html("""
    <div class="section-heading">
        What does the model see?
    </div>
    """)

    left, right = st.columns(
        [1.05, 0.95],
        gap="large"
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with left:

        st.image(
            image,
            use_container_width=True
        )

        html("""
        <div class="image-caption">
            Uploaded image
        </div>
        """)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with right:

        html("""
        <div class="prediction-container">

            <div class="prediction-label">
                Predicted expression
            </div>

            <div class="prediction-name">
        """ + predicted_class + """
            </div>

            <div class="prediction-confidence">
                confidence / """ + f"{confidence * 100:.1f}" + """%
            </div>

        </div>
        """)


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if confidence >= 0.80:

            html("""
            <div class="status-good">

                <strong>
                    High confidence
                </strong>

                <br><br>

                The model is relatively certain
                about this prediction.

            </div>
            """)

        elif confidence >= 0.60:

            html("""
            <div class="status-warning">

                <strong>
                    Moderate confidence
                </strong>

                <br><br>

                The prediction should be interpreted
                with some caution.

            </div>
            """)

        else:

            html("""
            <div class="status-warning">

                <strong>
                    Low confidence
                </strong>

                <br><br>

                This prediction should be considered
                for human review.

            </div>
            """)


    # ========================================================
    # 03 — IMAGE CONDITIONS
    # ========================================================

    html("""
    <div class="divider"></div>
    """)

    html("""
    <div class="section-label">
        03 / IMAGE CONDITIONS
    </div>
    """)

    html("""
    <div class="section-heading">
        How clear is the evidence?
    </div>
    """)

    html("""
    <div class="section-description">
        These measurements describe the image itself,
        not the person in it. They help identify
        conditions under which the classifier may
        become less reliable.
    </div>
    """)

    st.write("")

    q1, q2, q3 = st.columns(3)


    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    with q1:

        html(f"""
        <div class="metric">

            <div class="metric-label">
                Brightness
            </div>

            <div class="metric-value">
                {brightness:.1f}
            </div>

            <div class="metric-note">
                mean grayscale intensity
            </div>

        </div>
        """)


    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    with q2:

        html(f"""
        <div class="metric">

            <div class="metric-label">
                Contrast
            </div>

            <div class="metric-value">
                {contrast:.1f}
            </div>

            <div class="metric-note">
                grayscale variation
            </div>

        </div>
        """)


    # --------------------------------------------------------
    # SHARPNESS
    # --------------------------------------------------------

    with q3:

        html(f"""
        <div class="metric">

            <div class="metric-label">
                Sharpness
            </div>

            <div class="metric-value">
                {sharpness:.1f}
            </div>

            <div class="metric-note">
                Laplacian variance
            </div>

        </div>
        """)


    # ========================================================
    # 04 — GRAD-CAM
    # ========================================================

    html("""
    <div class="divider"></div>
    """)

    html("""
    <div class="section-label">
        04 / MODEL EXPLANATION
    </div>
    """)

    html("""
    <div class="section-heading">
        Where did it look?
    </div>
    """)

    html("""
    <div class="section-description">
        Grad-CAM highlights regions that contributed
        strongly to the selected prediction. It helps
        inspect what the network used as evidence
        rather than treating the prediction as a
        black box.
    </div>
    """)


    with st.spinner(
        "Mapping model attention..."
    ):

        cam_image = generate_gradcam(
            image,
            tensor,
            predicted_class
        )


    html("""
    <div class="explanation-box">
    """)

    st.image(
        cam_image,
        use_container_width=True
    )

    html(f"""
        <div class="explanation-note">
            Grad-CAM for the predicted class:
            <strong>{predicted_class}</strong>.
            Warmer regions represent stronger activation
            for this prediction.
        </div>
    </div>
    """)


    # ========================================================
    # 05 — RELIABILITY
    # ========================================================

    html("""
    <div class="divider"></div>
    """)

    html("""
    <div class="section-label">
        05 / RELIABILITY
    </div>
    """)

    html("""
    <div class="section-heading">
        Should this prediction be reviewed?
    </div>
    """)

    html("""
    <div class="section-description">
        The prototype combines prediction confidence
        with basic image-quality signals to flag cases
        that may deserve human attention.
    </div>
    """)

    st.write("")


    # --------------------------------------------------------
    # REVIEW RULE
    # --------------------------------------------------------

    review_required = (
        confidence < 0.60
        or brightness < 70
        or sharpness < 50
    )


    if review_required:

        html("""
        <div class="status-warning">

            <strong>
                HUMAN REVIEW RECOMMENDED
            </strong>

            <br><br>

            One or more current reliability checks
            were triggered. Treat the model output
            as uncertain rather than definitive.

        </div>
        """)

    else:

        html("""
        <div class="status-good">

            <strong>
                NO IMMEDIATE REVIEW FLAG
            </strong>

            <br><br>

            The prediction does not trigger the
            current confidence or image-condition
            checks.

        </div>
        """)


    # ========================================================
    # MODEL DETAILS
    # ========================================================

    html("""
    <div class="divider"></div>
    """)

    html("""
    <div class="section-label">
        ABOUT THE MODEL
    </div>
    """)

    a, b, c = st.columns(3)


    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    with a:

        html("""
        <div class="model-strip">

            <div class="model-strip-label">
                Architecture
            </div>

            <div class="model-strip-value">
                ResNet18
            </div>

        </div>
        """)


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    with b:

        html("""
        <div class="model-strip">

            <div class="model-strip-label">
                Dataset
            </div>

            <div class="model-strip-value">
                FER2013
            </div>

        </div>
        """)


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    with c:

        html("""
        <div class="model-strip">

            <div class="model-strip-label">
                Held-out test accuracy
            </div>

            <div class="model-strip-value">
                69.87%
            </div>

        </div>
        """)


# ============================================================
# FOOTER
# ============================================================

html("""
<div class="divider"></div>
""")

st.caption(
    "A NOTE ON INTERPRETATION"
)

st.write(
    "This system predicts facial-expression labels "
    "learned from FER2013. A predicted expression is "
    "not a definitive measurement of a person's actual "
    "emotional or psychological state."
)

st.caption(
    "ResNet18  ·  FER2013  ·  Grad-CAM  ·  "
    "Condition-based audit"
)