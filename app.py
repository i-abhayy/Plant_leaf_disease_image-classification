from google.genai import types
from io import BytesIO
from google import genai

from dotenv import load_dotenv
import os
import json
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# 1. FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "multiplant_mobilenetv2_best.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "models", "class_names.json"
)

IMG_SIZE = 224


# --------------------------------------------------
# 2. LOAD MODEL AND CLASS NAMES
# --------------------------------------------------

print("Loading LeafCare AI model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if len(class_names) != 38:
    raise ValueError(
        f"Expected 38 classes, but found {len(class_names)}"
    )

print("Model loaded successfully!")
print("Total classes:", len(class_names))

# Load environment variables and initialize Gemini.
load_dotenv(os.path.join(BASE_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

print("Gemini:", "Connected" if gemini_client else "API key not found")
if gemini_client:
    try:
        test_response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents="Reply with exactly: GEMINI TEST SUCCESS"
        )
        print("Gemini API test:", test_response.text)
    except Exception as e:
        print("Gemini API test failed:", str(e))


# --------------------------------------------------
# 3. DISEASE INFORMATION
# --------------------------------------------------

DISEASE_INFO = {

    "healthy": {
        "description": (
            "The model classified this leaf as healthy. "
            "No target disease symptoms were identified."
        ),
        "causes": (
            "No specific disease was identified by the model."
        ),
        "solutions": (
            "Continue regular watering, provide suitable sunlight, "
            "maintain soil health and monitor the plant regularly."
        )
    },

    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease that can cause dark, olive-coloured spots on leaves.",
        "causes": "It is associated with the fungus Venturia inaequalis and is favoured by wet conditions.",
        "solutions": "Remove fallen infected leaves, improve airflow and consult local agricultural guidance about suitable fungicides."
    },

    "Apple___Black_rot": {
        "description": "Black rot can cause leaf spots and fruit or branch damage in apple plants.",
        "causes": "It is caused by the fungus Botryosphaeria obtusa.",
        "solutions": "Remove infected plant material, prune affected branches appropriately and improve orchard sanitation."
    },

    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is a fungal disease that can produce orange or yellow spots on apple leaves.",
        "causes": "It is caused by Gymnosporangium fungi and involves juniper or cedar plants in its life cycle.",
        "solutions": "Remove heavily affected plant material where practical and seek local advice about resistant varieties and disease management."
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Powdery mildew can produce a white, powder-like coating on cherry leaves.",
        "causes": "It is caused by powdery mildew fungi and is favoured by certain warm, humid conditions.",
        "solutions": "Improve airflow, remove severely affected leaves and follow locally recommended disease management."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray leaf spot causes elongated grey or tan lesions on corn leaves.",
        "causes": "It is caused by Cercospora fungi and can spread under prolonged humid conditions.",
        "solutions": "Use resistant varieties, rotate crops and manage crop residue according to local agricultural recommendations."
    },

    "Corn_(maize)___Common_rust_": {
        "description": "Common rust produces reddish-brown raised pustules on corn leaves.",
        "causes": "It is caused by the fungus Puccinia sorghi.",
        "solutions": "Grow resistant varieties where available and monitor disease development with local agricultural guidance."
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Northern leaf blight causes long, cigar-shaped lesions on corn leaves.",
        "causes": "It is caused by Exserohilum turcicum.",
        "solutions": "Use resistant varieties, rotate crops and manage infected crop residues."
    },

    "Grape___Black_rot": {
        "description": "Grape black rot can produce brown leaf lesions and damage grape berries.",
        "causes": "It is caused by the fungus Guignardia bidwellii.",
        "solutions": "Remove infected berries and plant debris, improve airflow and follow locally recommended disease management."
    },

    "Grape___Esca_(Black_Measles)": {
        "description": "Esca is a grapevine disease complex that can cause leaf discoloration and wood damage.",
        "causes": "It is associated with several wood-infecting fungi.",
        "solutions": "Remove severely affected plant material where appropriate and consult a viticulture specialist."
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Grape leaf blight can cause brown or dark lesions on leaves.",
        "causes": "It is associated with fungal infection and favourable wet conditions.",
        "solutions": "Remove infected leaves, improve ventilation and seek local agricultural advice."
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Citrus greening can cause blotchy leaf mottling, yellow shoots and poor fruit development.",
        "causes": "It is caused by a bacterium associated with the Asian citrus psyllid.",
        "solutions": "Contact a local agricultural expert for confirmation, monitor psyllid populations and follow official citrus disease management guidance."
    },

    "Peach___Bacterial_spot": {
        "description": "Bacterial spot can produce dark lesions on peach leaves and fruit.",
        "causes": "It is caused by Xanthomonas bacteria.",
        "solutions": "Use suitable resistant varieties, avoid unnecessary leaf wetness and follow local bacterial disease management guidance."
    },

    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial spot can cause small dark lesions on pepper leaves.",
        "causes": "It is associated with Xanthomonas bacteria and can spread through splashing water and infected material.",
        "solutions": "Use clean planting material, avoid overhead watering and remove infected plant debris."
    },

    "Potato___Early_blight": {
        "description": "Early blight causes brown spots, often with concentric rings, on potato leaves.",
        "causes": "It is caused by Alternaria solani.",
        "solutions": "Remove infected leaves where practical, rotate crops and follow local recommendations for disease management."
    },

    "Potato___Late_blight": {
        "description": "Late blight can cause water-soaked dark lesions and rapid leaf damage.",
        "causes": "It is caused by Phytophthora infestans and can spread rapidly in cool, wet conditions.",
        "solutions": "Seek prompt local agricultural advice, remove infected plant material safely and follow recommended late-blight management."
    },

    "Squash___Powdery_mildew": {
        "description": "Powdery mildew produces white powder-like patches on squash leaves.",
        "causes": "It is caused by powdery mildew fungi.",
        "solutions": "Improve airflow, avoid overcrowding and remove severely affected leaves when appropriate."
    },

    "Strawberry___Leaf_scorch": {
        "description": "Leaf scorch causes reddish-purple spots and brown, dry areas on strawberry leaves.",
        "causes": "It is associated with the fungus Diplocarpon earlianum.",
        "solutions": "Remove infected leaves, improve airflow and use healthy planting material."
    },

    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot causes small dark lesions on tomato leaves and can affect fruit.",
        "causes": "It is associated with Xanthomonas bacteria.",
        "solutions": "Use disease-free seeds and seedlings, avoid overhead irrigation and remove infected plant debris."
    },

    "Tomato___Early_blight": {
        "description": "Early blight produces brown spots with concentric ring patterns, often on older leaves.",
        "causes": "It is commonly caused by Alternaria solani.",
        "solutions": "Remove affected leaves, avoid wetting foliage, rotate crops and follow local disease management advice."
    },

    "Tomato___Late_blight": {
        "description": "Late blight can cause water-soaked lesions that turn brown or black and spread quickly.",
        "causes": "It is caused by Phytophthora infestans.",
        "solutions": "Seek prompt agricultural advice, remove infected material safely and follow recommended late-blight control practices."
    },

    "Tomato___Leaf_Mold": {
        "description": "Tomato leaf mold can cause yellow patches on the upper leaf surface and fuzzy growth underneath.",
        "causes": "It is caused by Passalora fulva and favoured by high humidity.",
        "solutions": "Improve ventilation, reduce leaf wetness and remove severely affected leaves."
    },

    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot produces small circular spots with pale centres and dark edges.",
        "causes": "It is caused by the fungus Septoria lycopersici.",
        "solutions": "Remove affected lower leaves, avoid overhead watering and rotate crops."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Spider mite damage can cause tiny pale speckles, yellowing and sometimes fine webbing.",
        "causes": "It is caused by feeding from two-spotted spider mites.",
        "solutions": "Inspect the undersides of leaves, reduce dust and seek locally appropriate integrated pest management advice."
    },

    "Tomato___Target_Spot": {
        "description": "Target spot can cause brown circular lesions with concentric patterns on tomato leaves.",
        "causes": "It is associated with the fungus Corynespora cassiicola.",
        "solutions": "Remove infected plant debris, improve airflow and avoid prolonged leaf wetness."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Tomato yellow leaf curl virus can cause upward leaf curling, yellowing and stunted growth.",
        "causes": "It is a viral disease commonly transmitted by whiteflies.",
        "solutions": "Monitor and manage whiteflies, remove confirmed infected plants according to local guidance and use resistant varieties."
    },

    "Tomato___Tomato_mosaic_virus": {
        "description": "Tomato mosaic virus can cause mottled leaves, distortion and uneven plant growth.",
        "causes": "It is caused by a plant virus that can spread through contaminated hands, tools and plant material.",
        "solutions": "Remove infected plants where appropriate, disinfect tools and use clean seeds and planting material."
    }
}


# --------------------------------------------------
# 4. GET DISEASE INFORMATION
# --------------------------------------------------

def get_disease_info(class_name):

    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    # Generic fallback for classes without a detailed entry
    return {
        "description": (
            f"The model classified the leaf as {class_name}. "
            "This category indicates a possible plant disease."
        ),
        "causes": (
            "The exact cause should be confirmed through "
            "additional plant inspection."
        ),
        "solutions": (
            "Remove severely affected material where appropriate, "
            "maintain good plant hygiene and consult a local "
            "agricultural expert for suitable treatment."
        )
    }


# --------------------------------------------------
# 5. PREPROCESS UPLOADED IMAGE
# --------------------------------------------------

def preprocess_leaf_image(image_file):

    image = Image.open(image_file).convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image, dtype=np.float32)

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# 6. PREDICTION FUNCTION
# --------------------------------------------------

def predict_leaf(image_file):

    processed_image = preprocess_leaf_image(image_file)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_class = class_names[predicted_index]

    confidence = float(probabilities[predicted_index]) * 100

    disease_info = get_disease_info(predicted_class)

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "description": disease_info["description"],
        "causes": disease_info["causes"],
        "solutions": disease_info["solutions"]
    }


def generate_plant_explanation(disease, confidence, description, causes, solutions):

    if gemini_client is None:
        raise RuntimeError("Gemini API key is not configured.")

    prompt = f"""
    You are LeafCare AI, an educational plant health assistant.

    Our trained MobileNetV2 image classifier has produced this result:

    Predicted condition: {disease}
    Model confidence: {confidence}%
    Existing description: {description}
    Possible causes: {causes}
    Existing suggestions: {solutions}

    Explain this prediction in simple English.

    Organize your answer using these headings:

    1. What does this prediction mean?
    2. Visible symptoms to look for
    3. Possible causes
    4. Recommended care and treatment
    5. Prevention tips

    Important instructions:
    - Treat the classifier result as a prediction, not a confirmed diagnosis.
    - Do not invent symptoms that cannot be established from the supplied information.
    - Give practical, safe plant-care suggestions.
    - Do not recommend mixing chemicals or pesticides.
    - If the prediction is Healthy, explain how to maintain plant health.
    - Never claim that you personally inspected the image.
    - Keep the explanation easy to understand.
    """

    response = gemini_client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    return response.text

def verify_with_gemini(image_bytes, mime_type, model_result):

    if gemini_client is None:
        raise RuntimeError("Gemini API key is not configured.")

    # Give Gemini the actual image.
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    # Supply the available model classes for a grounded comparison.
    prompt = f"""
You are the visual verification system for LeafCare AI.

Examine the attached original plant leaf image independently.

Our MobileNetV2 classifier produced:
Predicted class: {model_result["disease"]}
Model confidence: {model_result["confidence"]}%

Available classifier classes:
{json.dumps(class_names)}

Your tasks:

1. Determine whether the image clearly contains a plant leaf.
2. Determine whether the image is clear enough to assess.
3. Independently inspect visible symptoms.
4. Identify the most likely condition supported by visible evidence.
5. Compare your assessment with the MobileNetV2 prediction.
6. If the image is unclear, not a leaf, or evidence is insufficient,
   return an uncertain assessment rather than guessing.
7. If you identify a condition, use an exact class name from the
   available classifier classes whenever possible.
8. Do not claim that a disease is confirmed from an image alone.
9. Provide safe, general plant-care recommendations.
10. Never recommend mixing pesticides or chemicals.

Return ONLY valid JSON with these fields:

{{
    "image_is_leaf": true,
    "image_is_clear": true,
    "plant_name": "Plant name or uncertain",
    "final_condition": "Exact classifier class or Uncertain",
    "agrees_with_model": true,
    "visual_observations": [
        "Visible observation 1",
        "Visible observation 2"
    ],
    "explanation": "Simple explanation of the assessment",
    "symptoms_to_check": [
        "Additional symptoms to inspect"
    ],
    "possible_causes": [
        "Possible cause"
    ],
    "care_recommendations": [
        "Safe care recommendation"
    ],
    "prevention_tips": [
        "Prevention recommendation"
    ],
    "caution": "A short diagnostic limitation statement"
}}

Important:
- agrees_with_model must be true, false, or null.
- Use null if you cannot make a reliable comparison.
- Never force agreement with MobileNetV2.
- Be honest about uncertainty.
"""

    response = gemini_client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    assessment = json.loads(response.text)

    required_keys = [
        "image_is_leaf",
        "image_is_clear",
        "final_condition",
        "agrees_with_model",
        "visual_observations",
        "explanation",
        "care_recommendations",
        "prevention_tips"
    ]

    for key in required_keys:
        if key not in assessment:
            raise ValueError(f"Gemini response is missing: {key}")

    return assessment



# --------------------------------------------------
# 7. HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# 8. PREDICTION API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No image was uploaded."
        }), 400

    image_file = request.files["file"]

    if image_file.filename == "":
        return jsonify({
            "error": "Please select an image."
        }), 400

    try:

        # Verify the uploaded image
        image_file.stream.seek(0)

        with Image.open(image_file.stream) as image:
            image.verify()

        image_file.stream.seek(0)

        result = predict_leaf(image_file.stream)

        return jsonify(result)

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "Invalid or corrupted image. Please upload a valid JPG or PNG."
        }), 400

    except Exception as error:

        print("Prediction error:", str(error))

        return jsonify({
            "error": "Unable to analyze this image. Please try again."
        }), 500

# --------------------------------------------------
# 8A. GEMINI IMAGE VERIFICATION API
# --------------------------------------------------

@app.route("/verify", methods=["POST"])
def verify():

    if gemini_client is None:
        return jsonify({
            "success": False,
            "error": "Gemini API key is not configured."
        }), 503

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image was uploaded."
        }), 400

    image_file = request.files["file"]

    if image_file.filename == "":
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400

    try:
        # Read original image bytes.
        image_bytes = image_file.read()

        # Validate image and identify its actual format.
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            image_format = image.format

        mime_types = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp"
        }

        mime_type = mime_types.get(image_format)

        if mime_type is None:
            return jsonify({
                "success": False,
                "error": "Please upload a JPG, PNG or WEBP image."
            }), 400

        # Step 1: MobileNetV2 prediction.
        model_result = predict_leaf(BytesIO(image_bytes))

        # Step 2: Gemini independently examines the original image.
        gemini_result = verify_with_gemini(
            image_bytes=image_bytes,
            mime_type=mime_type,
            model_result=model_result
        )

        # Step 3: Determine agreement status.
        if (
            gemini_result["image_is_leaf"] is not True
            or gemini_result["image_is_clear"] is not True
            or gemini_result["agrees_with_model"] is None
        ):
            verification_status = "uncertain"

        elif gemini_result["agrees_with_model"] is True:
            verification_status = "agree"

        else:
            verification_status = "disagree"

        # Gemini is the final AI assessment, but not a confirmed diagnosis.
        final_result = {
            "disease": gemini_result["final_condition"],
            "plant": gemini_result.get("plant_name", "Uncertain"),
            "status": verification_status,
            "explanation": gemini_result["explanation"],
            "care_recommendations": gemini_result["care_recommendations"],
            "prevention_tips": gemini_result["prevention_tips"],
            "caution": gemini_result.get(
                "caution",
                "AI assessment only. Professional confirmation may be needed."
            )
        }

        return jsonify({
            "success": True,
            "model_prediction": model_result,
            "gemini_assessment": gemini_result,
            "final_result": final_result
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "success": False,
            "error": "Invalid or corrupted image."
        }), 400

    except Exception:
        app.logger.exception("LeafCare AI verification failed")

        return jsonify({
            "success": False,
            "error": "Image verification failed. Please check the backend terminal."
        }), 502



@app.route("/ai-explain", methods=["POST"])
def ai_explain():

    if gemini_client is None:
        return jsonify({
            "error": "Gemini API key is not configured."
        }), 503

    data = request.get_json(silent=True) or {}

    required_fields = [
        "disease",
        "confidence",
        "description",
        "causes",
        "solutions"
    ]

    if not all(field in data for field in required_fields):
        return jsonify({
            "error": "Missing prediction information."
        }), 400

    try:
        explanation = generate_plant_explanation(
            disease=str(data["disease"])[:200],
            confidence=str(data["confidence"])[:20],
            description=str(data["description"])[:1000],
            causes=str(data["causes"])[:1000],
            solutions=str(data["solutions"])[:1000]
        )

        return jsonify({
            "success": True,
            "explanation": explanation
        })

    except Exception as error:
        app.logger.exception("Gemini API request failed")

        return jsonify({
            "success": False,
            "error": "Could not generate AI explanation. Please try again."
        }), 502


# --------------------------------------------------
# 9. FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Image is too large. Maximum allowed size is 8 MB."
    }), 413


# --------------------------------------------------
# 10. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
