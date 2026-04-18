import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from flask import Flask, render_template, request
import keras
import numpy as np
from PIL import Image
import gdown

MODEL_PATH = "model/crop_model.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1Pk--I6eXVErjViq0uGO_BoiE_zRNxeqW"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded!")

print("Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model Loaded")

with open("model/classes.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"{len(class_names)} classes loaded")

DISEASE_DB = {
    "Tomato___Late_blight": {
        "display": "Tomato - Late Blight",
        "severity": "high", "spread": "Very High", "crop": "Tomato",
        "desc": "Caused by Phytophthora infestans. Dark brown water-soaked spots on leaves and stems with white mould on the underside. Spreads rapidly in cool wet conditions and can destroy an entire crop within days.",
        "treatments": [
            "Remove and destroy all infected leaves and stems immediately - never compost them",
            "Apply copper-based fungicide (Blitox 50 or Bordeaux mixture) every 7 days",
            "Avoid overhead watering - water only at the base in the morning",
            "Improve air circulation by pruning overcrowded branches",
            "Use certified disease-free seeds for next planting season"
        ],
        "alert": "Act within 24 hours - this disease spreads extremely fast. Warn neighbouring farmers immediately.",
        "prevention": ["Use resistant tomato varieties", "Maintain proper plant spacing", "Avoid working in wet fields", "Rotate crops every season"]
    },
    "Tomato___Early_blight": {
        "display": "Tomato - Early Blight",
        "severity": "medium", "spread": "Medium", "crop": "Tomato",
        "desc": "Caused by Alternaria solani. Dark brown circular spots with concentric rings like a target appear on older leaves first. Progresses upward causing significant defoliation.",
        "treatments": [
            "Remove infected lower leaves and destroy them immediately",
            "Apply Mancozeb or Chlorothalonil fungicide every 10 days",
            "Mulch around plants to prevent soil splash onto leaves",
            "Ensure good drainage around plant roots",
            "Provide balanced fertilization - avoid excess nitrogen"
        ],
        "alert": "Start treatment early before the disease spreads to upper leaves and fruits.",
        "prevention": ["Water at the base of plants only", "Stake plants for better airflow", "Rotate with non-solanaceous crops", "Remove crop debris after harvest"]
    },
    "Tomato___Bacterial_spot": {
        "display": "Tomato - Bacterial Spot",
        "severity": "medium", "spread": "Medium", "crop": "Tomato",
        "desc": "Caused by Xanthomonas bacteria. Small dark water-soaked spots on leaves stems and fruits. Spots develop yellow halos. Fruits develop raised scab-like spots reducing market value.",
        "treatments": [
            "Apply copper-based bactericide (Copper Oxychloride) immediately",
            "Remove heavily infected plant parts and destroy them",
            "Avoid overhead irrigation and working in wet conditions",
            "Disinfect garden tools with bleach solution after each use",
            "Apply Streptomycin sulfate in severe cases"
        ],
        "alert": "Bacterial diseases spread rapidly through water splash. Stop all overhead irrigation immediately.",
        "prevention": ["Use disease-free certified seeds", "Treat seeds with hot water before planting", "Avoid working in wet fields", "Maintain crop hygiene"]
    },
    "Tomato___Leaf_Mold": {
        "display": "Tomato - Leaf Mold",
        "severity": "medium", "spread": "Medium", "crop": "Tomato",
        "desc": "Caused by Passalora fulva. Yellow patches on upper leaf surface with olive-green or brown velvety mould on underside. Common in greenhouses and humid conditions.",
        "treatments": [
            "Improve ventilation and reduce humidity immediately",
            "Apply Chlorothalonil or Mancozeb fungicide",
            "Remove and destroy infected leaves",
            "Avoid overhead watering",
            "Space plants wider for better airflow"
        ],
        "alert": "Leaf mold spreads fast in humid conditions. Improve ventilation immediately.",
        "prevention": ["Good ventilation", "Avoid excess humidity", "Proper plant spacing", "Resistant varieties"]
    },
    "Tomato___Septoria_leaf_spot": {
        "display": "Tomato - Septoria Leaf Spot",
        "severity": "medium", "spread": "Medium", "crop": "Tomato",
        "desc": "Caused by Septoria lycopersici. Small circular spots with dark borders and light grey centers on lower leaves. Causes yellowing and early leaf drop reducing fruit quality.",
        "treatments": [
            "Remove and destroy infected lower leaves immediately",
            "Apply Mancozeb or Copper fungicide every 7-10 days",
            "Mulch soil surface to prevent spore splash",
            "Stake plants to keep foliage off ground",
            "Avoid working among wet plants"
        ],
        "alert": "Remove infected leaves immediately to prevent spread upward through the plant.",
        "prevention": ["Crop rotation", "Mulching", "Staking plants", "Avoiding leaf wetness"]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display": "Tomato - Spider Mites",
        "severity": "medium", "spread": "High", "crop": "Tomato",
        "desc": "Tiny spider mites cause yellow stippling on leaves. Fine webbing visible on undersides. Thrives in hot dry conditions. Can cause severe leaf damage and early plant death.",
        "treatments": [
            "Spray plants with strong water jet to dislodge mites",
            "Apply Abamectin or Bifenazate miticide",
            "Increase humidity around plants - mites hate moisture",
            "Introduce natural predators like ladybirds if available",
            "Remove heavily infested leaves and destroy"
        ],
        "alert": "Spider mites multiply extremely fast in hot weather. Act immediately before population explodes.",
        "prevention": ["Regular monitoring", "Maintain adequate soil moisture", "Avoid dusty conditions", "Encourage natural predators"]
    },
    "Tomato___Target_Spot": {
        "display": "Tomato - Target Spot",
        "severity": "medium", "spread": "Medium", "crop": "Tomato",
        "desc": "Caused by Corynespora cassiicola. Brown circular lesions with concentric rings on leaves fruits and stems. Causes defoliation and fruit rot in severe infections.",
        "treatments": [
            "Apply Azoxystrobin or Chlorothalonil fungicide",
            "Remove infected plant material immediately",
            "Improve air circulation around plants",
            "Avoid overhead irrigation",
            "Maintain proper plant nutrition"
        ],
        "alert": "Target spot can cause significant fruit losses if left untreated.",
        "prevention": ["Good airflow", "Proper spacing", "Crop rotation", "Remove debris after harvest"]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Tomato - Yellow Leaf Curl Virus",
        "severity": "high", "spread": "High", "crop": "Tomato",
        "desc": "Viral disease spread by whiteflies. Leaves curl upward and turn yellow. Stunted plant growth and very low fruit production. No cure once infected - prevention is essential.",
        "treatments": [
            "Remove and destroy all infected plants immediately",
            "Control whitefly population with Imidacloprid or yellow sticky traps",
            "Cover young plants with insect-proof netting",
            "Spray Neem oil every 5-7 days to repel whiteflies",
            "Do not plant new tomatoes near infected ones"
        ],
        "alert": "No cure exists for this virus. Remove infected plants immediately to protect healthy ones.",
        "prevention": ["Use virus-resistant varieties", "Control whitefly early", "Use reflective mulch", "Install insect-proof nets"]
    },
    "Tomato___Tomato_mosaic_virus": {
        "display": "Tomato - Mosaic Virus",
        "severity": "high", "spread": "Medium", "crop": "Tomato",
        "desc": "Viral disease causing mottled light and dark green mosaic pattern on leaves. Leaves may be distorted and curled. Reduces fruit quality and yield significantly.",
        "treatments": [
            "Remove and destroy all infected plants - no chemical cure exists",
            "Wash hands thoroughly before handling healthy plants",
            "Disinfect all tools with bleach between uses",
            "Control aphids which spread the virus with Imidacloprid",
            "Do not smoke near plants - tobacco can carry the virus"
        ],
        "alert": "No cure for mosaic virus. Remove infected plants immediately. Wash hands and tools before touching other plants.",
        "prevention": ["Use resistant varieties", "Control aphids", "Disinfect tools regularly", "Avoid tobacco near plants"]
    },
    "Tomato___healthy": {
        "display": "Tomato - Healthy",
        "severity": "low", "spread": "None", "crop": "Tomato",
        "desc": "Your tomato crop appears perfectly healthy! No signs of disease fungal infection or pest damage detected. Excellent crop management!",
        "treatments": [
            "Continue regular monitoring every 3-4 days",
            "Maintain balanced NPK fertilization as per growth stage",
            "Apply preventive fungicide spray during high-humidity periods",
            "Ensure consistent watering at the base of plants",
            "Keep a crop diary to record your successful practices"
        ],
        "alert": "Great news! Your crop is healthy. Keep up your excellent management practices.",
        "prevention": ["Monitor regularly", "Maintain soil health", "Proper plant spacing", "Timely nutrition application"]
    },
    "Rice___Blast": {
        "display": "Rice - Blast Disease",
        "severity": "high", "spread": "High", "crop": "Rice",
        "desc": "Caused by Magnaporthe oryzae. Diamond-shaped grey-white lesions with brown borders on leaves. Can attack the neck causing entire panicle death called neck blast with devastating yield losses.",
        "treatments": [
            "Spray Tricyclazole (Beam) at the very first sign of lesions",
            "Apply Isoprothiolane as an alternative fungicide option",
            "Drain field water for 3-4 days to reduce humidity",
            "Avoid excess nitrogen fertilizer - it promotes blast spread",
            "Collect and burn all infected plant debris after harvest"
        ],
        "alert": "Neck blast can cause up to 70% yield loss. Spray fungicide preventively before heading stage.",
        "prevention": ["Use blast-resistant varieties", "Avoid dense planting", "Apply silicon fertilizers", "Monitor during humid weather"]
    },
    "Rice___Brown_spot": {
        "display": "Rice - Brown Spot",
        "severity": "medium", "spread": "Medium", "crop": "Rice",
        "desc": "Caused by Cochliobolus miyabeanus. Oval to circular brown spots with grey or whitish centres on leaves. Associated with nutrient deficiency especially potassium.",
        "treatments": [
            "Apply Mancozeb or Iprodione fungicide at early stage",
            "Correct potassium deficiency with muriate of potash",
            "Improve drainage in waterlogged fields",
            "Treat seeds with Thiram or Captan before planting",
            "Apply balanced fertilizers especially potassium"
        ],
        "alert": "Brown spot is often linked to poor soil nutrition. Check soil health and correct deficiencies.",
        "prevention": ["Balanced soil nutrition", "Use healthy certified seeds", "Avoid waterlogging", "Apply potassium regularly"]
    },
    "Rice___Leaf_scald": {
        "display": "Rice - Leaf Scald",
        "severity": "medium", "spread": "Medium", "crop": "Rice",
        "desc": "Caused by Microdochium oryzae. Irregular water-soaked lesions that dry to straw color. Affects leaf tips and margins causing premature drying.",
        "treatments": [
            "Apply Propiconazole or Tebuconazole fungicide",
            "Avoid excess nitrogen fertilization",
            "Improve field drainage",
            "Use certified disease-free seeds",
            "Remove and destroy infected crop debris"
        ],
        "alert": "Leaf scald reduces photosynthesis significantly. Begin treatment at first signs.",
        "prevention": ["Balanced fertilization", "Good drainage", "Certified seeds", "Crop rotation"]
    },
    "Wheat___Yellow_Rust": {
        "display": "Wheat - Yellow Rust",
        "severity": "medium", "spread": "High", "crop": "Wheat",
        "desc": "Caused by Puccinia striiformis. Yellow-orange powdery stripes along leaf veins. Thrives in cool moist conditions between 10-15 degrees C. Early detection is key to saving the crop.",
        "treatments": [
            "Apply Propiconazole (Tilt 25 EC) at 500ml per acre immediately",
            "Spray in the morning for best leaf absorption",
            "Repeat spray after 15 days if disease persists",
            "Ensure adequate potassium fertilization for resistance",
            "Harvest early if disease is widespread to minimise losses"
        ],
        "alert": "Yellow rust spreads through wind over large distances. Report outbreak to your local agriculture officer.",
        "prevention": ["Plant rust-resistant varieties", "Monitor weekly during cool weather", "Avoid late planting", "Remove volunteer wheat plants"]
    },
    "Wheat___Brown_rust": {
        "display": "Wheat - Brown Rust",
        "severity": "medium", "spread": "High", "crop": "Wheat",
        "desc": "Caused by Puccinia triticina. Orange-brown powdery pustules scattered on upper leaf surface. More common in warm humid conditions than yellow rust.",
        "treatments": [
            "Apply Propiconazole or Tebuconazole fungicide immediately",
            "Spray entire field coverage for best results",
            "Repeat application after 14-21 days",
            "Monitor field regularly after treatment",
            "Consider early harvest if severely affected"
        ],
        "alert": "Brown rust can cause up to 30% yield loss. Treat immediately when detected.",
        "prevention": ["Resistant varieties", "Timely sowing", "Monitor regularly", "Balanced nutrition"]
    },
    "Corn_(maize)___Common_rust_": {
        "display": "Corn - Common Rust",
        "severity": "medium", "spread": "Medium", "crop": "Maize / Corn",
        "desc": "Caused by Puccinia sorghi. Small oval powdery brick-red pustules appear on both leaf surfaces. Favoured by cool temperatures and high humidity. Reduces photosynthesis and grain yield.",
        "treatments": [
            "Apply Propiconazole or Azoxystrobin fungicide when pustules first appear",
            "Spray in early morning for maximum leaf coverage",
            "Repeat application after 14 days in severe cases",
            "Ensure proper plant spacing for air circulation",
            "Remove severely infected lower leaves to slow spread"
        ],
        "alert": "Yield losses up to 40% possible in severe cases. Begin treatment immediately.",
        "prevention": ["Plant resistant hybrid varieties", "Avoid late planting", "Rotate crops", "Monitor during cool humid periods"]
    },
    "Corn_(maize)___Gray_leaf_spot": {
        "display": "Corn - Gray Leaf Spot",
        "severity": "medium", "spread": "Medium", "crop": "Maize / Corn",
        "desc": "Caused by Cercospora zeae-maydis. Long rectangular grey-tan lesions running parallel to leaf veins. Thrives in warm humid conditions. Significantly reduces photosynthesis.",
        "treatments": [
            "Apply Azoxystrobin or Mancozeb fungicide when lesions first appear",
            "Rotate crops - avoid planting maize in same field two seasons",
            "Plough under crop residues after harvest to reduce spores",
            "Use certified resistant hybrid seeds for next season",
            "Ensure proper row spacing for good air circulation"
        ],
        "alert": "Yield losses up to 50% possible with severe infection. Begin treatment immediately.",
        "prevention": ["Crop rotation", "Resistant hybrids", "Proper spacing", "Remove crop residues"]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display": "Corn - Northern Leaf Blight",
        "severity": "medium", "spread": "Medium", "crop": "Maize / Corn",
        "desc": "Caused by Exserohilum turcicum. Large cigar-shaped tan lesions on leaves. Favoured by moderate temperatures and high humidity. Can cause major yield losses in susceptible varieties.",
        "treatments": [
            "Apply Propiconazole or Azoxystrobin fungicide at first sign",
            "Spray at whorl stage for best protection",
            "Use resistant hybrid varieties next season",
            "Rotate crops away from maize",
            "Till in crop residues after harvest"
        ],
        "alert": "Northern leaf blight can cause 30-50% yield loss. Treat early for best results.",
        "prevention": ["Resistant varieties", "Crop rotation", "Residue management", "Timely planting"]
    },
    "Corn_(maize)___healthy": {
        "display": "Corn - Healthy",
        "severity": "low", "spread": "None", "crop": "Maize / Corn",
        "desc": "Your maize crop appears healthy! No signs of disease detected. Your field management is excellent.",
        "treatments": [
            "Continue monitoring every 5-7 days",
            "Maintain adequate nitrogen at vegetative stages",
            "Ensure consistent moisture especially at silking",
            "Scout for pest damage regularly",
            "Record your management practices for future reference"
        ],
        "alert": "Your corn crop is healthy! Continue your current management practices.",
        "prevention": ["Regular monitoring", "Balanced nutrition", "Proper spacing", "Timely irrigation"]
    },
    "Potato___Late_blight": {
        "display": "Potato - Late Blight",
        "severity": "high", "spread": "Very High", "crop": "Potato",
        "desc": "Caused by Phytophthora infestans. Water-soaked dark lesions on leaves and stems. Infected tubers develop reddish-brown internal rot. Most devastating potato disease worldwide.",
        "treatments": [
            "Apply Metalaxyl plus Mancozeb (Ridomil Gold) immediately",
            "Spray every 5-7 days during wet and cool weather",
            "Hill up soil around plants to protect tubers",
            "Remove and destroy all infected plants and tubers",
            "Never store infected tubers - they spread disease to healthy ones"
        ],
        "alert": "This disease caused the Irish Famine. Act immediately - entire fields can be lost within one week.",
        "prevention": ["Certified disease-free seed potatoes", "Well-drained soil", "Avoid overhead irrigation", "Harvest promptly"]
    },
    "Potato___Early_blight": {
        "display": "Potato - Early Blight",
        "severity": "medium", "spread": "Medium", "crop": "Potato",
        "desc": "Caused by Alternaria solani. Dark brown circular target-like spots on older leaves. Causes premature defoliation and reduced tuber size.",
        "treatments": [
            "Apply Chlorothalonil or Mancozeb fungicide at first symptoms",
            "Remove infected leaves and dispose of them properly",
            "Avoid water stress - maintain consistent irrigation",
            "Apply balanced fertilizer to reduce plant stress",
            "Spray every 7-10 days during humid weather"
        ],
        "alert": "Early blight significantly reduces tuber yield. Start treatment early for best results.",
        "prevention": ["Certified healthy seed tubers", "Adequate plant nutrition", "Crop rotation", "Remove residues after harvest"]
    },
    "Potato___healthy": {
        "display": "Potato - Healthy",
        "severity": "low", "spread": "None", "crop": "Potato",
        "desc": "Your potato crop appears healthy! No disease detected. Continue your current management for a successful harvest.",
        "treatments": [
            "Continue monitoring every 3-5 days",
            "Hill up soil around plants as they grow",
            "Maintain consistent soil moisture",
            "Apply balanced fertilizer at key growth stages",
            "Scout for pest damage regularly"
        ],
        "alert": "Your potato crop is healthy! Keep up your excellent management.",
        "prevention": ["Regular monitoring", "Proper hilling", "Good drainage", "Certified seed potatoes"]
    },
    "Apple___Apple_scab": {
        "display": "Apple - Apple Scab",
        "severity": "medium", "spread": "Medium", "crop": "Apple",
        "desc": "Caused by Venturia inaequalis. Olive-green to brown velvety spots on leaves and fruits. Infected fruits develop corky distorted scabs reducing market value significantly.",
        "treatments": [
            "Apply Captan or Myclobutanil fungicide starting at bud break",
            "Continue spray program every 7-10 days through spring",
            "Remove and destroy all fallen infected leaves from orchard",
            "Prune trees to improve airflow and reduce humidity",
            "Avoid wetting foliage during irrigation - use drip irrigation"
        ],
        "alert": "Start fungicide program before symptoms appear - prevention is far more effective than cure.",
        "prevention": ["Plant scab-resistant varieties", "Rake fallen leaves in autumn", "Prune for open canopy", "Apply lime sulfur dormant season"]
    },
    "Apple___Black_rot": {
        "display": "Apple - Black Rot",
        "severity": "high", "spread": "High", "crop": "Apple",
        "desc": "Caused by Botryosphaeria obtusa. Purple spots on leaves that enlarge with yellow halos. Fruits develop brown rot that turns black and shrivels. Also causes cankers on branches.",
        "treatments": [
            "Apply Captan or Thiophanate-methyl fungicide immediately",
            "Remove all mummified fruits from trees and ground",
            "Prune out all infected and dead wood from trees",
            "Disinfect pruning tools with bleach between cuts",
            "Maintain tree vigor with proper fertilization and irrigation"
        ],
        "alert": "Remove all mummified fruits and infected wood immediately - they are source of future infections.",
        "prevention": ["Remove mummies annually", "Prune dead wood", "Good tree nutrition", "Preventive spray program"]
    },
    "Apple___Cedar_apple_rust": {
        "display": "Apple - Cedar Apple Rust",
        "severity": "medium", "spread": "Medium", "crop": "Apple",
        "desc": "Caused by Gymnosporangium juniperi-virginianae. Bright orange-yellow spots on leaves and fruits. Requires both apple and cedar or juniper trees to complete its life cycle.",
        "treatments": [
            "Apply Myclobutanil or Mancozeb fungicide at pink bud stage",
            "Continue sprays through petal fall period",
            "Remove nearby cedar or juniper trees if possible",
            "Apply protective fungicide before wet weather",
            "Use resistant apple varieties for future planting"
        ],
        "alert": "Rust requires nearby cedar trees to spread. Consider removing cedars within 500 meters of orchard.",
        "prevention": ["Plant resistant varieties", "Remove nearby cedars", "Preventive spray program", "Monitor cedars for galls in spring"]
    },
    "Apple___healthy": {
        "display": "Apple - Healthy",
        "severity": "low", "spread": "None", "crop": "Apple",
        "desc": "Your apple tree appears healthy! No signs of disease or pest damage detected. Your orchard management is working well.",
        "treatments": [
            "Continue regular monitoring of leaves and fruits",
            "Maintain dormant season spray program",
            "Practice proper pruning for good airflow",
            "Apply balanced fertilization",
            "Keep records of management practices"
        ],
        "alert": "Your apple tree is healthy! Continue your current orchard management practices.",
        "prevention": ["Regular monitoring", "Annual pruning", "Balanced nutrition", "Preventive spray program"]
    },
    "Grape___Black_rot": {
        "display": "Grape - Black Rot",
        "severity": "high", "spread": "High", "crop": "Grape",
        "desc": "Caused by Guignardia bidwellii. Tan-brown circular spots with dark borders on leaves. Infected berries shrivel into hard black mummified fruits. Can destroy entire grape harvest.",
        "treatments": [
            "Apply Myclobutanil or Mancozeb from bud break through fruit set",
            "Remove all mummified fruits from vines and ground",
            "Prune out all infected canes during winter dormancy",
            "Improve canopy airflow through proper training",
            "Spray every 10-14 days during wet season"
        ],
        "alert": "Mummified fruits spread disease for years. Remove every single one from vineyard immediately.",
        "prevention": ["Remove mummies before bud break", "Open canopy training", "Early spray program", "Resistant varieties"]
    },
    "Grape___Esca_(Black_Measles)": {
        "display": "Grape - Esca (Black Measles)",
        "severity": "high", "spread": "Low", "crop": "Grape",
        "desc": "Complex fungal disease causing tiger-stripe pattern of yellow and brown on leaves. Berries develop dark spots and shrivel. Internal wood shows brown streaking. Chronic vine disease.",
        "treatments": [
            "Remove and burn severely infected vines",
            "Prune out infected wood cutting back to healthy tissue",
            "Protect pruning wounds with wound sealant immediately",
            "Apply sodium arsenite to pruning wounds where permitted",
            "Maintain good vine nutrition to improve resistance"
        ],
        "alert": "Esca is a chronic disease with no complete cure. Focus on prevention and removing infected vines.",
        "prevention": ["Protect all pruning wounds", "Prune during dry weather", "Use clean pruning tools", "Avoid stress to vines"]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display": "Grape - Leaf Blight",
        "severity": "medium", "spread": "Medium", "crop": "Grape",
        "desc": "Caused by Isariopsis clavispora. Dark brown irregular spots on leaves that cause premature leaf drop. Weakens vines and reduces fruit quality.",
        "treatments": [
            "Apply Mancozeb or Copper fungicide at first signs",
            "Remove infected leaves from vines and ground",
            "Improve canopy ventilation through leaf removal",
            "Avoid overhead irrigation on foliage",
            "Apply fungicide every 10-14 days in wet conditions"
        ],
        "alert": "Leaf blight weakens vines over time. Consistent spray program is essential for control.",
        "prevention": ["Regular monitoring", "Good canopy management", "Timely fungicide applications", "Remove fallen leaves"]
    },
    "Grape___healthy": {
        "display": "Grape - Healthy",
        "severity": "low", "spread": "None", "crop": "Grape",
        "desc": "Your grape vines appear healthy! No signs of disease detected. Your vineyard management is excellent.",
        "treatments": [
            "Continue regular monitoring through the season",
            "Maintain preventive fungicide spray schedule",
            "Practice proper canopy management",
            "Apply balanced nutrition program",
            "Monitor for pest pressure regularly"
        ],
        "alert": "Your grape vines are healthy! Continue your excellent vineyard management.",
        "prevention": ["Regular monitoring", "Preventive sprays", "Good canopy management", "Balanced nutrition"]
    },
    "Pepper,_bell___Bacterial_spot": {
        "display": "Bell Pepper - Bacterial Spot",
        "severity": "medium", "spread": "Medium", "crop": "Bell Pepper",
        "desc": "Caused by Xanthomonas bacteria. Small water-soaked spots on leaves that turn brown with yellow halos. Fruits develop raised dark spots. Spreads rapidly through water splash.",
        "treatments": [
            "Apply copper-based bactericide immediately",
            "Remove heavily spotted leaves and destroy them",
            "Avoid all overhead irrigation - switch to drip irrigation",
            "Disinfect all garden tools with 10% bleach solution",
            "Apply Streptomycin sulfate in severe outbreaks"
        ],
        "alert": "Stop overhead irrigation immediately. Bacterial diseases spread rapidly through water splash.",
        "prevention": ["Certified disease-free seeds", "Treat seeds before planting", "Avoid wet conditions", "Rotate crops regularly"]
    },
    "Pepper,_bell___healthy": {
        "display": "Bell Pepper - Healthy",
        "severity": "low", "spread": "None", "crop": "Bell Pepper",
        "desc": "Your bell pepper crop appears healthy! No signs of disease detected. Continue your current management practices.",
        "treatments": [
            "Continue regular monitoring every few days",
            "Maintain consistent watering schedule",
            "Apply balanced fertilizer at fruiting stage",
            "Support heavy fruiting branches if needed",
            "Keep weeds controlled around plants"
        ],
        "alert": "Your pepper crop is healthy! Excellent work on your crop management.",
        "prevention": ["Monitor regularly", "Avoid overhead watering", "Maintain plant nutrition", "Proper spacing for airflow"]
    },
    "Strawberry___Leaf_scorch": {
        "display": "Strawberry - Leaf Scorch",
        "severity": "medium", "spread": "Medium", "crop": "Strawberry",
        "desc": "Caused by Diplocarpon earlianum. Small purple to reddish-purple spots on leaves that enlarge and cause leaf margins to turn brown and dry. Reduces plant vigor and fruit production.",
        "treatments": [
            "Apply Captan or Myclobutanil fungicide immediately",
            "Remove and destroy infected leaves",
            "Avoid overhead irrigation - use drip irrigation",
            "Improve air circulation by thinning plant density",
            "Apply fungicide every 10-14 days during wet periods"
        ],
        "alert": "Leaf scorch weakens strawberry plants and reduces fruit yield. Treat early for best results.",
        "prevention": ["Plant resistant varieties", "Avoid overhead irrigation", "Good plant spacing", "Remove old leaves"]
    },
    "Strawberry___healthy": {
        "display": "Strawberry - Healthy",
        "severity": "low", "spread": "None", "crop": "Strawberry",
        "desc": "Your strawberry plants appear healthy! No signs of disease detected. Your management practices are working well.",
        "treatments": [
            "Continue regular monitoring",
            "Maintain consistent moisture with drip irrigation",
            "Apply balanced fertilizer after harvest",
            "Remove old leaves and runners as needed",
            "Protect fruits from soil contact with straw mulch"
        ],
        "alert": "Your strawberry crop is healthy! Keep up your excellent management.",
        "prevention": ["Regular monitoring", "Drip irrigation", "Proper spacing", "Mulching"]
    },
    "Peach___Bacterial_spot": {
        "display": "Peach - Bacterial Spot",
        "severity": "medium", "spread": "Medium", "crop": "Peach",
        "desc": "Caused by Xanthomonas arboricola. Small water-soaked spots on leaves and fruits that turn brown. Causes premature leaf drop and fruit blemishes reducing market value significantly.",
        "treatments": [
            "Apply copper-based bactericide from bud swell through harvest",
            "Apply Oxytetracycline antibiotic spray in severe cases",
            "Remove infected fruits and leaves from orchard",
            "Avoid overhead irrigation on trees",
            "Maintain good tree nutrition especially potassium"
        ],
        "alert": "Bacterial spot can severely reduce fruit quality and market value. Begin treatment early in the season.",
        "prevention": ["Plant resistant varieties", "Copper sprays from dormancy", "Avoid wetting foliage", "Good tree nutrition"]
    },
    "Peach___healthy": {
        "display": "Peach - Healthy",
        "severity": "low", "spread": "None", "crop": "Peach",
        "desc": "Your peach tree appears healthy! No signs of disease detected. Your orchard management is excellent.",
        "treatments": [
            "Continue regular monitoring",
            "Maintain dormant season spray program",
            "Practice proper thinning for fruit size",
            "Apply balanced fertilization",
            "Monitor for pest pressure"
        ],
        "alert": "Your peach tree is healthy! Continue your current management practices.",
        "prevention": ["Regular monitoring", "Dormant sprays", "Proper pruning", "Balanced nutrition"]
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "display": "Cherry - Powdery Mildew",
        "severity": "medium", "spread": "Medium", "crop": "Cherry",
        "desc": "Caused by Podosphaera clandestina. White powdery coating on young leaves shoots and fruits. Causes leaf curling and fruit russeting. Thrives in warm dry conditions with high humidity.",
        "treatments": [
            "Apply Sulphur or Myclobutanil fungicide at first signs",
            "Improve air circulation through proper pruning",
            "Spray Neem oil as an organic alternative",
            "Remove heavily infected shoots",
            "Avoid excess nitrogen fertilization"
        ],
        "alert": "Powdery mildew spreads rapidly in warm weather. Begin treatment immediately upon detection.",
        "prevention": ["Plant resistant varieties", "Good pruning for airflow", "Avoid excess nitrogen", "Monitor young growth closely"]
    },
    "Cherry_(including_sour)___healthy": {
        "display": "Cherry - Healthy",
        "severity": "low", "spread": "None", "crop": "Cherry",
        "desc": "Your cherry tree appears healthy! No signs of disease detected. Your orchard management is working well.",
        "treatments": [
            "Continue regular monitoring",
            "Maintain proper pruning schedule",
            "Apply balanced fertilization",
            "Monitor for pest pressure",
            "Protect blossoms from late frosts"
        ],
        "alert": "Your cherry tree is healthy! Continue your excellent management.",
        "prevention": ["Regular monitoring", "Proper pruning", "Balanced nutrition", "Pest management"]
    },
    "Squash___Powdery_mildew": {
        "display": "Squash - Powdery Mildew",
        "severity": "medium", "spread": "High", "crop": "Squash",
        "desc": "Caused by Podosphaera xanthii. White powdery spots on upper leaf surfaces that spread to cover entire leaves. Causes premature leaf death and reduces fruit quality and yield.",
        "treatments": [
            "Apply Sulphur or Potassium bicarbonate spray immediately",
            "Use Neem oil spray as organic alternative",
            "Remove heavily infected leaves and destroy",
            "Improve air circulation around plants",
            "Apply Azoxystrobin fungicide in severe cases"
        ],
        "alert": "Powdery mildew spreads very fast in warm weather. Treat entire planting immediately.",
        "prevention": ["Plant resistant varieties", "Proper plant spacing", "Avoid overhead irrigation", "Monitor young leaves closely"]
    },
    "Soybean___healthy": {
        "display": "Soybean - Healthy",
        "severity": "low", "spread": "None", "crop": "Soybean",
        "desc": "Your soybean crop appears healthy! No signs of disease detected. Your field management is excellent.",
        "treatments": [
            "Continue monitoring every 5-7 days",
            "Maintain balanced nutrition especially phosphorus and potassium",
            "Scout for pest pressure regularly",
            "Ensure adequate nodulation for nitrogen fixation",
            "Monitor for iron deficiency chlorosis"
        ],
        "alert": "Your soybean crop is healthy! Keep up your excellent management practices.",
        "prevention": ["Regular monitoring", "Crop rotation", "Balanced nutrition", "Proper seed treatment"]
    },
    "Blueberry___healthy": {
        "display": "Blueberry - Healthy",
        "severity": "low", "spread": "None", "crop": "Blueberry",
        "desc": "Your blueberry plants appear healthy! No disease detected. Continue your current management for a great harvest.",
        "treatments": [
            "Continue regular monitoring",
            "Maintain soil pH between 4.5-5.0 for optimal growth",
            "Apply acid-forming fertilizer",
            "Mulch heavily to retain moisture",
            "Monitor for bird damage during harvest"
        ],
        "alert": "Your blueberry crop is healthy! Maintain soil acidity for continued good health.",
        "prevention": ["Maintain proper soil pH", "Good drainage", "Mulching", "Regular monitoring"]
    },
    "Raspberry___healthy": {
        "display": "Raspberry - Healthy",
        "severity": "low", "spread": "None", "crop": "Raspberry",
        "desc": "Your raspberry plants appear healthy! No disease detected. Your management practices are working well.",
        "treatments": [
            "Continue regular monitoring",
            "Prune out old canes after fruiting",
            "Maintain support trellising",
            "Apply balanced fertilizer in early spring",
            "Water consistently especially during fruiting"
        ],
        "alert": "Your raspberry crop is healthy! Continue your current excellent management.",
        "prevention": ["Regular pruning", "Good drainage", "Proper support", "Balanced nutrition"]
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display": "Orange - Citrus Greening (HLB)",
        "severity": "high", "spread": "High", "crop": "Orange / Citrus",
        "desc": "Caused by Candidatus Liberibacter bacteria spread by Asian citrus psyllid insects. Causes yellow mottling of leaves blotchy mottle pattern. Fruits are small lopsided and bitter. No cure exists.",
        "treatments": [
            "Remove and destroy all infected trees immediately",
            "Control Asian citrus psyllid with Imidacloprid systemic insecticide",
            "Apply reflective mulch to deter psyllid",
            "Inject antibiotics (Oxytetracycline) to slow disease progress",
            "Replace with certified disease-free trees from nursery"
        ],
        "alert": "Citrus greening has no cure and will kill the tree. Remove infected trees immediately to protect the rest of your orchard.",
        "prevention": ["Certified disease-free trees", "Psyllid control from planting", "Regular inspection", "Quarantine new trees"]
    }
}

DEFAULT_DISEASE = {
    "display": "Disease Detected",
    "severity": "medium", "spread": "Unknown", "crop": "Unknown",
    "desc": "The AI has detected a potential disease in your crop. Please consult your local agricultural extension officer for detailed treatment guidance specific to your region and crop variety.",
    "treatments": [
        "Remove and destroy visibly infected plant parts immediately",
        "Apply a broad-spectrum fungicide appropriate for this crop",
        "Improve drainage and air circulation around plants",
        "Avoid overhead watering - water only at the base",
        "Consult your local agriculture department for specific advice"
    ],
    "alert": "Please consult your local agricultural extension officer for region-specific treatment advice.",
    "prevention": ["Monitor crops regularly", "Practice good crop hygiene", "Maintain proper plant spacing", "Apply timely treatments"]
}

app = Flask(__name__)

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    return class_names[index], confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return "No file selected", 400
    os.makedirs("static/uploads", exist_ok=True)
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)
    result, confidence = predict_image(filepath)
    info_data = DISEASE_DB.get(result, DEFAULT_DISEASE)
    # override display name if not in db
    if result not in DISEASE_DB:
        info_data = dict(DEFAULT_DISEASE)
        info_data["display"] = result.replace("___", " - ").replace("_", " ")
        info_data["crop"] = result.split("___")[0].replace("_", " ") if "___" in result else "Unknown"
    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence, 1),
        img_path=filepath,
        info=info_data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)