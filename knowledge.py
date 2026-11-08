"""
Disease knowledge base for AI Plant Doctor.
Keys use the PlantVillage-style  Crop___Disease  naming.
Edit / extend this file to add more diseases, medicines or tips.
"""

disease_treatments = {
    "Apple___Apple_scab": {
        "medicines": "Copper fungicide, Mancozeb, Captan",
        "treatment": "Spray fungicides during early spring. Remove fallen leaves.",
        "suggestions": "Use resistant varieties, prune dense branches.",
        "nutrients": "Balanced NPK, emphasis on Calcium"
    },
    "Apple___Black_rot": {
        "medicines": "Thiophanate-methyl, Myclobutanil",
        "treatment": "Remove infected fruit mummies and cankers.",
        "suggestions": "Avoid overhead watering, prune infected limbs.",
        "nutrients": "Potassium rich fertilizer"
    },
    "Apple___Cedar_apple_rust": {
        "medicines": "Mancozeb, Myclobutanil",
        "treatment": "Apply fungicides before petal fall.",
        "suggestions": "Remove nearby juniper trees when possible.",
        "nutrients": "Maintain balanced NPK"
    },
    "Apple___healthy": {
        "medicines": "No treatment needed",
        "treatment": "Maintain proper watering & fertilizing.",
        "suggestions": "Prevent overwatering & monitor routinely.",
        "nutrients": "Balanced NPK"
    },
    "Blueberry___healthy": {
        "medicines": "No disease present",
        "treatment": "Good soil drainage, pH 5 – 5.5 recommended.",
        "suggestions": "Mulch and prune old stems.",
        "nutrients": "Acid-forming fertilizers (Ammonium sulfate)"
    },
    "Cherry___healthy": {
        "medicines": "No treatment needed",
        "treatment": "Balanced fertilizer, remove weeds.",
        "suggestions": "Ensure good sunlight and air flow.",
        "nutrients": "Balanced NPK"
    },
    "Cherry___Powdery_mildew": {
        "medicines": "Sulfur, Potassium bicarbonate",
        "treatment": "Spray fungicide at first sign of powder.",
        "suggestions": "Avoid overhead irrigation.",
        "nutrients": "Avoid excessive Nitrogen"
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "medicines": "Strobilurin fungicides, Propiconazole",
        "treatment": "Apply at VT stage (tasseling).",
        "suggestions": "Rotate crops, use resistant hybrids.",
        "nutrients": "Balanced NPK"
    },
    "Corn___Common_rust": {
        "medicines": "Triazole fungicides (Propiconazole)",
        "treatment": "Spray when rust pustules appear.",
        "suggestions": "Grow rust-resistant varieties.",
        "nutrients": "Zinc and Manganese"
    },
    "Corn___Northern_Leaf_Blight": {
        "medicines": "Azoxystrobin, Pyraclostrobin",
        "treatment": "Apply fungicides at tasseling.",
        "suggestions": "Use resistant seed, field sanitation.",
        "nutrients": "Balanced NPK"
    },
    "Corn___healthy": {
        "medicines": "No treatment needed",
        "treatment": "Maintain nitrogen and spacing.",
        "suggestions": "Avoid waterlogging.",
        "nutrients": "High Nitrogen"
    },
    "Grape___Black_rot": {
        "medicines": "Myclobutanil, Captan",
        "treatment": "Remove mummified berries and prune infected shoots.",
        "suggestions": "Improve air circulation, avoid overhead irrigation.",
        "nutrients": "Potassium and Magnesium"
    },
    "Grape___Esca_(Black_Measles)": {
        "medicines": "No cure, only prevention",
        "treatment": "Remove infected vines, disinfect pruning tools.",
        "suggestions": "Avoid drought stress.",
        "nutrients": "Boron and Zinc"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "medicines": "Mancozeb, Copper-based fungicide",
        "treatment": "Apply fungicide in early infection stages.",
        "suggestions": "Avoid wet foliage.",
        "nutrients": "Balanced NPK"
    },
    "Grape___healthy": {
        "medicines": "No disease",
        "treatment": "Regular pruning and fertigation.",
        "suggestions": "Maintain good air circulation.",
        "nutrients": "Balanced NPK"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "medicines": "No chemical cure",
        "treatment": "Remove infected trees immediately.",
        "suggestions": "Control psyllid insect using imidacloprid.",
        "nutrients": "Foliar Zinc, Manganese, and Boron"
    },
    "Peach___Bacterial_spot": {
        "medicines": "Copper fungicides, Oxytetracycline",
        "treatment": "Apply copper during dormancy.",
        "suggestions": "Use resistant cultivars.",
        "nutrients": "Calcium"
    },
    "Peach___healthy": {
        "medicines": "None",
        "treatment": "Maintain proper soil moisture.",
        "suggestions": "Use organic fertilizers.",
        "nutrients": "Balanced NPK"
    },
    "Pepper,_bell___Bacterial_spot": {
        "medicines": "Copper-based sprays, Streptomycin",
        "treatment": "Spray weekly during wet conditions.",
        "suggestions": "Rotate crops, avoid overhead irrigation.",
        "nutrients": "Calcium and Magnesium"
    },
    "Pepper,_bell___healthy": {
        "medicines": "Not applicable",
        "treatment": "Balanced NPK every 15 days.",
        "suggestions": "Proper sunlight and watering.",
        "nutrients": "Balanced NPK"
    },
    "Potato___Early_blight": {
        "medicines": "Chlorothalonil, Mancozeb",
        "treatment": "Spray at first appearance of leaf spots.",
        "suggestions": "Remove infected leaves.",
        "nutrients": "Potassium and Calcium"
    },
    "Potato___Late_blight": {
        "medicines": "Metalaxyl, Cymoxanil",
        "treatment": "Apply fungicides during cool/wet weather.",
        "suggestions": "Destroy infected tubers.",
        "nutrients": "Balanced NPK"
    },
    "Potato___healthy": {
        "medicines": "None",
        "treatment": "Maintain proper soil drainage.",
        "suggestions": "Rotate crops every 2-3 years.",
        "nutrients": "Potassium"
    },
    "Raspberry___healthy": {
        "medicines": "No disease",
        "treatment": "Organic compost, rooting hormone spray.",
        "suggestions": "Prune old canes.",
        "nutrients": "Balanced NPK"
    },
    "Soybean___healthy": {
        "medicines": "None",
        "treatment": "Maintain fertilizers and irrigation.",
        "suggestions": "Pest monitoring recommended.",
        "nutrients": "Phosphorus and Potassium"
    },
    "Squash___Powdery_mildew": {
        "medicines": "Sulfur, Neem oil, Bicarbonate spray",
        "treatment": "Spray early morning or evening.",
        "suggestions": "Increase spacing, remove infected leaves.",
        "nutrients": "Balanced NPK"
    },
    "Strawberry___Leaf_scorch": {
        "medicines": "Copper fungicides",
        "treatment": "Apply fungicide before fruiting.",
        "suggestions": "Use drip irrigation.",
        "nutrients": "Calcium"
    },
    "Strawberry___healthy": {
        "medicines": "Not required",
        "treatment": "Fertilize with NPK 10-10-10",
        "suggestions": "Avoid waterlogging.",
        "nutrients": "Balanced NPK"
    },
    "Tomato___Bacterial_spot": {
        "medicines": "Copper sprays, Streptomycin",
        "treatment": "Apply every 7–10 days.",
        "suggestions": "Avoid working on wet plants.",
        "nutrients": "Calcium and Magnesium"
    },
    "Tomato___Early_blight": {
        "medicines": "Mancozeb, Chlorothalonil",
        "treatment": "Spray fungicide every 14 days.",
        "suggestions": "Remove old infected leaves.",
        "nutrients": "Potassium and Magnesium"
    },
    "Tomato___healthy": {
        "medicines": "None",
        "treatment": "Provide support stakes.",
        "suggestions": "Mulch soil to avoid fungus splash.",
        "nutrients": "Balanced NPK"
    },
    "Tomato___Late_blight": {
        "medicines": "Fluazinam, Metalaxyl",
        "treatment": "Spray during humidity outbreaks.",
        "suggestions": "Burn infected plant parts.",
        "nutrients": "Potassium and Calcium"
    },
    "Tomato___Leaf_Mold": {
        "medicines": "Copper oxychloride, Chlorothalonil",
        "treatment": "Spray at first mold patches.",
        "suggestions": "Increase airflow, reduce humidity.",
        "nutrients": "Avoid excessive Nitrogen"
    },
    "Tomato___Septoria_leaf_spot": {
        "medicines": "Mancozeb, Copper fungicide",
        "treatment": "Start spraying when spots appear.",
        "suggestions": "Remove bottom leaves.",
        "nutrients": "Balanced NPK"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "medicines": "Neem oil, Abamectin",
        "treatment": "Spray underside of leaves.",
        "suggestions": "Maintain humidity to reduce mites.",
        "nutrients": "Silicon"
    },
    "Tomato___Target_Spot": {
        "medicines": "Copper oxychloride, Mancozeb",
        "treatment": "Start fungicide before fruiting.",
        "suggestions": "Avoid leaf wetness.",
        "nutrients": "Potassium"
    },
    "Tomato___Tomato_mosaic_virus": {
        "medicines": "No cure for virus",
        "treatment": "Remove affected plants completely.",
        "suggestions": "Use virus-free seeds, disinfect tools.",
        "nutrients": "Avoid stress and balance nutrients"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "medicines": "No chemical cure",
        "treatment": "Remove infected plants & control whiteflies.",
        "suggestions": "Use resistant cultivars & net protection.",
        "nutrients": "Avoid excessive Nitrogen"
    }
}