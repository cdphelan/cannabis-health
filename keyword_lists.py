#for the first "rough cut" of filtering down the full dataset
filtering_phrases = [
    "marijuana", "cannabis", "weed", "ganja", "reefer", "nug", "flower", 
"hash", "hashish", "kush", "sticky icky", "Mary Jane",
"THC", "CBD", "CBN", "CBG", "delta 8", "delta-8", "delta 9", "delta-9", 
"HHC", "THCV","RSO", "Rick Simpson Oil", "live resin", "distillate", 
"terpenes", "rosin", "tincture", "pre-roll", "hemp", 
"reggies", "mmj",
    "CBC", "THCA", "CBDA","CBDV","delta-10", "delta 10", "HHC", "HHCO",
"THCP", "CBL", "CBE", "exo-THC", "cannabinoids", "cannabinoid", 
"full spectrum extract", "broad spectrum extract", "entourage effect",     
"endocannabinoids", "anandamide", "2-AG", "CB1 receptor", 
"CB2 receptor","phytocannabinoids","terpenoids", "flavonoids", 
"cannabidiol",
    "pre-rolls", "blunts", "bongs", "dab rigs", "edibles", "edible gummies", 
"resin", "budder", "smokeables", "cannabutter", "cannaoil", 
"dry herb vaporizer", "concentrate vaporizer",
    "Blue Dream", "Sour Diesel", "OG Kush", "Girl Scout Cookies", "Pineapple Express", 
    "Granddaddy Purple", "White Widow", "AK-47", "Northern Lights", "Green Crack", "Trainwreck", 
    "Gorilla Glue", "Skywalker OG", "Purple Haze", "Maui Wowie", "Jack Herer", "Lemon Haze", 
    "Durban Poison", "Gelato", "Wedding Cake","Runtz", "Zkittlez", "Do-Si-Dos", "Super Silver Haze", 
    "Strawberry Cough", "Bubba Kush", "Cherry Pie", "Forbidden Fruit", "Bruce Banner", "Amnesia Haze",
    "Banana Kush", "LA Confidential", "Chemdawg", "Critical Mass", "Afghan Kush", "NYC Diesel", "Harlequin", 
    "ACDC", "Cannatonic", "Pennywise", "CBD Critical Mass", "Charlotte’s Web", "Sweet and Sour Widow", 
    "Lifter", "Elektra", "Suver Haze", "Hawaiian Haze", "Special Sauce", "Sour Space Candy",
    "indica", "sativa", "hybrid", "autoflowering", "photoperiod", "trichomes", "resinous", "dense buds", 
    "balanced strain", "medical strain", "recreational strain"]

#general cannabis terms
phrases = [
    "marijuana", "cannabis", "weed", "ganja", "reefer", 
    "nug", "flower", "hash", "hashish", "kush", "dank", "sticky icky", "Mary Jane",
    "THC", "CBD", "CBN", "CBG", "delta 8", "delta-8", "delta 9", "delta-9", "HHC", "THCV",
    "RSO", "Rick Simpson Oil", "live resin", "distillate",  "terpenes", 
     "rosin", "tincture", "pre-roll", "hemp", "reggies", "mmj"
] # removed: "pot", "bud", "herb", "dope", "grass", "green", "loud", "concentrate", "full spectrum", "broad spectrum", "shatter", "wax", "crumble", "sauce", "diamonds", "medical marijuana"

compounds = [
    "THC", "CBD", "CBN", "CBG", "CBC", "THCA", "CBDA", "THCV", "CBDV", "delta-8 THC", "delta 8", "delta-8", "delta 9", "delta-9",
    "delta-10", "HHC", "HHCO", "THCP", "CBL", "CBT", "CBE", "exo-THC", "cannabinoids",
    "acidic cannabinoids", "minor cannabinoids", "neutral cannabinoids", "cannabinoid acids", "cannabinoid analogs",
    "synthetic cannabinoids", "full spectrum extract", "broad spectrum extract",  "nanoemulsified THC",
    "nanoemulsified CBD", "liposomal CBD", "entourage effect", "psychoactive cannabinoids", "non-psychoactive cannabinoids",
    "endocannabinoids", "anandamide", "2-AG", "receptor agonist", "CB1 receptor", "CB2 receptor",
    "cannabinoid profile", "cannabinoid ratio", "cannabinoid content", "plant-derived cannabinoids", "phytocannabinoids",
    "terpenoids", "flavonoids", "cannabis metabolites", "cannabinoid tolerance", "cannabidiol"
] # removed: "isolate",

compounds_short = [
    "THC", "CBD", "CBN", "CBG", "CBC", "THCA", "CBDA", "THCV", "CBDV", "delta-8 THC", "delta 8", "delta-8", "delta 9", "delta-9",
    "delta-10", "HHC", "HHCO", "THCP", "CBL", "CBT", "CBE", "exo-THC", "cannabinoids", "marijuana", "cannabis", "weed"
]

ingestion_methods = [
    "smoking", "vaping", "edibles", "tinctures", "topicals", "transdermal patches", "dabbing", 
    "capsules", "sublingual drops", "tea",
    "infused beverages", "infused foods", "pre-rolls", "joints", "blunts", "pipes", "bongs", "gravity bongs", 
    "vaporizers", "dab rigs",
    "edible gummies", "cookies", "brownies", "chocolates", "syrup", "lozenges", "sprays", "cannabis oil", 
    "hash oil", "resin",
    "sugar wax", "budder", "live resin", "rosin", "suppositories", "bath bombs", "topical salves", "inhalers", 
    "oral sprays", "smokeables",
    "infused honey", "cannabutter", "cannaoil", "dry herb vaporizer", "concentrate vaporizer", "infused sugar", 
    "capsule form", "nano THC"
]

strains = [
    "Blue Dream", "Sour Diesel", "OG Kush", "Girl Scout Cookies", "Pineapple Express", "Granddaddy Purple", "White Widow", 
    "AK-47", "Northern Lights", "Green Crack",
    "Trainwreck", "Gorilla Glue", "Skywalker OG", "Purple Haze", "Maui Wowie", "Jack Herer", "Lemon Haze", "Durban Poison", 
    "Gelato", "Wedding Cake",
    "Runtz", "Zkittlez", "Do-Si-Dos", "Super Silver Haze", "Strawberry Cough", "Bubba Kush", "Cherry Pie", "Forbidden Fruit", 
    "Bruce Banner", "Amnesia Haze",
    "Banana Kush", "LA Confidential", "Chemdawg", "Critical Mass", "Afghan Kush", "NYC Diesel", "Harlequin", "ACDC", 
    "Cannatonic", "Pennywise",
    "CBD Critical Mass", "Charlotte’s Web", "Remedy", "Sweet and Sour Widow", "Lifter", "Elektra", "Suver Haze", 
    "Hawaiian Haze", "Special Sauce", "Sour Space Candy"
]

dosages = [
    "microdose", "low dose", "moderate dose", "high dose", "10mg THC", "20mg THC", "5mg THC", "2.5mg THC", 
    "50mg THC", "100mg THC",
    "1:1 THC:CBD", "2:1 THC:CBD", "20:1 CBD:THC", "5mg CBD", "10mg CBD", "25mg CBD", "50mg CBD", "0.3% THC", 
    "15% THC", "25% THC",
    "milligrams of THC", "milligrams of CBD", "CBD per serving", "THC per serving", "tolerance-based dosing", 
    "weight-based dosing",
    "bodyweight dosing", "individualized dosing", "start low and go slow", "effective dose", "edible dosage", 
    "vape dosage", "tincture dosage",
    "therapeutic dose", "psychoactive dose", "threshold dose", "sublingual dosage", "onset time", "duration", 
    "peak effects",
    "bioavailability", "half-life of THC", "half-life of CBD", "metabolism rate", "slow-release", "fast-acting", 
    "nano-dosing", "CBD isolate dose"
]

traits = [
    "indica", "sativa", "hybrid", "autoflowering", "photoperiod", "terpene profile", "cannabinoid profile", "aroma", "flavor", "color",
    "trichomes", "resinous", "dense buds", "sticky", "purple hues", "citrus notes", "earthy smell", "skunky", "sweet", "pine",
    "high THC", "high CBD", "balanced strain", "mellow", "energizing", "couch-lock", "euphoric", "creative", "relaxing", "uplifting",
    "anxiety-relief", "pain-relief", "anti-inflammatory", "appetite stimulant", "sedative", "head high", "body high", "entourage effect",
    "potency", "yield", "flowering time", "THC content", "CBD content", "flavor profile", "medical strain", "recreational strain", "cultivar", "heritage"
]

topicals = [
    "cream", "salve", "ointment", "lotion", "balm", "gel", "patch", "transdermal patch",
    "topical", "spray", "roll-on", "infused cream", "CBD cream", "THC cream",
    "CBD balm", "hemp balm", "CBD lotion", "infused lotion", "cannabis cream",
    "cannabis oil", "massage oil", "pain cream", "pain balm", "CBD gel", "CBD rub",
    "hemp salve", "cooling gel", "warming balm", "CBD roller", "topical CBD",
    "CBD stick", "THC balm", "CBD patch", "CBD topical", "CBD spray",
    "aromatherapy", "essential oil", "hemp cream", "THC salve", "numbing cream"
]

symptoms = ["anxiety", "pain", "sleep", "nausea", "appetite", "depression", "insomnia", "ptsd", "focus", "stress", "mood", "inflammation"]

