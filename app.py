import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import torch
from streamlit_chat import message
import tempfile
import re

load_dotenv()

st.set_page_config(
    page_title="🏮 Binondo Heritage Guide",
    page_icon="🏮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #dc2626, #b91c1c);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        max-height: 600px;
        overflow-y: auto;
    }
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: #dc2626;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background: #b91c1c;
    }
</style>
""", unsafe_allow_html=True)

BINONDO_KNOWLEDGE = {#knowledge base
    "heritage_sites": {
        "binondo_church": {
            "name": "Binondo Church (Minor Basilica of Saint Lorenzo Ruiz)",
            "founded": "1596",
            "description": "This beautiful neo-classical church with Chinese architectural influences is dedicated to Saint Lorenzo Ruiz, the first Filipino saint and martyr. It features a baroque altar with Chinese motifs and serves as the center of Catholic worship for the Chinese-Filipino community.",
            "highlights": ["First church in Binondo", "Dedicated to first Filipino saint", "Chinese architectural influences", "Baroque altar with Chinese motifs"],
            "history": "Founded in 1596, just two years after Binondo was established, this church was built to serve the growing Catholic Chinese community. It became the spiritual center where Chinese immigrants could practice their newly adopted Catholic faith while maintaining their cultural identity.",
            "architecture": "Neo-classical design with unique Chinese architectural elements, featuring a baroque altar decorated with Chinese motifs that represent the fusion of Spanish Catholic and Chinese artistic traditions.",
            "significance": "Home to the tomb and shrine of Saint Lorenzo Ruiz, the first Filipino saint who was of Chinese-Filipino heritage, making this church a symbol of successful cultural integration."
        },
        "escolta_street": {
            "name": "Escolta Street",
            "nickname": "Queen of Streets",
            "period": "Early 1900s to 1960s",
            "description": "Manila's premier shopping district featuring Art Deco and Neoclassical buildings. Currently undergoing heritage conservation and revitalization efforts.",
            "highlights": ["Historic commercial heart of Manila", "Art Deco architecture", "Featured in Filipino literature", "Heritage conservation ongoing"],
            "history": "During the American colonial period and post-war era, Escolta Street was the most fashionable shopping destination in Manila, rivaling major commercial streets in other Asian cities. It was home to the finest shops, theaters, and restaurants.",
            "architecture": "Features stunning Art Deco and Neoclassical buildings from the early 20th century, including the iconic Capitol Theater and various heritage commercial structures.",
            "decline_and_revival": "Declined in the 1970s as commercial activity moved to other areas, but is now experiencing a renaissance through heritage conservation efforts and cultural initiatives."
        },
        "plaza_san_lorenzo": {
            "name": "Plaza San Lorenzo Ruiz",
            "established": "Spanish colonial period (late 16th century)",
            "description": "The central plaza and heart of Binondo district, featuring a monument to Saint Lorenzo Ruiz and surrounded by heritage buildings.",
            "highlights": ["Central plaza of Binondo", "Monument to Saint Lorenzo Ruiz", "Gathering place for community events", "Traditional Chinese-style landscaping"],
            "history": "Originally called Plaza Calderon de la Barca, this plaza has been the heart of Binondo since the Spanish colonial period. It was renamed in 1988 to honor Saint Lorenzo Ruiz.",
            "monument": "The monument to Saint Lorenzo Ruiz was erected in 1996 to commemorate the canonization of the first Filipino saint, who was born in Binondo to a Chinese father and Filipino mother."
        },
        "ongpin_street": {
            "name": "Ongpin Street",
            "significance": "Main commercial artery of Binondo",
            "description": "Named after Roman Ongpin, this bustling street is lined with traditional Chinese businesses, medicine shops, gold shops, restaurants, and traditional goods stores.",
            "highlights": ["Traditional Chinese medicine shops", "Gold and jewelry shops", "Chinese restaurants", "Traditional goods stores", "Chinese signage and shop houses"],
            "history": "Named after Roman Ongpin, a prominent Chinese-Filipino businessman and philanthropist who contributed significantly to the development of Binondo's commercial district.",
            "businesses": "Home to generations-old family businesses specializing in traditional Chinese medicine, gold trading, authentic Chinese cuisine, and cultural goods."
        }
    },
    "food_spots": {
        "eng_bee_tin": {
            "name": "Eng Bee Tin Chinese Deli",
            "established": "1912",
            "significance": "Oldest Chinese bakery in the Philippines",
            "specialties": ["Hopia (Chinese pastries)", "Tikoy (rice cakes)", "Chinese delicacies"],
            "description": "Over 110 years old, this historic bakery is famous for traditional Chinese pastries and treats, especially during Chinese New Year.",
            "history": "Founded in 1912 by Guan Eng Bee, this family-owned bakery started as a small shop selling traditional Chinese pastries to the Binondo community. Over four generations, it has become an institution, preserving authentic Chinese baking traditions while adapting to Filipino tastes.",
            "founder": "Guan Eng Bee, a Chinese immigrant who brought traditional pastry-making techniques from Fujian province to the Philippines.",
            "evolution": "Started with just hopia and tikoy, but expanded to include various Chinese delicacies, mooncakes, and fusion pastries that blend Chinese and Filipino flavors.",
            "cultural_impact": "Became the go-to place for Chinese New Year treats and traditional celebrations, helping preserve Chinese culinary traditions in the Filipino-Chinese community.",
            "recipes": "Many recipes are closely guarded family secrets passed down through four generations, maintaining the authentic taste that has made them famous.",
            "modern_era": "Now has multiple branches but the original Binondo location remains the flagship, still operated by the founding family."
        },
        "dong_bei": {
            "name": "Dong Bei Dumplings",
            "specialties": ["Traditional Chinese dumplings", "Fresh noodles"],
            "description": "Authentic Chinese-style dumplings that locals love, serving traditional recipes passed down through generations.",
            "history": "Established by immigrants from Northeast China (Dongbei region), bringing authentic dumpling-making techniques and recipes from their homeland.",
            "specialty": "Known for hand-made dumplings with thin, delicate wrappers and flavorful fillings that represent authentic Northern Chinese cuisine."
        },
        "ma_mon_luk": {
            "name": "Ma Mon Luk",
            "significance": "Historic noodle house",
            "specialties": ["Wonton noodles", "Chinese noodle soups"],
            "description": "Famous for their wonton noodles and traditional Chinese noodle preparations.",
            "history": "Founded by Ma Mon Luk, a Chinese immigrant who popularized wonton noodles in the Philippines. The restaurant became legendary for its authentic Cantonese-style noodle soups.",
            "legacy": "Though the original location has moved, the Ma Mon Luk name remains synonymous with quality Chinese noodles in Manila."
        },
        "cafe_mezzanine": {
            "name": "Cafe Mezzanine",
            "type": "Filipino-Chinese fusion",
            "description": "Historic restaurant serving unique Filipino-Chinese fusion cuisine, blending the best of both culinary traditions.",
            "history": "Represents the evolution of Chinese cuisine in the Philippines, creating dishes that appeal to both Chinese and Filipino palates.",
            "fusion_concept": "Pioneered the concept of Filipino-Chinese fusion, creating unique dishes that reflect the cultural blending in Binondo."
        },
        "traditional_foods": {
            "hopia": {
                "description": "Traditional Chinese pastries with sweet or savory fillings",
                "history": "Brought by Chinese immigrants from Fujian province, adapted over time to include Filipino ingredients and flavors",
                "varieties": "Mongo (mung bean), ube (purple yam), pork, and other local adaptations"
            },
            "tikoy": {
                "description": "Sticky rice cakes, especially popular during Chinese New Year",
                "significance": "Symbol of good luck and prosperity in Chinese culture",
                "tradition": "Families gather to make tikoy together during Chinese New Year preparations"
            },
            "dim_sum": {
                "description": "Traditional Chinese small plates and tea culture",
                "history": "Cantonese tradition of small dishes served with tea, adapted to local tastes in Binondo"
            },
            "char_siu": {
                "description": "Chinese roasted pork and other Cantonese specialties",
                "technique": "Traditional Cantonese barbecue methods preserved by Chinese families in Binondo"
            }
        }
    },
    "cultural_traditions": {
        "festivals": {
            "chinese_new_year": {
                "description": "Grand celebrations with dragon dances, fireworks, and traditional performances",
                "history": "Celebrated in Binondo since the 1600s, making it one of the oldest continuous Chinese New Year celebrations outside of China",
                "traditions": "Dragon and lion dances, fireworks, traditional music, and special foods like tikoy and hopia"
            },
            "mooncake_festival": {
                "description": "Mid-Autumn Festival with traditional mooncake sharing and family gatherings",
                "significance": "Celebrates family unity and harvest, with families gathering to share mooncakes and admire the full moon"
            },
            "hungry_ghost_festival": {
                "description": "Ancestral worship traditions honoring deceased family members",
                "practices": "Burning incense, offering food to ancestors, and burning ceremonial paper money"
            }
        },
        "traditional_businesses": {
            "gold_trading": {
                "description": "Historic center for gold trading and jewelry craftsmanship with intricate Chinese designs",
                "history": "Chinese immigrants brought gold trading expertise, establishing Binondo as Manila's gold trading center"
            },
            "chinese_medicine": {
                "description": "Traditional herbal medicine shops with centuries-old practices, acupuncture, and medicinal herbs",
                "tradition": "Practitioners trained in traditional Chinese medicine continue ancient healing practices"
            }
        }
    },
    "history": {
        "establishment": "1594 by Spanish colonial government",
        "significance": "World's oldest Chinatown",
        "purpose": "Settlement for Catholic Chinese immigrants",
        "age": "Over 430 years of continuous Chinese-Filipino heritage",
        "role": "Historic trading hub connecting China and the Philippines"
    }
}

# Entity mapping for dynamic responses
ENTITY_MAPPING = {
    "eng bee tin": "eng_bee_tin",
    "engbeetin": "eng_bee_tin",
    "eng bee": "eng_bee_tin",
    "dong bei": "dong_bei",
    "dongbei": "dong_bei",
    "ma mon luk": "ma_mon_luk",
    "mamonluk": "ma_mon_luk",
    "cafe mezzanine": "cafe_mezzanine",
    "binondo church": "binondo_church",
    "saint lorenzo": "binondo_church",
    "lorenzo ruiz": "binondo_church",
    "escolta": "escolta_street",
    "escolta street": "escolta_street",
    "queen of streets": "escolta_street",
    "ongpin": "ongpin_street",
    "ongpin street": "ongpin_street",
    "plaza san lorenzo": "plaza_san_lorenzo",
    "plaza": "plaza_san_lorenzo",
    "hopia": "hopia",
    "tikoy": "tikoy",
    "dim sum": "dim_sum",
    "char siu": "char_siu"
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def extract_entities(query):
    """Extract specific entities from the query"""
    query_lower = query.lower()
    found_entities = []
    
    for entity_name, entity_key in ENTITY_MAPPING.items():
        if entity_name in query_lower:
            found_entities.append((entity_name, entity_key))
    
    return found_entities

def get_entity_info(entity_key, query_type="general"):
    """Get specific information about an entity"""
    if entity_key in BINONDO_KNOWLEDGE["food_spots"]:
        entity_data = BINONDO_KNOWLEDGE["food_spots"][entity_key]
        return format_specific_food_response(entity_data, entity_key, query_type)
    
    elif entity_key in BINONDO_KNOWLEDGE["heritage_sites"]:
        entity_data = BINONDO_KNOWLEDGE["heritage_sites"][entity_key]
        return format_specific_site_response(entity_data, entity_key, query_type)

    elif entity_key in BINONDO_KNOWLEDGE["food_spots"]["traditional_foods"]:
        entity_data = BINONDO_KNOWLEDGE["food_spots"]["traditional_foods"][entity_key]
        return format_specific_traditional_food_response(entity_data, entity_key, query_type)
    
    return None

def determine_query_type(query):
    """Determine what type of information the user is asking for"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['history', 'founded', 'established', 'started', 'began', 'origin']):
        return "history"
    elif any(word in query_lower for word in ['architecture', 'building', 'design', 'structure']):
        return "architecture"
    elif any(word in query_lower for word in ['significance', 'important', 'why', 'special']):
        return "significance"
    elif any(word in query_lower for word in ['food', 'eat', 'taste', 'specialty', 'famous for']):
        return "food"
    elif any(word in query_lower for word in ['location', 'where', 'address', 'find']):
        return "location"
    else:
        return "general"

def format_specific_food_response(entity_data, entity_key, query_type):
    """Format response for specific food establishments"""
    name = entity_data.get("name", "")
    
    if query_type == "history" and entity_key == "eng_bee_tin":
        return f"""🥟 **The Rich History of {name}**

📅 **Founded in 1912** - Over 110 years of tradition!

👨‍🍳 **The Founder:**
- Started by **Guan Eng Bee**, a Chinese immigrant from Fujian province
- Brought traditional pastry-making techniques from China to the Philippines
- Began as a small shop serving the Binondo Chinese community

🏪 **Evolution Through the Decades:**
- **1912-1930s**: Small family bakery specializing in hopia and tikoy
- **1940s-1960s**: Survived WWII and expanded offerings during post-war boom
- **1970s-1990s**: Became the go-to place for Chinese New Year treats
- **2000s-Present**: Four generations later, still family-owned with multiple branches

🥮 **Cultural Impact:**
- Helped preserve authentic Chinese baking traditions in the Philippines
- Became central to Filipino-Chinese celebrations and festivals
- Many recipes remain closely guarded family secrets
- The original Binondo location is still the flagship store

**Why It's Special:**
Eng Bee Tin represents the successful preservation of Chinese culinary heritage while adapting to Filipino tastes. It's not just a bakery - it's a living piece of Binondo's history! 🏮"""

    elif query_type == "food" or query_type == "general":
        specialties = entity_data.get("specialties", [])
        return f"""🥟 **{name} - Culinary Heritage Since {entity_data.get('established', '')}**

🌟 **Famous Specialties:**
{chr(10).join([f"- 🥮 **{specialty}**" for specialty in specialties])}

📖 **What Makes It Special:**
{entity_data.get('description', '')}

🏆 **Significance:**
{entity_data.get('significance', '')}

💡 **Did You Know?**
{entity_data.get('history', 'This establishment has been serving the Binondo community for generations, preserving traditional recipes and techniques.')}

Perfect for experiencing authentic Chinese-Filipino culinary traditions! 🏮"""

    else:
        return f"""🥟 **{name}**

{entity_data.get('description', '')}

**Established:** {entity_data.get('established', 'Historic establishment')}
**Significance:** {entity_data.get('significance', '')}
**Specialties:** {', '.join(entity_data.get('specialties', []))}

A true gem of Binondo's culinary heritage! 🏮"""

def format_specific_site_response(entity_data, entity_key, query_type):
    """Format response for specific heritage sites"""
    name = entity_data.get("name", "")
    
    if query_type == "history":
        return f"""🏛️ **The History of {name}**

📅 **Founded:** {entity_data.get('founded', 'Historic period')}

📖 **Historical Background:**
{entity_data.get('history', entity_data.get('description', ''))}

🌟 **Key Historical Points:**
{chr(10).join([f"- {highlight}" for highlight in entity_data.get('highlights', [])])}

🏗️ **Architectural Significance:**
{entity_data.get('architecture', 'Features traditional architectural elements that reflect the cultural heritage of Binondo.')}

**Why It Matters:**
This site represents the rich cultural heritage and successful integration of Chinese and Filipino traditions in Binondo! 🏮"""

    elif query_type == "architecture":
        return f"""🏗️ **Architecture of {name}**

🎨 **Architectural Style:**
{entity_data.get('architecture', entity_data.get('description', ''))}

🌟 **Notable Features:**
{chr(10).join([f"- {highlight}" for highlight in entity_data.get('highlights', [])])}

📅 **Built:** {entity_data.get('founded', 'Historic period')}

**Cultural Significance:**
The architecture reflects the unique blend of Chinese, Spanish, and Filipino influences that make Binondo special! 🏮"""

    else:
        return f"""🏛️ **{name}**

📅 **Established:** {entity_data.get('founded', 'Historic period')}

📖 **Description:**
{entity_data.get('description', '')}

🌟 **Highlights:**
{chr(10).join([f"- {highlight}" for highlight in entity_data.get('highlights', [])])}

**Significance:**
{entity_data.get('significance', 'An important part of Binondo\'s cultural heritage.')}

A must-visit site to understand Binondo's rich history! 🏮"""

def format_specific_traditional_food_response(entity_data, entity_key, query_type):
    """Format response for traditional foods"""
    food_name = entity_key.replace('_', ' ').title()
    
    if isinstance(entity_data, dict):
        description = entity_data.get('description', '')
        history = entity_data.get('history', '')
        significance = entity_data.get('significance', '')
    else:
        description = entity_data
        history = ''
        significance = ''
    
    return f"""🥟 **{food_name} - Traditional Chinese-Filipino Delicacy**

📖 **What It Is:**
{description}

{f"📚 **History & Origin:**{chr(10)}{history}" if history else ""}

{f"🌟 **Cultural Significance:**{chr(10)}{significance}" if significance else ""}

**Where to Try:**
You can find authentic {food_name} at traditional establishments throughout Binondo, especially at Eng Bee Tin and other historic Chinese bakeries.

A delicious taste of Binondo's culinary heritage! 🏮"""

def get_relevant_info(query):
    """Enhanced function to get relevant information based on query"""
    query_lower = query.lower()
    
    entities = extract_entities(query)
    query_type = determine_query_type(query)
    
    if entities:
        entity_name, entity_key = entities[0]  
        specific_response = get_entity_info(entity_key, query_type)
        if specific_response:
            return specific_response
    
    if any(word in query_lower for word in ['food', 'eat', 'restaurant', 'spots', 'dining', 'cuisine']):
        return format_food_response()
    
    elif any(word in query_lower for word in ['heritage', 'sites', 'church', 'plaza', 'street', 'buildings']):
        return format_heritage_sites_response()
    
    elif any(word in query_lower for word in ['cultural', 'traditions', 'festivals', 'culture', 'events', 'celebrations']):
        return format_cultural_response()
    
    elif any(word in query_lower for word in ['history', 'oldest', 'established', 'founded', 'chinatown']):
        return format_history_response()
       
    elif any(word in query_lower for word in ['binondo church', 'saint lorenzo', 'lorenzo ruiz']):
        return format_church_response()
    
    elif any(word in query_lower for word in ['escolta', 'queen of streets']):
        return format_escolta_response()
    
    elif any(word in query_lower for word in ['ongpin', 'commercial']):
        return format_ongpin_response()
       
    elif any(word in query_lower for word in ['all', 'everything', 'comprehensive', 'overview', 'about binondo']):
        return format_comprehensive_response()
    
    else:
        return format_default_response()

def format_food_response():
    """Format response about food spots"""
    response = """🍜 **Amazing Food Spots in Binondo!**

Here are the must-visit places for authentic Chinese-Filipino cuisine:

🥟 **Eng Bee Tin Chinese Deli** (Est. 1912)
- The oldest Chinese bakery in the Philippines!
- Famous for: Hopia (Chinese pastries) and Tikoy (rice cakes)
- Perfect for traditional Chinese New Year treats

🥢 **Dong Bei Dumplings**
- Authentic Chinese-style dumplings
- Fresh noodles made daily
- Local favorite for traditional recipes

🍜 **Ma Mon Luk**
- Historic noodle house
- Famous wonton noodles and Chinese soups
- A Binondo institution

🍽️ **Cafe Mezzanine**
- Filipino-Chinese fusion cuisine
- Unique blend of both culinary traditions
- Great for experiencing cultural fusion

**Traditional Foods to Try:**
- 🥟 Hopia - Sweet or savory Chinese pastries
- 🍰 Tikoy - Sticky rice cakes (especially during Chinese New Year)
- 🥢 Dim Sum - Traditional small plates with tea
- 🍖 Char Siu - Chinese roasted meats
- 🍜 Fresh noodles and wontons

The food scene here represents over 400 years of Chinese-Filipino culinary fusion! 🏮"""
    
    return response

def format_heritage_sites_response():
    """Format response about heritage sites"""
    response = """🏛️ **Binondo's Amazing Heritage Sites!**

Discover over 430 years of history in these iconic locations:

⛪ **Binondo Church (Minor Basilica of Saint Lorenzo Ruiz)**
- Founded: 1596 (just 2 years after Binondo!)
- Dedicated to Saint Lorenzo Ruiz, the first Filipino saint
- Beautiful neo-classical architecture with Chinese influences
- Features a baroque altar with Chinese motifs

🏛️ **Plaza San Lorenzo Ruiz**
- The heart and central plaza of Binondo
- Monument to Saint Lorenzo Ruiz (erected 1996)
- Gathering place for community events and celebrations
- Traditional Chinese-style landscaping

🛍️ **Escolta Street - "Queen of Streets"**
- Manila's premier shopping district (1900s-1960s)
- Beautiful Art Deco and Neoclassical buildings
- Currently undergoing heritage conservation
- Featured in Filipino literature and films

🏪 **Ongpin Street**
- Main commercial artery of Binondo
- Traditional Chinese businesses line the street
- Gold shops, medicine stores, restaurants
- Bustling atmosphere with Chinese signage

Each site tells the story of how Chinese immigrants built their community while preserving their heritage! 🏮"""
    
    return response

def format_cultural_response():
    """Format response about cultural traditions"""
    response = """🎭 **Rich Cultural Traditions of Binondo!**

Experience 430+ years of living Chinese-Filipino culture:

🎊 **Major Festivals:**
- 🧧 **Chinese New Year** - Grand celebrations with dragon dances, fireworks, and traditional performances
- 🥮 **Mooncake Festival** - Mid-Autumn celebration with family gatherings and mooncake sharing
- 👻 **Hungry Ghost Festival** - Ancestral worship honoring deceased family members
- 🐉 **Dragon Boat Festival** - Cultural performances and traditional foods

🏪 **Traditional Businesses:**
- 💰 **Gold Trading** - Historic center with intricate Chinese jewelry designs
- 🌿 **Chinese Medicine** - Herbal shops with centuries-old practices and acupuncture
- ✍️ **Calligraphy** - Traditional Chinese brush painting and custom calligraphy
- 📜 **Paper Goods** - Ceremonial items for ancestral worship and festivals

🗣️ **Living Culture:**
- Languages: Hokkien Chinese, Filipino, and English spoken daily
- Family businesses spanning multiple generations
- Unique blend of Catholic faith with Chinese ancestral traditions
- Traditional architecture mixed with modern adaptations

This isn't just history - it's a living, breathing culture that continues today! 🏮"""
    
    return response

def format_history_response():
    """Format response about Binondo's history"""
    response = """📚 **The Fascinating History of Binondo!**

🏮 **World's Oldest Chinatown - Since 1594!**

**The Beginning:**
- Established in 1594 by the Spanish colonial government
- Created as a settlement for Catholic Chinese immigrants
- That's over 430 years of continuous heritage!

**Why It's Special:**
- First Chinatown in the world (predates San Francisco's by over 250 years!)
- Built for Chinese who converted to Christianity
- Became a major trading hub connecting China and the Philippines
- Survived Spanish colonization, American occupation, Japanese invasion, and modernization

**Cultural Significance:**
- Home to Saint Lorenzo Ruiz, the first Filipino saint (Chinese-Filipino heritage)
- Preserved Chinese traditions while adapting to Filipino culture
- Created unique Chinese-Filipino fusion in food, architecture, and customs

**Today:**
- Still a thriving community with original families' descendants
- Maintains traditional businesses alongside modern establishments
- Living testament to successful cultural integration
- UNESCO recognition for its historical and cultural value

From a small settlement for Chinese Catholics to the world's oldest Chinatown - Binondo's story is truly remarkable! 🏮"""
    
    return response

def format_church_response():
    """Format specific response about Binondo Church"""
    response = """⛪ **Binondo Church - A Sacred Heritage Site!**

**Minor Basilica of Saint Lorenzo Ruiz**

🏛️ **Historical Significance:**
- Founded in 1596 (just 2 years after Binondo was established!)
- First church built in Binondo
- Dedicated to Saint Lorenzo Ruiz, the first Filipino saint and martyr

✨ **Architectural Beauty:**
- Neo-classical style with unique Chinese architectural influences
- Beautiful baroque altar featuring Chinese motifs
- Religious art blending Filipino, Chinese, and Spanish styles
- Historical artifacts from the Spanish colonial period

🙏 **Cultural Importance:**
- Center of Catholic worship for the Chinese-Filipino community
- Houses the tomb and shrine of Saint Lorenzo Ruiz
- Represents the successful blend of Chinese culture with Catholic faith
- Site of important community celebrations and religious festivals

**Why Visit:**
The church is a perfect example of how Binondo successfully blended different cultures. You'll see Chinese design elements in a Catholic church, representing the unique identity of Chinese-Filipino Catholics who built this community over 400 years ago! 🏮"""
    
    return response

def format_escolta_response():
    """Format specific response about Escolta Street"""
    escolta_data = BINONDO_KNOWLEDGE["heritage_sites"]["escolta_street"]
    
    response = f"""🛍️ **Escolta Street - {escolta_data['nickname']}**

**Historic "Queen of Streets"**

- **Period**: {escolta_data['period']}
- **Description**: {escolta_data['description']}
- **Highlights**:
  - {', '.join(escolta_data['highlights'])}

Escolta Street is a must-visit for its rich history and stunning architecture. Explore its Art Deco and Neoclassical buildings and experience the vibrant shopping culture that has defined Manila for over a century! 🏮"""
    
    return response

def format_ongpin_response():
    """Format specific response about Ongpin Street"""
    ongpin_data = BINONDO_KNOWLEDGE["heritage_sites"]["ongpin_street"]
    
    response = f"""🏪 **Ongpin Street - {ongpin_data['significance']}**

**Main Commercial Artery of Binondo**

- **Description**: {ongpin_data['description']}
- **Highlights**:
  - {', '.join(ongpin_data['highlights'])}

Ongpin Street is the heart of Binondo's commercial district, offering a unique blend of traditional Chinese businesses and modern conveniences. From gold shops to medicine stores, it's a bustling street that showcases the rich cultural heritage of Binondo! 🏮"""
    
    return response

def format_comprehensive_response():
    """Format comprehensive response about everything"""
    response = """🏮 **Complete Guide to Binondo - World's Oldest Chinatown!**

**🏛️ HERITAGE SITES:**
⛪ Binondo Church (1596) - First Filipino saint's basilica
🏛️ Plaza San Lorenzo Ruiz - Central heritage plaza  
🛍️ Escolta Street - Historic "Queen of Streets"
🏪 Ongpin Street - Main commercial artery

**🍜 MUST-TRY FOOD:**
🥟 Eng Bee Tin (1912) - Oldest Chinese bakery, famous hopia
🥢 Dong Bei Dumplings - Authentic Chinese dumplings
🍜 Ma Mon Luk - Historic wonton noodles
🍽️ Traditional: Tikoy, dim sum, char siu, fresh noodles

**🎭 CULTURAL TRADITIONS:**
🧧 Chinese New Year - Dragon dances & fireworks
🥮 Mooncake Festival - Mid-Autumn celebrations  
💰 Gold trading - Traditional jewelry craftsmanship
🌿 Chinese medicine - Herbal shops & acupuncture

**📚 AMAZING HISTORY:**
- Established 1594 - Over 430 years old!
- World's oldest Chinatown
- Created for Catholic Chinese immigrants
- Survived colonization while preserving heritage

**Why Binondo is Special:**
It's not just a tourist destination - it's a living, breathing community where 400+ years of Chinese-Filipino culture continues to thrive. From traditional businesses run by the same families for generations to festivals that blend Catholic and Chinese traditions, Binondo is truly unique! 🏮"""
    
    return response

def format_default_response():
    """Default response for unclear queries"""
    response = """🏮 **Welcome to Binondo Heritage Guide!**

I'm here to help you discover the amazing world of Binondo - the world's oldest Chinatown! 

**What would you like to know about?**

🏛️ **Heritage Sites** - Churches, plazas, historic streets
🍜 **Food & Restaurants** - Traditional cuisine and famous spots  
🎭 **Cultural Traditions** - Festivals, customs, and practices
📚 **History** - How Binondo became the world's oldest Chinatown
⛪ **Specific Sites** - Binondo Church, Escolta Street, Ongpin Street

**Try asking:**
- "Tell me about food spots in Binondo"
- "What are the heritage sites?"
- "What's the history of Binondo?"
- "What cultural festivals happen here?"
- "What is the history of Eng Bee Tin?"

I'm excited to share the rich 430+ year heritage of this amazing district with you! 🏮"""
    
    return response

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏮 Binondo Heritage Guide</h1>
        <p>Discover the rich cultural heritage of the world's oldest Chinatown</p>
        <p><strong>CPE124 Group 1 - AI-Powered Heritage Guide</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    with st.sidebar:
        st.markdown("### 🏛️ Heritage Sites")
        st.markdown("""
        <div class="sidebar-info">
        • <strong>Binondo Church</strong><br>
          Minor Basilica of Saint Lorenzo Ruiz<br><br>
        • <strong>Escolta Street</strong><br>
          Historic "Queen of Streets"<br><br>
        • <strong>Plaza San Lorenzo Ruiz</strong><br>
          Central heritage plaza<br><br>
        • <strong>Ongpin Street</strong><br>
          Main commercial artery
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🥟 Traditional Food")
        st.markdown("""
        <div class="sidebar-info">
        • <strong>Hopia & Tikoy</strong><br>
          Traditional pastries<br><br>
        • <strong>Dumplings</strong><br>
          Authentic Chinese style<br><br>
        • <strong>Chinese Tea</strong><br>
          Traditional tea houses
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎭 Cultural Events")
        st.markdown("""
        <div class="sidebar-info">
        • <strong>Chinese New Year</strong><br>
          Grand celebrations<br><br>
        • <strong>Mooncake Festival</strong><br>
          Mid-Autumn celebration<br><br>
        • <strong>Cultural Shows</strong><br>
          Traditional performances
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ℹ️ Did You Know?")
        st.markdown("""
        <div class="sidebar-info">
        • Binondo is 430+ years old<br>
        • World's oldest Chinatown<br>
        • Home to first Filipino saint<br>
        • Historic trading hub since 1594
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # chat css
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # chat messages
        if not st.session_state.messages:
            st.markdown("""
            ### 👋 Welcome to Binondo!
            I'm your heritage guide for the world's oldest Chinatown! Ask me about:
            
            **🍜 Food Spots** - "Give me food spots in Binondo"
            **🏛️ Heritage Sites** - "What are Binondo's heritage sites?"  
            **🎭 Cultural Traditions** - "Tell me about cultural festivals"
            **📚 History** - "How did Binondo become the oldest Chinatown?"
            **⛪ Specific Sites** - "Tell me about Binondo Church"
            **🥟 Specific Places** - "What is the history of Eng Bee Tin?"
            """)
        
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}")
            else:
                message(msg["content"], key=f"bot_{i}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # user input
        user_input = st.chat_input("Ask about Binondo's heritage sites, food, or cultural traditions...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                bot_response = get_relevant_info(user_input)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
            st.rerun()
    
   

if __name__ == "__main__":
    main()
