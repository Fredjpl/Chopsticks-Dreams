# Cook From What You Have: Your AI Kitchen Assistant for Chinese Recipes

## Description

**Cook From What You Have** is an **AI-powered recipe recommendation agent** that helps users discover authentic Chinese dishes based on ingredients they already have. By simply uploading a photo of their groceries or a shopping receipt, users receive customized recipe suggestions, missing ingredient lists, and smart substitutes—bridging daily cooking with intelligent assistance. 

### Target Audience

This tool is ideal for **home cooks, students, and busy individuals** who want to make the most of their available ingredients while minimizing food waste and enjoying culturally authentic meals.

### Access the Web Interface ✨🌐

To explore the application, follow these steps:

1. Open your web browser.
2. Navigate to the public IP: **[[https://chopsticks-dreams-g6b9bvcvhbcbedec.centralus-01.azurewebsites.net/](https://chopsticks-dreams-g6b9bvcvhbcbedec.centralus-01.azurewebsites.net/)]**
3. Explore the features: upload a grocery photo, speak to the assistant, or ask for recipes.

![Web Interface Image](https://raw.githubusercontent.com/Fredjpl/Chopsticks-Dreams/main/imgs/web_ui.png)

## User Manual

### Use cases

You can explore our agent using the following scenarios:

- “**I have eggs and peppers, what can I cook?**”
  - Upload a photo of your ingredients or type them in. The assistant will suggest recipes based on these items.
- “**I want to learn how to cook** green peppers and onions.”
  - Simply state your desired dish or ingredients. The assistant will provide recipe options and necessary ingredients.
- “**I want to know more about** stir-fried green peppers and onions.”
  - Ask the assistant for details on a specific recipe, including preparation steps and cooking tips.
- **I want to buy the missing ingredients.**
  - After receiving recipe suggestions, the assistant will list any missing ingredients and recommend nearby stores.

## Features

1. **Visual Ingredient Input**: Users can upload a **photo of ingredients** or **shopping receipt**.

2. **Recipe Recommendation**: The agent uses RAG (Retrieval-Augmented Generation) to suggest **Chinese recipes** based on detected ingredients. It also provides: 

   - A list of **missing ingredients**, and
   - A list of **possible substitutions**.

   In PFD parsing section, it offers paddleOCR and a sliding window approach for effective document parsing. To enhance retrieval, we combined Dense (M3E) and Sparse (BGE-M3) strategies with Faiss, BM25, and BGE-Reranker, achieving Q&A accuracy improvements of over 9% compared to GPT-4 with external knowledge.

3. **Voice Interaction**: Users can **speak their ingredients or ask relevant cooking questions**. The agent supports **speech-to-text and text-to-speech** via **Azure Whisper**, enabling a hands-free, conversational experience.

4. **Cooking Video & Smart Grocery Suggestions**: Our agent can recognize user's intention and interest in a recipe, then fetches **relevant YouTube videos** using the **YouTube Data API** to provide visual step-by-step cooking guidance and recommends **nearby grocery stores** where you can buy missing items.

## How to Run

### Deployment

Our AI assistant is deployed on an Amazon EC2 instance and is accessible at: **[http://3.19.22.54:5000](http://3.19.22.54:5000/)**. The backend runs on `Flask`, and the server listens on port `5000`. 

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/Fredjpl/Chopsticks-Dreams.git
    cd Chopsticks-Dreams
    ```
2. Create and activate a Conda environment:
    ```bash
    conda create --name chef_agent
    conda activate chef_agent
    
    # Install required Python dependencies
    pip install -r requirements.txt
    ```
3. Set up environment variables:
    ```bash
    # export your openai api key first
    export OPENAI_API_KEY="sk-…"
    # Google map api
    export GOOGLEMAP_API=""
    # Google service api to get fetch youtube videos
    export GOOGLE_API=""
    # Whisper api, the servise is deployed on Azure portal
    export SPEECH_KEY=""
    # Whisper api servise location
    export SPEECH_REGION =""
    # Coherence has the top-level bilingual rerank model
    export COHERE_API_KEY=""
    ```
### Launch the Program
Run the server with:
```bash
python -m server.server
```

## Innovation

While there are existing recipe search tools such as *[Supercook](https://www.supercook.com/#/desktop)*, they typically require manual ingredient selection through a clunky UI, and users have reported unreliable recipe results. Some AI-based apps generate recipes, but they often lack grounding in real ingredients. 

Our agent stands out by combining **multimodal input (CV + voice)**, **retrieval-augmented generation (RAG)** with **GPT-o4-mini**, and **localized grocery recommendations** into a seamless user workflow. To our knowledge, this is the first agent that offers a **context-aware, Chinese-cuisine-focused cooking assistant** based on real-world inputs like receipts or ingredient photos.

## **Practical Value & Scalability**

This agent is highly practical for students, busy professionals, or families who want to cook efficiently while minimizing food waste. This agent can easily be extended beyond Chinese cuisine to other culinary traditions, and integrated into smart kitchen devices or meal planning apps. Organizations such as meal-kit services or grocery platforms could incorporate this system to personalize recommendations based on what users actually have, demonstrating a wide range of commercial applications.

## Acknowledgement

- The recipe dataset used in our RAG system is sourced from this [GitHub repository](https://github.com/Anduin2017/HowToCook), which provides a comprehensive collection of Chinese recipes in structured Markdown format.
- We also reviewed user feedback on existing tools such as *[Supercook](https://www.reddit.com/r/cookingforbeginners/comments/l8ru1z/am_i_misusing_supercook/)* to better understand common pain points and inform the design of a more user-friendly and reliable alternative.
