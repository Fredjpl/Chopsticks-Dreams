# Chopsticks-Dreams
```bash
export OPENAI_API_KEY="sk-..."      # https://platform.openai.com/account/api-keys
export COHERE_API_KEY="c0h3r3..."   # https://dashboard.cohere.com/api-keys
```
# Build the indexes
```
pip install -r requirements.txt
python -m tools.rag.build_index

```
# Ask the chef
```
python3 -m tools.recipe_rag
```
