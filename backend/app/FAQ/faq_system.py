import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


##### POTREBA DA SE NAPRAVI DATASET I VO SOODNOS DA SE SMENAT PATH-OT DO DATASET-OT, 
# I DA SE INSTALIRAAT NEKOI OD PAKETITE KOI NE SE NAOGJAAT VO REQUIREMENTS.TXT
# ISTO I DA SE SMENAT QUERY-TO ZA TESTIRANJE NA RETRIEVAL SISTEMOT

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
faq_df = pd.read_csv("faq_dataset.csv")

### 1. Load model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

### 2.Creating embeddings for all FAQ questions
questions = faq_df['question'].tolist()

faq_embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Save embeddings for faster loading next time (it's just cache)
np.save('faq_embeddings.npy', faq_embeddings)

### 3. Retrieval Function 
# Here we retrive the top most relevant FAQs for our query
# As arguments we pass the User's query, the number of results to retrun, and the minimum similarity score/threshold
# We return the list of dictionaries with the question,answer,score,confidence
def retrieve_faq(query, top_k=3, threshold=0.0):
    # Encode query
    query_emb = model.encode([query], normalize_embeddings=True)
    
    # Calculate similarities
    similarities = cosine_similarity(query_emb, faq_embeddings)[0]
    
    # Get top-k indices
    top_idx = np.argsort(similarities)[::-1][:top_k]
    
    # Filter by threshold
    top_idx = [idx for idx in top_idx if similarities[idx] >= threshold]
    
    # Build results
    results = []
    for idx in top_idx:
        score = float(similarities[idx])
        results.append({
            "question": faq_df.iloc[idx]['question'],
            "answer": faq_df.iloc[idx]['answer'],
            "category": faq_df.iloc[idx]['category'],
            "language": faq_df.iloc[idx]['language'],
            "score": score,
            "confidence": get_confidence_level(score)
        })
    
    return results

### 4. Test Retrieval
# Test with different queries
# -------------------------
# MODE 1: Show normal Top-3 retrieval results
# -------------------------
test_queries = [
    "I forgot my password",
    "What are the payment options?",
    "Cannot log into account"
]

for query in test_queries:
    print(f"\n{'-'*70}")
    print(f"Query: '{query}'")
    print('-'*70)
    
    results = retrieve_faq(query, top_k=3)
    
    for i, r in enumerate(results, 1):
        confidence_symbol = ":)" if r['confidence'] == "High" else ":|" if r['confidence'] == "Medium" else ":("
        print(f"\n{confidence_symbol} Result {i} - {r['confidence']} Confidence ({r['score']:.3f})")
        print(f"   Q: {r['question']}")
        print(f"   A: {r['answer'][:80]}...")
        print(f"   [Category: {r['category']} | Language: {r['language']}]")


# -------------------------
# MODE 2: Production-style filtering with confidence threshold
# This MODE is just for filtering weak answers, we just adjusted the threshold to be 0.820
# Returns only answers we trust (>= 0.820 similarity)
# -------------------------

for query in test_queries:
    print(f"\n{'-'*70}")
    print(f"Query: '{query}'")
    print('-'*70)
    
    results = retrieve_faq(query, top_k=3,threshold=0.820)
    
    if not results:
        print("No confident match found. Escalating to support or asking user for clarification.")
        continue

    for i, r in enumerate(results, 1):
        confidence_symbol = ":)" if r['confidence'] == "High" else ":|" if r['confidence'] == "Medium" else ":("
        print(f"\n{confidence_symbol} Result {i} - {r['confidence']} Confidence ({r['score']:.3f})")
        print(f"   Q: {r['question']}")
        print(f"   A: {r['answer'][:80]}...")
        print(f"   [Category: {r['category']} | Language: {r['language']}]")


### 5. Evaluation Metrics
# Evaluating our system using ML metrics
# - **Hits@1**: Is the top result correct
# - **Hits@3**: Is the correct answer in top 3
# - **MRR**: Mean Reciprocal Rank

# We evaluate retrival system on test cases
# As arguments we pass a list of dicts with query and expected
# This returns the dictionary with evaluation metrics

def evaluate_system(test_cases):
    hits_1 = 0
    hits_3 = 0
    mrr_scores = []
    
    for test in test_cases:
        results = retrieve_faq(test['query'], top_k=3)
        expected_question = faq_df.iloc[test['expected_idx']]['question']
        
        # Check Hits@1
        if len(results) > 0 and results[0]['question'] == expected_question:
            hits_1 += 1
            mrr_scores.append(1.0)
        else:
            # Find rank of correct answer
            for rank, r in enumerate(results, 1):
                if r['question'] == expected_question:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)
        
        # Check Hits@3
        if any(r['question'] == expected_question for r in results):
            hits_3 += 1
    
    n = len(test_cases)
    return {
        'hits@1': hits_1 / n,
        'hits@3': hits_3 / n,
        'mrr': np.mean(mrr_scores),
        'total_queries': n
    }

# Define test cases
test_cases = [
    {'query': 'I forgot my password', 'expected_idx': 0},
    {'query': 'Which payment methods are supported?', 'expected_idx': 1},
    {'query': 'Cannot access my account', 'expected_idx': 2},
    {'query': 'reach customer support', 'expected_idx': 3},
    {'query': 'change plan', 'expected_idx': 4},
    {'query': 'get refund', 'expected_idx': 6},
]


metrics = evaluate_system(test_cases)

print("EVALUATION RESULTS\n")
print(f"Hits@1 (top result correct):     {metrics['hits@1']:.1%} ({int(metrics['hits@1']*metrics['total_queries'])}/{metrics['total_queries']})")
print(f"Hits@3 (answer in top 3):        {metrics['hits@3']:.1%} ({int(metrics['hits@3']*metrics['total_queries'])}/{metrics['total_queries']})")
print(f"MRR (Mean Reciprocal Rank):      {metrics['mrr']:.3f}")


### Interactive Demo 

def demo_search(query):
    print(f"\n{'-'*70}")
    print(f"*** Query: '{query}'")
    print('-'*70)
    
    results = retrieve_faq(query, top_k=3)
    
    if not results:
        print("No results found.")
        return
    
    for i, r in enumerate(results, 1):
        confidence_symbol = ":)" if r['confidence'] == "High" else ":|" if r['confidence'] == "Medium" else ":("
        
        print(f"\n{confidence_symbol} Result {i}: {r['confidence']} Confidence ({r['score']:.3f})")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}")
        print(f"[Category: {r['category']} | Language: {r['language']}]")

# Test with various queries
demo_queries = [
    "I forgot my password",
    "payment options available",
    "Не можам да се најавам"
]


for q in demo_queries:
    demo_search(q)