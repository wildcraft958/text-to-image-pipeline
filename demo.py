from services.trend_engine import TrendManager
tm = TrendManager()
# Try a high temperature to get more diversity
trending_prompts = tm.get_trending_prompts(
    hours_back=24*7,
    top_n=100,
    p_threshold=0.7,
    num_samples=10,  # Let's ask for 10 diverse samples
    temperature=200.0 # This will flatten the probabilities
)
print(f"Found {len(trending_prompts)} trending prompts.")
print(trending_prompts)



# from google import genai
# client = genai.Client(api_key="AIzaSyDGVlrguCu4TctJjx1bb1iemsgf9d5HYUc")
# resp = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="ping"
# )
# print(resp.text)          # should return “pong” (or similar)
