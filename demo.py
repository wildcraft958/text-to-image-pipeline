from services.trend_engine import TrendManager
tm = TrendManager()
print(tm.fetch_top_performing_content(hours_back=24*7, limit=20))
