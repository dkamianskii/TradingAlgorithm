from indicators import moving_averages as ma
import yfinance as yf

data = yf.download("AAPL", start="2021-01-01", end="2021-12-10")

#print(data.head(10)["Close"])
print(type(data["Close"].index[0]))
#print(ma.SMA(data.head(10)["Close"], 1))


