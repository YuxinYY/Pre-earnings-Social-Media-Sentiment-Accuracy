# Pre-earnings-Social-Media-Sentiment-Accuracy
Uses ML methods to conduct sentiment analysis of reddit submission and comment data and see whether the average sentiment towards a stock matches with its actual earnings outcome.

The files consists of three parts:
1. Data Collection: Loading data from **Academic Torrents** and Filter submission posts and comments between Jun 2021 and Aug 2021 (inclusive) from the following subreddits: r/wallstreetbets, r/stocks, r/Superstonk, r/pennystocks, r/investing.
2. Data Cleaning & Preprocessing: keep submissions and comments mentioning a specific stock and posted within the pre-announcement window.
3. Machine Learning:  
Apply sentiment analysis on the filtered submission posts and comments using a [model to be determined]. Each text is classified into positive or negative sentiment with respect to the mentioned stock. I then aggregate the sentiment scores across the pre-earnings window for each individual stock.

4. Analysis:  
Compare the aggregated sentiment signal to the actual Q2 2021 earnings results, obtained via the **Alpha Vantage EARNINGS API** (https://www.alphavantage.co/documentation/). Specifically, I test whether positive sentiment corresponds to earnings surprises above expectations, and whether negative sentiment corresponds to earnings disappointments. Accuracy, precision/recall, and confusion matrices are reported to evaluate the predictive power of social media sentiment on earnings outcomes.


Resources:
1. Academic Torrents:
   Link: https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13
   Content: 
   Tool: 
3. 
