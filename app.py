"""
Stock Analysis Agent - Web Interface
-------------------------------------
Flask web app wrapping the CrewAI stock analysis agent (Groq-powered).
Enter a ticker in the browser and get an investor report back.

Folder layout expected:
    app.py
    templates/
        index.html
    .env                 (GROQ_API_KEY=...)

Usage:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
from flask import Flask, render_template, request

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
import yfinance as yf
from duckduckgo_search import DDGS

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


# --------------------------------------------------------------------------
# Custom tools (same as the CLI version)
# --------------------------------------------------------------------------
class StockSearchTool(BaseTool):
    name: str = "StockNewsSearcher"
    description: str = "Search for the latest news and updates about a stock using DuckDuckGo"

    def _run(self, query: str) -> str:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=2)
            return "\n".join(r["body"] for r in results)


class YahooFinanceTool(BaseTool):
    name: str = "YahooFinanceFetcher"
    description: str = "Get the latest 1-month stock price history for a given ticker using yFinance"

    def _run(self, ticker: str) -> str:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            return f"No price data found for ticker '{ticker}'. It may be an invalid symbol."
        return hist.tail(3).to_string()


# --------------------------------------------------------------------------
# Crew builder (same structure as the CLI version)
# --------------------------------------------------------------------------
def build_crew(ticker: str, llm: LLM) -> Crew:
    stock_analyst = Agent(
        role="Stock Analyst",
        goal="Analyze recent stock data and news",
        backstory="Expert in financial trends, macro indicators, and company performance",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    report_writer = Agent(
        role="Report Generator",
        goal="Write investor-friendly summaries of stock analysis",
        backstory="Professional writer with expertise in finance reporting",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    search_tool = StockSearchTool()
    finance_tool = YahooFinanceTool()

    search_task = Task(
        description=f"Search latest news and updates about the stock '{ticker}' using DuckDuckGo.",
        expected_output=f"Summarized news highlights for {ticker} stock.",
        agent=stock_analyst,
        tools=[search_tool],
    )

    analysis_task = Task(
        description=f"Analyze {ticker} stock price trends using yFinance.",
        expected_output="Key trends and technical highlights for the past month.",
        agent=stock_analyst,
        tools=[finance_tool],
    )

    report_task = Task(
        description="Write a clean investor report using previous analysis and news insights.",
        expected_output="Concise report with market summary and investment outlook.",
        agent=report_writer,
    )

    return Crew(
        agents=[stock_analyst, report_writer],
        tasks=[search_task, analysis_task, report_task],
        verbose=False,
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    report = None
    error = None
    ticker = ""

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()

        if not GROQ_API_KEY:
            error = "GROQ_API_KEY is not set. Add it to your .env file (in this folder) and restart the server."
        elif not ticker:
            error = "Please enter a stock ticker (e.g. AAPL, MSFT, TSLA)."
        else:
            try:
                llm = LLM(
                    model="groq/llama-3.3-70b-versatile",
                    api_key=GROQ_API_KEY,
                    temperature=0.7,
                )
                crew = build_crew(ticker, llm)
                result = crew.kickoff()
                report = str(result)
            except Exception as e:
                error = f"Something went wrong while analyzing '{ticker}': {e}"

    return render_template("index.html", report=report, error=error, ticker=ticker)


if __name__ == "__main__":
    # debug=False is REQUIRED once this is publicly reachable — Flask's debug
    # mode includes an interactive debugger that lets visitors run arbitrary
    # Python code on the server if an error occurs. Never enable it publicly.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
