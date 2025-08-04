# Analyze a financial report with Mistral-24B AI
import streamlit as st
import requests
import pandas as pd
import json
from typing import Optional, Dict, Any
import io
import time

# Page configuration
st.set_page_config(
    page_title="AI Financial Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_S"

# Custom CSS to improve design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .analysis-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

def check_ollama_connection() -> bool:
    """Check connection with Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def analyze_financial_report(financial_data: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Analyze a financial report with Mistral-24B AI with enhanced timeout handling
    """
    analysis_prompts = {
        "comprehensive": f"""You are a senior financial analyst with 15+ years of experience. Analyze the following financial data comprehensively and provide actionable insights:

{financial_data}

Structure your analysis as follows:

## EXECUTIVE SUMMARY
- Overall financial health score (1-10)
- Key highlights and critical findings
- Primary recommendations (top 3)

## FINANCIAL METRICS ANALYSIS
### Revenue Analysis
- Revenue trends and growth rates
- Revenue quality and sustainability
- Market position indicators

### Profitability Analysis  
- Gross margin, operating margin, net margin
- Profitability trends and drivers
- Cost structure optimization opportunities

### Liquidity & Solvency
- Current ratio, quick ratio, cash ratio
- Debt-to-equity, interest coverage
- Working capital management

### Efficiency Ratios
- Asset turnover, inventory turnover
- Return on Assets (ROA), Return on Equity (ROE)
- Cash conversion cycle

## TREND ANALYSIS
- Historical performance patterns
- Seasonal variations and cyclical trends
- Growth trajectory assessment

## RISK ASSESSMENT
- Financial risks (liquidity, credit, market)
- Operational risks and vulnerabilities
- Early warning indicators

## STRATEGIC RECOMMENDATIONS
1. **Immediate Actions** (0-3 months)
2. **Short-term Initiatives** (3-12 months)  
3. **Long-term Strategy** (1-3 years)

Provide specific numbers, percentages, and calculations throughout your analysis.""",
        
        "anomaly": f"""You are a forensic financial analyst specializing in fraud detection and anomaly identification. Examine this financial data for irregularities:

{financial_data}

## ANOMALY DETECTION FRAMEWORK

### 1. REVENUE ANOMALIES
- Revenue recognition timing issues
- Unusual revenue spikes/drops (>25% variance)
- Channel or customer concentration risks
- Related party transaction flags

### 2. EXPENSE IRREGULARITIES  
- Unexpected cost increases or decreases
- Missing typical expense categories
- Cost allocation manipulations
- Timing of expense recognition

### 3. BALANCE SHEET RED FLAGS
- Asset quality deterioration
- Unusual changes in reserves/allowances
- Off-balance sheet arrangements
- Related party balances

### 4. CASH FLOW WARNINGS
- Operating cash flow vs. net income gaps
- Unusual investing activities
- Financing activity red flags
- Free cash flow sustainability

### 5. STATISTICAL ANALYSIS
- Calculate variance from industry norms
- Benford's law analysis (if applicable)
- Ratio analysis outliers
- Time series anomaly detection

### 6. FRAUD RISK INDICATORS
- Management override opportunities
- Internal control weaknesses
- Pressure/incentive factors
- Rationalization indicators

## INVESTIGATION PRIORITIES
Rank findings by:
1. **High Risk** - Immediate investigation required
2. **Medium Risk** - Further analysis needed
3. **Low Risk** - Monitor trends

For each anomaly, provide: Detection method, Financial impact, Investigation steps.""",
        
        "performance": f"""You are a performance management consultant analyzing financial performance. Evaluate this data against best practices:

{financial_data}

## PERFORMANCE EVALUATION FRAMEWORK

### 1. GROWTH PERFORMANCE
- Revenue growth (YoY, QoQ, CAGR)
- Profit growth sustainability
- Market share expansion
- Organic vs. inorganic growth

### 2. PROFITABILITY EXCELLENCE
- Margin improvement trends
- Operating leverage analysis
- Cost management effectiveness
- Pricing power indicators

### 3. OPERATIONAL EFFICIENCY
- Asset utilization rates
- Process efficiency metrics
- Technology enablement impact
- Supply chain optimization

### 4. CAPITAL EFFICIENCY
- Return on Invested Capital (ROIC)
- Economic Value Added (EVA)
- Capital allocation effectiveness
- Working capital optimization

### 5. FINANCIAL STRENGTH
- Balance sheet quality
- Debt management capability
- Cash generation power
- Financial flexibility

### 6. STAKEHOLDER VALUE
- Shareholder returns
- Customer satisfaction metrics
- Employee productivity
- ESG performance indicators

## BENCHMARKING ANALYSIS
- Industry comparison (if determinable)
- Best-in-class performance gaps
- Historical performance trends
- Peer group positioning

## PERFORMANCE IMPROVEMENT ROADMAP
### Quick Wins (0-6 months)
- Immediate optimization opportunities
- Low-hanging fruit identification

### Strategic Initiatives (6-18 months)
- Process improvements
- Technology investments
- Organizational changes

### Transformational Changes (18+ months)
- Business model evolution
- Market expansion strategies
- Innovation investments

Include quantitative analysis, specific KPIs, and measurable improvement targets."""
    }
    
    prompt = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower for more precise financial analysis
            "top_p": 0.9,
            "max_tokens": 3000,  # Reduced for faster processing
            "num_predict": 3000,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,     # Context window
            "num_batch": 512,    # Batch size for efficiency
            "num_thread": 8      # Use multiple threads
        }
    }
    
    try:
        # Show progress with detailed status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🚀 Initializing Mistral-24B model...")
        progress_bar.progress(10)
        
        status_text.text("📊 Processing financial data...")
        progress_bar.progress(25)
        
        # Increased timeout to 5 minutes for large model
        with st.spinner("🤖 Mistral-24B analyzing (this may take 2-5 minutes)..."):
            status_text.text("🧠 Model thinking and analyzing...")
            progress_bar.progress(50)
            
            response = requests.post(
                OLLAMA_URL, 
                json=payload, 
                timeout=300,  # 5 minutes timeout
                headers={'Connection': 'keep-alive'}
            )
            
        progress_bar.progress(90)
        status_text.text("✅ Analysis complete!")
        
        if response.status_code == 200:
            result = response.json()
            progress_bar.progress(100)
            status_text.text("🎉 Results ready!")
            
            # Clear progress indicators after a moment
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return {
                "success": True,
                "analysis": result.get("response", "No analysis generated."),
                "model_used": MODEL_NAME,
                "analysis_type": analysis_type
            }
        else:
            progress_bar.empty()
            status_text.empty()
            return {
                "success": False,
                "error": f"API Error: {response.status_code} - {response.text}"
            }
            
    except requests.exceptions.Timeout:
        progress_bar.empty()
        status_text.empty()
        return {
            "success": False,
            "error": "⏰ Analysis timeout. Try: 1) Reduce data size, 2) Restart Ollama, 3) Check system resources. Large models need more time."
        }
    except requests.exceptions.ConnectionError:
        progress_bar.empty()
        status_text.empty()
        return {
            "success": False,
            "error": "🔌 Connection failed. Ensure Ollama is running: `ollama serve`"
        }
    except requests.RequestException as e:
        progress_bar.empty()
        status_text.empty()
        return {
            "success": False,
            "error": f"Connection error: {str(e)}"
        }

def analyze_financial_report_streaming(financial_data: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Alternative streaming analysis for better performance with large models
    """
    analysis_prompts = {
        "comprehensive": f"""You are a senior financial analyst. Analyze this financial data briefly but thoroughly:

{financial_data}

Provide:
1. **Executive Summary** (key findings)
2. **Financial Health** (Strong/Good/Concerning/Poor)
3. **Key Metrics** (Revenue, Profit, Ratios)
4. **Top 3 Risks** identified
5. **Top 3 Recommendations**

Be concise but insightful.""",
        
        "anomaly": f"""You are a forensic analyst. Examine this financial data for irregularities:

{financial_data}

Focus on:
1. **Revenue Anomalies** (unusual patterns)
2. **Expense Red Flags** (unexpected costs)
3. **Cash Flow Issues** (timing problems)
4. **Risk Assessment** (1-10 scale)
5. **Investigation Priority** (High/Medium/Low)

Be specific about concerns found.""",
        
        "performance": f"""You are a performance consultant. Evaluate this financial data:

{financial_data}

Analyze:
1. **Growth Performance** (trends)
2. **Profitability** (margins)
3. **Efficiency** (ratios)
4. **Performance Score** (1-10)
5. **Improvement Areas** (top 3)

Provide actionable insights."""
    }
    
    prompt = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,  # Enable streaming
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 2000,  # Shorter for streaming
            "num_predict": 2000,
            "repeat_penalty": 1.1
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            timeout=180,  # 3 minutes for streaming
            stream=True
        )
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if 'response' in json_response:
                            full_response += json_response['response']
                        if json_response.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            return {
                "success": True,
                "analysis": full_response,
                "model_used": MODEL_NAME,
                "analysis_type": f"{analysis_type} (streaming)"
            }
        else:
            return {
                "success": False,
                "error": f"Streaming Error: {response.status_code}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Streaming failed: {str(e)}"
        }

def process_uploaded_file(uploaded_file) -> Optional[str]:
    """Process uploaded files (CSV/Excel)"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please use CSV or Excel.")
            return None
        
        # Display first rows
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
        
        # File statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Size", f"{uploaded_file.size} bytes")
        
        return df.to_string()
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 AI Financial Report Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Check Ollama connection
    if check_ollama_connection():
        st.sidebar.success("✅ Ollama connected")
        st.sidebar.info("🤖 Using: Mistral-Small-24B")
    else:
        st.sidebar.error("❌ Ollama unavailable")
        st.error("⚠️ Please ensure Ollama is running on localhost:11434")
        return
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "🔍 Analysis Type",
        ["comprehensive", "anomaly", "performance"],
        format_func=lambda x: {
            "comprehensive": "📈 Comprehensive Analysis",
            "anomaly": "🚨 Anomaly Detection", 
            "performance": "⚡ Performance Analysis"
        }[x]
    )
    
    # Processing mode
    processing_mode = st.sidebar.radio(
        "⚙️ Processing Mode",
        ["Standard", "Fast Streaming"],
        help="Use Fast Streaming if you experience timeouts"
    )
    
    # Troubleshooting section
    with st.sidebar.expander("🔧 Troubleshooting Timeouts"):
        st.markdown("""
        **If you get timeout errors:**
        
        1. **Use Fast Streaming mode** ⚡
        2. **Reduce data size** (< 1000 lines)
        3. **Restart Ollama:**
           ```bash
           ollama serve
           ```
        4. **Check system resources:**
           - RAM: 16GB+ recommended
           - CPU: 8+ cores preferred
           - Free disk space: 15GB+
        
        5. **Model optimization:**
           ```bash
           # Load model in memory
           ollama run hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_S
           # Type /bye to exit but keep in memory
           ```
        
        **24B models need patience! 🕐**
        """)
    
    # Input mode
    input_mode = st.sidebar.radio(
        "📝 Input Mode",
        ["Manual Text", "File Upload"],
        index=0
    )
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Financial Data")
        
        financial_data = ""
        
        if input_mode == "Manual Text":
            financial_data = st.text_area(
                "Paste your financial data here",
                height=300,
                placeholder="Example:\nRevenue Q4 2023: $2.5M\nExpenses: $1.8M\nNet Profit: $700K\n..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Supported formats: CSV, Excel (.xlsx, .xls)"
            )
            
            if uploaded_file:
                financial_data = process_uploaded_file(uploaded_file)
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Launch Analysis",
            type="primary",
            disabled=not financial_data,
            use_container_width=True
        )
    
    with col2:
        st.subheader("🤖 Analysis Results")
        
        if analyze_button and financial_data:
            # Choose analysis method based on processing mode
            if processing_mode == "Fast Streaming":
                result = analyze_financial_report_streaming(financial_data, analysis_type)
            else:
                result = analyze_financial_report(financial_data, analysis_type)
            
            if result["success"]:
                # Display results
                st.markdown(f'<div class="analysis-container">', unsafe_allow_html=True)
                st.markdown("### 📊 Generated Analysis")
                st.write(result["analysis"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Analysis information
                processing_info = f"🔧 Model: Mistral-24B | Type: {result['analysis_type']} | Mode: {processing_mode}"
                st.info(processing_info)
                
                # Download option
                st.download_button(
                    label="💾 Download Analysis",
                    data=result["analysis"],
                    file_name=f"financial_analysis_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.error(f"❌ {result['error']}")
                
                # Show additional help for timeout errors
                if "timeout" in result['error'].lower():
                    st.warning("💡 **Timeout Solutions:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("✅ Try **Fast Streaming** mode")
                        st.info("✅ Reduce your data size")
                    with col2:
                        st.info("✅ Restart Ollama service")
                        st.info("✅ Pre-load the model")
                        
        elif analyze_button:
            st.warning("⚠️ Please provide financial data to analyze.")
    
    # Footer with information
    st.markdown("---")
    
    with st.expander("ℹ️ About this application"):
        st.markdown("""
        **AI Financial Report Analyzer** uses the **Mistral-Small-24B** model via Ollama to analyze your financial reports.
        
        **Features:**
        - 📈 Comprehensive analysis of financial metrics
        - 🚨 Anomaly and risk detection
        - ⚡ Performance evaluation
        - 📁 CSV and Excel file support
        - 💾 Results download
        
        **Model Information:**
        - **Mistral-Small-3.1-24B** - 24 billion parameters
        - **High-performance** financial analysis
        - **Advanced reasoning** capabilities
        - **Professional-grade** insights
        
        **Requirements:**
        - Ollama installed and running
        - Mistral-Small model: `ollama pull hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_S`
        """)
    
    # Quick setup guide
    with st.expander("🚀 Quick Setup Guide for Mistral-24B"):
        st.markdown("""
        **Optimize your system for the 24B model:**
        
        ```bash
        # 1. Ensure Ollama is running
        ollama serve
        
        # 2. Pre-load model (recommended)
        ollama run hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_S
        # Wait for load, then type: /bye
        
        # 3. Check if model is loaded
        ollama list
        ```
        
        **System Requirements:**
        - **RAM:** 16GB+ (24GB recommended)
        - **Storage:** 15GB+ free space
        - **CPU:** 8+ cores preferred
        
        **Performance Tips:**
        - Close other applications
        - Use "Fast Streaming" for large datasets
        - Keep data under 1000 rows for best performance
        """)
    
    # Sample data examples
    with st.expander("📝 Sample financial data"):
        st.code("""
Revenue Q4 2023: $2,500,000
Revenue Q3 2023: $2,200,000
Revenue Q2 2023: $2,000,000
Revenue Q1 2023: $1,800,000

Operating Expenses Q4: $1,800,000
Marketing Expenses Q4: $400,000
R&D Expenses Q4: $300,000

Net Profit Q4: $700,000
Cash Flow Q4: $850,000
Total Assets: $15,000,000
Total Liabilities: $8,500,000
        """)

if __name__ == "__main__":
    main()