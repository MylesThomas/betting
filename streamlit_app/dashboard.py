"""
TQS Props Dashboard

Multi-sport player props arbitrage opportunities dashboard.

Deployed at: https://tqs-props-dashboard.streamlit.app/

Navigation:
- NBA Arbs: NBA player props arbitrage opportunities
- NFL Arbs: NFL player props arbitrage opportunities

Usage:
    streamlit run streamlit_app/dashboard.py
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="TQS Props Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Show "Home" instead of "dashboard" in sidebar nav */
    [data-testid="stSidebarNav"] li:first-child span {
        visibility: hidden;
    }
    [data-testid="stSidebarNav"] li:first-child span::before {
        content: "Home";
        visibility: visible;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sport-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        transition: transform 0.3s ease;
    }
    
    .sport-card:hover {
        transform: translateY(-5px);
    }
    
    .sport-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .sport-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .sport-description {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.75rem;
        }
        .sport-emoji {
            font-size: 3rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">🎯 TQS Props Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Find arbitrage opportunities across multiple sportsbooks.<br>
        Select a sport from the sidebar to get started.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dashboard cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #C9082A 0%, #17408B 100%); 
                    border-radius: 15px; padding: 2rem; text-align: center; color: white;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏀</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">NBA Arbs</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">
                Player props arbitrage opportunities<br>
                Points, Rebounds, Assists, Threes, and more
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        if st.button("🏀 Go to NBA Dashboard", use_container_width=True, key="nba_btn"):
            st.switch_page("pages/1_NBA_Arbs.py")
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #013369 0%, #D50A0A 100%); 
                    border-radius: 15px; padding: 2rem; text-align: center; color: white;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏈</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">NFL Arbs</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">
                Player props arbitrage opportunities<br>
                Passing, Rushing, Receiving, TDs
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        if st.button("🏈 Go to NFL Dashboard", use_container_width=True, key="nfl_btn"):
            st.switch_page("pages/2_NFL_Arbs.py")
    
    st.markdown("---")
    
    # Quick stats section
    st.subheader("📊 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Data Collection**
        
        Our scripts fetch live odds from multiple sportsbooks via The Odds API:
        - FanDuel, DraftKings, BetMGM
        - Caesars, PointsBet, and more
        - Updated daily before game time
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Arbitrage Detection**
        
        We analyze every player prop market:
        - Compare Over/Under odds across books
        - Calculate combined probability
        - Flag opportunities under 100%
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Profit Calculation**
        
        For each arb opportunity:
        - Optimal stake allocation
        - Guaranteed profit amount
        - Which books to bet at
        """)
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About Arbitrage Betting"):
        st.markdown("""
        ### What is Arbitrage?
        
        Arbitrage betting (arbing) exploits differences in odds between sportsbooks to guarantee 
        a profit regardless of the outcome.
        
        **Example:**
        - Book A: Patrick Mahomes Over 275.5 Pass Yards @ +115
        - Book B: Patrick Mahomes Under 275.5 Pass Yards @ +105
        - Combined implied probability: 95.3% (< 100% = arb!)
        - Guaranteed profit: ~4.9% on total stake
        
        ### How We Find Arbs
        
        1. Fetch live odds from 10+ sportsbooks
        2. For each player prop, find best Over odds and best Under odds
        3. Calculate combined implied probability
        4. If < 100%, it's an arbitrage opportunity!
        
        ### Important Notes
        
        - Lines move fast - act quickly on opportunities
        - Some books may limit or ban arb bettors
        - Always verify odds before placing bets
        - Start with small stakes to test the process
        """)
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 🎯 TQS Props Dashboard")
        st.markdown("---")
        st.markdown("""
        **Select a sport from the sidebar** to view arbitrage opportunities.
        
        📊 **Data Sources:**
        - The Odds API
        - 10+ US Sportsbooks
        
        ⏰ **Update Schedule:**
        - NBA: 12:00 PM ET daily
        - NFL: Before game time
        
        💡 **Pro Tips:**
        - Higher profit % = better opportunity
        - Check both books before betting
        - Use the exact line shown
        """)


if __name__ == "__main__":
    main()
