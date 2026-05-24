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
    page_title="TQS",
    page_icon="📊",
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
    st.markdown('<div class="main-header">TQS</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Quantitative betting strategies.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dashboard cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
                    border-radius: 15px; padding: 2rem; text-align: center; color: white;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">NBA Rebounds</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">
                Production model track record<br>
                P&amp;L, hit rates, today's plays
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("📊 Go to Rebounds Dashboard", use_container_width=True, key="reb_btn"):
            st.switch_page("pages/3_NBA_Rebounds_Strategy.py")

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #17408B 0%, #1d4ed8 100%);
                    border-radius: 15px; padding: 2rem; text-align: center; color: white;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏀</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">NCAAB Away Revenge</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">
                Bet the revenge team (away)<br>
                Season record &amp; today's plays
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("🏀 Go to NCAAB Dashboard", use_container_width=True, key="ncaab_btn"):
            st.switch_page("pages/4_NCAAB_Revenge_Spot.py")
    
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### TQS")
        st.markdown("---")
        st.markdown("""
        📊 **Data Sources:**
        - The Odds API
        - 10+ US Sportsbooks
        """)


if __name__ == "__main__":
    main()
