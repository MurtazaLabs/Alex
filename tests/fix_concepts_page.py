with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the main function
main_start = content.find('def main():')
if main_start == -1:
    print("main() not found!")
else:
    print(f"main() found at position {main_start}")

# Find the end of main function (before if __name__ == "__main__":)
main_end = content.find('\nif __name__ == "__main__":')
if main_end == -1:
    main_end = len(content)

print(f"main() starts at {main_start}, ends at {main_end}")

# New main function
new_main = '''def main():
    """Main entry point for Concept Explorer page."""
    st.set_page_config(page_title="Concept Explorer", page_icon="🧠", layout="wide")
    
    # Initialize session state
    if "concepts" not in st.session_state:
        # Demo concepts
        st.session_state.concepts = [
            {
                "id": "concept_001",
                "name": "Liquidity Sweep",
                "display_name": "Liquidity Sweep / Sweep the Lows/Highs",
                "definition": "Price moves beyond a previous high/low to trigger stop losses/orders before reversing",
                "detection_rule": {"indicator": "price_action", "pattern": "sweep_then_reverse"},
                "required_conditions": [
                    "Price sweeps beyond previous swing high/low",
                    "Volume increases during sweep",
                    "Price closes back inside previous range"
                ],
                "optional_conditions": [
                    "Volume spike > 1.5x average",
                    "Sweep occurs at key level (support/resistance)",
                    "RSI divergence"
                ],
                "false_positives": [
                    "Breakout continuation",
                    "News-driven move",
                    "Low volume sweep"
                ],
                "limitations": [
                    "Less reliable in strong trends",
                    "Requires sufficient liquidity"
                ],
                "market_regimes": ["trending", "ranging", "volatile"],
                "research_status": "well_understood",
                "confidence_score": 0.85,
                "relationships": [
                    {"target_id": "concept_002", "relationship_type": "implies", "target_name": "Stop Hunt", "strength": 0.8},
                    {"target_id": "concept_003", "relationship_type": "part_of", "target_name": "Smart Money Concepts", "strength": 0.7},
                ],
                "evidence": [
                    "Multiple academic papers on stop hunting",
                    "Empirical backtest on EUR/USD 2020-2023"
                ],
                "id": "concept_001",
                "name": "Liquidity Sweep",
                "definition": "Price moves beyond a previous high/low to trigger stop losses/orders before reversing",
                "created_at": "2024-01-15",
                "confidence_score": 0.85,
            },
            {
                "id": "concept_002",
                "name": "Order Block",
                "display_name": "Order Block / Institutional Order Block",
                "definition": "A price area where institutional orders were placed, often acting as support/resistance",
                "detection_rule": {"indicator": "price_action", "pattern": "strong_move_from_consolidation"},
                "required_conditions": [
                    "Strong impulsive move away from consolidation",
                    "Clear imbalance in price action",
                    "Unmitigated (not yet revisited)"
                ],
                "optional_conditions": [
                    "Fair Value Gap (FVG) present",
                    "Volume imbalance",
                    "Institutional candle structure"
                ],
                "false_positives": [
                    "Regular consolidation zone",
                    "Breakout without volume"
                ],
                "limitations": [
                    "Subjective identification",
                    "Multiple conflicting definitions"
                ],
                "market_regimes": ["trending", "ranging"],
                "research_status": "partially_understood",
                "confidence_score": 0.65,
                "relationships": [
                    {"target_id": "concept_001", "relationship_type": "enhances", "target_name": "Liquidity Sweep", "strength": 0.7},
                    {"target_id": "concept_004", "relationship_type": "part_of", "target_name": "Smart Money Concepts", "strength": 0.9},
                ],
                "evidence": [
                    "Inner Circle Trader (ICT) methodology",
                    "Limited academic research"
                ],
                "id": "concept_002",
                "name": "Order Block",
                "definition": "A price area where institutional orders were placed, often acting as support/resistance",
                "created_at": "2024-02-01",
                "confidence_score": 0.65,
            },
            {
                "id": "concept_003",
                "name": "Fair Value Gap",
                "display_name": "Fair Value Gap (FVG) / Imbalance",
                "definition": "Price gap created by rapid movement where no trading occurred, often acts as magnet for price return",
                "detection_rule": {"indicator": "price_action", "pattern": "three_candle_imbalance"},
                "required_conditions": [
                    "Three candles with non-overlapping bodies",
                    "Gap between candle 1 high/low and candle 3 low/high"
                ],
                "optional_conditions": [
                    "Large candle in middle",
                    "Volume spike on middle candle"
                ],
                "false_positives": [
                    "Normal gap in trending market",
                    "News-driven gap"
                ],
                "limitations": [
                    "Fills not guaranteed",
                    "Time to fill varies widely"
                ],
                "market_regimes": ["trending", "volatile"],
                "research_status": "researching",
                "confidence_score": 0.55,
                "relationships": [
                    {"target_id": "concept_001", "relationship_type": "enhances", "target_name": "Liquidity Sweep", "strength": 0.6},
                    {"target_id": "concept_004", "relationship_type": "part_of", "target_name": "Imbalance Theory", "strength": 0.9},
                ],
                "evidence": [
                    "ICT methodology",
                    "Some quantitative studies on gap fills"
                ],
                "id": "concept_003",
                "name": "Fair Value Gap",
                "definition": "Price gap created by rapid movement where no trading occurred, often acts as magnet for price return",
                "created_at": "2024-03-01",
                "confidence_score": 0.55,
            },
            {
                "id": "concept_004",
                "name": "Smart Money Concepts",
                "display_name": "SMC / Smart Money Concepts",
                "definition": "Collection of institutional trading concepts including order blocks, FVGs, liquidity sweeps",
                "detection_rule": None,
                "required_conditions": [],
                "optional_conditions": [],
                "false_positives": [],
                "limitations": ["Umbrella term, not a specific pattern"],
                "market_regimes": ["all"],
                "research_status": "well_understood",
                "confidence_score": 0.75,
                "relationships": [
                    {"target_id": "concept_002", "relationship_type": "part_of", "target_name": "Order Block", "strength": 0.9},
                    {"target_id": "concept_003", "relationship_type": "part_of", "target_name": "Fair Value Gap", "strength": 0.9},
                ],
                "evidence": ["ICT methodology", "Trader community consensus"],
                "id": "concept_004",
                "name": "Smart Money Concepts",
                "definition": "Collection of institutional trading concepts",
                "created_at": "2024-01-01",
                "confidence_score": 0.75,
            }
        ]
    
    # Handle routing
    if st.session_state.get("show_create_concept"):
        render_concept_creator()
    elif st.session_state.get("selected_concept"):
        render_concept_detail(st.session_state.selected_concept)
    elif st.session_state.get("selected_concept_for_graph"):
        render_concept_graph(st.session_state.concepts, st.session_state.selected_concept_for_graph)
    else:
        # Sidebar
        with st.sidebar:
            st.markdown("### 🧠 Concept Explorer")
            
            # Search
            st.subheader("🔍 Search")
            query = st.text_input("Search concepts", placeholder="Search by name, definition...")
            if st.button("Search", use_container_width=True):
                st.session_state.search_triggered = True
            
            st.markdown("---")
            
            # Filters
            st.subheader("🔍 Filters")
            status_filter = st.selectbox("Research Status", ["All", "unresearched", "researching", "partially_understood", "well_understood", "debunked"])
            category_filter = st.selectbox("Category", ["All", "Pattern", "Indicator", "Regime", "Risk", "Setup"])
            min_confidence = st.slider("Min Confidence", 0, 100, 0)
            
            st.markdown("---")
            
            # Quick actions
            if st.button("➕ New Concept", use_container_width=True):
                st.session_state.show_create_concept = True
                st.rerun()
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            if st.button("📊 View Graph", use_container_width=True):
                st.session_state.selected_concept_for_graph = st.session_state.concepts[0]["id"] if st.session_state.concepts else None
                st.rerun()
            
            # Main content
            if st.session_state.get("show_create_concept"):
                render_concept_creator()
            elif st.session_state.get("selected_concept"):
                render_concept_detail(st.session_state.selected_concept)
            elif st.session_state.get("selected_concept_for_graph"):
                render_concept_graph(st.session_state.concepts, st.session_state.selected_concept_for_graph)
            else:
                # Main content
                render_concept_search()
                st.markdown("---")
                render_concept_list(st.session_state.concepts)


if __name__ == "__main__":
    main()

# Concept status colors
CONCEPT_STATUS_COLORS = {
    "unresearched": "#9e9e9e",
    "researching": "#ffa726",
    "partially_understood": "#ffa726",
    "well_understood": "#26a69a",
    "debunked": "#ef5350",
}

RELATIONSHIP_TYPES = [
    "implies",
    "contradicts",
    "supports",
    "is_a",
    "part_of",
    "parameter_of",
    "regime_of",
    "risk_factor_of",
    "requires",
    "enhances",
]

# Status colors for graph
status_colors = {
    "well_understood": "#26a69a",
    "partially_understood": "#ffa726",
    "researching": "#ffa726",
    "unresearched": "#9e9e9e",
    "debunked": "#ef5350",
}

if __name__ == "__main__":
    main()

'''
# Read the file
with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the main function
main_start = content.find('def main():')
if main_start == -1:
    print("main() not found!")
else:
    print(f"main() found at position {main_start}")

# Find the end of main function (before if __name__ == "__main__":)
main_end = content.find('\nif __name__ == "__main__":')
if main_end == -1:
    main_end = len(content)

print(f"main() starts at {main_start}, ends at {main_end}")

# Replace the main function
new_content = content[:content.find('def main():')] + new_main + content[content.find('\nif __name__ == "__main__":'):]

with open(r'E:\Forex trading bot\dashboard\pages\4_Concepts\page.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Fixed!')