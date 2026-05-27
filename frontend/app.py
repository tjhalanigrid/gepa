#!/usr/bin/env python3
"""
Streamlit Claims Executive Dashboard
Provides interactive claim image uploads, invokes pipeline API, and renders beautiful cost breakdowns.
"""

import os
import requests
import streamlit as st
import pandas as pd

# Set Page Config for Sleek Look
st.set_page_config(
    page_title="AI Claims Adjuster Executive Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f1115;
    }
    .main-header {
        font-size: 38px;
        font-weight: 800;
        color: #3b82f6;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 15px;
    }
    .severity-badge-PRISTINE { background-color: #059669; color: white; padding: 4px 10px; border-radius: 8px; font-weight: bold; }
    .severity-badge-MINOR { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 8px; font-weight: bold; }
    .severity-badge-MODERATE { background-color: #f59e0b; color: white; padding: 4px 10px; border-radius: 8px; font-weight: bold; }
    .severity-badge-SEVERE { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allowed_html=True)

# Sidebar configurations
st.sidebar.image("https://img.icons8.com/color/120/car-collision.png", width=90)
st.sidebar.title("AI Claims Adjuster")
st.sidebar.markdown("---")
api_host = st.sidebar.text_input("FastAPI Service Host", "http://localhost:8000")
st.sidebar.info("Upload claim images to automatically run object detections, part segmentations, and cost sheet estimations.")

st.markdown('<div class="main-header">🚗 AUTOMOTIVE AI CLAIMS EXECUTIVE DASHBOARD</div>', unsafe_allowed_html=True)
st.markdown("---")

# File Upload Panel
st.subheader("1. Submit Damage claim Images")
uploaded_file = st.file_uploader(
    "Drag and drop vehicle image representing the damage...", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allowed_html=True)
        st.image(uploaded_file, caption="Claim Evidence Image", use_column_width=True)
        st.markdown('</div>', unsafe_allowed_html=True)
        
    with col2:
        st.subheader("2. Automated Claims Processing pipeline")
        run_pipeline = st.button("🚀 Execute Visual & Cost Reconciliations", use_container_width=True)
        
        if run_pipeline:
            with st.spinner("Executing neural detections & VLM Claims Adjuster..."):
                try:
                    # Submit file upload to API Gateway
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    api_url = f"{api_host}/api/assessment/evaluate"
                    
                    response = requests.post(api_url, files=files, timeout=60)
                    
                    if response.status_code == 200:
                        claim_data = response.json()
                        st.success("Claims pipeline completed successfully!")
                        
                        # 1. Executive Summary Panel
                        st.markdown('<div class="metric-card">', unsafe_allowed_html=True)
                        st.markdown(f"### Claim ID: `{claim_data.get('claim_id', 'N/A')}`")
                        
                        overall_sev = claim_data.get("overall_severity", "Moderate").upper()
                        st.markdown(f"**Overall Severity**: <span class='severity-badge-{overall_sev}'>{overall_sev}</span>", unsafe_allowed_html=True)
                        st.write(f"**Executive Summary**:\n{claim_data.get('overall_summary', 'N/A')}")
                        st.markdown('</div>', unsafe_allowed_html=True)
                        
                        # 2. Visual Damages Table
                        st.subheader("📋 Visual Detections & Bounding Box Audits")
                        damages_list = claim_data.get("damages", [])
                        
                        if damages_list:
                            df_damage = pd.DataFrame(damages_list)
                            # Display clean table
                            st.dataframe(
                                df_damage[["part", "damage_type", "severity", "confidence", "reasoning"]],
                                use_container_width=True
                            )
                        else:
                            st.info("No visual damage items detected on the panels.")
                            
                        # 3. Claims Cost Sheets
                        cost_est = claim_data.get("cost_estimation", {})
                        if cost_est:
                            st.subheader("💰 Mitchell-style Parts & Repair Valuation Sheet")
                            totals = cost_est.get("summary_totals", {})
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Grand Total Cost", f"${totals.get('grand_total_estimate', 0.0):,.2f}")
                            c2.metric("Total Parts Cost", f"${totals.get('total_parts_cost', 0.0):,.2f}")
                            c3.metric("Total Labor hours", f"{totals.get('total_body_labor_hours', 0.0) + totals.get('total_paint_labor_hours', 0.0)} hrs")
                            
                            st.markdown("#### Repair Invoice Line Items")
                            line_items = cost_est.get("line_items", [])
                            if line_items:
                                df_costs = pd.DataFrame(line_items)
                                st.dataframe(
                                    df_costs[["part", "severity", "decision", "part_cost", "labor_hours_body", "labor_hours_paint", "total_item_cost", "justification"]],
                                    use_container_width=True
                                )
                    else:
                        st.error(f"API Error (HTTP {response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with API service: {e}")
else:
    st.info("Please upload a vehicle damage image to begin claims assessment.")
