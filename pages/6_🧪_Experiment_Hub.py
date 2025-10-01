# pages/6_🧪_Experiment_Hub.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Path Configuration and Module Import
# ============================================================================

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from utils.data_loader import DataLoader
    from components.sidebar import render_sidebar
    from components.headers import render_page_header, render_section_header, render_info_box
    from config.data_config import DataConfig
except ImportError as e:
    logger.error(f"Module import failed: {e}")
    st.error(f"⚠️ Critical module import error: {e}")
    st.stop()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Experiment Hub - SYPHU iGEM",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Constants and Configuration
# ============================================================================

EXPERIMENT_TYPES = {
    "Microscopy": {
        "icon": "🔬",
        "description": "Fluorescence and brightfield imaging",
        "typical_duration": 1,
        "equipment": ["Microscope", "Camera", "Image analysis software"]
    },
    "Molecular Biology": {
        "icon": "🧬",
        "description": "PCR, cloning, transformation",
        "typical_duration": 3,
        "equipment": ["Thermocycler", "Gel electrophoresis", "Incubator"]
    },
    "Protein Analysis": {
        "icon": "🔬",
        "description": "Western blot, protein purification",
        "typical_duration": 2,
        "equipment": ["Gel apparatus", "Transfer system", "Antibodies"]
    },
    "Sequencing": {
        "icon": "📊",
        "description": "DNA/RNA sequencing and analysis",
        "typical_duration": 5,
        "equipment": ["Sequencer", "Library prep kit", "Bioinformatics tools"]
    },
    "Cell Culture": {
        "icon": "🧫",
        "description": "Cell maintenance and assays",
        "typical_duration": 7,
        "equipment": ["Incubator", "Biosafety cabinet", "Culture media"]
    },
    "Other": {
        "icon": "⚗️",
        "description": "General laboratory work",
        "typical_duration": 1,
        "equipment": []
    }
}

EXPERIMENT_STATUS = {
    "Planning": {"color": "#FFC107", "icon": "📋"},
    "Active": {"color": "#4CAF50", "icon": "▶️"},
    "On Hold": {"color": "#FF9800", "icon": "⏸️"},
    "Completed": {"color": "#2196F3", "icon": "✅"},
    "Failed": {"color": "#F44336", "icon": "❌"}
}

PRIORITY_LEVELS = ["Low", "Medium", "High", "Urgent"]


# ============================================================================
# Data Management Functions
# ============================================================================

def initialize_experiment_storage():
    """Initialize experiment storage in session state."""
    if 'experiment_records' not in st.session_state:
        st.session_state.experiment_records = {}
    if 'experiment_notes' not in st.session_state:
        st.session_state.experiment_notes = {}


def generate_experiment_id() -> str:
    """Generate unique experiment ID."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    count = len(st.session_state.get('experiment_records', {}))
    return f"EXP_{timestamp}_{count:03d}"


def save_experiment(experiment_data: Dict[str, Any]) -> str:
    """
    Save experiment to session state.

    Parameters
    ----------
    experiment_data : Dict[str, Any]
        Experiment information dictionary.

    Returns
    -------
    str
        Experiment ID.
    """
    exp_id = experiment_data.get('id', generate_experiment_id())
    experiment_data['id'] = exp_id
    experiment_data['last_modified'] = datetime.now().isoformat()

    st.session_state.experiment_records[exp_id] = experiment_data
    logger.info(f"Experiment saved: {exp_id}")

    return exp_id


def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve experiment by ID."""
    return st.session_state.experiment_records.get(exp_id)


def update_experiment_status(exp_id: str, new_status: str) -> bool:
    """Update experiment status."""
    if exp_id in st.session_state.experiment_records:
        st.session_state.experiment_records[exp_id]['status'] = new_status
        st.session_state.experiment_records[exp_id]['last_modified'] = datetime.now().isoformat()
        return True
    return False


def delete_experiment(exp_id: str) -> bool:
    """Delete experiment from records."""
    if exp_id in st.session_state.experiment_records:
        del st.session_state.experiment_records[exp_id]
        if exp_id in st.session_state.experiment_notes:
            del st.session_state.experiment_notes[exp_id]
        logger.info(f"Experiment deleted: {exp_id}")
        return True
    return False


def export_experiments_to_csv() -> pd.DataFrame:
    """Export all experiments to DataFrame."""
    if not st.session_state.experiment_records:
        return pd.DataFrame()

    records = []
    for exp_id, exp_data in st.session_state.experiment_records.items():
        record = {
            'ID': exp_id,
            'Name': exp_data.get('name', ''),
            'Type': exp_data.get('type', ''),
            'Researcher': exp_data.get('researcher', ''),
            'Status': exp_data.get('status', ''),
            'Priority': exp_data.get('priority', ''),
            'Start Date': exp_data.get('start_date', ''),
            'Duration (days)': exp_data.get('duration', 0),
            'Created': exp_data.get('created_at', ''),
            'Modified': exp_data.get('last_modified', '')
        }
        records.append(record)

    return pd.DataFrame(records)


# ============================================================================
# Main Page Rendering
# ============================================================================

def main():
    """Main function to render Experiment Hub page."""

    render_sidebar()

    render_page_header(
        title="Experiment Management Hub",
        icon="🧪",
        subtitle="Laboratory experiment tracking and protocol management"
    )

    # Initialize storage
    initialize_experiment_storage()

    # Display overview metrics
    render_experiment_metrics()

    # Main tabs
    render_experiment_tabs()


def render_experiment_metrics():
    """Display overview metrics for experiments."""

    experiments = st.session_state.experiment_records

    if not experiments:
        return

    # Calculate metrics
    total = len(experiments)
    active = sum(1 for exp in experiments.values() if exp.get('status') == 'Active')
    completed = sum(1 for exp in experiments.values() if exp.get('status') == 'Completed')
    high_priority = sum(1 for exp in experiments.values() if exp.get('priority') in ['High', 'Urgent'])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Experiments", total)
    with col2:
        st.metric("Active", active, delta=f"{active / total * 100:.0f}%" if total > 0 else "0%")
    with col3:
        st.metric("Completed", completed, delta=f"{completed / total * 100:.0f}%" if total > 0 else "0%")
    with col4:
        st.metric("High Priority", high_priority)

    st.markdown("---")


def render_experiment_tabs():
    """Render main experiment management tabs."""

    tab1, tab2, tab3, tab4 = st.tabs([
        "➕ New Experiment",
        "📊 Active Experiments",
        "📁 Experiment Archive",
        "📈 Analytics"
    ])

    with tab1:
        render_new_experiment_tab()

    with tab2:
        render_active_experiments_tab()

    with tab3:
        render_archive_tab()

    with tab4:
        render_analytics_tab()


# ============================================================================
# Tab 1: New Experiment
# ============================================================================

def render_new_experiment_tab():
    """Render new experiment creation interface."""

    render_section_header("Create New Experiment", "➕")

    render_info_box(
        content="""
        **Experiment Documentation Guidelines:**

        - Use clear, descriptive experiment names
        - Document objectives in measurable terms
        - Include detailed methodology for reproducibility
        - Specify all required materials and equipment
        - Set realistic timelines and milestones
        """,
        box_type="info",
        title="Best Practices"
    )

    with st.form("new_experiment_form", clear_on_submit=True):
        render_experiment_form()


def render_experiment_form():
    """Render experiment creation form fields."""

    st.markdown("### Basic Information")

    col1, col2 = st.columns(2)

    with col1:
        exp_name = st.text_input(
            "Experiment Name *",
            placeholder="e.g., GFP Expression in E. coli",
            help="Descriptive name for the experiment"
        )

        exp_type = st.selectbox(
            "Experiment Type *",
            options=list(EXPERIMENT_TYPES.keys()),
            format_func=lambda x: f"{EXPERIMENT_TYPES[x]['icon']} {x}",
            help="Select the category that best describes this experiment"
        )

        researcher = st.text_input(
            "Principal Researcher *",
            placeholder="Full name",
            help="Primary person responsible for this experiment"
        )

        collaborators = st.text_input(
            "Collaborators",
            placeholder="Comma-separated names",
            help="Other team members involved"
        )

    with col2:
        start_date = st.date_input(
            "Start Date *",
            value=datetime.now(),
            help="Planned experiment start date"
        )

        # Suggest duration based on experiment type
        suggested_duration = EXPERIMENT_TYPES[exp_type]['typical_duration']
        duration = st.number_input(
            "Estimated Duration (days) *",
            min_value=1,
            max_value=365,
            value=suggested_duration,
            help=f"Typical duration for {exp_type}: {suggested_duration} day(s)"
        )

        priority = st.select_slider(
            "Priority *",
            options=PRIORITY_LEVELS,
            value="Medium",
            help="Experiment urgency level"
        )

        status = st.selectbox(
            "Initial Status",
            options=list(EXPERIMENT_STATUS.keys()),
            index=0,
            help="Current experiment state"
        )

    st.markdown("---")
    st.markdown("### Scientific Details")

    col1, col2 = st.columns(2)

    with col1:
        hypothesis = st.text_area(
            "Hypothesis",
            height=100,
            placeholder="State your hypothesis clearly...",
            help="What are you testing?"
        )

        objectives = st.text_area(
            "Objectives *",
            height=100,
            placeholder="List specific, measurable objectives...",
            help="What do you aim to achieve?"
        )

    with col2:
        methodology = st.text_area(
            "Methodology *",
            height=100,
            placeholder="Describe experimental procedure step-by-step...",
            help="Detailed protocol for reproducibility"
        )

        expected_outcomes = st.text_area(
            "Expected Outcomes",
            height=100,
            placeholder="Describe anticipated results...",
            help="What results do you expect?"
        )

    st.markdown("---")
    st.markdown("### Resources")

    col1, col2 = st.columns(2)

    with col1:
        # Display equipment for selected type
        suggested_equipment = EXPERIMENT_TYPES[exp_type]['equipment']
        if suggested_equipment:
            st.info(f"**Typical equipment for {exp_type}:**\n" +
                    "\n".join([f"• {eq}" for eq in suggested_equipment]))

        equipment = st.text_area(
            "Required Equipment",
            height=80,
            placeholder="List all equipment needed...",
            value="\n".join(suggested_equipment) if suggested_equipment else ""
        )

    with col2:
        materials = st.text_area(
            "Required Materials",
            height=80,
            placeholder="List all reagents, consumables, etc...",
            help="Include catalog numbers if available"
        )

        estimated_cost = st.number_input(
            "Estimated Cost ($)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            help="Approximate budget requirement"
        )

    st.markdown("---")

    # Safety and compliance
    with st.expander("Safety & Compliance", expanded=False):
        safety_concerns = st.text_area(
            "Safety Considerations",
            height=60,
            placeholder="List any safety concerns, PPE requirements, etc..."
        )

        ethics_approval = st.checkbox("Ethics approval required/obtained")
        biosafety_level = st.selectbox("Biosafety Level", ["N/A", "BSL-1", "BSL-2", "BSL-3"])

    # File attachments
    with st.expander("Attachments", expanded=False):
        protocol_file = st.file_uploader(
            "Upload Protocol Document",
            type=['pdf', 'docx', 'txt'],
            help="Detailed protocol file"
        )

        reference_files = st.file_uploader(
            "Upload References",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Relevant literature or protocols"
        )

    # Submit buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        submitted = st.form_submit_button(
            "✅ Create Experiment",
            type="primary",
            use_container_width=True
        )

    with col2:
        save_draft = st.form_submit_button(
            "💾 Save Draft",
            use_container_width=True
        )

    with col3:
        clear_form = st.form_submit_button(
            "🔄 Clear",
            use_container_width=True
        )

    # Form submission handling
    if submitted or save_draft:
        if not exp_name or not researcher or not objectives or not methodology:
            st.error("⚠️ Please fill all required fields (marked with *)")
        else:
            experiment_data = {
                'id': generate_experiment_id(),
                'name': exp_name,
                'type': exp_type,
                'researcher': researcher,
                'collaborators': collaborators,
                'start_date': start_date.isoformat(),
                'duration': duration,
                'priority': priority,
                'status': 'Planning' if save_draft else status,
                'hypothesis': hypothesis,
                'objectives': objectives,
                'methodology': methodology,
                'expected_outcomes': expected_outcomes,
                'equipment': equipment,
                'materials': materials,
                'estimated_cost': estimated_cost,
                'safety_concerns': safety_concerns,
                'ethics_approval': ethics_approval,
                'biosafety_level': biosafety_level,
                'created_at': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat()
            }

            exp_id = save_experiment(experiment_data)

            action = "saved as draft" if save_draft else "created"
            st.success(f"✅ Experiment '{exp_name}' {action} successfully!")
            st.info(f"📋 Experiment ID: **{exp_id}**")
            st.balloons()

            # Auto-rerun to refresh the active experiments tab
            st.rerun()


# ============================================================================
# Tab 2: Active Experiments
# ============================================================================

def render_active_experiments_tab():
    """Render active experiments overview."""

    render_section_header("Active Experiments", "📊")

    experiments = st.session_state.experiment_records

    if not experiments:
        render_info_box(
            content="""
            **No experiments recorded yet.**

            Create your first experiment using the "New Experiment" tab to start 
            tracking your laboratory work.
            """,
            box_type="info",
            title="Get Started"
        )
        return

    # Filter controls
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=list(EXPERIMENT_STATUS.keys()),
            default=["Planning", "Active", "On Hold"]
        )

    with col2:
        type_filter = st.multiselect(
            "Filter by Type",
            options=list(EXPERIMENT_TYPES.keys()),
            default=list(EXPERIMENT_TYPES.keys())
        )

    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Start Date", "Priority", "Name", "Status"]
        )

    # Filter experiments
    filtered_experiments = {
        exp_id: exp_data
        for exp_id, exp_data in experiments.items()
        if exp_data.get('status') in status_filter
           and exp_data.get('type') in type_filter
    }

    if not filtered_experiments:
        st.info("No experiments match the current filters.")
        return

    st.markdown(f"**Showing {len(filtered_experiments)} experiment(s)**")
    st.markdown("---")

    # Display experiments
    for exp_id, exp_data in filtered_experiments.items():
        render_experiment_card(exp_id, exp_data)


def render_experiment_card(exp_id: str, exp_data: Dict[str, Any]):
    """Render individual experiment card."""

    status = exp_data.get('status', 'Planning')
    status_info = EXPERIMENT_STATUS.get(status, EXPERIMENT_STATUS['Planning'])

    with st.expander(
            f"{status_info['icon']} **{exp_data['name']}** ({exp_id})",
            expanded=False
    ):
        # Header with quick actions
        col1, col2 = st.columns([3, 1])

        with col1:
            # Status badge
            st.markdown(
                f"<span style='background-color: {status_info['color']}; "
                f"color: white; padding: 0.25rem 0.5rem; border-radius: 4px; "
                f"font-size: 0.85rem;'>{status}</span>",
                unsafe_allow_html=True
            )

        with col2:
            # Quick action buttons
            action_col1, action_col2 = st.columns(2)

            with action_col1:
                if st.button("✏️ Edit", key=f"edit_{exp_id}", use_container_width=True):
                    st.session_state[f'editing_{exp_id}'] = True
                    st.rerun()

            with action_col2:
                if st.button("🗑️ Delete", key=f"delete_{exp_id}", use_container_width=True):
                    if st.session_state.get(f'confirm_delete_{exp_id}'):
                        delete_experiment(exp_id)
                        st.success(f"Experiment {exp_id} deleted")
                        st.rerun()
                    else:
                        st.session_state[f'confirm_delete_{exp_id}'] = True
                        st.warning("⚠️ Click again to confirm deletion")

        st.markdown("---")

        # Experiment details
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**General Information**")
            st.write(f"**Type:** {EXPERIMENT_TYPES[exp_data['type']]['icon']} {exp_data['type']}")
            st.write(f"**Researcher:** {exp_data['researcher']}")
            if exp_data.get('collaborators'):
                st.write(f"**Collaborators:** {exp_data['collaborators']}")
            st.write(f"**Priority:** {exp_data.get('priority', 'Medium')}")

        with col2:
            st.markdown("**Timeline**")
            st.write(f"**Start Date:** {exp_data['start_date'][:10]}")
            st.write(f"**Duration:** {exp_data['duration']} days")

            # Calculate end date and progress
            start = datetime.fromisoformat(exp_data['start_date'])
            end = start + timedelta(days=exp_data['duration'])
            st.write(f"**End Date:** {end.strftime('%Y-%m-%d')}")

            if status == 'Active':
                days_elapsed = (datetime.now() - start).days
                progress = min(days_elapsed / exp_data['duration'] * 100, 100)
                st.progress(progress / 100)
                st.caption(f"{progress:.0f}% complete ({days_elapsed}/{exp_data['duration']} days)")

        with col3:
            st.markdown("**Metadata**")
            st.write(f"**Created:** {exp_data['created_at'][:10]}")
            st.write(f"**Modified:** {exp_data.get('last_modified', 'N/A')[:10]}")
            if exp_data.get('estimated_cost'):
                st.write(f"**Est. Cost:** ${exp_data['estimated_cost']:.2f}")

        # Scientific details in tabs
        detail_tabs = st.tabs(["📋 Details", "⚗️ Protocol", "💬 Notes", "⚙️ Settings"])

        with detail_tabs[0]:
            if exp_data.get('hypothesis'):
                st.markdown("**Hypothesis:**")
                st.write(exp_data['hypothesis'])

            st.markdown("**Objectives:**")
            st.write(exp_data.get('objectives', 'Not specified'))

            if exp_data.get('expected_outcomes'):
                st.markdown("**Expected Outcomes:**")
                st.write(exp_data['expected_outcomes'])

        with detail_tabs[1]:
            st.markdown("**Methodology:**")
            st.write(exp_data.get('methodology', 'Not specified'))

            if exp_data.get('equipment'):
                st.markdown("**Equipment:**")
                st.text(exp_data['equipment'])

            if exp_data.get('materials'):
                st.markdown("**Materials:**")
                st.text(exp_data['materials'])

        with detail_tabs[2]:
            render_experiment_notes(exp_id)

        with detail_tabs[3]:
            render_experiment_settings(exp_id, exp_data)


def render_experiment_notes(exp_id: str):
    """Render experiment notes section."""

    if exp_id not in st.session_state.experiment_notes:
        st.session_state.experiment_notes[exp_id] = []

    notes = st.session_state.experiment_notes[exp_id]

    # Display existing notes
    if notes:
        st.markdown("**Laboratory Notes:**")
        for i, note in enumerate(reversed(notes)):
            st.markdown(
                f"<div style='background: #f0f2f6; padding: 0.5rem; "
                f"border-radius: 4px; margin: 0.5rem 0;'>"
                f"<small>{note['timestamp']}</small><br>{note['content']}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No notes yet. Add your first observation below.")

    # Add new note
    new_note = st.text_area("Add Note", key=f"note_input_{exp_id}", height=100)
    if st.button("💾 Save Note", key=f"save_note_{exp_id}"):
        if new_note:
            st.session_state.experiment_notes[exp_id].append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'content': new_note
            })
            st.success("Note saved!")
            st.rerun()


def render_experiment_settings(exp_id: str, exp_data: Dict[str, Any]):
    """Render experiment settings panel."""

    # Status update
    new_status = st.selectbox(
        "Update Status",
        options=list(EXPERIMENT_STATUS.keys()),
        index=list(EXPERIMENT_STATUS.keys()).index(exp_data.get('status', 'Planning')),
        key=f"status_{exp_id}"
    )

    if new_status != exp_data.get('status'):
        if st.button("Update Status", key=f"update_status_{exp_id}"):
            update_experiment_status(exp_id, new_status)
            st.success(f"Status updated to: {new_status}")
            st.rerun()

    # Safety information
    if exp_data.get('safety_concerns') or exp_data.get('biosafety_level'):
        st.markdown("---")
        st.markdown("**Safety Information:**")
        if exp_data.get('safety_concerns'):
            st.warning(f"⚠️ {exp_data['safety_concerns']}")
        if exp_data.get('biosafety_level') != 'N/A':
            st.info(f"🔒 Biosafety Level: {exp_data['biosafety_level']}")


# ============================================================================
# Tab 3: Archive
# ============================================================================

def render_archive_tab():
    """Render experiment archive."""

    render_section_header("Experiment Archive", "📁")

    experiments = st.session_state.experiment_records

    # Filter for completed/failed experiments
    archived = {
        exp_id: exp_data
        for exp_id, exp_data in experiments.items()
        if exp_data.get('status') in ['Completed', 'Failed']
    }

    if not archived:
        st.info("No archived experiments. Completed experiments will appear here.")
        return

    st.markdown(f"**{len(archived)} archived experiment(s)**")

    # Export option
    if st.button("📥 Export Archive to CSV"):
        df = export_experiments_to_csv()
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"experiments_archive_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    st.markdown("---")

    # Display archived experiments
    for exp_id, exp_data in archived.items():
        with st.expander(f"{EXPERIMENT_STATUS[exp_data['status']]['icon']} {exp_data['name']} ({exp_id})"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Type:** {exp_data['type']}")
                st.write(f"**Researcher:** {exp_data['researcher']}")

            with col2:
                st.write(f"**Status:** {exp_data['status']}")
                st.write(f"**Duration:** {exp_data['duration']} days")

            with col3:
                st.write(f"**Completed:** {exp_data.get('last_modified', 'N/A')[:10]}")


# ============================================================================
# Tab 4: Analytics
# ============================================================================

def render_analytics_tab():
    """Render experiment analytics and insights."""

    render_section_header("Experiment Analytics", "📈")

    experiments = st.session_state.experiment_records

    if not experiments:
        st.info("No data available for analysis. Create experiments to see analytics.")
        return

    df = export_experiments_to_csv()

    col1, col2 = st.columns(2)

    with col1:
        # Status distribution
        import plotly.express as px
        status_counts = df['Status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Experiments by Status"
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col2:
        # Type distribution
        type_counts = df['Type'].value_counts()
        fig_type = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title="Experiments by Type",
            labels={'x': 'Type', 'y': 'Count'}
        )
        st.plotly_chart(fig_type, use_container_width=True)

    # Timeline view
    st.markdown("---")
    st.markdown("### Experiment Timeline")

    timeline_df = df[['Name', 'Start Date', 'Duration (days)', 'Status']].copy()
    timeline_df['End Date'] = pd.to_datetime(timeline_df['Start Date']) + pd.to_timedelta(
        timeline_df['Duration (days)'], unit='D')

    st.dataframe(timeline_df, use_container_width=True)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
