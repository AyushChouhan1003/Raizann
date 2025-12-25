# ============================================================================
# FILE: src/report_service.py - Mental Health Report Generation
# ============================================================================
"""Mental health report generation service."""

import logging
import json
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


async def generate_report(user_id: str, storage, llm_service) -> Dict:
    """
    Generate comprehensive mental health report for a user.
    NEW LOGIC: Time-based (2-day cooldown), works with partial assessments.
    
    Args:
        user_id: User ID to generate report for
        storage: StorageManager instance
        llm_service: LLMService instance
    
    Returns:
        Dict containing report data or instructions
    """
    try:
        # Step 1: Get user
        user = storage.get_user(user_id)
        if not user:
            return _no_user_response()
        
        # Step 2: Check minimum data requirement (at least 10 messages)
        message_count = storage.count_user_messages(user_id)
        if message_count < 10:
            return {
                "status": "insufficient_data",
                "message": "Not enough conversation history to generate a meaningful report. Please continue chatting with the bot (minimum 10 messages required)."
            }
        
        # Step 3: Check report generation cooldown (2 days = 48 hours)
        last_report_date = user.get("last_report_generated_date")
        
        if last_report_date:
            hours_since_last = (datetime.utcnow() - last_report_date).total_seconds() / 3600
            
            if hours_since_last < 48:
                # Still in cooldown - return latest report
                hours_remaining = 48 - hours_since_last
                logger.info(f"Report cooldown active for {user_id} ({hours_remaining:.1f}h remaining)")
                
                latest_report = storage.get_latest_report(user_id)
                if latest_report:
                    return latest_report
                else:
                    # No report found but in cooldown (shouldn't happen)
                    return {
                        "status": "cooldown_active",
                        "message": f"Report generation is on cooldown. Please try again in {hours_remaining:.1f} hours.",
                        "hours_remaining": round(hours_remaining, 1)
                    }
        
        # Step 4: Cooldown expired or first report - generate new report
        logger.info(f"Generating new mental health report for user {user_id}")
        
        # Fetch recent conversation history (up to 100 messages)
        messages = storage.get_user_recent_conversations(user_id, limit=100)
        conversation_history = _format_messages_for_llm(messages)
        
        # Get assessment data (may be partial or complete)
        assessment_data = user.get("assessment_data", {})
        questions_answered = assessment_data.get("questions_answered", [])
        answered_count = sum(1 for q in questions_answered if q.get("answered", False))
        
        # Prepare user profile with assessment context
        user_profile = user.copy()
        user_profile["assessment_context"] = {
            "total_questions": 8,
            "answered_count": answered_count,
            "questions_data": questions_answered
        }
        
        # Generate report using LLM (works with partial assessment data)
        report_json = await llm_service.generate_mental_health_report(
            conversation_history=conversation_history,
            user_profile=user_profile,
            user_id=user_id
        )
        
        # Store report in collection
        storage.store_mental_health_report(user_id, report_json)
        
        # Update user's last report generation timestamp
        storage.update_last_report_date(user_id)
        
        logger.info(f"Successfully generated and stored report for user {user_id} (with {answered_count}/8 assessment answers)")
        return report_json
        
    except Exception as e:
        logger.error(f"Error generating report for user {user_id}: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"An error occurred while generating the report: {str(e)}"
        }


def _no_user_response() -> Dict:
    """Response when user doesn't exist."""
    return {
        "status": "user_not_found",
        "message": "User not found. Please register first."
    }


def _assessment_instructions_response() -> Dict:
    """Response when no assessment has been completed."""
    return {
        "status": "no_assessment",
        "message": "Mental Health Report Not Yet Available",
        "instructions": {
            "title": "How to Get Your Mental Health Report",
            "steps": [
                "Continue chatting with Raizann AI regularly",
                "After one week of conversation, a mental health assessment will automatically activate",
                "During the assessment week, Raizann will organically ask you 8 questions about your wellbeing",
                "Answer the questions naturally during your conversations",
                "After the assessment week completes, your comprehensive mental health report will be generated",
                "You can access your report anytime using this endpoint"
            ],
            "assessment_questions": [
                "Overall Mood: How often have you felt generally good or emotionally balanced?",
                "Life Satisfaction: How often have you felt satisfied with your life or direction?",
                "Energy Level: How often have you had enough energy for daily tasks?",
                "Ability to Cope: How often have you felt able to handle stress effectively?",
                "Low Mood or Stress: How often have you felt sad, stressed, or overwhelmed?",
                "Concentration & Clarity: How often have you struggled to concentrate?",
                "Social Connection: How often have you felt connected and supported?",
                "Safety & Risk: How often have you experienced thoughts of self-harm?"
            ],
            "timeline": "Assessment activates automatically after 7 days, runs for 7 days, then report generates",
            "note": "This is an AI-generated educational tool, not a clinical diagnosis. Professional evaluation is recommended for mental health concerns."
        }
    }


def _assessment_in_progress_response() -> Dict:
    """Response when assessment is currently active."""
    return {
        "status": "assessment_in_progress",
        "message": "Mental health assessment is currently in progress",
        "instructions": "Continue your conversations with Raizann AI. The assessment questions will be asked organically over the next few days. Your report will be generated once the assessment period completes (7 days).",
        "note": "You can continue chatting normally. The assessment questions will be naturally integrated into your conversations."
    }


def _is_report_recent(report: Dict, last_assessment_date: datetime) -> bool:
    """
    Check if existing report is for the current assessment period.
    
    Args:
        report: Existing report dictionary
        last_assessment_date: Date of last assessment start
    
    Returns:
        True if report is recent/current, False otherwise
    """
    try:
        report_date_str = report.get("date", "")
        if not report_date_str:
            return False
        
        report_date = datetime.fromisoformat(report_date_str.split('T')[0])
        
        # Report is recent if generated after or on the same day as assessment
        return report_date.date() >= last_assessment_date.date()
    except Exception as e:
        logger.error(f"Error checking report recency: {e}")
        return False


def _format_messages_for_llm(messages: list) -> str:
    """
    Format message list into readable conversation history for LLM.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        Formatted string of conversation history
    """
    formatted_lines = []
    for msg in messages:
        role = "User" if msg.get("role") == "user" else "Raizann AI"
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        formatted_lines.append(f"[{timestamp}] {role}: {content}")
    
    return "\n\n".join(formatted_lines)


def render_report_html(report_data: Dict) -> str:
    """
    Render report data as HTML using template.
    
    Args:
        report_data: Report JSON dictionary
    
    Returns:
        Rendered HTML string
    """
    from jinja2 import Template
    import os
    
    try:
        # Load HTML template
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "utils", 
            "report_template.html"
        )
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Create Jinja2 template
        template = Template(template_content)
        
        # Render with report data
        html = template.render(**report_data)
        
        return html
        
    except Exception as e:
        logger.error(f"Error rendering HTML template: {e}")
        # Fallback to simple HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Report Error</title></head>
        <body>
            <h1>Error Rendering Report</h1>
            <p>Unable to render HTML template: {str(e)}</p>
            <pre>{json.dumps(report_data, indent=2)}</pre>
        </body>
        </html>
        """

