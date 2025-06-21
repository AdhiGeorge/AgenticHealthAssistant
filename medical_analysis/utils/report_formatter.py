from datetime import datetime
from medical_analysis.utils.logger import get_logger

logger = get_logger(__name__)

def format_final_report(patient_info, summary, agent_results):
    """Format the final comprehensive medical report with professional structure."""
    try:
        if isinstance(summary, dict) and 'presenting_complaint' in summary:
            comprehensive_summary = summary
        else:
            comprehensive_summary = {}
        report = f"""
{'='*80}
                    MEDICAL DIAGNOSTIC ANALYSIS REPORT
{'='*80}

📋 **PATIENT INFORMATION**
{'-'*50}
{patient_info[:500]}{'...' if len(patient_info) > 500 else ''}

📊 **CLINICAL ASSESSMENT**
{'-'*50}
🔸 **Presenting Complaint:** {comprehensive_summary.get('presenting_complaint', 'Not specified')}
🔸 **Clinical History:** {comprehensive_summary.get('history', 'Not specified')}
🔸 **Physical Examination:** {comprehensive_summary.get('examination', 'Not specified')}
🔸 **Investigations:** {comprehensive_summary.get('investigations', 'Not specified')}

🏥 **SPECIALTY CONSULTATIONS**
{'-'*50}
"""
        for specialty, result in agent_results.items():
            if isinstance(result, dict):
                findings = result.get('findings', 'No specific findings')
                recommendations = result.get('recommendations', 'Standard consultation recommended')
                diagnosis = result.get('diagnosis', 'Requires specialist evaluation')
            else:
                findings = str(result)
                recommendations = f"Consult {specialty.title()} specialist"
                diagnosis = f"Requires {specialty.title()} assessment"
            if "Let's analyze" in findings or "Let's assume" in findings:
                findings = "Analysis indicates need for specialist evaluation. Specific findings require detailed clinical assessment."
            report += f"""
🔹 **{specialty.upper()} CONSULTATION**
   **Clinical Assessment:** {diagnosis}
   **Key Findings:** {findings}
   **Recommendations:** {recommendations}
"""
        report += f"""
📋 **FINAL DIAGNOSIS**
{'-'*50}
🔸 **Primary Diagnosis:** {comprehensive_summary.get('diagnosis', 'Requires further evaluation')}
🔸 **Differential Diagnoses:** {comprehensive_summary.get('differentials', 'Multiple differentials considered')}

💊 **TREATMENT RECOMMENDATIONS**
{'-'*50}
{comprehensive_summary.get('recommendations', 'Standard medical care recommended')}

📅 **FOLLOW-UP PLAN**
{'-'*50}
{comprehensive_summary.get('followup', 'Regular follow-up recommended')}

📝 **CLINICAL SUMMARY**
{'-'*50}
{comprehensive_summary.get('clinical_summary', 'Patient requires comprehensive medical evaluation and management')}

{'='*80}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AI-Powered Medical Analysis System
{'='*80}
"""
        return report.strip()
    except Exception as e:
        logger.error(f"Failed to format final report: {e}")
        return format_fallback_report(patient_info, agent_results)

def format_fallback_report(patient_info, agent_results):
    fallback_report = f"""
{'='*80}
                    MEDICAL DIAGNOSTIC ANALYSIS REPORT
{'='*80}

📋 **PATIENT INFORMATION**
{'-'*50}
{patient_info[:500]}{'...' if len(patient_info) > 500 else ''}

🏥 **SPECIALTY CONSULTATIONS**
{'-'*50}
"""
    for specialty, result in agent_results.items():
        if isinstance(result, dict):
            findings = result.get('findings', 'No specific findings')
        else:
            findings = str(result)
        fallback_report += f"""
🔹 **{specialty.upper()} CONSULTATION**
   **Assessment:** Requires specialist evaluation
   **Findings:** {findings[:200]}{'...' if len(findings) > 200 else ''}
"""
    fallback_report += f"""
📋 **RECOMMENDATIONS**
{'-'*50}
• Consult relevant medical specialists for comprehensive evaluation
• Schedule follow-up appointments as recommended by specialists
• Monitor symptoms and report any changes

{'='*80}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AI-Powered Medical Analysis System
{'='*80}
"""
    return fallback_report.strip() 