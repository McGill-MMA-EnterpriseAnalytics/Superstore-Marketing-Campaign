#!/usr/bin/env python
"""
Scheduled Drift Check Script

This script runs the enhanced drift detector on a schedule and logs results to MLflow.
It can be called manually or set up as a cron job/scheduled task.

Usage:
    python scheduled_drift_check.py --config config.yaml --log-to-mlflow

Options:
    --config: Path to configuration file (default: config.yaml)
    --log-to-mlflow: Whether to log results to MLflow (default: True)
    --send-alerts: Whether to send alert notifications (default: False)
    --email: Email address to send alerts to (optional)
"""

import argparse
import logging
import os
import smtplib
import sys
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# Add parent directory to path to allow running from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.monitoring.enhanced_drift_detector import EnhancedDriftDetector
from src.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drift_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def send_email_alert(email_to, drift_results, config):
    """
    Send email alert for drift detection.
    
    Parameters:
        email_to (str): Recipient email address
        drift_results (dict): Drift detection results
        config (dict): Email configuration
    
    Returns:
        bool: Whether the email was sent successfully
    """
    if not email_to:
        logger.warning("No email address provided for alerts")
        return False
    
    try:
        # Get email configuration
        email_config = config.get("email", {})
        email_from = email_config.get("from", "drift-monitor@example.com")
        smtp_server = email_config.get("smtp_server", "localhost")
        smtp_port = email_config.get("smtp_port", 25)
        smtp_user = email_config.get("smtp_user", "")
        smtp_password = email_config.get("smtp_password", "")
        
        # Count alerts by dataset
        alerts_by_dataset = {}
        for dataset, results in drift_results.items():
            if results["alerts"]:
                alerts_by_dataset[dataset] = results["alerts"]
        
        if not alerts_by_dataset:
            logger.info("No alerts to send")
            return True
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = f"Drift Monitor Alert: {len(alerts_by_dataset)} datasets with drift detected"
        
        # Create HTML for the body
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h2 {{ color: #d9534f; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .alert {{ color: #d9534f; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h2>Drift Monitor Alert</h2>
            <p>Drift has been detected in {len(alerts_by_dataset)} datasets.</p>
            
            <h3>Alert Summary:</h3>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Alerts</th>
                    <th>Feature Drift</th>
                    <th>Target Drift</th>
                    <th>Prediction Drift</th>
                    <th>Concept Drift</th>
                </tr>
        """
        
        for dataset, alerts in alerts_by_dataset.items():
            results = drift_results[dataset]
            html += f"""
                <tr>
                    <td>{dataset}</td>
                    <td><span class="alert">{', '.join(alerts)}</span></td>
                    <td>{results['feature_drift']['avg_wasserstein']:.4f}</td>
                    <td>{results['target_drift']['l1_distance']:.4f}</td>
                    <td>{results['prediction_drift']['l1_distance']:.4f}</td>
                    <td>{results['concept_drift']['f1_drop']:.4f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h3>Recommendations:</h3>
            <ul>
                <li>Review the drift history and model performance in MLflow</li>
                <li>Check the visualization notebook for detailed analysis</li>
                <li>Consider retraining the model if drift persists</li>
            </ul>
            
            <p>This is an automated alert from the Drift Monitoring System.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if smtp_user and smtp_password:
                server.starttls()
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Sent alert email to {email_to}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to send email alert: {str(e)}")
        return False


def main():
    """Main entry point for scheduled drift check."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run scheduled drift check")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--log-to-mlflow", action="store_true", help="Log results to MLflow")
    parser.add_argument("--send-alerts", action="store_true", help="Send alert notifications")
    parser.add_argument("--email", type=str, help="Email address to send alerts to")
    args = parser.parse_args()
    
    # Log startup
    logger.info(f"Starting scheduled drift check at {datetime.now().isoformat()}")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Log to MLflow: {args.log_to_mlflow}")
    logger.info(f"Send alerts: {args.send_alerts}")
    
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize and run drift detector
        detector = EnhancedDriftDetector(args.config)
        results = detector.run_all_checks(log_to_mlflow=args.log_to_mlflow)
        
        # Send alerts if configured
        if args.send_alerts and args.email:
            send_email_alert(args.email, results, config)
        
        # Log completion
        elapsed_time = time.time() - start_time
        dataset_count = len(results)
        alert_count = sum(len(r["alerts"]) for r in results.values())
        
        logger.info(f"Completed drift detection for {dataset_count} datasets in {elapsed_time:.2f} seconds")
        logger.info(f"Total alerts: {alert_count}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running scheduled drift check: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 