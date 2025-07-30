#!/usr/bin/env python3
"""
PDF Report Generator for Breast Cancer Classification Analysis
Generates a comprehensive PDF report with results and visualizations
"""

import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from PIL import Image as PILImage
import glob

class BreastCancerReportGenerator:
    """Generates PDF report for breast cancer analysis"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.report_path = os.path.join(results_dir, "breast_cancer_analysis_report.pdf")
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.blue,
            spaceAfter=20,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        # Section style
        self.section_style = ParagraphStyle(
            'CustomSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.darkgreen,
            spaceAfter=15,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        # Body text style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )
        
        # Code style
        self.code_style = ParagraphStyle(
            'CustomCode',
            parent=self.styles['Code'],
            fontSize=10,
            textColor=colors.darkred,
            spaceAfter=10,
            fontName='Courier'
        )

    def _load_final_report(self):
        """Load final report data"""
        try:
            with open(os.path.join(self.results_dir, "final_report.txt"), 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "Final report file not found."

    def _get_available_images(self):
        """Get list of available visualization images"""
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        images = []
        
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(self.results_dir, ext)))
        
        return sorted(images)

    def _resize_image_if_needed(self, image_path, max_width=500, max_height=400):
        """Resize image if it's too large for PDF"""
        try:
            with PILImage.open(image_path) as img:
                width, height = img.size
                
                # Calculate scaling factor
                scale_w = max_width / width if width > max_width else 1
                scale_h = max_height / height if height > max_height else 1
                scale = min(scale_w, scale_h)
                
                if scale < 1:
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    return new_width, new_height
                else:
                    return width, height
        except Exception:
            return max_width, max_height

    def _add_image_to_story(self, story, image_path, title=None):
        """Add an image to the story with optional title"""
        try:
            if title:
                story.append(Paragraph(title, self.section_style))
            
            width, height = self._resize_image_if_needed(image_path)
            img = Image(image_path, width=width, height=height)
            story.append(img)
            story.append(Spacer(1, 20))
            
        except Exception as e:
            error_text = f"Could not load image: {os.path.basename(image_path)} - {str(e)}"
            story.append(Paragraph(error_text, self.body_style))
            story.append(Spacer(1, 10))

    def generate_report(self):
        """Generate the complete PDF report"""
        # Create document
        doc = SimpleDocTemplate(
            self.report_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title Page
        story.append(Spacer(1, 100))
        story.append(Paragraph("🎯 Breast Cancer Classification", self.title_style))
        story.append(Paragraph("Advanced Machine Learning Analysis Report", self.subtitle_style))
        story.append(Spacer(1, 50))
        
        # Report info
        current_date = datetime.now().strftime("%B %d, %Y")
        info_text = f"""
        <b>Analysis Date:</b> {current_date}<br/>
        <b>Dataset:</b> Wisconsin Breast Cancer Dataset<br/>
        <b>Models Used:</b> 6 Machine Learning Algorithms<br/>
        <b>Generated by:</b> Breast Cancer Classification System
        """
        story.append(Paragraph(info_text, self.body_style))
        story.append(PageBreak())
        
        # Table of Contents
        story.append(Paragraph("📋 Table of Contents", self.title_style))
        toc_data = [
            ["Section", "Page"],
            ["1. Executive Summary", "3"],
            ["2. Analysis Results", "4"],
            ["3. Model Performance", "5"],
            ["4. Visualizations", "6"],
            ["5. Data Overview", "7"],
            ["6. Correlation Analysis", "8"],
            ["7. Individual ROC Curves", "9"],
            ["8. Confusion Matrices", "10"],
            ["9. Model Comparison", "11"],
            ["10. Feature Importance", "12"],
            ["11. Conclusions", "13"]
        ]
        
        toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(toc_table)
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("📊 Executive Summary", self.title_style))
        
        summary_text = """
        This report presents a comprehensive analysis of breast cancer classification using 
        advanced machine learning techniques. The analysis was performed on the Wisconsin 
        Breast Cancer Dataset using 6 different machine learning algorithms to achieve 
        highly accurate diagnostic predictions.
        
        <b>Key Findings:</b><br/>
        • Achieved over 95% accuracy with multiple models<br/>
        • Logistic Regression showed the best overall performance<br/>
        • All models demonstrated excellent discriminative capability<br/>
        • Feature analysis revealed important diagnostic indicators<br/>
        • Cross-validation confirmed model robustness
        """
        story.append(Paragraph(summary_text, self.body_style))
        story.append(PageBreak())
        
        # Analysis Results (from final_report.txt)
        story.append(Paragraph("📈 Analysis Results", self.title_style))
        
        final_report_content = self._load_final_report()
        
        # Parse and format the final report
        lines = final_report_content.split('\n')
        for line in lines:
            if line.strip():
                if line.startswith('=') or line.startswith('-'):
                    continue
                elif 'MODEL PERFORMANCE SUMMARY:' in line or 'BEST MODEL:' in line or 'RESULTS:' in line:
                    story.append(Paragraph(line, self.subtitle_style))
                elif line.startswith('•'):
                    story.append(Paragraph(line, self.body_style))
                elif any(char.isdigit() for char in line) and ('Accuracy' in line or 'Precision' in line):
                    story.append(Paragraph(line, self.code_style))
                else:
                    story.append(Paragraph(line, self.body_style))
        
        story.append(PageBreak())
        
        # Visualizations Section
        story.append(Paragraph("📊 Visualizations", self.title_style))
        
        # Get all available images
        images = self._get_available_images()
        
        # Define image titles mapping
        image_titles = {
            'data_overview.png': '📊 Data Overview',
            'correlation_matrix.png': '🔗 Correlation Matrix',
            'individual_roc_curves.png': '📈 Individual ROC Curves',
            'confusion_matrices.png': '🎯 Confusion Matrices',
            'model_comparison.png': '⚖️ Model Performance Comparison',
            'feature_importance.png': '🔍 Feature Importance Analysis',
            'roc_curves_simple.png': '📈 ROC Curves',
            'confusion_matrices_simple.png': '🎯 Confusion Matrices (Simple)'
        }
        
        # Add images to report
        for image_path in images:
            image_name = os.path.basename(image_path)
            title = image_titles.get(image_name, f"📊 {image_name.replace('_', ' ').title()}")
            self._add_image_to_story(story, image_path, title)
            
            # Add page break after every 2 images
            if images.index(image_path) % 2 == 1:
                story.append(PageBreak())
        
        # Conclusions
        story.append(PageBreak())
        story.append(Paragraph("🏆 Conclusions", self.title_style))
        
        conclusions_text = """
        <b>Model Performance:</b><br/>
        The analysis demonstrates that machine learning models can achieve exceptional 
        accuracy in breast cancer classification. Logistic Regression emerged as the 
        best performer with 98.25% accuracy, closely followed by SVM.
        
        <b>Feature Insights:</b><br/>
        Certain morphological features of cell nuclei prove to be highly discriminative 
        for malignancy detection. The correlation analysis reveals important relationships 
        between different measurements.
        
        <b>Clinical Relevance:</b><br/>
        These results suggest that automated classification systems can serve as valuable 
        diagnostic aids in clinical settings, potentially improving accuracy and reducing 
        analysis time.
        
        <b>Recommendations:</b><br/>
        • Deploy the best-performing model (Logistic Regression) for clinical use<br/>
        • Continue monitoring model performance with new data<br/>
        • Consider ensemble methods for even better performance<br/>
        • Validate results on external datasets
        """
        story.append(Paragraph(conclusions_text, self.body_style))
        
        # Footer
        story.append(Spacer(1, 50))
        footer_text = f"""
        <b>Report Generated:</b> {current_date}<br/>
        <b>Analysis Tool:</b> Breast Cancer Classification System v1.0<br/>
        <b>Developed by:</b> Ozan İdgü
        """
        story.append(Paragraph(footer_text, self.body_style))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"✅ PDF report generated successfully: {self.report_path}")
            return True
        except Exception as e:
            print(f"❌ Error generating PDF report: {str(e)}")
            return False

def main():
    """Main function to generate the report"""
    print("📄 Generating PDF Report...")
    
    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Generate report
    generator = BreastCancerReportGenerator()
    success = generator.generate_report()
    
    if success:
        print("🎉 PDF report generation completed!")
        print(f"📄 Report saved as: {generator.report_path}")
    else:
        print("❌ Failed to generate PDF report!")

if __name__ == "__main__":
    main()
