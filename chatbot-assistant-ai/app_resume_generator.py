# AI-Powered Resume Generator
import streamlit as st
import requests
from fpdf import FPDF
import re
from langdetect import detect
import json
from datetime import datetime
import base64
import os
import tempfile
import time
from io import BytesIO

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_API_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "mistral-nemo:latest"

class ResumeGenerator:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'fr': 'French', 
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        
        self.ats_prompts = {
            'technical': "Focus on technical skills, certifications, and quantifiable achievements. Use industry-standard keywords and metrics.",
            'creative': "Emphasize creative projects, portfolios, and innovative solutions. Highlight artistic and design skills.",
            'management': "Focus on leadership experience, team management, and strategic achievements. Include budget management and team size.",
            'sales': "Highlight sales figures, targets achieved, client relationships, and revenue generation.",
            'general': "Create a balanced resume with clear sections and ATS-friendly formatting."
        }

    def check_ollama_connection(self):
        """Vérifier la connexion à Ollama et la disponibilité du modèle"""
        try:
            # Test de connexion basique
            response = requests.get(OLLAMA_API_URL, timeout=10)
            if response.status_code != 200:
                return False, "Ollama service not responding"
            
            # Vérifier les modèles disponibles
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if not any(DEFAULT_MODEL in name for name in model_names):
                available_models = ', '.join(model_names) if model_names else "None"
                return False, f"Model {DEFAULT_MODEL} not found. Available models: {available_models}"
            
            return True, "Connection successful"
            
        except requests.exceptions.Timeout:
            return False, "Timeout connecting to Ollama service"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama service. Is it running on localhost:11434?"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def detect_language(self, text):
        """Detect the language of input text"""
        try:
            detected_lang = detect(text)
            return detected_lang if detected_lang in self.supported_languages else 'en'
        except:
            return 'en'

    def get_language_specific_prompt(self, language_code):
        """Get language-specific formatting instructions"""
        language_instructions = {
            'en': "Format the resume in English with standard US/UK formatting conventions.",
            'fr': "Format the resume in French following European CV standards.",
            'es': "Format the resume in Spanish with Latin American/Spanish formatting.",
            'de': "Format the resume in German following German CV conventions.",
            'it': "Format the resume in Italian with European formatting standards.",
            'pt': "Format the resume in Portuguese with Brazilian/Portuguese conventions."
        }
        return language_instructions.get(language_code, language_instructions['en'])

    def generate_fallback_resume(self, name, job_role, experience, skills, education, summary):
        """Generate a basic resume template when AI is not available"""
        return f"""
===========================================
PROFESSIONAL RESUME
===========================================

{name.upper()}
Target Position: {job_role}

===========================================
PROFESSIONAL SUMMARY
===========================================

{summary}

Experience Level: {experience} years in the field

===========================================
CORE COMPETENCIES
===========================================

{skills}

===========================================
PROFESSIONAL EXPERIENCE
===========================================

{job_role} | {experience} Years
• Developed and implemented solutions using core competencies
• Collaborated with cross-functional teams to deliver projects
• Applied technical skills to solve complex business challenges
• Achieved measurable results through systematic approach

===========================================
EDUCATION
===========================================

{education}

===========================================
KEY ACHIEVEMENTS
===========================================

• Strong foundation in {skills.split(',')[0].strip() if skills else 'relevant technologies'}
• Proven track record of {experience} years in professional development
• Expertise in multiple technical domains and methodologies

Contact Information: [Your Phone] | [Your Email] | [Your LinkedIn]
"""

    def generate_structured_prompt(self, name, job_role, experience, skills, education, 
                                 summary, language, ats_type, additional_sections):
        """Generate a comprehensive, structured prompt for resume generation"""
        
        detected_lang = self.detect_language(f"{name} {job_role} {summary}")
        language_instruction = self.get_language_specific_prompt(detected_lang)
        ats_instruction = self.ats_prompts.get(ats_type, self.ats_prompts['general'])
        
        base_prompt = f"""
Create a professional, ATS-optimized resume with the following specifications:

PERSONAL INFORMATION:
- Name: {name}
- Target Role: {job_role}
- Years of Experience: {experience}

CORE DETAILS:
- Skills: {skills}
- Education: {education}
- Professional Summary: {summary}

FORMATTING REQUIREMENTS:
- {language_instruction}
- {ats_instruction}
- Use clear section headers: PROFESSIONAL SUMMARY, EXPERIENCE, EDUCATION, SKILLS, etc.
- Include bullet points for achievements with quantifiable metrics where possible
- Ensure ATS compatibility with standard formatting
- Professional tone and language

ADDITIONAL SECTIONS TO INCLUDE:
{additional_sections}

STRUCTURE:
1. Header with contact information placeholder
2. Professional Summary (3-4 lines)
3. Core Competencies/Skills
4. Professional Experience (with achievements and metrics)
5. Education
6. Additional sections as specified

Generate a complete, professional resume that would pass ATS screening and impress hiring managers.
"""
        return base_prompt

    def generate_resume(self, name, job_role, experience, skills, education, 
                       summary, language, ats_type, additional_sections=""):
        """Generate resume using Mistral-Nemo model with fallback"""
        
        # Check connection first
        is_connected, connection_message = self.check_ollama_connection()
        
        if not is_connected:
            st.warning(f"AI Service Issue: {connection_message}")
            st.info("Using fallback template generation...")
            return self.generate_fallback_resume(name, job_role, experience, skills, education, summary)
        
        prompt = self.generate_structured_prompt(
            name, job_role, experience, skills, education, 
            summary, language, ats_type, additional_sections
        )
        
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 2000
            }
        }
        
        try:
            # Increased timeout and better error handling
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json().get("response", "")
                if result:
                    return self.format_resume_sections(result)
                else:
                    st.warning("Empty response from AI. Using fallback template...")
                    return self.generate_fallback_resume(name, job_role, experience, skills, education, summary)
            else:
                st.error(f"AI Service Error: {response.status_code} - {response.text}")
                return self.generate_fallback_resume(name, job_role, experience, skills, education, summary)
                
        except requests.exceptions.Timeout:
            st.warning("AI generation timed out. Using fallback template...")
            return self.generate_fallback_resume(name, job_role, experience, skills, education, summary)
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {str(e)}")
            return self.generate_fallback_resume(name, job_role, experience, skills, education, summary)

    def format_resume_sections(self, resume_text):
        """Format resume into clear sections"""
        # Add section separators and formatting
        sections = [
            "PROFESSIONAL SUMMARY", "CORE COMPETENCIES", "PROFESSIONAL EXPERIENCE",
            "EDUCATION", "SKILLS", "CERTIFICATIONS", "PROJECTS", "ACHIEVEMENTS"
        ]
        
        formatted_text = resume_text
        for section in sections:
            formatted_text = re.sub(
                f"({section})", 
                f"\n{'='*50}\n\\1\n{'='*50}\n", 
                formatted_text, 
                flags=re.IGNORECASE
            )
        
        return formatted_text

    def clean_text_for_pdf(self, text):
        """Clean text to remove problematic Unicode characters for PDF generation"""
        # Dictionary of common Unicode characters and their ASCII replacements
        unicode_replacements = {
            '\u0153': 'oe',  # œ
            '\u0152': 'OE',  # Œ
            '\u00e0': 'a',   # à
            '\u00e1': 'a',   # á
            '\u00e2': 'a',   # â
            '\u00e4': 'a',   # ä
            '\u00e8': 'e',   # è
            '\u00e9': 'e',   # é
            '\u00ea': 'e',   # ê
            '\u00eb': 'e',   # ë
            '\u00ec': 'i',   # ì
            '\u00ed': 'i',   # í
            '\u00ee': 'i',   # î
            '\u00ef': 'i',   # ï
            '\u00f2': 'o',   # ò
            '\u00f3': 'o',   # ó
            '\u00f4': 'o',   # ô
            '\u00f6': 'o',   # ö
            '\u00f9': 'u',   # ù
            '\u00fa': 'u',   # ú
            '\u00fb': 'u',   # û
            '\u00fc': 'u',   # ü
            '\u00f1': 'n',   # ñ
            '\u00e7': 'c',   # ç
            '\u2013': '-',   # en dash
            '\u2014': '-',   # em dash
            '\u2018': "'",   # left single quotation mark
            '\u2019': "'",   # right single quotation mark
            '\u201c': '"',   # left double quotation mark
            '\u201d': '"',   # right double quotation mark
            '\u2022': '•',   # bullet point
            '\u2026': '...' # ellipsis
        }
        
        # Replace Unicode characters
        cleaned_text = text
        for unicode_char, replacement in unicode_replacements.items():
            cleaned_text = cleaned_text.replace(unicode_char, replacement)
        
        # Remove any remaining non-ASCII characters
        cleaned_text = ''.join(char if ord(char) < 128 else '?' for char in cleaned_text)
        
        return cleaned_text

    def generate_multi_page_pdf(self, resume_text, name):
        """Generate a multi-page PDF with proper formatting and encoding handling"""
        
        # Clean the text first
        clean_resume_text = self.clean_text_for_pdf(resume_text)
        clean_name = self.clean_text_for_pdf(name)
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'Professional Resume', 0, 1, 'C')
                self.ln(5)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
            def safe_cell(self, w, h, txt='', border=0, ln=0, align=''):
                """Safe cell method that handles encoding issues"""
                try:
                    # Clean the text before adding to cell
                    safe_txt = txt.encode('latin-1', 'replace').decode('latin-1')
                    self.cell(w, h, safe_txt, border, ln, align)
                except UnicodeEncodeError:
                    # Fallback: remove problematic characters
                    safe_txt = ''.join(char if ord(char) < 128 else '?' for char in txt)
                    self.cell(w, h, safe_txt, border, ln, align)

        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Process text line by line
        lines = clean_resume_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                pdf.ln(3)
                continue
                
            try:
                # Handle section headers
                if '=' in line and len(line) > 20:  # Section separator
                    pdf.ln(5)
                    continue
                elif line.isupper() and len(line.split()) <= 4:  # Section title
                    pdf.set_font('Arial', 'B', 14)
                    pdf.safe_cell(0, 8, line, 0, 1, 'L')
                    pdf.ln(2)
                elif line.startswith('•') or line.startswith('-'):  # Bullet points
                    pdf.set_font('Arial', '', 10)
                    # Handle long bullet points with word wrap
                    wrapped_lines = self.wrap_text(line, 85)
                    for wrapped_line in wrapped_lines:
                        pdf.safe_cell(0, 6, wrapped_line, 0, 1, 'L')
                else:  # Regular text
                    pdf.set_font('Arial', '', 11)
                    # Handle long lines with word wrap
                    wrapped_lines = self.wrap_text(line, 90)
                    for wrapped_line in wrapped_lines:
                        pdf.safe_cell(0, 7, wrapped_line, 0, 1, 'L')
            
            except Exception as e:
                # Skip problematic lines but continue processing
                continue
        
        # Generate PDF bytes safely without temp files
        try:
            # Use BytesIO instead of temp files to avoid Windows file locking issues
            from io import BytesIO
            pdf_buffer = BytesIO()
            
            # Get PDF content as string and encode properly
            pdf_content = pdf.output(dest='S')
            
            # Handle encoding - FPDF returns bytes or string depending on version
            if isinstance(pdf_content, str):
                pdf_bytes = pdf_content.encode('latin-1')
            else:
                pdf_bytes = pdf_content
                
            return pdf_bytes
        
        except Exception as e:
            # Ultimate fallback - create a simple text-based PDF
            st.warning(f"Advanced PDF generation failed: {str(e)}. Using simple format.")
            return self.generate_simple_pdf(clean_resume_text, clean_name)

    def generate_simple_pdf(self, resume_text, name):
        """Generate a simple PDF as fallback without temp files"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        
        # Split text into lines and add each line
        lines = resume_text.split('\n')
        for line in lines:
            if len(line.strip()) > 0:
                # Ensure line fits in PDF width
                if len(line) > 90:
                    words = line.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) <= 90:
                            current_line += (" " + word) if current_line else word
                        else:
                            if current_line:
                                try:
                                    pdf.cell(0, 6, current_line, 0, 1)
                                except:
                                    pdf.cell(0, 6, "Content encoding error", 0, 1)
                            current_line = word
                    if current_line:
                        try:
                            pdf.cell(0, 6, current_line, 0, 1)
                        except:
                            pdf.cell(0, 6, "Content encoding error", 0, 1)
                else:
                    try:
                        pdf.cell(0, 6, line, 0, 1)
                    except:
                        pdf.cell(0, 6, "Content encoding error", 0, 1)
            else:
                pdf.ln(3)
        
        # Return as bytes without temp file
        try:
            # Get PDF content directly
            pdf_content = pdf.output(dest='S')
            
            # Handle encoding - FPDF returns bytes or string depending on version
            if isinstance(pdf_content, str):
                pdf_bytes = pdf_content.encode('latin-1')
            else:
                pdf_bytes = pdf_content
                
            return pdf_bytes
        except Exception as e:
            # If all else fails, create a minimal text PDF
            st.error(f"All PDF generation methods failed: {str(e)}")
            # Return a basic PDF with error message
            error_pdf = FPDF()
            error_pdf.add_page()
            error_pdf.set_font('Arial', '', 12)
            error_pdf.cell(0, 10, "PDF generation encountered encoding issues.", 0, 1)
            error_pdf.cell(0, 10, "Please use the text download option.", 0, 1)
            
            try:
                error_content = error_pdf.output(dest='S')
                if isinstance(error_content, str):
                    return error_content.encode('latin-1', 'replace')
                else:
                    return error_content
            except:
                # Return empty bytes if even error PDF fails
                return b""

    def wrap_text(self, text, max_length):
        """Wrap text to fit PDF width"""
        if len(text) <= max_length:
            return [text]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines

def main():
    st.set_page_config(
        page_title="AI Resume Generator Pro",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🚀 AI-Powered Resume Generator Pro")
    st.markdown("*Powered by Mistral-Nemo with Advanced ATS Optimization*")
    
    # Initialize generator
    generator = ResumeGenerator()
    
    # Check Ollama status in sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Connection status
        with st.spinner("Checking AI service..."):
            is_connected, status_message = generator.check_ollama_connection()
        
        if is_connected:
            st.success(f"✅ AI Service: Connected")
        else:
            st.error(f"❌ AI Service: {status_message}")
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **To fix AI connection issues:**
                
                1. **Start Ollama service:**
                   ```bash
                   ollama serve
                   ```
                
                2. **Install the model:**
                   ```bash
                   ollama pull mistral-nemo:latest
                   ```
                
                3. **Verify installation:**
                   ```bash
                   ollama list
                   ```
                
                **Note:** The app will use a template generator if AI is unavailable.
                """)
        
        st.divider()
        st.header("⚙️ Advanced Options")
        
        # Language selection
        selected_language = st.selectbox(
            "Resume Language",
            options=list(generator.supported_languages.keys()),
            format_func=lambda x: generator.supported_languages[x]
        )
        
        # ATS optimization type
        ats_type = st.selectbox(
            "ATS Optimization Type",
            options=list(generator.ats_prompts.keys()),
            help="Select the type of role for optimized keyword usage"
        )
        
        # Additional sections
        st.subheader("Additional Sections")
        additional_sections = []
        
        if st.checkbox("Certifications"):
            additional_sections.append("- Professional Certifications section")
        if st.checkbox("Projects"):
            additional_sections.append("- Key Projects section with descriptions")
        if st.checkbox("Achievements/Awards"):
            additional_sections.append("- Achievements and Awards section")
        if st.checkbox("Languages"):
            additional_sections.append("- Language Proficiency section")
        if st.checkbox("Publications"):
            additional_sections.append("- Publications and Research section")
        
        additional_sections_text = "\n".join(additional_sections)

    # Main form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Personal Information")
        name = st.text_input("Full Name *", placeholder="John Doe")
        job_role = st.text_input("Target Job Role *", placeholder="Software Engineer")
        experience = st.slider("Years of Experience", 0, 50, 5)
        
        st.subheader("🎓 Education")
        education = st.text_area(
            "Education Details *", 
            placeholder="B.Sc. Computer Science\nUniversity Name, Year",
            height=100
        )
    
    with col2:
        st.subheader("💼 Professional Details")
        skills = st.text_area(
            "Skills (comma-separated) *",
            placeholder="Python, Machine Learning, Cloud Computing, Project Management",
            height=100
        )
        
        summary = st.text_area(
            "Professional Summary *",
            placeholder="Experienced software engineer with 5+ years in developing scalable applications...",
            height=120
        )

    # Generate button
    if st.button("🎯 Generate Professional Resume", type="primary"):
        if not all([name, job_role, skills, education, summary]):
            st.error("Please fill in all required fields marked with *")
        else:
            with st.spinner("Generating your professional resume..."):
                resume_text = generator.generate_resume(
                    name, job_role, experience, skills, education, 
                    summary, selected_language, ats_type, additional_sections_text
                )
                
            # Store in session state for downloads
            st.session_state.resume_text = resume_text
            st.session_state.user_name = name
                
            # Display results
            st.success("Resume generated successfully!")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📄 Resume Preview", "📊 ATS Analysis", "⬇️ Download"])
            
            with tab1:
                st.text_area("Generated Resume", resume_text, height=600)
            
            with tab2:
                st.subheader("ATS Optimization Analysis")
                
                # Simple ATS analysis
                word_count = len(resume_text.split())
                skills_list = [skill.strip() for skill in skills.split(',')]
                keyword_count = len([skill for skill in skills_list if skill.lower() in resume_text.lower()])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Word Count", word_count)
                col2.metric("Keywords Found", f"{keyword_count}/{len(skills_list)}")
                col3.metric("ATS Score", f"{min(100, (keyword_count/len(skills_list))*100):.0f}%")
                
                st.info(f"**Selected ATS Type:** {ats_type.title()}\n\n**Optimization:** {generator.ats_prompts[ats_type]}")
            
            with tab3:
                st.subheader("Download Options")
                
                # Text download
                st.download_button(
                    label="📄 Download as Text",
                    data=resume_text,
                    file_name=f"{name.replace(' ', '_')}_resume.txt",
                    mime="text/plain"
                )
                
                # PDF generation and download with improved error handling
                if st.button("📋 Generate PDF Resume"):
                    try:
                        with st.spinner("Creating PDF..."):
                            # Add small delay to ensure file system is ready
                            time.sleep(0.1)
                            pdf_bytes = generator.generate_multi_page_pdf(resume_text, name)
                            
                            if pdf_bytes and len(pdf_bytes) > 0:
                                st.download_button(
                                    label="⬇️ Download PDF Resume",
                                    data=pdf_bytes,
                                    file_name=f"{name.replace(' ', '_')}_resume.pdf",
                                    mime="application/pdf",
                                    key="main_pdf_download"
                                )
                                st.success("PDF ready for download!")
                            else:
                                st.error("PDF generation failed - empty file created")
                                st.info("Please use the text download option above.")
                    
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
                        st.info("Please try the text download option above.")
                        
                        # Offer alternative - immediate text download
                        st.markdown("**Alternative download:**")
                        st.download_button(
                            label="📄 Backup Text Download",
                            data=resume_text,
                            file_name=f"{name.replace(' ', '_')}_resume_backup.txt",
                            mime="text/plain",
                            key="backup_text_download"
                        )

    # Show download section if resume exists in session
    elif hasattr(st.session_state, 'resume_text'):
        st.info("Previous resume available for download")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Previous Resume (Text)",
                data=st.session_state.resume_text,
                file_name=f"{st.session_state.user_name.replace(' ', '_')}_resume.txt",
                mime="text/plain",
                key="previous_text_download"
            )
        
        with col2:
            if st.button("📋 Generate Previous Resume PDF"):
                try:
                    with st.spinner("Creating PDF from previous resume..."):
                        time.sleep(0.1)
                        pdf_bytes = generator.generate_multi_page_pdf(
                            st.session_state.resume_text, 
                            st.session_state.user_name
                        )
                        
                        if pdf_bytes and len(pdf_bytes) > 0:
                            st.download_button(
                                label="⬇️ Download Previous Resume (PDF)",
                                data=pdf_bytes,
                                file_name=f"{st.session_state.user_name.replace(' ', '_')}_resume.pdf",
                                mime="application/pdf",
                                key="previous_pdf_download"
                            )
                            st.success("Previous resume PDF ready!")
                        else:
                            st.error("Previous resume PDF generation failed")
                            
                except Exception as e:
                    st.error(f"PDF generation error: {str(e)}")
                    st.info("Use the text download option instead.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>🤖 Powered by Mistral-Nemo AI | 📈 ATS-Optimized | 🌐 Multi-Language Support</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()