from flask import Flask, render_template, request, send_file
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import re
import os
import time
from dotenv import load_dotenv
from google.api_core import exceptions


load_dotenv()

app = Flask(__name__)


api_key = os.getenv('GEMINI_API_KEY')


if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add your API key to the .env file.")

if not api_key.startswith('AIza'):
    raise ValueError("Invalid API key format. Gemini API keys should start with 'AIza'.")


try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("✓ API key loaded and configured successfully")
except Exception as e:
    raise ValueError(f"Failed to configure API key: {str(e)}. Please check your API key in the .env file.")

def get_response(prompt, max_retries=3, initial_delay=2):
    """
    Get response from Gemini API with retry logic and error handling.
    
    Args:
        prompt: The prompt to send to the API
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before retrying
    
    Returns:
        str: The response text from the API
    
    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Configure generation with timeout
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response and response.text:
                return response.text
            else:
                raise ValueError("Empty response from API")
                
        except exceptions.RetryError as e:
            last_exception = e
            error_msg = str(e).lower()
            
            if "timeout" in error_msg or "503" in error_msg or "failed to connect" in error_msg or "socket is null" in error_msg:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⚠ Network error (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(
                        "Network connection error: Unable to connect to the API service. "
                        "Please check your internet connection and try again later. "
                        f"Error details: {str(e)}"
                    )
            else:
                raise Exception(f"API request failed: {str(e)}")
                
        except (exceptions.ServiceUnavailable, exceptions.DeadlineExceeded, exceptions.Unavailable) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"⚠ Service unavailable (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                raise Exception(
                    "Service temporarily unavailable. The API service may be experiencing high load. "
                    "Please try again in a few moments."
                )
        
        except (ConnectionError, TimeoutError, OSError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"⚠ Connection error (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                raise Exception(
                    "Network connection error: Unable to connect to the API service. "
                    "Please check your internet connection and try again later."
                )
                
        except Exception as e:
            # Check if it's a network-related error by examining the error message
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["timeout", "503", "failed to connect", "socket", "connection", "network"]):
                last_exception = e
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"⚠ Network-related error (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(
                        "Network connection error: Unable to connect to the API service. "
                        "Please check your internet connection and try again later. "
                        f"Error: {str(e)}"
                    )
            else:
                # For other exceptions, don't retry
                raise Exception(f"API error: {str(e)}")
    
    # If we've exhausted all retries
    if last_exception:
        raise Exception(
            f"Failed to get response after {max_retries} attempts. "
            "Please check your internet connection and try again. "
            f"Last error: {str(last_exception)}"
        )

def clean_special_characters(text):
    
    text = re.sub(r'\*\*', '', text)  
    text = re.sub(r'#+\s*', '', text)  
    text = re.sub(r'`', '', text)  
    text = re.sub(r'~~', '', text)  
    text = re.sub(r'__', '', text)  
    text = re.sub(r'_', ' ', text)  
    text = re.sub(r'^\s*[-]\s+', '• ', text, flags=re.MULTILINE)  
    text = re.sub(r'^\s*\d+[\.\)]\s+', '• ', text, flags=re.MULTILINE)  
    text = re.sub(r'\n{3,}', '\n\n', text) 
    return text.strip()

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/form')
def form():
    return render_template('index2.html')

@app.route('/result', methods=['POST'])
def result():
    name = request.form.get('name', '')
    edu = request.form.get('edu', '')
    skills = request.form.get('skills', '')
    intrest = request.form.get('intrest', '')
    hobbies = request.form.get('hobbies', '')
    look = request.form.get('look', '')
    project_recommendation = request.form.get('project_recommendation', 'no')
    
    
    prompt = f"""Provide career guidance for the following person in simple bullet points only. Keep it short and clear - no long paragraphs.

Name: {name}
Education: {edu}
Skills: {skills}
Area of Interest: {intrest}
Hobbies: {hobbies}
Career Goals: {look}
Project Recommendation: {project_recommendation}

Provide ONLY main points in bullet format:
1. Suitable Career Paths (list 3-4 options)
2. Recommended Job Roles (list 3-4 roles)
3. Skills to Develop (list 3-4 skills)
4. Next Steps (list 3-4 action items)
{"5. Project Recommendations (list 2-3 projects)" if project_recommendation == "yes" else ""}

Keep each point short - maximum one line per bullet. No explanations or paragraphs."""
    
    
    try:
        ai_response = get_response(prompt)
        ai_response = clean_special_characters(ai_response)
    except Exception as e:
        error_message = str(e)
        # Return to form with error message
        return render_template('r.html', 
                             name=name,
                             edu=edu,
                             skills=skills,
                             intrest=intrest,
                             hobbies=hobbies,
                             look=look,
                             result=f"Error: {error_message}\n\nPlease try again later or check your internet connection.",
                             error=True)
    
    return render_template('r.html', 
                         name=name,
                         edu=edu,
                         skills=skills,
                         intrest=intrest,
                         hobbies=hobbies,
                         look=look,
                         result=ai_response)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    name = request.form.get('name', '')
    edu = request.form.get('edu', '')
    skills = request.form.get('skills', '')
    intrest = request.form.get('intrest', '')
    hobbies = request.form.get('hobbies', '')
    look = request.form.get('look', '')
    result = request.form.get('result', '')
 
    result = clean_special_characters(result)
    
    # Create PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor='#2c3e50',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#34495e',
        spaceAfter=12,
        spaceBefore=12
    )
    
    
    story.append(Paragraph("AI Career Guidance Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    
    story.append(Paragraph("<b>Personal Information</b>", heading_style))
    story.append(Paragraph(f"<b>Name:</b> {name}", styles['Normal']))
    story.append(Paragraph(f"<b>Education:</b> {edu}", styles['Normal']))
    story.append(Paragraph(f"<b>Skills:</b> {skills}", styles['Normal']))
    story.append(Paragraph(f"<b>Area of Interest:</b> {intrest}", styles['Normal']))
    story.append(Paragraph(f"<b>Hobbies:</b> {hobbies}", styles['Normal']))
    story.append(Paragraph(f"<b>Career Goals:</b> {look}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
   
    story.append(Paragraph("<b>Career Guidance</b>", heading_style))
    
    result_lines = result.split('\n')
    for line in result_lines:
        if line.strip():
            story.append(Paragraph(line.strip(), styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    doc.build(story)
    buffer.seek(0)
    
    return send_file(buffer, 
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=f'Career_Guidance_{name.replace(" ", "_")}.pdf')

if __name__ == '__main__':
    app.run(debug=True)
