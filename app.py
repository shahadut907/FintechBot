"""
FinSolve RAG Chatbot - Enhanced Professional Streamlit Frontend
Beautiful, responsive chat interface with role-based access control and advanced features
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, List
import time
import io
import base64

# Optional PDF generation - gracefully handle missing reportlab
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="FinSolve AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
# Debug function for login testing
def debug_login_test():
    st.subheader("🔍 Login Debug Test")
    
    def test_login(email, password):
        url = f"{API_BASE_URL}/auth/login"
        
        data = {
            "email": email,
            "password": password
        }
        
        st.write("📤 Sending data:", data)
        st.write("🌐 URL:", url)
        
        try:
            response = requests.post(url, json=data)
            
            st.write("📊 Status Code:", response.status_code)
            st.write("📋 Response Text:", response.text)
            
            if response.status_code == 200:
                st.success("✅ Login Successful!")
                st.json(response.json())
            else:
                st.error(f"❌ Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("🔌 Backend server running নেই!")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    
    email = st.text_input("📧 Email Address", value="user@finsolve.com", key="debug_email")
    password = st.text_input("🔒 Password", type="password", key="debug_password")
    
    if st.button("🚀 Test Login", key="debug_login"):
        if email and password:
            test_login(email, password)
        else:
            st.warning("Email ও Password দিন!")
# Call debug function
    # Sidebar এ debug option
    if st.sidebar.checkbox("🔍 Show Debug"):
        debug_login_test()

#  Enhanced Professional CSS
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main > div {
        padding-top: 1rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px 15px 0 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Chat container */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        background: #fafbfc;
        border-radius: 15px;
        border: 1px solid #e1e8ed;
        margin-bottom: 1.5rem;
    }
    
    /* Message bubbles */
    .chat-message {
        display: flex;
        margin: 1rem 0;
        animation: fadeInUp 0.3s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 1rem 1.2rem;
        border-radius: 20px;
        font-size: 0.95rem;
        line-height: 1.5;
        word-wrap: break-word;
        position: relative;
    }
    
    .user-message {
        justify-content: flex-end;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 8px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        justify-content: flex-start;
    }
    
    .assistant-bubble {
        background: white;
        color: #2c3e50;
        border: 1px solid #e1e8ed;
        border-bottom-left-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin: 0 0.8rem;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        order: 2;
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }
    
    .message-info {
        font-size: 0.75rem;
        color: #8899a6;
        margin-top: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .message-timestamp {
        opacity: 0.7;
    }
    
    .message-sources {
        color: #1da1f2;
        cursor: pointer;
    }
    
    /* Input area */
    .input-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        border: 2px solid #e1e8ed;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        transition: border-color 0.3s ease;
    }
    
    .input-container:focus-within {
        border-color: #667eea;
        box-shadow: 0 2px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-title {
        font-weight: 600;
        font-size: 1rem;
        color: #2c3e50;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.85rem;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-weight: 500;
    }
    
    .status-healthy {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .status-degraded {
        background: #fff3e0;
        color: #f57c00;
    }
    
    .status-offline {
        background: #ffebee;
        color: #d32f2f;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sample query buttons */
    .sample-query-btn {
        background: transparent;
        border: 1px solid #667eea;
        color: #667eea;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sample-query-btn:hover {
        background: #667eea;
        color: white;
        transform: translateY(-1px);
    }
    
    /* Login container */
    .login-container {
        max-width: 450px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e8ed;
    }
    
    .login-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    
    /* Role badges */
    .role-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    
    /* Demo users table */
    .demo-user-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem;
        margin: 0.3rem 0;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    /* Utility classes */
    .text-center { text-align: center; }
    .text-small { font-size: 0.85rem; }
    .text-muted { color: #8899a6; }
    .mt-2 { margin-top: 1rem; }
    .mb-2 { margin-bottom: 1rem; }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main > div {
            margin: 0.5rem;
            border-radius: 15px;
        }
        
        .app-title {
            font-size: 1.8rem;
        }
        
        .message-bubble {
            max-width: 85%;
        }
        
        .login-container {
            margin: 1rem;
            padding: 1.5rem;
        }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


class ChatUtils:
    """Utility class for chat-related operations"""
    
    @staticmethod
    def format_timestamp(timestamp: str) -> str:
        """Format timestamp for display"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%I:%M %p")
        except:
            return datetime.now().strftime("%I:%M %p")
    
    @staticmethod
    def create_download_link(data: str, filename: str, file_type: str) -> str:
        """Create download link for chat history"""
        if file_type == "txt":
            b64 = base64.b64encode(data.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="{filename}" class="download-link">📄 Download as TXT</a>'
        else:  # PDF
            b64 = base64.b64encode(data).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-link">📄 Download as PDF</a>'
        return href
    
    @staticmethod
    def export_chat_to_txt(chat_history: List[Dict]) -> str:
        """Export chat history to TXT format"""
        lines = []
        lines.append("FinSolve AI Assistant - Chat History")
        lines.append("=" * 50)
        lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        for message in chat_history:
            timestamp = ChatUtils.format_timestamp(message.get('timestamp', ''))
            if message['type'] == 'user':
                lines.append(f"[{timestamp}] You:")
                lines.append(f"{message['content']}")
            else:
                lines.append(f"[{timestamp}] FinSolve AI:")
                lines.append(f"{message['content']['response']}")
                if message['content'].get('sources'):
                    lines.append(f"Sources: {', '.join(message['content']['sources'])}")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def export_chat_to_pdf(chat_history: List[Dict]) -> bytes:
        """Export chat history to PDF format using reportlab or fallback to HTML"""
        if REPORTLAB_AVAILABLE:
            return ChatUtils._export_pdf_reportlab(chat_history)
        else:
            # Fallback to HTML-based PDF (requires user to save as PDF manually)
            return ChatUtils._export_html_for_pdf(chat_history)
    
    @staticmethod
    def _export_pdf_reportlab(chat_history: List[Dict]) -> bytes:
        """Export using reportlab library"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=20,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=10,
            textColor=colors.HexColor('#667eea')
        )
        
        user_style = ParagraphStyle(
            'UserMessage',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            leftIndent=20,
            backgroundColor=colors.HexColor('#f0f0f0'),
            borderWidth=1,
            borderColor=colors.HexColor('#e0e0e0'),
            borderPadding=10
        )
        
        ai_style = ParagraphStyle(
            'AIMessage',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=15,
            rightIndent=20,
            backgroundColor=colors.HexColor('#f8f9fa'),
            borderWidth=1,
            borderColor=colors.HexColor('#e0e0e0'),
            borderPadding=10
        )
        
        # Build content
        story = []
        story.append(Paragraph("FinSolve AI Assistant - Chat History", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        for message in chat_history:
            timestamp = ChatUtils.format_timestamp(message.get('timestamp', ''))
            
            if message['type'] == 'user':
                story.append(Paragraph(f"<b>[{timestamp}] You:</b>", heading_style))
                story.append(Paragraph(message['content'], user_style))
            else:
                story.append(Paragraph(f"<b>[{timestamp}] FinSolve AI:</b>", heading_style))
                story.append(Paragraph(message['content']['response'], ai_style))
                
                if message['content'].get('sources'):
                    sources_text = f"<i>Sources: {', '.join(message['content']['sources'])}</i>"
                    story.append(Paragraph(sources_text, styles['Normal']))
            
            story.append(Spacer(1, 10))
        
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data
    
    @staticmethod
    def _export_html_for_pdf(chat_history: List[Dict]) -> bytes:
        """Export as HTML that can be saved as PDF by the browser"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>FinSolve AI Assistant - Chat History</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }}
                .header {{
                    text-align: center;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .message {{
                    margin: 20px 0;
                    padding: 15px;
                    border-radius: 10px;
                }}
                .user-message {{
                    background-color: #e3f2fd;
                    margin-left: 50px;
                }}
                .ai-message {{
                    background-color: #f5f5f5;
                    margin-right: 50px;
                }}
                .timestamp {{
                    font-size: 0.8em;
                    color: #666;
                    margin-bottom: 5px;
                }}
                .sources {{
                    font-size: 0.8em;
                    color: #1976d2;
                    margin-top: 10px;
                    font-style: italic;
                }}
                @media print {{
                    body {{ margin: 0; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 FinSolve AI Assistant - Chat History</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for message in chat_history:
            timestamp = ChatUtils.format_timestamp(message.get('timestamp', ''))
            
            if message['type'] == 'user':
                html_content += f"""
                <div class="message user-message">
                    <div class="timestamp">[{timestamp}] You:</div>
                    <div>{message['content']}</div>
                </div>
                """
            else:
                sources_html = ""
                if message['content'].get('sources'):
                    sources_list = ', '.join(message['content']['sources'])
                    sources_html = f'<div class="sources">Sources: {sources_list}</div>'
                
                html_content += f"""
                <div class="message ai-message">
                    <div class="timestamp">[{timestamp}] FinSolve AI:</div>
                    <div>{message['content']['response']}</div>
                    {sources_html}
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content.encode('utf-8')


class FinSolveChat:
    """Enhanced FinSolve chatbot application class"""
    
    def __init__(self):
        self.api_base = API_BASE_URL
        self.chat_utils = ChatUtils()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        default_states = {
            'authenticated': False,
            'user_info': None,
            'access_token': None,
            'chat_history': [],
            'system_status': None,
            'current_query': '',
            'show_sources': True,
            'auto_scroll': True
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def make_api_request(self, endpoint: str, method: str = "GET", data: Dict = None, auth_required: bool = True) -> Dict[str, Any]:
        """Make API request with comprehensive error handling"""
        url = f"{self.api_base}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if auth_required and st.session_state.access_token:
            headers["Authorization"] = f"Bearer {st.session_state.access_token}"
        
        try:
            if method == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 401:
                st.session_state.authenticated = False
                st.session_state.access_token = None
                st.error("🔒 Your session has expired. Please log in again.")
                return {"success": False, "error": "Authentication required"}
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the AI service. Please ensure the backend server is running.")
            return {"success": False, "error": "Connection error"}
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The AI might be processing a complex query.")
            return {"success": False, "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def render_app_header(self):
        """Render the application header"""
        st.markdown("""
        <div class="app-header">
            <div class="app-title">🤖 FinSolve AI Assistant</div>
            <div class="app-subtitle">Intelligent Document Search & Analysis Platform</div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_login_page(self):
        """Render enhanced login page"""
        self.render_app_header()
        
        # Main login form
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔐 Sign In</div>', unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=True):
            email = st.text_input(
                "📧 Email Address", 
                placeholder="user@finsolve.com",
                help="Enter your FinSolve email address"
            )
            password = st.text_input(
                "🔒 Password", 
                type="password", 
                placeholder="Enter your password",
                help="Enter your account password"
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button(
                    "🚀 Sign In", 
                    use_container_width=True
                )
            
            if submit_button:
                if email and password:
                    self.authenticate(email, password)
                else:
                    st.error("⚠️ Please enter both email and password.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demo users section
        self.render_demo_users()
    
    def render_demo_users(self):
        """Render demo users section"""
        st.markdown("---")
        with st.expander("👥 Demo User Accounts", expanded=False):
            st.markdown("**For testing purposes, you can use these demo accounts:**")
            
            demo_users = [
                {"email": "tony.sharma@finsolve.com", "password": "ceo123", "role": "C-Level Executive", "access": "Full company access"},
                {"email": "peter.pandey@finsolve.com", "password": "eng123", "role": "Engineering", "access": "Technical architecture"},
                {"email": "finance.lead@finsolve.com", "password": "fin123", "role": "Finance", "access": "Financial & marketing data"},
                {"email": "marketing.lead@finsolve.com", "password": "mkt123", "role": "Marketing", "access": "Marketing campaigns"},
                {"email": "hr.lead@finsolve.com", "password": "hr123", "role": "HR", "access": "Employee data & policies"},
                {"email": "employee@finsolve.com", "password": "emp123", "role": "Employee", "access": "Company handbook only"}
            ]
            
            for user in demo_users:
                st.markdown(f"""
                <div class="demo-user-row">
                    <div>
                        <strong>{user['email']}</strong><br>
                        <span class="text-small text-muted">Password: {user['password']}</span>
                    </div>
                    <div>
                        <span class="role-badge">{user['role']}</span><br>
                        <span class="text-small">{user['access']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def authenticate(self, email: str, password: str):
        """Authenticate user with enhanced feedback"""
        with st.spinner("🔐 Authenticating..."):
            result = self.make_api_request("/auth/login", "POST", {
                "email": email,
                "password": password
            }, auth_required=False)
            
            if result.get("access_token"):
                st.session_state.authenticated = True
                st.session_state.access_token = result["access_token"]
                st.session_state.user_info = result["user"]
                st.success(f"✅ Welcome back, {result['user']['name']}!")
                time.sleep(1.5)
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please check your email and password.")
    
    def logout(self):
        """Logout user and clear session"""
        st.session_state.authenticated = False
        st.session_state.access_token = None
        st.session_state.user_info = None
        st.session_state.chat_history = []
        st.session_state.current_query = ''
        st.success("👋 You have been logged out successfully!")
        time.sleep(1)
        st.rerun()
    
    def render_sidebar(self):
        """Render enhanced sidebar with user info and controls"""
        with st.sidebar:
            # User information
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">👤 User Profile</div>', unsafe_allow_html=True)
            
            user = st.session_state.user_info
            st.markdown(f"**{user['name']}**")
            st.markdown(f"<span class='role-badge'>{user['role']}</span>", unsafe_allow_html=True)
            st.markdown(f"📧 {user['email']}")
            st.markdown(f"🏢 {user['department']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # System status
            self.render_system_status()
            
            # Chat controls
            self.render_chat_controls()
            
            # Sample queries
            self.render_sample_queries()
            
            # Download section
            self.render_download_section()
            
            # Logout button
            st.markdown("---")
            if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
                self.logout()
    
    def render_system_status(self):
        """Render system status section"""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 System Status</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Health Check**")
        with col2:
            if st.button("🔄", help="Refresh status", key="refresh_status"):
                self.get_system_status()
        
        if st.session_state.system_status:
            status = st.session_state.system_status
            
            # Overall status
            if status['status'] == 'healthy':
                st.markdown('<span class="status-indicator status-healthy">🟢 Operational</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-indicator status-degraded">🟡 Degraded</span>', unsafe_allow_html=True)
            
            # AI Model status
            if status.get('ollama_status'):
                st.markdown('<span class="status-indicator status-healthy">🤖 AI Online</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-indicator status-offline">🤖 AI Offline</span>', unsafe_allow_html=True)
            
            # Statistics
            st.markdown(f"📄 **Documents:** {status.get('total_documents', 0):,} chunks")
            st.markdown(f"🧠 **Model:** {status.get('model', 'Unknown')}")
        else:
            st.markdown('<span class="status-indicator status-offline">❌ Unavailable</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_chat_controls(self):
        """Render chat control options"""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">⚙️ Chat Settings</div>', unsafe_allow_html=True)
        
        st.session_state.show_sources = st.checkbox(
            "📄 Show Sources", 
            value=st.session_state.get('show_sources', True),
            help="Display source documents for AI responses"
        )
        
        st.session_state.auto_scroll = st.checkbox(
            "📜 Auto-scroll", 
            value=st.session_state.get('auto_scroll', True),
            help="Automatically scroll to latest message"
        )
        
        if st.button("🗑️ Clear Chat", help="Clear all chat history"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sample_queries(self):
        """Render sample queries based on user role"""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">💡 Quick Queries</div>', unsafe_allow_html=True)
        
        sample_queries = {
            "C-Level Executive": [
                "📊 Give me a company overview",
                "💰 What are our key financial metrics?",
                "📈 Compare department performance",
                "⚠️ What are our biggest risks?"
            ],
            "Finance": [
                "💵 What was our Q4 revenue?",
                "📉 Show expense breakdown",
                "📊 Display ROI metrics",
                "💳 Budget vs actual spending"
            ],
            "Marketing": [
                "🎯 Q2 campaign performance",
                "💰 Customer acquisition cost",
                "📈 Marketing ROI analysis",
                "👥 Lead conversion rates"
            ],
            "HR": [
                "👨‍💼 Employee count by department",
                "⭐ Average performance ratings",
                "📅 Attendance patterns",
                "🏃‍♂️ Turnover analysis"
            ],
            "Engineering": [
                "🏗️ System architecture overview",
                "🔧 Technical documentation",
                "📊 Development metrics",
                "🚀 Deployment processes"
            ]
        }
        
        user_role = st.session_state.user_info['role']
        role_queries = sample_queries.get(user_role, [
            "📚 Company policies",
            "🎯 How to apply for leave?",
            "💰 Reimbursement process",
            "🏢 Company benefits"
        ])
        
        for query in role_queries:
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                st.session_state.current_query = query.split(" ", 1)[1]  # Remove emoji
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_download_section(self):
        """Render chat history download section"""
        if not st.session_state.chat_history:
            return
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">💾 Export Chat</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 TXT", use_container_width=True, help="Download as text file"):
                txt_content = self.chat_utils.export_chat_to_txt(st.session_state.chat_history)
                filename = f"finsolve_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                st.download_button(
                    label="⬇️ Download TXT",
                    data=txt_content,
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True
                )
        
        with col2:
            if REPORTLAB_AVAILABLE:
                if st.button("📑 PDF", use_container_width=True, help="Download as PDF file"):
                    try:
                        pdf_content = self.chat_utils.export_chat_to_pdf(st.session_state.chat_history)
                        filename = f"finsolve_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_content,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
            else:
                if st.button("🌐 HTML", use_container_width=True, help="Download as HTML (can be saved as PDF)"):
                    try:
                        html_content = self.chat_utils.export_chat_to_pdf(st.session_state.chat_history)
                        filename = f"finsolve_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        
                        st.download_button(
                            label="⬇️ Download HTML",
                            data=html_content,
                            file_name=filename,
                            mime="text/html",
                            use_container_width=True
                        )
                        
                        st.info("💡 To convert to PDF: Open the HTML file in your browser and use 'Print → Save as PDF'")
                    except Exception as e:
                        st.error(f"HTML generation failed: {str(e)}")
        
        st.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
        
        if not REPORTLAB_AVAILABLE:
            st.markdown('<div style="font-size: 0.8em; color: #666; margin-top: 0.5rem;">💡 Install reportlab for direct PDF export: pip install reportlab</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def get_system_status(self):
        """Get system status from backend"""
        result = self.make_api_request("/system/status")
        if result.get("status"):
            st.session_state.system_status = result
        return result
    
    def send_chat_message(self, message: str) -> Dict[str, Any]:
        """Send chat message to backend"""
        return self.make_api_request("/chat", "POST", {
            "message": message,
            "user_name": st.session_state.user_info.get("name", "User")
        })
    
    def render_chat_message(self, message: Dict[str, Any], is_user: bool = False):
        """Render chat message with enhanced styling"""
        timestamp = self.chat_utils.format_timestamp(message.get('timestamp', ''))
        
        if is_user:
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-avatar user-avatar">👤</div>
                <div class="message-bubble user-bubble">
                    {message['content']}
                    <div class="message-info">
                        <span class="message-timestamp">{timestamp}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # AI message
            content = message.get('content', message)
            response_text = content.get('response', '')
            sources = content.get('sources', [])
            processing_time = content.get('processing_time', '')
            
            sources_display = ""
            if sources and st.session_state.show_sources:
                sources_list = ", ".join(sources)
                sources_display = f'<div class="message-sources" title="{sources_list}">📄 {len(sources)} sources</div>'
            
            processing_display = ""
            if processing_time:
                processing_display = f'<div class="message-timestamp">⏱️ {processing_time}</div>'
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-avatar assistant-avatar">🤖</div>
                <div class="message-bubble assistant-bubble">
                    {response_text}
                    <div class="message-info">
                        <div>{sources_display}</div>
                        <div>{processing_display}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        self.render_app_header()
        
        # Add user greeting
        st.markdown(f"""
        <div class="text-center mb-2">
            <span style="color: #667eea; font-weight: 500;">
                Welcome back, {st.session_state.user_info['name']} 
                ({st.session_state.user_info['role']})
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat container
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message['type'] == 'user':
                    self.render_chat_message({'content': message['content'], 'timestamp': message.get('timestamp')}, is_user=True)
                else:
                    self.render_chat_message({'content': message['content'], 'timestamp': message.get('timestamp')})
        else:
            # Welcome message
            st.markdown("""
            <div class="chat-message assistant-message">
                <div class="message-avatar assistant-avatar">🤖</div>
                <div class="message-bubble assistant-bubble">
                    👋 Hello! I'm your FinSolve AI Assistant. I can help you find information from company documents, 
                    analyze data, and answer questions based on your role permissions. 
                    <br><br>
                    Try asking me about company policies, financial data, or use the sample queries in the sidebar!
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "",
                placeholder="💬 Ask me anything about FinSolve...",
                key="user_input",
                value=st.session_state.get('current_query', ''),
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("🚀 Send", use_container_width=True, type="primary")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle input
        if (send_button or st.session_state.get('current_query')) and user_input.strip():
            self.handle_user_input(user_input.strip())
    
    def handle_user_input(self, user_input: str):
        """Handle user input and generate AI response"""
        # Clear the current query
        if 'current_query' in st.session_state:
            st.session_state.current_query = ''
        
        # Add user message to history
        user_message = {
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        # Show AI thinking indicator
        with st.spinner("🤖 AI is analyzing your request..."):
            result = self.send_chat_message(user_input)
            
            if result.get('success', False):
                # Add AI response to history
                ai_message = {
                    'type': 'assistant',
                    'content': result,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.chat_history.append(ai_message)
            else:
                # Add error response
                error_response = {
                    'response': "I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists.",
                    'sources': [],
                    'processing_time': "0.00s"
                }
                ai_message = {
                    'type': 'assistant',
                    'content': error_response,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.chat_history.append(ai_message)
        
        # Auto-scroll to bottom
        if st.session_state.auto_scroll:
            st.markdown("""
            <script>
                var chatContainer = document.getElementById('chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            </script>
            """, unsafe_allow_html=True)
        
        # Rerun to update the interface
        st.rerun()
    
    def run(self):
        """Main application runner"""
        # Initialize system status on first run
        if not st.session_state.system_status and st.session_state.authenticated:
            self.get_system_status()
        
        # Render appropriate interface
        if st.session_state.authenticated:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                self.render_chat_interface()
            
            with col2:
                self.render_sidebar()
        else:
            self.render_login_page()


# Main execution
if __name__ == "__main__":
    try:
        app = FinSolveChat()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or contact support.")
        
        # Show helpful information about dependencies
        if not REPORTLAB_AVAILABLE:
            st.warning("📦 Note: ReportLab is not installed. PDF export will use HTML format instead.")
            st.code("pip install reportlab", language="bash")