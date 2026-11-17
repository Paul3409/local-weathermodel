"""Visual-only Streamlit theme for ELBTAL V2. No map data or image pixels changed."""
import streamlit as st

def apply():
    st.markdown("""
<style>
:root { --el-navy:#071c32; --el-panel:#102c46; --el-cyan:#38c8ee; --el-border:#275575; }
.stApp {
 background: radial-gradient(ellipse 110% 440px at 50% -100px, #163e5e 0%, #0a2139 52%, #07182b 100%) fixed;
 color:#edf6ff;
}
.block-container { max-width:1050px; padding-top:1.35rem; padding-bottom:3rem; }
h1,h2,h3 { letter-spacing:-.025em; color:#f3f8ff!important; }
h1 { font-size:clamp(1.65rem,4vw,2.3rem)!important; font-weight:800!important; }
h2,h3 { border-left:3px solid var(--el-cyan); padding-left:.75rem; }
[data-testid="stCaptionContainer"] { color:#abc4d8!important; }
div[data-testid="stPopover"] > button {
 background:linear-gradient(135deg,#113550,#0c2943)!important;
 border:1px solid #32617e!important;
 border-radius:12px!important;
 color:#f2f8ff!important;
 min-height:48px;
 font-weight:600;
 box-shadow:0 3px 12px #0000001a;
}
div[data-testid="stPopover"] > button:hover { border-color:#42c8eb!important; background:#164365!important; }
div[data-testid="stPopoverBody"] { background:#0c253d!important; border:1px solid #35617b!important; border-radius:12px!important; }
div[data-testid="stPopoverBody"] button { min-height:43px; text-align:left; }
div.stButton > button[kind="primary"] {
 background:linear-gradient(110deg,#078db5,#15afd2)!important;
 border:1px solid #59d6ee!important; border-radius:11px!important;
 color:white!important; font-weight:700; min-height:46px;
}
div.stButton > button[kind="secondary"] { border-radius:10px; border-color:#315b78; }
div[data-testid="stImage"] img { border-radius:9px; max-width:100%; }
hr { border-color:#284862!important; }
@media(max-width:640px) {
 .block-container { padding:1rem .8rem 2.5rem; }
 div[data-testid="stPopover"] > button { min-height:52px; }
 h1 { line-height:1.2!important; }
}
</style>
""", unsafe_allow_html=True)
