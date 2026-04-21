"""style_loader.py — place next to app.py"""
import streamlit as st
from pathlib import Path


def inject_styles():
    css_path = Path(__file__).parent / "style.css"
    fonts = '<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">'
    st.markdown(fonts, unsafe_allow_html=True)
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_theme_toggle():
    """Inject the floating dark/light theme toggle button + JS."""
    st.markdown("""
    <div class="theme-toggle-wrap">
        <button class="theme-toggle-btn" id="themeToggleBtn" onclick="toggleTheme()">
            <span id="themeIcon">☀️</span>
            <span id="themeLabel">Light</span>
        </button>
    </div>

    <script>
    (function() {
        const root = document.documentElement;
        const stored = localStorage.getItem('skyrisk-theme') || 'dark';
        applyTheme(stored);

        function applyTheme(theme) {
            root.setAttribute('data-theme', theme);
            localStorage.setItem('skyrisk-theme', theme);
            const icon  = document.getElementById('themeIcon');
            const label = document.getElementById('themeLabel');
            if (icon)  icon.textContent  = theme === 'dark' ? '☀️' : '🌙';
            if (label) label.textContent = theme === 'dark' ? 'Light' : 'Dark';
        }

        window.toggleTheme = function() {
            const current = root.getAttribute('data-theme') || 'dark';
            applyTheme(current === 'dark' ? 'light' : 'dark');
        };

        // Re-apply after Streamlit re-renders
        const observer = new MutationObserver(() => {
            const t = localStorage.getItem('skyrisk-theme') || 'dark';
            if (root.getAttribute('data-theme') !== t) applyTheme(t);
            const icon  = document.getElementById('themeIcon');
            const label = document.getElementById('themeLabel');
            if (icon)  icon.textContent  = t === 'dark' ? '☀️' : '🌙';
            if (label) label.textContent = t === 'dark' ? 'Light' : 'Dark';
        });
        observer.observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """, unsafe_allow_html=True)
