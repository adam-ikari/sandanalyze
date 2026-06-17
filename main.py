"""SandAnalyze - 沙粒形态分析系统入口。

Usage:
    sandanalyze              # Launch Streamlit web app
    sandanalyze --port 8080  # Custom port
"""

import sys
from streamlit.web import cli as stcli


def main() -> None:
    """Launch the SandAnalyze Streamlit application."""
    sys.argv = ["streamlit", "run", "app.py", *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
