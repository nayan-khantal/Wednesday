import sys
import webbrowser
import uvicorn
from threading import Timer


def open_browser():
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🕷️  Wednesday Addams — Web Interface")
    print("=" * 50)
    print("Launching server at: http://127.0.0.1:8000")
    print("Opening your browser...\n")

    # Open browser 1.5 seconds after launching server
    Timer(1.5, open_browser).start()

    uvicorn.run(
        "src.web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
