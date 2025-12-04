from dotenv import load_dotenv
load_dotenv()

import argparse
from src.webui.interface import theme_map, create_ui
import gradio as gr


def main():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7800, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    # Create UI
    demo = create_ui(theme_name=args.theme)

    # Launch safely (no queue = no port crash)
    demo.launch(
        server_name=args.ip,
        server_port=args.port,
        show_error=True
    )


if __name__ == '__main__':
    main()
