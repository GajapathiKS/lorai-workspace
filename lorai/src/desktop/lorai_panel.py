"""LorAI Desktop Panel — Sets wallpaper and manages desktop appearance.

Runs as a background daemon inside the LorAI container to maintain
the desktop environment's visual state.
"""

from __future__ import annotations

import os
import subprocess
import time

WALLPAPER_COLOR = "#1a1a2e"  # Dark blue-purple
DISPLAY = os.environ.get("DISPLAY", ":1")


def set_wallpaper():
    """Set a solid color wallpaper using xsetroot."""
    try:
        subprocess.run(
            ["xsetroot", "-solid", WALLPAPER_COLOR],
            env={**os.environ, "DISPLAY": DISPLAY},
            capture_output=True,
        )
        print(f"LorAI Panel: Wallpaper set to {WALLPAPER_COLOR}")
    except FileNotFoundError:
        print("LorAI Panel: xsetroot not found, skipping wallpaper.")


def create_desktop_shortcuts():
    """Create .desktop files on the user's Desktop."""
    desktop_dir = os.path.expanduser("~/Desktop")
    os.makedirs(desktop_dir, exist_ok=True)

    shortcuts = [
        {
            "name": "LorAI Terminal",
            "exec": "xterm -fa 'Monospace' -fs 12 -title 'LorAI Terminal' -e 'python3 /opt/lorai/src/ai-shell/lorai_terminal.py'",
            "icon": "utilities-terminal",
        },
        {
            "name": "File Manager",
            "exec": "pcmanfm",
            "icon": "system-file-manager",
        },
    ]

    for shortcut in shortcuts:
        path = os.path.join(desktop_dir, f"{shortcut['name'].replace(' ', '-')}.desktop")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(f"""[Desktop Entry]
Type=Application
Name={shortcut['name']}
Exec={shortcut['exec']}
Icon={shortcut['icon']}
Terminal=false
""")
            os.chmod(path, 0o755)

    print(f"LorAI Panel: Created {len(shortcuts)} desktop shortcuts.")


def main():
    """Initialize desktop and keep running."""
    set_wallpaper()
    create_desktop_shortcuts()

    # Keep alive — periodically refresh wallpaper in case it's reset
    while True:
        time.sleep(300)
        set_wallpaper()


if __name__ == "__main__":
    main()
