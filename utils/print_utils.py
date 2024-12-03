DEBUG_ON = True

# Pretty text
def printc(text, color):
    colors = {
        "r": "\033[91m",    # Red
        "g": "\033[92m",    # Green
        "y": "\033[93m",    # Yellow
        "b": "\033[94m",    # Blue
        "p": "\033[95m",    # Purple
        "v": "\033[35m"     # Violet (Magenta/Purple)
    }
    end_color = "\033[0m"
    if not DEBUG_ON and color == 'p':
        return
    print(colors.get(color, ""), text, end_color)


# Emoji, because emoji
def pemji(name):
    emojis = {
        "gear": "\U00002699",          # Gear
        "lightning": "\U000026A1",     # High Voltage / Lightning
        "download": "\U00002B07",      # Down Arrow
        "rocket": "\U0001F680",        # Rocket
        "toolbox": "\U0001F9F0",       # Toolbox
        "flashlight": "\U0001F526",    # Flashlight
        "wrench": "\U0001F527",        # Wrench
        "red_square": "\U0001F7E5",    # Red Square
        "red_exclamation": "\U00002757", # Red Exclamation Mark
        "red_cross": "\U0000274C",     # Red Cross Mark
        "syringe": "\U0001F489",       # Syringe
        "trashcan": "\U0001F5D1",      # Trash Can
        "check_mark": "\U00002714",    # Check Mark
        "hourglass": "\U0000231B",     # Hourglass
        "fire": "\U0001F525",          # Fire
        "bomb": "\U0001F4A3",          # Bomb
        "bulb": "\U0001F4A1",          # Light Bulb
        "hammer": "\U0001F528",        # Hammer
        "clipboard": "\U0001F4CB",     # Clipboard
        "warning": "\U000026A0",       # Warning Sign
        "no_entry": "\U000026D4",      # No Entry
        "green_check": "\U00002705",   # Green Check Mark
        "hourglass_not_done": "\U000023F3",   # Hourglass Not Done
        "clipboard_with_check": "\U0001F5D2", # Clipboard With Check
        "magnifying_glass": "\U0001F50D",     # Magnifying Glass
        "battery": "\U0001F50B",       # Battery
        "electric_plug": "\U0001F50C", # Electric Plug
        "radioactive": "\U00002622",   # Radioactive
        "biohazard": "\U00002623",     # Biohazard
        "floppy_disk": "\U0001F4BE",   # Floppy Disk
        "cd": "\U0001F4BF",            # CD
        "dvd": "\U0001F4C0",           # DVD
        "lock": "\U0001F512",          # Lock
        "unlock": "\U0001F513",        # Unlock
        "battery_full": "\U0001F50B",  # Battery Full
        "battery_low": "\U0001F50B",   # Battery Low
        "no_battery": "\U00002620",    # No Battery
        "satellite": "\U0001F6F0",     # Satellite
        "antenna_bars": "\U0001F4F6",  # Antenna Bars
        "bell": "\U0001F514",          # Bell
        "bell_with_slash": "\U0001F515" # Bell with Slash
    }
    return emojis.get(name, "❓")  # If no emoji selection, return question mark emoji