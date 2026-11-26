"""Mouse and keyboard control: a thin wrapper around pyautogui. Failures return False instead of raising (except FailSafe)."""
import logging
import os
import time

import pyautogui

log = logging.getLogger("gui_agent")


class DesktopController:
    def __init__(self, move_duration=0.3, paste_delay=0.15, wheel_unit=None):
        pyautogui.FAILSAFE = True            # moving the mouse into a screen corner triggers an emergency stop
        self.width, self.height = pyautogui.size()
        self.move_duration = move_duration
        self.paste_delay = paste_delay
        # On Windows pyautogui passes the scroll amount straight to the system as a wheel delta,
        # and one wheel notch is 120, so scroll(5) barely moves. We count in notches and multiply by 120 on Windows.
        self.wheel_unit = wheel_unit if wheel_unit is not None else (120 if os.name == "nt" else 1)

    def is_valid_coord(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def click(self, x, y, left_or_right="left", num_of_clicks=1):
        if not self.is_valid_coord(x, y):
            log.warning("click out of bounds: (%s, %s), screen %sx%s", x, y, self.width, self.height)
            return False
        try:
            pyautogui.moveTo(x, y, self.move_duration)
            pyautogui.click(button=left_or_right, clicks=num_of_clicks)
            return True
        except pyautogui.FailSafeException:
            raise                             # must propagate - never swallow the fail-safe
        except Exception as e:
            log.warning("click failed: %s", e)
            return False

    def type_text(self, text, interval=0.05, press_enter=False):
        text = "" if text is None else str(text)
        try:
            if text:
                if text.isascii():
                    pyautogui.typewrite(text, interval=interval)
                elif not self._paste(text):
                    return False
            if press_enter:
                pyautogui.press("enter")
            return True
        except pyautogui.FailSafeException:
            raise
        except Exception as e:
            log.warning("type_text failed: %s", e)
            return False

    def hotkey(self, *keys):
        """Press a key combination, e.g. hotkey("ctrl", "s"); a single key is a plain press."""
        keys = [str(k).lower() for k in keys if k]
        if not keys:
            return False
        try:
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            return True
        except pyautogui.FailSafeException:
            raise
        except Exception as e:
            log.warning("hotkey %s failed: %s", "+".join(keys), e)
            return False

    def scroll(self, notches, x=None, y=None):
        """Scroll the wheel: positive is up, negative is down, in notches. If coordinates are given, move there first (scrolling acts on the window under the mouse)."""
        try:
            if x is not None and y is not None:
                if not self.is_valid_coord(x, y):
                    return False
                pyautogui.moveTo(x, y, self.move_duration)
            pyautogui.scroll(int(notches) * self.wheel_unit)
            return True
        except pyautogui.FailSafeException:
            raise
        except Exception as e:
            log.warning("scroll failed: %s", e)
            return False

    def _paste(self, text):
        # typewrite cannot type Chinese, so non-ASCII text goes through the clipboard + Ctrl+V
        try:
            import pyperclip
        except ImportError:
            log.warning("typing non-ASCII text requires: pip install pyperclip")
            return False
        old = None
        try:
            old = pyperclip.paste()
        except Exception:
            pass
        pyperclip.copy(text)
        time.sleep(self.paste_delay)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(self.paste_delay)
        if old is not None:
            try:
                pyperclip.copy(old)           # restore the original clipboard content
            except Exception:
                pass
        return True


class DryRunController:
    """Logs only, never touches the mouse or keyboard. Used by --dry-run."""
    width, height = 1920, 1080

    def click(self, x, y, left_or_right="left", num_of_clicks=1):
        log.info("[dry-run] click (%s, %s) x%s", x, y, num_of_clicks)
        return True

    def type_text(self, text, interval=0.05, press_enter=False):
        log.info("[dry-run] type %r enter=%s", text, press_enter)
        return True

    def hotkey(self, *keys):
        log.info("[dry-run] hotkey %s", "+".join(keys))
        return True

    def scroll(self, notches, x=None, y=None):
        log.info("[dry-run] scroll %+d notches @ (%s, %s)", notches, x, y)
        return True
