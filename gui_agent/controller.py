"""鼠标键盘控制，pyautogui 的一层薄封装。失败一律返回 False，不往外抛（FailSafe 除外）。"""
import logging
import os
import time

import pyautogui

log = logging.getLogger("gui_agent")


class DesktopController:
    def __init__(self, move_duration=0.3, paste_delay=0.15, wheel_unit=None):
        pyautogui.FAILSAFE = True            # 鼠标甩到屏幕角落就紧急停止
        self.width, self.height = pyautogui.size()
        self.move_duration = move_duration
        self.paste_delay = paste_delay
        # pyautogui 在 Windows 上把 scroll 的参数直接当滚轮增量传给系统，
        # 而系统里一格是 120，scroll(5) 几乎不动。所以这里按"格"来算，Windows 上乘 120
        self.wheel_unit = wheel_unit if wheel_unit is not None else (120 if os.name == "nt" else 1)

    def is_valid_coord(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def click(self, x, y, left_or_right="left", num_of_clicks=1):
        if not self.is_valid_coord(x, y):
            log.warning("click 越界: (%s, %s)，屏幕 %sx%s", x, y, self.width, self.height)
            return False
        try:
            pyautogui.moveTo(x, y, self.move_duration)
            pyautogui.click(button=left_or_right, clicks=num_of_clicks)
            return True
        except pyautogui.FailSafeException:
            raise                             # 这个必须一路抛上去，不能吞
        except Exception as e:
            log.warning("click 失败: %s", e)
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
            log.warning("type_text 失败: %s", e)
            return False

    def hotkey(self, *keys):
        """按组合键，比如 hotkey("ctrl", "s")；只有一个键就是 press。"""
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
            log.warning("hotkey %s 失败: %s", "+".join(keys), e)
            return False

    def scroll(self, notches, x=None, y=None):
        """滚轮，正数向上、负数向下，单位是"格"。给了坐标就先把鼠标移过去（滚动作用在鼠标下面的窗口）。"""
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
            log.warning("scroll 失败: %s", e)
            return False

    def _paste(self, text):
        # typewrite 敲不出中文，所以非 ASCII 的内容走剪贴板 + Ctrl+V
        try:
            import pyperclip
        except ImportError:
            log.warning("要输入非 ASCII 文字，需要先 pip install pyperclip")
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
                pyperclip.copy(old)           # 把原来的剪贴板内容放回去
            except Exception:
                pass
        return True


class DryRunController:
    """只打日志，不动鼠标键盘。--dry-run 用。"""
    width, height = 1920, 1080

    def click(self, x, y, left_or_right="left", num_of_clicks=1):
        log.info("[dry-run] 点击 (%s, %s) x%s", x, y, num_of_clicks)
        return True

    def type_text(self, text, interval=0.05, press_enter=False):
        log.info("[dry-run] 输入 %r enter=%s", text, press_enter)
        return True

    def hotkey(self, *keys):
        log.info("[dry-run] 按键 %s", "+".join(keys))
        return True

    def scroll(self, notches, x=None, y=None):
        log.info("[dry-run] 滚动 %+d 格 @ (%s, %s)", notches, x, y)
        return True
