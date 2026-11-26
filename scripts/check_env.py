"""Environment check: can we read the screen size, move the mouse and take a screenshot? Run it after installing."""
import cv2
import numpy as np
import pyautogui


def main():
    w, h = pyautogui.size()
    print(f"screen resolution: {w} x {h}")

    pyautogui.moveTo(w // 2, h // 2, duration=1.0)
    print(f"mouse moved to the screen centre ({w // 2}, {h // 2})")

    shot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_screenshot.png", img)
    print("screenshot saved as test_screenshot.png")


if __name__ == "__main__":
    main()
