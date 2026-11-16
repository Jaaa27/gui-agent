"""环境自检：能不能拿到屏幕尺寸、移动鼠标、截图。装完依赖先跑一下。"""
import cv2
import numpy as np
import pyautogui


def main():
    w, h = pyautogui.size()
    print(f"屏幕分辨率: {w} x {h}")

    pyautogui.moveTo(w // 2, h // 2, duration=1.0)
    print(f"鼠标已移到屏幕中央 ({w // 2}, {h // 2})")

    shot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_screenshot.png", img)
    print("截图已保存为 test_screenshot.png")


if __name__ == "__main__":
    main()
