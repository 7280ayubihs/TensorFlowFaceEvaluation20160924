# coding: utf-8
import os
import cv2


#
# カメラに接続して、’t’ が入力されたら、ファイルに保存する関数
#
# @param device_no カメラデバイスの番号（default: 0）
# @param size 取得するときのサイズ（default: カメラデバイスの最大サイズ）
# @param save_file 保存するファイルパス（default: picture.png）
#
# @return rtn_code 指定された device_no が無効な場合には -1 を返します。
#                     それ以外の場合には、0 を返します。
#
def take_picture(device_no=0, size=None, save_file='picture.png'):
    # カメラ接続
    cap = cv2.VideoCapture(device_no)
    if not cap.isOpened():
        return -1

    # 画面に表示
    while True:
        ret, frame = cap.read()

        # リサイズが指定されている場合はリサイズ
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # カメラの映像を表示
        cv2.imshow('take picture', frame)

        # キー入力
        key = cv2.waitKey(1)

        # 't' が入力されたら、画像を保存して終了
        if key == ord('t'):
            cv2.imwrite(save_file, frame)
            break

        # 'q' が入力されたら、画像を保存せずに終了
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0


#
# メイン関数
# ※動作確認用コード
#
if __name__ == "__main__":
    # 画像を保存するディレクトリを作成
    SAVE_DIR = "./data/"
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    take_picture(1, (400, 300), SAVE_DIR + 'cap.jpg')
